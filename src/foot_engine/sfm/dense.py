"""Sparse SfM 결과 기반 OpenMVS Dense 포인트클라우드/메쉬 복원 모듈.

OpenMVS CLI 실행파일 필요(`OPENMVS_BIN_DIR`), CPU 빌드로 동작.

흐름:
    undistort_for_dense() -> convert_masks_for_openmvs() -> run_interface_colmap()
    -> run_densify_point_cloud() -> clean_dense_point_cloud()
    -> (선택) filter_by_reprojection_consistency()/filter_grazing_points()/
       filter_point_cloud_visibility()+restore_point_cloud_views()
    -> run_reconstruct_mesh() -> (선택) run_refine_mesh()
    -> mesh_postprocess.postprocess_mesh() -- 배경/파편 제거 + 스무딩(별도 모듈,
       사진/카메라 정보 불필요 -- 완성된 STL 등에도 독립적으로 재적용 가능)

사진 -> 메쉬 생성(이 모듈 고유 부분)과 배경/파편 제거·스무딩(메쉬만 입력받는
`mesh_postprocess.py`)은 서로 다른 모듈로 분리돼 있다 -- 후자는
`scripts/postprocess_mesh.py`로 기존 STL에 단독으로도 돌릴 수 있다.

기본값 메모:
- 마스크는 dilate=0으로 densify 전에 적용.
- 이상치 제거는 통계적 방식만(DBSCAN·fusion 합의 강도는 안 씀 — 성긴 진짜
  표면까지 지움).
- 메쉬 생성은 스무딩 OFF, 최대 연결 요소만 유지.
- 축 정렬은 `find_sole_direction()` — 접점이 가장 많은 방향을 발바닥(-Y)으로.
- `filter_by_reprojection_consistency()`는 켜지 말 것 — 관측 카메라가 적은
  부위(발등/발뒤꿈치 등)를 통째로 지울 수 있다.
- `RefineMesh`는 `decimate=1`(단순화 끔) 기본.
- 중간 산출물은 기본 정리(`keep_intermediates=True`로 보존).
- `prune_protrusions`은 실행마다 결과가 흔들려 기본 꺼짐.

알려진 한계:
- 발바닥 미촬영 프로토콜 특성상 접지면에 큰 구멍이 남는다.
- 발목 부근 뿔/스파이크 결함이 남을 수 있다.
- 실루엣 경계 깊이 노이즈(flying pixel)가 완전히 제거되지 않는다.
- 오목 부위(아치/뒤꿈치)에 관측 부족 크레이터가 남을 수 있다.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import trimesh

from .geometry import measured_length, pca_axes
from .mesh_postprocess import (
    DEFAULT_CURVATURE_ALPHA,
    DEFAULT_CURVATURE_ITERATIONS,
    DEFAULT_CURVATURE_MAX_RADIUS_MULT,
    DEFAULT_CURVATURE_MIN_RADIUS_MULT,
    DEFAULT_CURVATURE_MU,
    DEFAULT_CURVATURE_PERCENTILE,
    fill_small_holes,
    finish_smooth_mesh,
    keep_largest_component,
    postprocess_mesh,
    prune_thin_protrusions,
    sand_surface,
    smooth_high_curvature_regions,
)
from .reconstruction import filter_outlier_points

#: OpenMVS CLI 실행파일이 있는 폴더. 환경변수로 덮어쓸 것 -- 설치 방법은 README 참고.
DEFAULT_OPENMVS_BIN_DIR = os.environ.get("OPENMVS_BIN_DIR", "")

#: `DensifyPointCloud`의 간헐적 네이티브 크래시를 피하기 위한 스레드 상한.
DEFAULT_MAX_THREADS = 8

#: `--postprocess-dmaps` 비트마스크: 1=remove-speckles, 2=fill-gaps.
#: 저텍스처 평면(발등/발바닥)의 깊이 추정 공백을 일부 메운다.
DEFAULT_POSTPROCESS_DMAPS = 3

#: 스케일 보정 기준 삼는 자기신고 발길이가 없을 때 쓰는 임시값(mm). 절대
#: 축척이 아니라 형태 비교/시각화용 placeholder.
DEFAULT_REFERENCE_LENGTH_MM = 250.0


def _resolve_openmvs_bin(openmvs_bin: str | Path | None) -> Path:
    # Path("").is_dir()는 "."(cwd)로 취급돼 늘 True다 -- 값이 실제로 비어있는지는
    # Path로 감싸기 전에 문자열로 따로 걸러내야 한다.
    bin_str = str(openmvs_bin) if openmvs_bin else DEFAULT_OPENMVS_BIN_DIR
    resolved = Path(bin_str) if bin_str else None
    if resolved is None or not resolved.is_dir():
        where = str(resolved) if resolved is not None else "(지정되지 않음)"
        raise FileNotFoundError(
            f"OpenMVS 실행파일 폴더를 찾을 수 없습니다: {where} -- "
            "OPENMVS_BIN_DIR 환경변수를 설정하거나 openmvs_bin 인자로 직접 넘기세요. "
            "설치 방법은 README의 'Dense MVS(선택)' 절 참고."
        )
    return resolved


def largest_sparse_dir(sparse_root: Path) -> Path:
    """`sparse/0`, `sparse/1`, ... 중 등록 이미지가 가장 많은 폴더를 찾는다.

    번호가 항상 크기순은 아니므로 `sparse/0`을 무조건 가정하면 안 된다.
    """
    best_dir, best_n = None, -1
    for d in sorted(sparse_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            n = pycolmap.Reconstruction(d).num_reg_images()
        except Exception:
            continue
        if n > best_n:
            best_dir, best_n = d, n
    if best_dir is None:
        raise FileNotFoundError(f"유효한 sparse 재구성을 찾을 수 없습니다: {sparse_root}")
    return best_dir


def undistort_for_dense(sparse_dir: Path, images_dir: Path, output_dir: Path) -> Path:
    """sparse 카메라를 PINHOLE(왜곡보정)로 바꿔 OpenMVS가 기대하는 COLMAP dense
    워크스페이스(`images/`, `sparse/`, `stereo/`)를 만든다.

    `pycolmap.undistort_images()`로 처리한다 -- 별도 `colmap.exe` 실행파일이
    필요 없다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # pycolmap이 상위 폴더까지는 안 만들어줌
    pycolmap.undistort_images(
        output_path=str(output_dir),
        input_path=str(sparse_dir),
        image_path=str(images_dir),
        output_type="COLMAP",
    )
    return output_dir


def convert_masks_for_openmvs(masks_dir: Path, out_dir: Path) -> Path:
    """`masking.py` 마스크(`<원본파일명>.png`)를 OpenMVS 명명 규칙
    (`<확장자 뺀 파일명>.mask.png`)으로 복사한다.

    OpenMVS의 `Util::getFileName()`이 확장자를 뗀 이름 뒤에 `.mask.png`를
    붙여 마스크를 찾으므로(소스 확인), 원본 마스크 파일명 그대로는 못 찾는다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in masks_dir.glob("*.png"):
        # "frame_00000.jpg.png" -> "frame_00000" (원본 확장자까지 통째로 뗀다)
        stem = f.name
        for ext in (".jpg.png", ".jpeg.png", ".png.png", ".bmp.png"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        else:
            stem = f.stem
        (out_dir / f"{stem}.mask.png").write_bytes(f.read_bytes())
        n += 1
    print(f"[dense] 마스크 {n}개를 OpenMVS 명명 규칙으로 변환: {out_dir}")
    return out_dir


def _run_openmvs(
    exe_name: str,
    args: list[str],
    workdir: Path,
    openmvs_bin: Path,
    log_name: str,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(openmvs_bin / exe_name)] + args + ["-w", str(workdir), "-v", "3"]
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    log_path = workdir / log_name
    log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"{exe_name} 실패 (exit {result.returncode}) -- 상세 로그: {log_path} "
            f"(OpenMVS 자체 로그는 workdir의 {exe_name}-*.log 참고)"
        )


def run_interface_colmap(dense_dir: Path, workdir: Path, *, openmvs_bin: str | Path | None = None) -> Path:
    """COLMAP dense 워크스페이스를 OpenMVS 씬 파일(`scene.mvs`)로 변환한다."""
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    _run_openmvs(
        "InterfaceCOLMAP.exe",
        ["-i", str(Path(dense_dir).resolve()), "-o", "scene.mvs"],
        workdir, bin_dir, "log_interface_colmap.txt",
    )
    return workdir / "scene.mvs"


def run_densify_point_cloud(
    scene_mvs: Path,
    workdir: Path,
    *,
    masks_dir: Path | None = None,
    openmvs_bin: str | Path | None = None,
    max_threads: int = DEFAULT_MAX_THREADS,
    postprocess_dmaps: int = DEFAULT_POSTPROCESS_DMAPS,
    resolution_level: int | None = None,
    number_views_fuse: int | None = None,
    output_name: str = "scene_dense.mvs",
) -> Path:
    """마스크 기반 dense 포인트클라우드를 만든다 (`scene_dense.ply` + `.mvs`).

    Args:
        masks_dir: `convert_masks_for_openmvs()`로 변환된(=`.mask.png`
            규칙) 마스크 폴더. 지정하면 배경 픽셀은 깊이 계산 자체를
            생략한다. `masking.generate_masks()`를 `dilate=0`으로 호출해
            만들 것.
        postprocess_dmaps: 기본 3(remove-speckles+fill-gaps). 저텍스처 평면
            깊이 공백을 메운다. 0이면 비활성.
        resolution_level: 뎁스맵 계산 전 이미지를 몇 단계 축소할지(0=원본
            해상도, 숫자가 클수록 더 축소돼 빠르지만 점군이 성겨진다).
            `None`(기본)이면 OpenMVS 자체 기본값을 그대로 쓴다.
        number_views_fuse: 점 하나를 살리는 데 필요한 최소 동의 뷰 개수.
            `None`(기본)이면 OpenMVS 자체 기본값(2).
    """
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    args = [scene_mvs.name, "-o", output_name, "--max-threads", str(max_threads)]
    if postprocess_dmaps:
        args += ["--postprocess-dmaps", str(postprocess_dmaps)]
    if resolution_level is not None:
        args += ["--resolution-level", str(resolution_level)]
    if number_views_fuse is not None:
        args += ["--number-views-fuse", str(number_views_fuse)]
    if masks_dir is not None:
        args += ["--mask-path", str(Path(masks_dir).resolve()) + os.sep, "--ignore-mask-label", "0"]
    _run_openmvs("DensifyPointCloud.exe", args, workdir, bin_dir, "log_densify.txt")
    return workdir / output_name.replace(".mvs", ".ply")


def _parse_dense_ply(ply_path: Path) -> tuple[str, bytes, np.ndarray, np.ndarray]:
    """OpenMVS dense PLY(xyz+rgb+normal+view_indices+view_weights)를 레코드
    단위로 파싱한다. 고정 필드 27바이트 뒤로 가변 길이 리스트(view_indices,
    view_weights)가 이어지는 스키마 — `trimesh`/`open3d` 라운드트립은 이
    필드를 날려버려서 원본 바이트를 직접 다룬다.

    Returns:
        (header 텍스트, body 바이트열, 레코드 경계 offset 배열(N+1,), xyz 배열(N,3)).
    """
    data = ply_path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    body = data[header_end:]
    n = int([line for line in header.splitlines() if line.startswith("element vertex")][0].split()[-1])

    fixed_size = 27
    offsets = np.empty(n + 1, dtype=np.int64)
    xyz = np.empty((n, 3), dtype=np.float32)
    pos = 0
    for i in range(n):
        offsets[i] = pos
        xyz[i] = struct.unpack_from("<3f", body, pos)
        pos += fixed_size
        n_idx = body[pos]
        pos += 1 + 4 * n_idx
        n_w = body[pos]
        pos += 1 + 4 * n_w
    offsets[n] = pos
    if pos != len(body):
        raise ValueError(
            f"{ply_path} 파싱 실패 -- 예상 스키마(xyz+rgb+normal+view_indices"
            "+view_weights)와 다른 형식일 수 있습니다."
        )
    return header, body, offsets, xyz


def _extract_normals_and_views(body: bytes, offsets: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """`_parse_dense_ply()`가 찾아둔 레코드 경계로 법선과 view_indices를 추가로 뽑는다.

    Returns:
        (법선 배열(N,3), 점마다 다른 길이의 view_indices 배열 리스트(N개)).
    """
    n = len(offsets) - 1
    normals = np.empty((n, 3), dtype=np.float32)
    views: list[np.ndarray] = []
    for i in range(n):
        pos = offsets[i]
        normals[i] = struct.unpack_from("<3f", body, pos + 15)  # xyz(12)+rgb(3) 다음
        n_idx = body[pos + 27]
        if n_idx:
            views.append(np.frombuffer(body, dtype="<u4", count=n_idx, offset=pos + 28).copy())
        else:
            views.append(np.empty(0, dtype=np.uint32))
    return normals, views


def filter_grazing_points(
    dense_ply_path: Path,
    sparse_dir: Path,
    out_path: Path,
    *,
    min_score: float = 0.3,
) -> tuple[int, int]:
    """표면 법선이 관측 카메라 시선과 거의 접선(grazing)인 점을 제거한다.

    occlusion 경계의 "flying pixel" 노이즈를 겨냥한 필터 — 어느 카메라에서
    봐도 표면이 옆으로 누워 보이는 점을 지운다.

    Args:
        dense_ply_path: view_indices/view_weights/normal 필드가 있는 dense PLY.
        sparse_dir: OpenMVS가 실제로 쓴 언디스토션된 sparse 재구성
            (`undistort_for_dense()` 출력) — `run_sparse_sfm()`의 원본이
            아님(카메라 인덱스가 다르게 매겨짐).
        min_score: 관측된 모든 뷰에 대한 평균 |cos(법선, 시선방향)|이 이
            미만이면 제거(0=접선, 1=정면). 관측 정보 없는 점은 유지.

    Returns:
        (원본 점 개수, 유지된 점 개수).
    """
    header, body, offsets, xyz = _parse_dense_ply(dense_ply_path)
    normals, views = _extract_normals_and_views(body, offsets)
    n = len(xyz)

    recon = pycolmap.Reconstruction(str(sparse_dir))
    imgs_sorted = sorted(recon.images.items())
    camera_centers = np.array([img.projection_center() for _, img in imgs_sorted])

    scores = np.ones(n, dtype=np.float32)  # 관측 정보 없는 점은 "정면"으로 취급(제거 안 함)
    for i in range(n):
        v = views[i]
        if len(v) == 0:
            continue
        dirs = camera_centers[v] - xyz[i]
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        scores[i] = np.abs(dirs @ normals[i]).mean()

    keep = scores >= min_score
    kept_n = int(keep.sum())
    print(
        f"[dense] grazing-angle 필터: {n:,} -> {kept_n:,}개 ({kept_n/n:.1%} 유지, "
        f"min_score={min_score})"
    )

    chunks = [body[offsets[i]:offsets[i + 1]] for i in np.where(keep)[0]]
    new_body = b"".join(chunks)
    new_header = header.replace(f"element vertex {n}\n", f"element vertex {kept_n}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(new_header.encode("ascii") + new_body)
    return n, kept_n


def filter_by_reprojection_consistency(
    dense_ply_path: Path,
    sparse_dir: Path,
    masks_dir: Path,
    out_path: Path,
    *,
    min_vote_ratio: float = 0.6,
) -> tuple[int, int]:
    """씬의 카메라 전부(점의 원래 관측 뷰가 아니라)에 재투영해 최종 마스크와
    동의하는 비율이 낮은 점을 제거한다.

    특정 프레임 하나의 마스크 오분류(예: 배경 물체가 사람/피부로 오분류)에
    낚이지 않고, 다수 뷰의 합의를 따르는 필터.

    Args:
        masks_dir: `masking.generate_masks()` 원본 마스크 폴더(OpenMVS 변환
            전, `<원본파일명>.png`).
        min_vote_ratio: 이 미만이면 제거(0~1). 관측 카메라 수가 적은 부위
            (발등/발뒤꿈치 등)를 통째로 지울 수 있다 -- 일반적으로 켜지 말 것.

    Returns:
        (원본 점 개수, 유지된 점 개수).
    """
    header, body, offsets, xyz = _parse_dense_ply(dense_ply_path)
    n = len(xyz)

    recon = pycolmap.Reconstruction(str(sparse_dir))
    imgs_sorted = sorted(recon.images.items())

    votes = np.zeros(n, dtype=np.float32)
    totals = np.zeros(n, dtype=np.float32)
    for _, img in imgs_sorted:
        mask_path = masks_dir / f"{img.name}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape
        cam = recon.cameras[img.camera_id]
        cfw = img.cam_from_world()
        pts_cam = xyz @ cfw.rotation.matrix().T + cfw.translation
        z = pts_cam[:, 2]
        valid = z > 1e-6
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        u = np.full(n, -1.0)
        v = np.full(n, -1.0)
        u[valid] = pts_cam[valid, 0] / z[valid] * fx + cx
        v[valid] = pts_cam[valid, 1] / z[valid] * fy + cy
        in_bounds = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui = u[in_bounds].astype(int)
        vi = v[in_bounds].astype(int)
        fg = np.zeros(n, dtype=bool)
        fg[in_bounds] = mask[vi, ui] > 0
        totals[in_bounds] += 1
        votes[in_bounds] += fg[in_bounds]

    vote_ratio = np.divide(votes, totals, out=np.zeros_like(votes), where=totals > 0)
    keep = vote_ratio >= min_vote_ratio
    kept_n = int(keep.sum())
    print(
        f"[dense] 재투영 다수결 필터: {n:,} -> {kept_n:,}개 ({kept_n/n:.1%} 유지, "
        f"min_vote_ratio={min_vote_ratio})"
    )

    chunks = [body[offsets[i]:offsets[i + 1]] for i in np.where(keep)[0]]
    new_body = b"".join(chunks)
    new_header = header.replace(f"element vertex {n}\n", f"element vertex {kept_n}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(new_header.encode("ascii") + new_body)
    return n, kept_n


def clean_dense_point_cloud(
    dense_ply_path: Path,
    out_path: Path,
    *,
    k: int = 8,
    std_ratio: float = 2.0,
    prune_protrusions: bool = False,
    max_protrusion_ratio: float = 0.08,
) -> tuple[int, int]:
    """dense 포인트클라우드에서 통계적 이상치를 제거한다 (DBSCAN 사용 안 함).

    view_indices/view_weights 필드를 원본 바이트 그대로 보존해야 하는 이유는
    `_parse_dense_ply()` 참고. DBSCAN(최대 군집 유지)은 의도적으로 안 쓴다.

    Args:
        prune_protrusions: 국소 밀도 기준으로 뿔/스파이크 후보 점을 미리
            지운다. 판정 기준(`_protrusion_remove_mask()`)이 그 점군 자체의
            상대적 분포라서, SfM/MVS 재구성의 실행 간 편차만으로 얼마나
            지워지는지 크게 흔들린다 -- 기본 꺼짐.
        max_protrusion_ratio: `prune_protrusions`가 이 비율보다 많은 점을
            지우려 하면 판정 기준 자체가 흔들린 것으로 보고 이번 실행에서는
            통째로 건너뛴다(`cleaning.py`의 `max_plane_ratio`와 같은 패턴).

    Returns:
        (원본 점 개수, 정리 후 점 개수).
    """
    header, body, offsets, xyz = _parse_dense_ply(dense_ply_path)
    n = len(xyz)

    inliers = filter_outlier_points(xyz, k=k, std_ratio=std_ratio)
    stat_kept_n = int(inliers.sum())
    print(f"[dense] 점단위 정리(통계적 이상치 제거): {n:,} -> {stat_kept_n:,} ({stat_kept_n/n:.1%} 유지)")

    if prune_protrusions:
        # 통계적 이상치 제거로 남은 점들 기준으로 뿔/스파이크 판정(전체 배경
        # 노이즈가 섞인 상태에서 밀도를 재면 뿔 판정 기준 자체가 흔들린다).
        protrusion_mask_subset = _protrusion_remove_mask(
            xyz[inliers], adjacency_edges=None,
            density_radius_nn_mult=4.0, far_percentile=97.0, density_ratio=0.6,
        )
        n_protrusion = int(protrusion_mask_subset.sum())
        if n_protrusion > stat_kept_n * max_protrusion_ratio:
            print(
                f"[dense] 뿔/스파이크 판정이 점 {n_protrusion:,}개"
                f"({n_protrusion/stat_kept_n:.1%})를 지우려 함 -- "
                f"안전장치(max_protrusion_ratio={max_protrusion_ratio:.0%}) 초과로 이번엔 건너뜀"
            )
        elif n_protrusion:
            inlier_idx = np.where(inliers)[0]
            inliers[inlier_idx[protrusion_mask_subset]] = False
            print(f"[dense] 뿔/스파이크 제거: 점 {n_protrusion:,}개")

    kept_n = int(inliers.sum())
    chunks = [body[offsets[i]:offsets[i + 1]] for i in np.where(inliers)[0]]
    new_body = b"".join(chunks)
    new_header = header.replace(f"element vertex {n}\n", f"element vertex {kept_n}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(new_header.encode("ascii") + new_body)
    return n, kept_n


def run_reconstruct_mesh(
    scene_mvs: Path,
    point_cloud_ply: Path,
    workdir: Path,
    *,
    openmvs_bin: str | Path | None = None,
    smooth: int = 0,
    free_space_support: bool = False,
    thickness_factor: float = 1.0,
    quality_factor: float = 1.0,
    output_name: str = "scene_mesh.mvs",
) -> Path:
    """Delaunay 사면체화 + 그래프컷으로 점군을 메쉬로 만든다.

    Args:
        smooth: 기본 0(끔) -- OpenMVS 기본값(2)은 형상 정교함을 깎아낸다.
        free_space_support / thickness_factor / quality_factor: 그래프컷
            가중치 튜닝. 기본값은 OpenMVS 자체 기본값 그대로(꺼짐/1.0).
    """
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    args = [
        scene_mvs.name, "-p", point_cloud_ply.name, "-o", output_name, "--smooth", str(smooth),
        "--thickness-factor", str(thickness_factor), "--quality-factor", str(quality_factor),
    ]
    if free_space_support:
        args += ["--free-space-support", "1"]
    _run_openmvs("ReconstructMesh.exe", args, workdir, bin_dir, "log_reconstruct_mesh.txt")
    return workdir / output_name.replace(".mvs", ".ply")


def filter_point_cloud_visibility(
    scene_mvs: Path,
    point_cloud_ply: Path,
    workdir: Path,
    *,
    threshold: int = -1,
    openmvs_bin: str | Path | None = None,
    output_name: str = "scene_dense_vf",
) -> Path:
    """OpenMVS 내장 가시성 기반 점군 필터. `DensifyPointCloud`를 필터 전용
    모드로 재호출한다(densify와 같은 호출에 넣으면 무효 -- 별도 2차 호출 필요).

    **주의**: 출력 PLY는 view_indices/view_weights 필드가 없다 --
    `run_reconstruct_mesh()`에 그대로 넘기면 크래시하니 `restore_point_cloud_views()`로
    복원 후 넘길 것(`run_dense_pipeline()`은 이미 그렇게 엮여 있음).

    Args:
        point_cloud_ply: 필터링할 기존 점군(view 필드 있어야 함).
        threshold: 음수만 필터 발동(절댓값 클수록 공격적).

    Returns:
        필터링된(view 필드 없는) 점군 경로.
    """
    if threshold >= 0:
        raise ValueError(f"threshold는 음수여야 필터가 발동합니다(소스 확인) -- 받은 값: {threshold}")
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    _run_openmvs(
        "DensifyPointCloud.exe",
        [scene_mvs.name, "-p", point_cloud_ply.name, "-o", output_name, "--filter-point-cloud", str(threshold)],
        workdir, bin_dir, "log_filter_point_cloud.txt",
    )
    return workdir / f"{output_name}_filtered.ply"


def restore_point_cloud_views(
    view_source_ply: Path,
    filtered_ply: Path,
    out_path: Path,
) -> tuple[int, int]:
    """`filter_point_cloud_visibility()`가 지운 view 필드를 좌표 매칭으로 복원한다.

    필터는 좌표를 안 건드리고 부분집합만 골라내므로, 필터를 통과한 각 점을
    `view_source_ply`(필터 입력, view 필드 있음)에서 좌표로 찾아 그 레코드를
    그대로 복사해 붙인다.

    Args:
        view_source_ply: 필터의 입력으로 쓴, view 필드가 있는 점군.
        filtered_ply: `filter_point_cloud_visibility()`의 출력.
        out_path: view 필드가 복원된 결과 -- `run_reconstruct_mesh()`에 넘길 것.

    Returns:
        (필터링된 점 개수, 좌표 매칭에 성공해 복원된 점 개수).
    """
    header, body, offsets, xyz = _parse_dense_ply(view_source_ply)
    xyz_to_idx = {(float(x), float(y), float(z)): i for i, (x, y, z) in enumerate(xyz)}

    filt_data = filtered_ply.read_bytes()
    filt_header_end = filt_data.index(b"end_header\n") + len(b"end_header\n")
    filt_header = filt_data[:filt_header_end].decode("ascii")
    filt_body = filt_data[filt_header_end:]
    filt_n = int([line for line in filt_header.splitlines() if line.startswith("element vertex")][0].split()[-1])

    FIXED_SIZE = 27  # xyz+rgb+normal만 있는 필터 출력 스키마(view 필드 없음)
    kept_records: list[bytes] = []
    missing = 0
    for i in range(filt_n):
        x, y, z = struct.unpack_from("<3f", filt_body, i * FIXED_SIZE)
        idx = xyz_to_idx.get((float(x), float(y), float(z)))
        if idx is None:
            missing += 1
            continue
        kept_records.append(body[offsets[idx]:offsets[idx + 1]])
    if missing:
        print(
            f"[dense][경고] 필터 결과 {missing}/{filt_n}개 점이 원본과 좌표 매칭 실패 -- "
            "view 필드 없이는 못 살리므로 건너뜀(예상외로 많으면 filter_point_cloud_visibility()가 "
            "좌표를 실제로 바꾸는 다른 OpenMVS 버전일 수 있음, 재검증 필요)."
        )

    new_body = b"".join(kept_records)
    new_header = header.replace(
        f"element vertex {len(xyz)}\n", f"element vertex {len(kept_records)}\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(new_header.encode("ascii") + new_body)
    return filt_n, len(kept_records)


def run_refine_mesh(
    scene_mvs: Path,
    mesh_ply: Path,
    workdir: Path,
    *,
    openmvs_bin: str | Path | None = None,
    decimate: float = 1.0,
    regularity_weight: float | None = None,
    resolution_level: int = 0,
    scales: int = 2,
    output_name: str = "scene_mesh_refined.mvs",
) -> Path:
    """사진 광도일관성 기반으로 메쉬 정점 위치를 보정한다 (선택적, 가장 느린 단계).

    CPU로도 동작한다(`--cuda-device` 기본값 -2). 전체 파이프라인 소요시간의
    대부분을 차지하는 병목이다 -- 빠른 반복 실험에서는 건너뛸 것.

    Args:
        decimate: refine 전 입력 메쉬 단순화 정도(0~1). OpenMVS 기본값 0은
            "auto"로 공격적으로 단순화한다 -- 이 함수 기본값은 1(단순화 끔,
            해상도 보존).
        regularity_weight: photo-consistency 대 표면 정규화(smoothness) 항
            가중치. `None`이면 OpenMVS 기본값(0.2) 사용.
        resolution_level: 계산 전 이미지를 몇 단계 축소할지(기본 0=원본
            해상도). 소요 시간에 가장 직접적으로 영향 -- 1~2로 올리면
            빨라지는 대신 정밀도가 낮아진다.
        scales: 다단계 최적화 반복 횟수(기본 2). 줄이면 빨라지는 대신
            거칠어진다.
    """
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    args = [
        scene_mvs.name, "-m", mesh_ply.name, "-o", output_name, "--decimate", str(decimate),
        "--resolution-level", str(resolution_level), "--scales", str(scales),
    ]
    if regularity_weight is not None:
        args += ["--regularity-weight", str(regularity_weight)]
    _run_openmvs(
        "RefineMesh.exe", args, workdir, bin_dir, "log_refine_mesh.txt",
    )
    return workdir / output_name.replace(".mvs", ".ply")


def align_principal_axes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """PCA 주축을 좌표축(X=최장, Y=중간, Z=최단)에 맞춰 정렬한다(중심은 원점).

    `fitting.fit_point_cloud_to_template()`가 하던 "템플릿 좌표계에 강체
    정렬"을 대체하는 최소 버전이다 -- 템플릿이 없어졌으니 맞출 대상이 없고,
    발 하나만 놓고는 축의 부호(양/음, 즉 어느 쪽이 발끝/뒤꿈치·위/아래인지)를
    PCA만으로는 결정할 근거가 없다. 축만 좌표축에 맞추고 부호(어느 쪽이
    위/앞인지)는 정하지 않는 최소 버전 -- 발바닥 방향까지 정하려면
    `align_sole_down()`을 대신 쓸 것(발바닥 검출로 Y축 부호까지 정함).

    반사(reflection)가 섞이면 메쉬가 뒤집혀(면 방향이 반전돼) 나오므로,
    행렬식이 -1이면 마지막 축 부호를 뒤집어 순수 회전(det=+1)만 적용한다.
    """
    axes = pca_axes(mesh.vertices)
    if np.linalg.det(axes) < 0:
        axes = axes.copy()
        axes[:, -1] *= -1.0
    centroid = mesh.vertices.mean(axis=0)
    aligned = mesh.copy()
    aligned.vertices = (mesh.vertices - centroid) @ axes
    return aligned


def _fibonacci_sphere(n: int) -> np.ndarray:
    """구 표면에 n개 방향을 고르게 뿌린다((n, 3) 단위벡터) -- 피보나치 나선."""
    i = np.arange(n)
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = 2.0 * np.pi * i / golden
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def find_sole_direction(
    surface_points: np.ndarray,
    length_axis: np.ndarray,
    *,
    n_directions: int = 360,
    exclude_cone_deg: float = 40.0,
    contact_band_ratio: float = 0.03,
) -> np.ndarray:
    """구 위의 모든 방향을 검사해, 그 방향으로 투영했을 때 최솟값 근방에
    점이 가장 많이 몰리는(=접지면 후보) 방향을 발바닥(아래) 방향으로 고른다.

    길이축(`length_axis`) 기준 `exclude_cone_deg` 이내(발끝/발목 쪽)
    방향은 후보에서 제외한다 -- 발바닥은 길이축과 대략 수직이라는 가정.

    Args:
        surface_points: 중심이 원점으로 이동된, **표면적 기준 균등 샘플**
            점(`trimesh.sample.sample_surface()` 등). 정점(vertex)을 그대로
            쓰면 안 된다 -- 곡률이 큰 부위(돌기/스파이크)는 삼각형이 잘게
            쪼개져 정점이 몰리므로, 진짜 넓은 발바닥보다 작은 돌기 하나가
            "접점이 더 많다"고 잘못 이길 수 있다.
        length_axis: 발 길이 방향 단위벡터(원뿔 제외 기준).
        contact_band_ratio: 접점으로 칠 허용 오차 -- bounding diagonal 대비
            비율.

    Returns:
        아래(발바닥) 방향 단위벡터.
    """
    sample = surface_points
    directions = _fibonacci_sphere(n_directions)
    cos_thresh = np.cos(np.radians(exclude_cone_deg))
    keep = np.abs(directions @ length_axis) <= cos_thresh
    directions = directions[keep]

    diag = float(np.linalg.norm(sample.max(axis=0) - sample.min(axis=0)))
    band = diag * contact_band_ratio

    proj = sample @ directions.T  # (n_sample, n_kept_directions)
    mins = proj.min(axis=0)
    contact_counts = (proj <= (mins + band)).sum(axis=0)

    # 접점은 뽑힌 방향의 "최솟값" 쪽에 있으므로, 발바닥이 실제로 있는 방향은
    # 부호를 뒤집은 쪽이다.
    best = int(np.argmax(contact_counts))
    return -directions[best]


def align_sole_down(
    mesh: trimesh.Trimesh,
    *,
    n_directions: int = 360,
    exclude_cone_deg: float = 40.0,
    contact_band_ratio: float = 0.03,
    n_surface_samples: int = 20_000,
    rng: np.random.Generator | None = None,
) -> trimesh.Trimesh:
    """`align_principal_axes()`의 다음 단계 -- 발바닥까지 검출해 최종 좌표계를
    X=길이축, Y=높이축(발바닥이 -Y), Z=너비축으로 맞춘다(중심은 원점).

    `find_sole_direction()`(위 참고)으로 발바닥 방향을 고른다 -- 표면적
    기준 균등 샘플(정점이 아니라)을 넘겨 메쉬 삼각화 밀도 편향을 피한다.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    centroid = mesh.vertices.mean(axis=0)
    c = mesh.vertices - centroid

    length_axis = pca_axes(c)[:, 0]
    surface_points, _ = trimesh.sample.sample_surface(mesh, n_surface_samples, seed=rng)
    down = find_sole_direction(
        surface_points - centroid, length_axis,
        n_directions=n_directions, exclude_cone_deg=exclude_cone_deg,
        contact_band_ratio=contact_band_ratio,
    )

    y_axis = -down
    x_axis = length_axis - (length_axis @ y_axis) * y_axis  # y_axis에 재직교화
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)  # 순수 회전(det=+1) 보장

    final_axes = np.stack([x_axis, y_axis, z_axis], axis=1)

    aligned = mesh.copy()
    aligned.vertices = c @ final_axes
    return aligned


def find_ankle_cut_height(
    mesh: trimesh.Trimesh,
    *,
    n_bins: int = 60,
    smooth_window: int = 1,
    exclude_top_ratio: float = 0.15,
    rebound_lookahead_bins: int = 10,
    rebound_ratio: float = 1.15,
    extra_margin_bins: int = 3,
    max_length_ratio: float = 0.40,
    near_floor_ratio: float = 0.12,
    n_surface_samples: int = 20_000,
    rng: np.random.Generator | None = None,
) -> float | None:
    """`align_sole_down()` 직후(Y=높이축, 발바닥이 -Y 쪽) 호출 전제 -- 다리까지
    찍힌 스캔에서 발목 높이를 찾는다.

    높이(Y)를 구간으로 나눠 구간별 단면 폭(구간 내 XZ 중심 기준 반경의 90
    퍼센타일)을 재면, 발(아래, 넓고 불규칙) -> 발목(잘록) -> 종아리 근육(다시
    굵어짐) 순으로 폭이 오목한 패턴을 그린다. 발 쪽 정점(첫 1/3 구간)에서
    최댓값을 찾은 뒤, 그 위쪽으로 훑으며 "국소 최솟값 다음 구간들에서 다시
    확실히 굵어지는" 첫 지점을 찾는다 -- 발가락/뒤꿈치처럼 그냥 가늘어지며
    끝나는 경우(다시 안 굵어짐)와 구분하기 위함.

    이 국소 최솟값 지점을 바로 자르지는 않는다 -- 복사뼈(malleolus)가
    거기서 살짝 더 올라간 곳에 튀어나와 있어서, 최솟값 지점 자체가 이미
    복사뼈 바로 아래이거나 복사뼈를 관통하는 높이일 수 있다. 대신
    반등을 확인하는 데 쓴 `rebound_lookahead_bins` 구간 안에서 폭이 실제로
    가장 커지는 지점(반등의 정점)까지 올라간다 -- 반등 직후 잠깐 주춤하는
    구간이 있으면(복사뼈 두 개가 서로 다른 높이에 튀어나온 경우 등) 그
    잠깐의 주춤함에 속아 진짜 반등 정점 전에 멈추지 않도록.

    그 반등 정점에서도 `extra_margin_bins`만큼 한 번 더 올라간 지점을
    최종 발목 높이로 삼는다 -- 반등이 아주 미세해서(복사뼈 돌출이 작아
    정점이 최솟값 바로 다음 구간인 경우) 위 보정만으로는 복사뼈 바로 위
    몇 mm까지밖에 못 올라가는 사례가 있어, 여유를 더 두는 안전 마진.
    관측 경계 바로 아래(`exclude_top_ratio`)는 후보에서 제외한다 --
    카메라 프레임에 잘린 다리 끝단이 노이즈로 국소 최솟값처럼 보일 수
    있어서다.

    위 과정은 각 스캔의 국소적인 폭 곡선 모양(복사뼈/종아리 근육이
    어디서 얼마나 튀어나왔는지)을 따라가므로, 스캔마다 그 모양이 다르면
    "발끝부터 절단선까지"가 발 길이 대비 서로 다른 비율이 될 수 있다
    (실측 결과 0.37~0.70로 거의 두 배 차이 -- 어떤 샘플은 발목 바로 위,
    어떤 샘플은 종아리 중간까지 남는 문제). 이를 막기 위해 결과를
    `max_length_ratio * 발_길이`로 한 번 더 상한을 씌운다 -- 발_길이는
    바닥 근처(`near_floor_ratio`, 전체 높이의 하위 12%) 점들의 X축
    범위로 근사한다(다리 포함 여부와 무관하게 발 부분만 잡힘).

    이전 버전(`find_leg_cut_plane`, 삭제됨)은 다리+발을 합친 PCA 축을 다시
    구해 그 축을 따라 훑었는데, 다리가 섞이면 그 축 자체가 다리 쪽으로
    쏠려 신뢰할 수 없었다. `align_sole_down()`이 이미 발바닥 검출로 정한
    Y축은 다리 포함 여부와 무관하게 안정적이라 이 축을 그대로 쓴다.

    Returns:
        자를 높이(Y, `mesh` 기준). 패턴이 뚜렷하지 않으면 `None` -- 멀쩡한
        발을 잘못 잘라내는 것보다 다리 포함 케이스를 놓치는 쪽이 안전하다는
        원칙(`clean_dense_point_cloud`의 `max_protrusion_ratio`와 같은 태도).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    surface_points, _ = trimesh.sample.sample_surface(mesh, n_surface_samples, seed=rng)
    x = surface_points[:, 0]
    y = surface_points[:, 1]
    xz = surface_points[:, [0, 2]]

    floor_thresh = y.min() + near_floor_ratio * (y.max() - y.min())
    near_floor = y <= floor_thresh
    foot_length = float(x[near_floor].max() - x[near_floor].min()) if near_floor.sum() > 20 else None

    edges = np.linspace(y.min(), y.max(), n_bins + 1)
    idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    widths = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = xz[idx == b]
        if len(sel) > 20:
            bin_centroid = sel.mean(axis=0)
            widths[b] = np.percentile(np.linalg.norm(sel - bin_centroid, axis=1), 90)
    centers = (edges[:-1] + edges[1:]) / 2

    valid = np.where(~np.isnan(widths))[0]
    n = len(valid)
    if n < rebound_lookahead_bins * 3:
        return None
    w = widths[valid]
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        w = np.convolve(w, kernel, mode="same")

    peak_pos = int(np.argmax(w[: max(n // 3, 3)]))
    search_hi = int(n * (1 - exclude_top_ratio))
    for i in range(peak_pos + 1, search_hi):
        if w[i] <= w[i - 1] and w[i] <= w[i + 1]:
            lookahead = w[i + 1: min(i + 1 + rebound_lookahead_bins, n)]
            if len(lookahead) and lookahead.max() >= w[i] * rebound_ratio:
                bump_i = i + 1 + int(np.argmax(lookahead))  # 반등 구간 안의 실제 정점
                final_i = min(bump_i + extra_margin_bins, search_hi - 1, n - 1)
                cut_height = float(centers[valid[final_i]])
                if foot_length is not None:
                    cut_height = min(cut_height, max_length_ratio * foot_length)
                return cut_height
    return None


def trim_leg_above_ankle(mesh: trimesh.Trimesh, **kwargs) -> trimesh.Trimesh:
    """`align_sole_down()` 직후(Y=높이축) 호출 전제 -- `find_ankle_cut_height()`로
    다리 포함 패턴이 확인되면 발목 높이에서 잘라내고, 아니면 원본 그대로
    반환한다(패턴이 애매하면 아무것도 안 함).
    """
    cut_y = find_ankle_cut_height(mesh, **kwargs)
    if cut_y is None:
        return mesh
    trimmed = mesh.slice_plane([0.0, cut_y, 0.0], [0.0, -1.0, 0.0], cap=True)
    if trimmed is None or len(trimmed.vertices) == 0:
        print("[trim] 다리 패턴이 감지됐지만 자르기 결과가 비어 원본을 유지합니다")
        return mesh
    # slice_plane 절단면 근처에서 원래 몸통과 얇게만 이어져 있던 부분이
    # 떨어져 나가 부유 조각이 될 수 있다 -- 절단 직후 다시 한번 정리.
    trimmed, faces_before, faces_after = keep_largest_component(trimmed)
    if faces_after < faces_before:
        print(f"[trim] 절단 후 부유 조각 제거: 면 {faces_before:,} -> {faces_after:,}")
    print(
        f"[trim] 다리 포함 패턴 감지 -- 발목 높이(Y={cut_y:.4g})에서 잘라냄 "
        f"(정점 {len(mesh.vertices):,} -> {len(trimmed.vertices):,})"
    )
    return trimmed


def rest_on_floor(
    mesh: trimesh.Trimesh,
    *,
    floor_percentile: float = 0.5,
    n_surface_samples: int = 20_000,
    rng: np.random.Generator | None = None,
) -> trimesh.Trimesh:
    """발바닥이 Y=0에 오도록 Y축으로만 평행이동한다.

    단일 최저 정점(`.min()`)이 아니라 `floor_percentile`(기본 0.5%) 백분위를
    기준으로 삼는다 -- 진짜 접지면이 아니라 노이즈 스파이크 하나가 최저점을
    차지하고 있으면(발목 뿔 결함 등) 그 점 하나에 전체 메쉬가 매달려
    나머지는 뜨는 문제를 피하기 위함이다. 정점이 아니라 표면적 기준 균등
    샘플로 백분위를 계산한다 -- `find_sole_direction()`과 같은 이유로,
    정점 그대로 쓰면 삼각화가 촘촘한 작은 돌기가 과대표집된다.
    `align_sole_down()`이 정한 좌표계(발바닥=-Y)를 전제로 한다.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    surface_points, _ = trimesh.sample.sample_surface(mesh, n_surface_samples, seed=rng)
    floor_y = float(np.percentile(surface_points[:, 1], floor_percentile))
    resting = mesh.copy()
    resting.apply_translation([0.0, -floor_y, 0.0])
    return resting


def to_z_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Y=높이 좌표계를 Z=높이로 바꾼다(X축 기준 90도 회전, 형태 왜곡 없음).

    대부분의 3D 뷰어/슬라이서는 Z를 "위"로 가정한다 -- `align_sole_down()`/
    `rest_on_floor()`가 만드는 Y=높이 좌표계 그대로 내보내면 그런 뷰어에서
    발이 옆으로 누운 것처럼 보인다.
    """
    rotated = mesh.copy()
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    rotated.vertices = np.stack([x, -z, y], axis=1)
    return rotated


def to_y_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """`to_z_up()`의 역변환 -- Z=높이 좌표계를 Y=높이로 되돌린다.

    glTF(.glb/.gltf) 스펙은 Y-up이 규약이라, `finalize_mesh()`가 기본으로
    내보내는 Z-up 결과(STL/슬라이서 관례)를 glb로도 같이 저장하고 싶을 때
    이 함수로 되돌린 다음 내보낸다.
    """
    rotated = mesh.copy()
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    rotated.vertices = np.stack([x, z, -y], axis=1)
    return rotated


def decimate_mesh(
    mesh: trimesh.Trimesh,
    *,
    target_vertices: int,
    smooth_after: bool = True,
    smooth_lamb: float = 0.5,
    smooth_iterations: int = 5,
    max_tuning_iterations: int = 3,
) -> trimesh.Trimesh:
    """쿼드릭 에지 축약(quadric edge collapse)으로 정점 수를 `target_vertices`
    근방까지 줄인다 -- GNN 학습 데이터(예: 논문 기준 mesh size=15, 정점
    ~18,000개)와 해상도를 맞출 때 사용.

    `trimesh.simplify_quadric_decimation()`은 목표를 면(face) 개수로 받는데,
    정점 개수와 면 개수의 관계가 메쉬 형태마다 달라 한 번에 정확히 못 맞춘다
    -- 삼각메쉬는 오일러 공식상 면이 정점의 약 2배(F ≈ 2V)라는 근사로 첫 값을
    잡은 뒤, 실제 결과의 정점 수를 보고 목표 대비 비율만큼 면 개수를 보정해
    최대 `max_tuning_iterations`번 다시 시도한다(매번 원본에서 다시 축약 --
    이미 축약된 메쉬를 또 축약하면 품질이 누적으로 나빠짐).

    축약은 국소적으로 뾰족한 아티팩트를 남기기 쉬워, 기본으로 가벼운
    라플라시안 마감(`finish_smooth_mesh()`, 이미 파이프라인에서 쓰던 것
    재사용)을 이어 붙인다. `smooth_iterations` 기본값 5는 실측 근거 있음 --
    이 마감 스무딩은 `volume_constraint`를 꺼놓고 쓰기 때문에(비-watertight
    메쉬라 부피보존 계산이 NaN을 냄, 모듈독스트링의 `finish_smooth_mesh()`
    참고) 반복할수록 부피가 줄어드는 편향이 생긴다 -- 20회는 PCA 길이가
    ~2.5% 줄고 정점당 평균 3mm(최대 12mm) 이동하는데, 5회면 각진 표면은
    거의 그대로 지우면서 수축은 ~0.8%(평균 1.7mm)로 준다. 입력이 이미
    깨끗한 경우(예: 재삼각화된 저해상도 예측 결과) 굳이 20회씩 돌릴 필요가
    없어서 더 가벼운 값을 기본으로 삼음 -- 입력이 노이즈가 많으면 직접
    올릴 것.
    """
    n_before = len(mesh.vertices)
    if n_before <= target_vertices:
        print(f"[decimate] 이미 목표({target_vertices:,}) 이하(정점 {n_before:,}) -- 건너뜀")
        return mesh

    target_faces = target_vertices * 2
    simplified = mesh
    for _ in range(max_tuning_iterations):
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        n_after = len(simplified.vertices)
        if n_after == 0:
            break
        ratio = target_vertices / n_after
        if 0.9 <= ratio <= 1.1:
            break
        target_faces = max(int(target_faces * ratio), 4)

    print(f"[decimate] 쿼드릭 단순화: 정점 {n_before:,} -> {len(simplified.vertices):,} "
          f"(목표 {target_vertices:,})")
    if smooth_after:
        simplified = finish_smooth_mesh(simplified, lamb=smooth_lamb, iterations=smooth_iterations)
    return simplified


def find_floor_contact_mask(
    mesh: trimesh.Trimesh,
    *,
    tolerance_mm: float = 2.0,
    up_axis: int = 2,
) -> np.ndarray:
    """바닥(Y=0 또는 Z=0, `up_axis`로 지정)에서 `tolerance_mm` 이내인 정점을
    "접지 노드"로 표시한 불리언 배열(정점 순서와 1:1 대응)을 반환한다.

    `rest_on_floor()`가 이미 발바닥을 0 근방에 붙여놨다는 전제 -- 즉
    `finalize_mesh()` 이후(스케일까지 끝나 `tolerance_mm`이 실제 mm 단위로
    맞는) 메쉬에 대해서만 의미가 있다. `up_axis`는 기본 2(Z) --
    `finalize_mesh(z_up=True)` 기본값과 맞춤; Y=높이 메쉬라면 1을 넘길 것.
    """
    heights = mesh.vertices[:, up_axis]
    mask = heights <= (heights.min() + tolerance_mm)
    print(f"[floor-contact] 접지 노드 {int(mask.sum()):,}/{len(mask):,}개 "
          f"({100 * mask.mean():.1f}%, 허용오차 {tolerance_mm:.1f}mm)")
    return mask


def finalize_mesh(
    mesh: trimesh.Trimesh,
    *,
    reference_length_mm: float | None = None,
    z_up: bool = True,
    trim_leg: bool = False,
    target_vertices: int | None = None,
) -> tuple[trimesh.Trimesh, float]:
    """`run_dense_pipeline()`이 만든 원본 메쉬(임의 좌표계/임의 스케일)를
    축 정렬 + 스케일링 + 바닥 정착까지 마친 최종 메쉬로 만든다.

    `run_pipeline()`과 `run_dense_pipeline.py`가 공유한다.

    Args:
        z_up: 기본 True -- 내부적으로는 Y=높이로 계산하지만, 최종 결과는
            `to_z_up()`으로 Z=높이 좌표계로 내보낸다(대부분의 뷰어 관례).
        trim_leg(False): 발목 위 다리까지 찍힌 스캔에서 `trim_leg_above_ankle()`로
            다리를 잘라낸다(정렬 뒤, `align_sole_down()`이 정한 Y=높이축 기준 --
            발목 높이 검출은 이 축이 먼저 정해져야 신뢰할 수 있다). 패턴이
            뚜렷할 때만 자르지만(`find_ankle_cut_height()` 참고), 다리가 안
            찍힌 정상 스캔에는 영향 없어야 함.
        target_vertices(None): 지정하면 `decimate_mesh()`로 이 정점 수 근방까지
            단순화 + 마감 스무딩한다(예: 다른 데이터셋/모델의 해상도에 맞출 때).
            단순화+스무딩이 바닥 접지를 살짝 흐트러뜨릴 수 있어, 그 뒤
            `rest_on_floor()`를 한 번 더 적용해 바로잡는다.

    Returns:
        (정렬/스케일/접지 완료된 메쉬, 적용된 스케일 배율)
    """
    mesh, faces_before, faces_after = keep_largest_component(mesh)
    if faces_after < faces_before:
        print(f"[정리] 몸통과 떨어진 부유 조각 제거: 면 {faces_before:,} -> {faces_after:,}")

    mesh = align_sole_down(mesh)
    if trim_leg:
        mesh = trim_leg_above_ankle(mesh)

    resolved_reference_length_mm = reference_length_mm
    if resolved_reference_length_mm is None:
        resolved_reference_length_mm = DEFAULT_REFERENCE_LENGTH_MM
        print(
            f"[스케일] 자기신고 발길이 없음 — placeholder {DEFAULT_REFERENCE_LENGTH_MM:.0f}mm 기준으로"
            " 스케일링(절대 축척 아님, 형태 비교/시각화용 임시값 — 실사용 전 반드시 확인할 것)"
        )
    own_length = measured_length(mesh.vertices)
    scale_factor = resolved_reference_length_mm / own_length
    mesh.apply_scale(scale_factor)
    print(
        f"[스케일] 메쉬 자체 PCA 길이 {own_length:.4f}(SfM 임의 단위) -> "
        f"{resolved_reference_length_mm:.1f}mm 기준(x{scale_factor:.4f})"
    )

    if target_vertices is not None:
        mesh = decimate_mesh(mesh, target_vertices=target_vertices)

    mesh = rest_on_floor(mesh)
    if z_up:
        mesh = to_z_up(mesh)
    return mesh, scale_factor


def run_dense_pipeline(
    *,
    sparse_dir: Path,
    images_dir: Path,
    masks_dir: Path,
    workdir: Path,
    openmvs_bin: str | Path | None = None,
    refine: bool = False,
    postprocess_dmaps: int = DEFAULT_POSTPROCESS_DMAPS,
    max_threads: int = DEFAULT_MAX_THREADS,
    densify_resolution_level: int | None = None,
    densify_number_views_fuse: int | None = None,
    visibility_filter_threshold: int | None = None,
    grazing_filter_min_score: float | None = None,
    reprojection_consistency_min_vote: float | None = None,
    free_space_support: bool = False,
    thickness_factor: float = 1.0,
    quality_factor: float = 1.0,
    refine_decimate: float = 1.0,
    refine_regularity_weight: float | None = None,
    smooth_high_curvature: bool = True,
    curvature_percentile: float = DEFAULT_CURVATURE_PERCENTILE,
    curvature_min_radius_mult: float = DEFAULT_CURVATURE_MIN_RADIUS_MULT,
    curvature_max_radius_mult: float = DEFAULT_CURVATURE_MAX_RADIUS_MULT,
    curvature_iterations: int = DEFAULT_CURVATURE_ITERATIONS,
    curvature_alpha: float = DEFAULT_CURVATURE_ALPHA,
    curvature_mu: float = DEFAULT_CURVATURE_MU,
    fill_holes: bool = True,
    sand_surface_enabled: bool = True,
    sand_min_neighbors: int = 16,
    sand_max_neighbors: int = 32,
    sand_iterations: int = 3,
    finish_smooth: bool = True,
    finish_smooth_lambda: float = 0.5,
    finish_smooth_iterations: int = 40,
    prune_protrusions: bool = False,
    keep_intermediates: bool = False,
) -> Path:
    """위 단계 전부를 엮는 오케스트레이션. 최종 메쉬 경로를 반환한다.

    Args:
        sparse_dir(필수): sparse 재구성 폴더. 확실치 않으면 `largest_sparse_dir()`.
        masks_dir(필수): `masking.generate_masks(..., dilate=0)` 결과 폴더.
        refine(False): RefineMesh(느림) 실행 여부.
        densify_resolution_level(None): `run_densify_point_cloud()`의
            resolution_level 그대로 전달(0=원본 해상도, 더 조밀한 점군).
        densify_number_views_fuse(None): `run_densify_point_cloud()`의
            number_views_fuse 그대로 전달.
        visibility_filter_threshold(None): 음수(예: -1)면 가시성 필터 활성화.
        grazing_filter_min_score(None): grazing 필터 임계값, visibility보다 먼저 적용.
        reprojection_consistency_min_vote(None): 배경 오염 필터, 권장 안 함.
        free_space_support/thickness_factor/quality_factor: `run_reconstruct_mesh()` 전달.
        refine_decimate/refine_regularity_weight: `refine=True`일 때 `run_refine_mesh()`
            전달. resolution_level/scales는 항상 최고 정밀도(0/2)로 고정 -- 낮추면 폭
            치수가 실행마다 달라지는 문제가 있어 노출하지 않는다.
        smooth_high_curvature(True): 고곡률 스무딩 여부, curvature_* 로 강도 조절.
        fill_holes/sand_surface_enabled(True): 구멍 메움/사포질 후처리.
        sand_min_neighbors/sand_max_neighbors/sand_iterations: `sand_surface()`
            강도(이웃 범위/반복 횟수) — 키울수록 매끈해지지만 디테일도 죽는다.
        finish_smooth(True): 라플라시안 마감 스무딩 -- 위 단계들로 안 빠지는
            잔여 고주파 표면 노이즈 정리(비용 미미, 실측 몇 초). `finish_smooth_lambda`/
            `finish_smooth_iterations`로 강도 조절.
        prune_protrusions(False): 포인트클라우드 단계 뿔 프루닝.
        keep_intermediates(False): 중간 산출물 보존 여부.
    """
    # OpenMVS 서브프로세스는 -w(workdir)를 cwd로 실행되므로, 그 외 입력 경로는
    # 전부 절대경로로 넘겨야 한다 -- 상대경로면 cwd가 바뀐 뒤 엉뚱한 곳을
    # 가리키게 된다.
    sparse_dir = Path(sparse_dir).resolve()
    images_dir = Path(images_dir).resolve()
    masks_dir = Path(masks_dir).resolve()
    workdir = Path(workdir).resolve()

    dense_dir = undistort_for_dense(sparse_dir, images_dir, workdir / "dense")
    openmvs_masks_dir = convert_masks_for_openmvs(masks_dir, workdir / "openmvs_masks")
    openmvs_dir = workdir / "openmvs"

    scene_mvs = run_interface_colmap(dense_dir, openmvs_dir, openmvs_bin=openmvs_bin)
    dense_ply = run_densify_point_cloud(
        scene_mvs, openmvs_dir, masks_dir=openmvs_masks_dir, openmvs_bin=openmvs_bin,
        max_threads=max_threads, postprocess_dmaps=postprocess_dmaps,
        resolution_level=densify_resolution_level, number_views_fuse=densify_number_views_fuse,
    )
    cleaned_ply = openmvs_dir / "scene_dense_cleaned.ply"
    clean_dense_point_cloud(dense_ply, cleaned_ply, prune_protrusions=prune_protrusions)

    mesh_input_ply = cleaned_ply
    if reprojection_consistency_min_vote is not None:
        consistency_ply = openmvs_dir / "scene_dense_consistency.ply"
        filter_by_reprojection_consistency(
            mesh_input_ply, dense_dir / "sparse", masks_dir, consistency_ply,
            min_vote_ratio=reprojection_consistency_min_vote,
        )
        mesh_input_ply = consistency_ply

    if grazing_filter_min_score is not None:
        grazing_ply = openmvs_dir / "scene_dense_degrazed.ply"
        filter_grazing_points(
            mesh_input_ply, dense_dir / "sparse", grazing_ply, min_score=grazing_filter_min_score,
        )
        mesh_input_ply = grazing_ply

    if visibility_filter_threshold is not None:
        filtered_ply = filter_point_cloud_visibility(
            scene_mvs, mesh_input_ply, openmvs_dir,
            threshold=visibility_filter_threshold, openmvs_bin=openmvs_bin,
        )
        restored_ply = openmvs_dir / "scene_dense_vf_restored.ply"
        restore_point_cloud_views(mesh_input_ply, filtered_ply, restored_ply)
        mesh_input_ply = restored_ply

    mesh_ply = run_reconstruct_mesh(
        scene_mvs, mesh_input_ply, openmvs_dir, openmvs_bin=openmvs_bin,
        free_space_support=free_space_support, thickness_factor=thickness_factor,
        quality_factor=quality_factor,
    )

    if refine:
        mesh_ply = run_refine_mesh(
            scene_mvs, mesh_ply, openmvs_dir, openmvs_bin=openmvs_bin,
            decimate=refine_decimate, regularity_weight=refine_regularity_weight,
        )

    # 뿔/스파이크 사후 프루닝(prune_thin_protrusions)은 메쉬 위상을 망가뜨려
    # 안 쓴다 -- 점 단위 사전 제거(clean_dense_point_cloud의
    # prune_protrusions, 기본 꺼짐)를 대신 쓸 것. 배경/파편 제거 + 스무딩
    # 자체는 `mesh_postprocess.postprocess_mesh()`(사진/카메라 정보 불필요 --
    # 완성된 STL 등에도 독립적으로 적용 가능, `scripts/postprocess_mesh.py` 참고)로 위임한다.
    mesh = trimesh.load(mesh_ply, process=False)
    mesh, post_stats = postprocess_mesh(
        mesh, keep_largest=True, prune_protrusions=False,
        fill_holes=fill_holes,
        sand_surface_enabled=sand_surface_enabled, sand_min_neighbors=sand_min_neighbors,
        sand_max_neighbors=sand_max_neighbors, sand_iterations=sand_iterations,
        smooth_high_curvature=smooth_high_curvature, curvature_percentile=curvature_percentile,
        curvature_min_radius_mult=curvature_min_radius_mult, curvature_max_radius_mult=curvature_max_radius_mult,
        curvature_iterations=curvature_iterations, curvature_alpha=curvature_alpha, curvature_mu=curvature_mu,
        finish_smooth=finish_smooth, finish_smooth_lambda=finish_smooth_lambda,
        finish_smooth_iterations=finish_smooth_iterations,
    )
    if post_stats.steps_applied:
        mesh.export(mesh_ply)

    if not keep_intermediates:
        mesh_ply = _keep_final_mesh_only(mesh_ply, workdir)

    print(f"[dense] 완료: {mesh_ply}")
    return mesh_ply


def _keep_final_mesh_only(mesh_ply: Path, workdir: Path) -> Path:
    """`mesh_ply`를 `<workdir>/mesh.ply`로 옮기고 `workdir`의 나머지(undistort
    워크스페이스, depth map, OpenMVS 로그/중간 씬 파일 등)는 전부 지운다.

    `keep_intermediates=False`(기본)일 때만 호출된다.
    """
    final_path = workdir / f"mesh{mesh_ply.suffix}"
    if mesh_ply.resolve() != final_path.resolve():
        shutil.copy2(mesh_ply, final_path)
    for child in workdir.iterdir():
        if child.resolve() == final_path.resolve():
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink()
    return final_path
