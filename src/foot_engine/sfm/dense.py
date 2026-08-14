"""Sparse SfM 결과 기반 OpenMVS Dense 포인트클라우드/메쉬 복원 모듈.

OpenMVS CLI 실행파일 필요(`OPENMVS_BIN_DIR`), CPU 빌드로 동작.

흐름:
    undistort_for_dense() -> convert_masks_for_openmvs() -> run_interface_colmap()
    -> run_densify_point_cloud() -> clean_dense_point_cloud()
    -> (선택) filter_by_reprojection_consistency()/filter_grazing_points()/
       filter_point_cloud_visibility()+restore_point_cloud_views()
    -> run_reconstruct_mesh() -> (선택) run_refine_mesh() -> keep_largest_component()
    -> (선택) fill_small_holes()/sand_surface() -> smooth_high_curvature_regions()

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
import networkx as nx
import numpy as np
import pycolmap
import trimesh
from scipy.spatial import cKDTree

from .geometry import measured_length, pca_axes
from .reconstruction import filter_outlier_points

#: OpenMVS CLI 실행파일이 있는 폴더. 환경변수로 덮어쓸 것 -- 설치 방법은 README 참고.
DEFAULT_OPENMVS_BIN_DIR = os.environ.get("OPENMVS_BIN_DIR", "")

#: `DensifyPointCloud`의 간헐적 네이티브 크래시를 피하기 위한 스레드 상한.
DEFAULT_MAX_THREADS = 8

#: `--postprocess-dmaps` 비트마스크: 1=remove-speckles, 2=fill-gaps.
#: 저텍스처 평면(발등/발바닥)의 깊이 추정 공백을 일부 메운다.
DEFAULT_POSTPROCESS_DMAPS = 3

#: `smooth_high_curvature_regions()` 기본 강도(곡률 백분위 임계값).
DEFAULT_CURVATURE_PERCENTILE = 60.0
DEFAULT_CURVATURE_RINGS = 6
DEFAULT_CURVATURE_ITERATIONS = 15
DEFAULT_CURVATURE_ALPHA = 0.7

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


def _protrusion_remove_mask(
    points: np.ndarray,
    *,
    density_radius_nn_mult: float,
    far_percentile: float,
    density_ratio: float,
    adjacency_edges: np.ndarray | None = None,
) -> np.ndarray:
    """뿔/스파이크에 해당하는 점(정점) 인덱스 마스크를 계산한다 -- 메쉬 정점과
    raw 포인트클라우드 양쪽에서 재사용하는 공통 로직.

    `adjacency_edges`를 주면(메쉬) 그 인접성을 그대로 쓰고, `None`이면(raw
    포인트클라우드, 메쉬 엣지가 없음) "얇음" 후보 점들끼리 반경 기반으로
    인접 그래프를 새로 구성한다.
    """
    n = len(points)
    if n < 20:
        return np.zeros(n, dtype=bool)

    tree = cKDTree(points)
    nn_dist, _ = tree.query(points, k=min(11, n))
    typical_spacing = float(np.median(nn_dist[:, 1:]))
    if typical_spacing <= 0:
        return np.zeros(n, dtype=bool)

    centroid = np.median(points, axis=0)
    dist = np.linalg.norm(points - centroid, axis=1)
    density = tree.query_ball_point(points, typical_spacing * density_radius_nn_mult, return_length=True)
    thin_mask = density < np.median(density) * density_ratio
    far_mask = dist > np.percentile(dist, far_percentile)

    graph = nx.Graph()
    thin_idx_arr = np.where(thin_mask)[0]
    if adjacency_edges is not None:
        graph.add_edges_from(adjacency_edges)
    else:
        graph.add_nodes_from(thin_idx_arr.tolist())
        if len(thin_idx_arr) >= 2:
            thin_tree = cKDTree(points[thin_idx_arr])
            local_pairs = thin_tree.query_pairs(typical_spacing * 2.0)
            graph.add_edges_from((thin_idx_arr[a], thin_idx_arr[b]) for a, b in local_pairs)

    thin_idx = set(thin_idx_arr.tolist())
    far_idx = set(np.where(far_mask)[0].tolist())
    remove_idx: set[int] = set()
    for component in nx.connected_components(graph.subgraph(thin_idx & set(graph.nodes))):
        if component & far_idx:
            remove_idx |= component

    remove_mask = np.zeros(n, dtype=bool)
    if remove_idx:
        remove_mask[list(remove_idx)] = True
    return remove_mask


def prune_thin_protrusions(
    mesh: trimesh.Trimesh,
    *,
    density_radius_nn_mult: float = 4.0,
    far_percentile: float = 97.0,
    density_ratio: float = 0.6,
) -> tuple[trimesh.Trimesh, int]:
    """몸통에 이어져(fused) 붙은 뿔/스파이크를 통째로 잘라낸다.

    `keep_largest_component()`는 이미 분리된 파편만 잡는다 -- 뿔은 몸통과
    같은 연결 요소라 무력하다. 대신 "뿔은 국소 정점 밀도가 몸통보다
    뚜렷이 낮다"는 걸 핵심 단서로 쓴다 -- 얇고 길쭉한 구조라 단위
    부피당 정점 수가 적다:

        1. 각 정점의 반경 `density_radius_nn_mult * 전형적 10-최근접 간격`
           안 이웃 수(밀도) 계산 -- 메쉬 해상도(정점 간격)에 비례하는
           반경이라야 정점 수/전체 크기가 달라져도 "밀도" 수치가 일관된
           의미를 가진다.
        2. 밀도가 전체 중앙값의 `density_ratio` 미만인 정점을 "얇음"으로 표시.
        3. 중심에서 먼 상위 `far_percentile`% 정점을 뿔 후보 씨앗으로 표시.
        4. "얇음" 정점들만으로 메쉬 인접 그래프의 연결 요소를 구하고, 씨앗과
           겹치는 연결 요소(뿔의 끝~뿌리 전체)를 통째로 제거한다.

    Returns:
        (프루닝된 메쉬, 제거된 정점 수).
    """
    remove_mask = _protrusion_remove_mask(
        mesh.vertices, adjacency_edges=mesh.edges_unique,
        density_radius_nn_mult=density_radius_nn_mult, far_percentile=far_percentile,
        density_ratio=density_ratio,
    )
    n_removed = int(remove_mask.sum())
    if n_removed == 0:
        return mesh, 0

    pruned = mesh.copy()
    pruned.update_vertices(~remove_mask)
    return pruned, n_removed


def keep_largest_component(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int, int]:
    """면(face) 기준 가장 큰 연결 요소만 남기고 부유 파편을 지운다.

    `ReconstructMesh` 출력은 보통 발 하나가 98%+를 차지하는 단일 덩어리이고,
    나머지는 배경에서 떨어져 나온 작은 파편이다. 이미 공간적으로 분리된
    덩어리 단위로만 자르므로 DBSCAN(성긴 진짜 부위를 노이즈로 오판)과 달리
    안전하다 -- 다만 발 표면에 이어져(fused) 붙은 경계 노이즈는 같은
    덩어리라 이걸로 안 떨어진다.

    Returns:
        (필터링된 메쉬, 원본 face 수, 남은 face 수).
    """
    total_faces = len(mesh.faces)
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh, total_faces, total_faces
    largest = max(components, key=lambda c: len(c.faces))
    return largest, total_faces, len(largest.faces)


def fill_small_holes(
    mesh: trimesh.Trimesh,
    *,
    max_hole_diameter_ratio: float = 0.05,
    use_fan: bool = True,
) -> tuple[trimesh.Trimesh, int]:
    """작은 구멍(핀홀/관측 누락 조각)만 팬 삼각분할로 메운다.

    발바닥처럼 원래 안 찍은 큰 구멍은 일부러 그대로 둔다 -- 억지로 메우면
    평평한 가짜 뚜껑이 씌워져 실제 형태를 왜곡한다(알려진 한계, 모듈
    docstring 참고). 구멍 하나의 바운딩박스 대각선이 메쉬 전체 대각선의
    `max_hole_diameter_ratio`보다 작을 때만 메운다.

    삼각형은 추가만 되고(폴리곤 감소 없음) 기존 정점/면은 그대로 둔다.
    `trimesh.repair.fill_holes()`와 같은 방식(경계 사이클 탐색 + 팬
    삼각분할)이지만 크기 필터가 없는 그 함수와 달리 큰 구멍은 건너뛴다.

    Args:
        max_hole_diameter_ratio: 구멍 자체 바운딩박스 대각선 / 메쉬 전체
            바운딩박스 대각선 비율 상한(기본 0.05 = 5%).
        use_fan: 볼록하지 않은 구멍도 팬 삼각분할로 메울지.

    Returns:
        (구멍 메운 메쉬, 실제로 메운 구멍 개수).
    """
    if len(mesh.faces) < 3 or mesh.is_watertight:
        return mesh, 0

    boundary_groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    if len(boundary_groups) < 3:
        return mesh, 0

    boundary = mesh.edges[boundary_groups]
    holes = nx.cycle_basis(nx.from_edgelist(boundary))
    if not holes:
        return mesh, 0

    mesh_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    max_diag = mesh_diag * max_hole_diameter_ratio
    small_holes = []
    for loop in holes:
        pts = mesh.vertices[loop]
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        if diag <= max_diag:
            small_holes.append(loop)
    if not small_holes:
        return mesh, 0

    new_faces = trimesh.geometry.triangulate_quads(small_holes, use_fan=use_fan)
    if len(new_faces) == 0:
        return mesh, 0

    # trimesh.repair.fill_holes()와 같은 winding 보정 -- 새 face의 경계
    # edge가 기존 경계와 같은 방향이면 뒤집는다(반대 방향이어야 정상).
    new_edges = trimesh.geometry.faces_to_edges(new_faces)
    hashable_new = trimesh.grouping.hashable_rows(new_edges)
    hashable_old = trimesh.grouping.hashable_rows(boundary)
    needs_reverse = np.isin(hashable_new, hashable_old).reshape((-1, 3)).any(axis=1)
    new_faces[needs_reverse] = np.fliplr(new_faces[needs_reverse])

    out = mesh.copy()
    out.extend_faces(new_faces)
    return out, len(small_holes)


def _ring_neighbors_padded(
    adjacency: list, *, min_neighbors: int, max_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """위상 인접(1-ring)을 BFS로 확장해 정점별 이웃 최소 개수를 채운다.

    공간(유클리드) 최근접이 아니라 메쉬 표면을 따라간 위상 인접을 쓴다 --
    발처럼 접힌/오목한 형태에서는 공간적으로 가까워도 표면상으로는 먼 두
    지점(예: 발목 반대쪽, 발가락 사이)이 유클리드 최근접 이웃으로 섞여
    들어가 국소 곡면 피팅을 망가뜨린다.

    Returns:
        (idx_padded, mask) -- 둘 다 (n, max_neighbors) 모양. `mask`가
        False인 자리의 `idx_padded` 값은 의미 없다(0으로 채워짐).
    """
    n = len(adjacency)
    idx_padded = np.zeros((n, max_neighbors), dtype=np.int64)
    mask = np.zeros((n, max_neighbors), dtype=bool)
    for i in range(n):
        visited = {i}
        frontier = {i}
        neighbors: set[int] = set()
        while len(neighbors) < min_neighbors and frontier:
            next_frontier = {nb for node in frontier for nb in adjacency[node] if nb not in visited}
            visited |= next_frontier
            neighbors |= next_frontier
            frontier = next_frontier
        chosen = np.fromiter(neighbors, dtype=np.int64, count=len(neighbors))[:max_neighbors]
        idx_padded[i, : len(chosen)] = chosen
        mask[i, : len(chosen)] = True
    return idx_padded, mask


def sand_surface(
    mesh: trimesh.Trimesh,
    *,
    min_neighbors: int = 16,
    max_neighbors: int = 32,
    iterations: int = 3,
    regularization: float = 1e-6,
    max_offset_ratio: float = 1.0,
) -> trimesh.Trimesh:
    """모든 정점을 국소 이차곡면(quadric) 근사에 투영해 다듬는다("사포질").

    각 정점 주변 위상(표면) 인접을 BFS로 확장해 최소 `min_neighbors`개를
    모은 뒤, 그 이웃들로 국소 접평면(PCA)을 구하고 접평면 좌표계에서 2차
    곡면 `h = a*u^2+b*uv+c*v^2+d*u+e*v+f`를 최소제곱으로 피팅해, 그 정점을
    이웃들의 추세가 예측하는 위치(`f`, 법선 방향 오프셋)로 옮긴다. 이웃
    평균으로 등방적으로 당기는 라플라시안(`smooth_high_curvature_regions()`)과
    달리 법선 방향으로만 움직이므로 접평면 방향의 진짜 형태(2차 항으로
    표현되는 국소 굴곡)는 보존하면서 그 정점만 튀는 고주파 노이즈를 깎아낸다.

    `smooth_high_curvature_regions()`와 달리 곡률 임계값으로 일부만 고르지
    않고 전체 정점에 균일하게 적용한다 -- 발 전체를 다듬는 일반 노이즈
    완화용이며, 정점/면 개수·위상은 그대로다(폴리곤 감소 없음).

    구현 노트:
    - 이웃은 공간(유클리드) 최근접이 아니라 위상(표면) 인접이다 --
      `_ring_neighbors_padded()` docstring 참고.
    - 이웃 좌표(u, w)는 피팅 전에 국소 이웃 거리 스케일로 정규화한다 --
      정규화 없이는 좌표 스케일에 따라 정규방정식(AᵀA) 조건수가 나빠져
      일부 정점의 피팅이 극단값으로 튄다. 남는 이상치에 대비해
      `max_offset_ratio`로 오프셋을 국소 스케일의 배수로 clamp한다.

    한계: 관측 부족으로 생긴 오목 부위(아치/뒤꿈치) 크레이터처럼 이웃
    전체가 같은 방향으로 치우친 저주파 왜곡은 이웃들의 이차곡면 추세
    자체가 이미 왜곡돼 있어 이 방식으로도 못 없앤다 -- 크레이터 완화는
    여전히 `smooth_high_curvature_regions()` 몫이고, 이 함수는 그것과
    별개로 전반적인 표면 노이즈를 줄이는 보완 단계다.

    Args:
        min_neighbors: 국소 곡면 피팅에 쓸 최소 이웃 수 -- 이차곡면
            미지수(6개)보다 충분히 많아야 안정적이다. 1-ring으로 부족하면
            2-ring, 3-ring... 순으로 확장한다.
        max_neighbors: 이웃이 이보다 많아지면 자른다(배열 패딩 크기 상한).
        iterations: 반복 횟수. 이웃 집합(위상 기준이라 메쉬가 안 변하는 한
            고정)은 한 번만 계산하고, 매 반복 그 이웃들의 현재 위치로
            다시 피팅한다.
        regularization: 정규방정식(AᵀA, 정규화된 u/w 기준이라 대각 성분이
            대략 O(min_neighbors) 스케일)에 더하는 상대적 대각 성분.
        max_offset_ratio: 오프셋 크기를 국소 이웃 평균 거리의 이 배수로
            제한하는 안전장치(기본 1.0).
    """
    n = len(mesh.vertices)
    if n < max(min_neighbors, 6) + 1:
        return mesh  # 정점이 너무 적어 이차곡면을 안정적으로 못 피팅함

    idx, mask = _ring_neighbors_padded(
        mesh.vertex_neighbors, min_neighbors=min_neighbors, max_neighbors=max_neighbors
    )
    valid = mask.sum(axis=1) >= 6  # 이차곡면 미지수(6개) 미만이면 그 정점은 안 건드림
    maskf = mask.astype(np.float64)

    v = mesh.vertices.copy().astype(np.float64)
    for _ in range(iterations):
        rel = (v[idx] - v[:, None, :]) * maskf[:, :, None]  # (n, K, 3) -- 패딩은 0으로 무효화
        cov = np.einsum("nki,nkj->nij", rel, rel)
        evals, evecs = np.linalg.eigh(cov)  # 오름차순: 0=법선, 1/2=접평면
        normal = evecs[:, :, 0]
        u_axis, v_axis = evecs[:, :, 2], evecs[:, :, 1]

        dist = np.linalg.norm(rel, axis=2)
        scale = np.where(valid, dist.sum(axis=1) / np.maximum(mask.sum(axis=1), 1), 1.0)
        scale = np.maximum(scale, 1e-9)

        # u/w를 국소 스케일로 정규화 -- 상수항(f)은 (u,w)=(0,0)에서의 값이라
        # 스케일 무관, 그대로 실제 법선 방향 오프셋(길이 단위)이다.
        u = (np.einsum("nki,ni->nk", rel, u_axis) / scale[:, None]) * maskf
        w = (np.einsum("nki,ni->nk", rel, v_axis) / scale[:, None]) * maskf
        h = np.einsum("nki,ni->nk", rel, normal) * maskf

        design = np.stack([u * u, u * w, w * w, u, w, maskf], axis=-1)  # (n,K,6) -- 패딩 행은 전부 0
        AtA = np.einsum("nki,nkj->nij", design, design)
        AtA += regularization * min_neighbors * np.eye(6)
        Ath = np.einsum("nki,nk->ni", design, h)
        coeffs = np.linalg.solve(AtA, Ath[..., None])[..., 0]  # (n, 6)

        offset = np.where(valid, coeffs[:, 5], 0.0)  # 이차곡면이 예측하는 (0,0) 지점의 높이
        max_offset = max_offset_ratio * scale
        offset = np.clip(offset, -max_offset, max_offset)
        v = v + offset[:, None] * normal

    out = mesh.copy()
    out.vertices = v
    print(f"[dense] 사포질(전체 이차곡면 투영): 정점 {n:,}개, {iterations}회 반복")
    return out


def smooth_high_curvature_regions(
    mesh: trimesh.Trimesh,
    *,
    curvature_percentile: float = 90.0,
    rings: int = 5,
    iterations: int = 10,
    alpha: float = 0.6,
) -> trimesh.Trimesh:
    """곡률이 튀는 정점과 그 주변 링만 라플라시안으로 스무딩한다. 나머지
    정점은 그대로 둔다.

    관측 부족으로 생긴 크레이터형 결함 완화용 -- 노이즈와 진짜 굴곡을
    구분하지 못해 발가락 사이 같은 진짜 디테일도 함께 뭉개진다.

    Args:
        curvature_percentile: 이 백분위 이상 |곡률|인 정점을 코어로 삼는다.
        rings: 코어에서 몇 단계 인접 정점까지 감쇠 가중치로 확산시킬지.
    """
    v = mesh.vertices.copy()
    n = len(v)
    edge_len = np.linalg.norm(v[mesh.edges[:, 0]] - v[mesh.edges[:, 1]], axis=1)
    radius = float(np.median(edge_len)) * 4
    curv = trimesh.curvature.discrete_mean_curvature_measure(mesh, v, radius)

    thresh = np.percentile(np.abs(curv), curvature_percentile)
    core_mask = np.abs(curv) > thresh

    neighbors = mesh.vertex_neighbors
    weight = np.zeros(n)
    weight[core_mask] = 1.0
    frontier = set(np.where(core_mask)[0].tolist())
    visited = set(frontier)
    for w in np.linspace(1.0, 0.2, rings):
        next_frontier = {nb for idx in frontier for nb in neighbors[idx] if nb not in visited}
        for idx in next_frontier:
            weight[idx] = max(weight[idx], w)
        visited |= next_frontier
        frontier = next_frontier

    new_v = v.copy()
    for _ in range(iterations):
        avg = np.array([new_v[neighbors[i]].mean(axis=0) if len(neighbors[i]) else new_v[i] for i in range(n)])
        new_v = new_v + weight[:, None] * alpha * (avg - new_v)

    out = mesh.copy()
    out.vertices = new_v
    print(
        f"[dense] 고곡률 국소 스무딩: 정점 {int((weight > 0).sum()):,}/{n:,}개 영향"
        f"(코어 {int(core_mask.sum()):,}개, curvature_percentile={curvature_percentile})"
    )
    return out


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


def finalize_mesh(
    mesh: trimesh.Trimesh,
    *,
    reference_length_mm: float | None = None,
    z_up: bool = True,
) -> tuple[trimesh.Trimesh, float]:
    """`run_dense_pipeline()`이 만든 원본 메쉬(임의 좌표계/임의 스케일)를
    축 정렬 + 스케일링 + 바닥 정착까지 마친 최종 메쉬로 만든다.

    `run_pipeline()`과 `run_dense_pipeline.py`가 공유한다.

    Args:
        z_up: 기본 True -- 내부적으로는 Y=높이로 계산하지만, 최종 결과는
            `to_z_up()`으로 Z=높이 좌표계로 내보낸다(대부분의 뷰어 관례).

    Returns:
        (정렬/스케일/접지 완료된 메쉬, 적용된 스케일 배율)
    """
    mesh = align_sole_down(mesh)

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
    curvature_rings: int = DEFAULT_CURVATURE_RINGS,
    curvature_iterations: int = DEFAULT_CURVATURE_ITERATIONS,
    curvature_alpha: float = DEFAULT_CURVATURE_ALPHA,
    fill_holes: bool = True,
    sand_surface_enabled: bool = True,
    sand_min_neighbors: int = 16,
    sand_max_neighbors: int = 32,
    sand_iterations: int = 3,
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
    # prune_protrusions, 기본 꺼짐)를 대신 쓸 것. 여기서는 이미 분리된
    # 부유 파편만 정리한다.
    mesh = trimesh.load(mesh_ply, process=False)
    mesh, faces_before, faces_after = keep_largest_component(mesh)
    changed = faces_after < faces_before
    if changed:
        print(f"[dense] 부유 파편 제거: {faces_before:,} -> {faces_after:,} faces")

    if fill_holes:
        mesh, n_filled = fill_small_holes(mesh)
        if n_filled:
            print(f"[dense] 작은 구멍 메움: {n_filled}개")
            changed = True

    if sand_surface_enabled:
        mesh = sand_surface(
            mesh, min_neighbors=sand_min_neighbors, max_neighbors=sand_max_neighbors,
            iterations=sand_iterations,
        )
        changed = True

    if smooth_high_curvature:
        mesh = smooth_high_curvature_regions(
            mesh, curvature_percentile=curvature_percentile, rings=curvature_rings,
            iterations=curvature_iterations, alpha=curvature_alpha,
        )
        changed = True

    if changed:
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
