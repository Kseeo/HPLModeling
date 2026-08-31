"""Sparse SfM -> OpenMVS Dense 포인트클라우드/메쉬 복원.

- OpenMVS CLI 필요(`OPENMVS_BIN_DIR`), CPU 빌드 동작.
- 흐름: undistort_for_dense -> convert_masks_for_openmvs -> run_interface_colmap
  -> run_densify_point_cloud -> clean_dense_point_cloud
  -> (선택) filter_by_reprojection_consistency / filter_grazing_points /
     filter_point_cloud_visibility+restore_point_cloud_views
  -> run_reconstruct_mesh -> (선택) run_refine_mesh
  -> mesh_postprocess.postprocess_mesh (배경/파편 제거+스무딩, 별도 모듈)
- 사진->메쉬(이 모듈)와 배경/파편 제거·스무딩(mesh_postprocess.py, 메쉬만
  입력)은 분리돼 있음 -- 후자는 scripts/postprocess_mesh.py로 단독 실행 가능.

기본값:
- 마스크는 dilate=0, densify 전에 적용.
- 이상치 제거는 통계적 방식만(DBSCAN/fusion 합의 강도 안 씀).
- 메쉬 생성은 스무딩 OFF, 최대 연결 요소만 유지.
- 축 정렬은 find_sole_direction()(접점 최다 방향=발바닥).
- filter_by_reprojection_consistency()는 기본 OFF(비권장).
- RefineMesh는 decimate=1(단순화 끔) 기본.
- 중간 산출물은 기본 정리(keep_intermediates=True로 보존).
- prune_protrusions은 결과 불안정으로 기본 OFF.

알려진 한계:
- 접지면에 큰 구멍(발바닥 미촬영 프로토콜 특성).
- 발목 부근 뿔/스파이크 결함 잔존 가능.
- 실루엣 경계 flying pixel 노이즈 완전 제거 안 됨.
- 오목 부위(아치/뒤꿈치) 관측 부족 크레이터 잔존 가능.
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

#: OpenMVS CLI 폴더. 환경변수로 덮어쓸 것 -- 설치는 README 참고.
DEFAULT_OPENMVS_BIN_DIR = os.environ.get("OPENMVS_BIN_DIR", "")

#: DensifyPointCloud 간헐적 크래시 방지용 스레드 상한.
DEFAULT_MAX_THREADS = 8

#: --postprocess-dmaps 비트마스크(1=remove-speckles, 2=fill-gaps).
DEFAULT_POSTPROCESS_DMAPS = 3

#: 자기신고 발길이 없을 때 쓰는 임시 스케일 기준값(mm). 절대 축척 아님.
DEFAULT_REFERENCE_LENGTH_MM = 250.0


def _resolve_openmvs_bin(openmvs_bin: str | Path | None) -> Path:
    # Path("").is_dir()는 "."(cwd)로 True가 되므로 문자열 단계에서 먼저 거른다.
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
    """등록 이미지가 가장 많은 sparse/N 폴더를 찾는다. 번호=크기순 아님."""
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
    """sparse 카메라를 PINHOLE로 바꿔 COLMAP dense 워크스페이스를 만든다.

    pycolmap.undistort_images() 사용 -- 별도 colmap.exe 불필요.
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # pycolmap이 상위 폴더까지는 안 만듦
    pycolmap.undistort_images(
        output_path=str(output_dir),
        input_path=str(sparse_dir),
        image_path=str(images_dir),
        output_type="COLMAP",
    )
    return output_dir


def convert_masks_for_openmvs(masks_dir: Path, out_dir: Path) -> Path:
    """마스크 파일명을 OpenMVS 규칙(<확장자 뗀 이름>.mask.png)으로 복사한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in masks_dir.glob("*.png"):
        # "frame_00000.jpg.png" -> "frame_00000"
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
    """COLMAP dense 워크스페이스를 OpenMVS 씬 파일(scene.mvs)로 변환한다."""
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
    """마스크 기반 dense 포인트클라우드를 만든다 (scene_dense.ply + .mvs).

    - masks_dir: convert_masks_for_openmvs() 결과 폴더. 지정 시 배경 픽셀은
      깊이 계산 생략(masking.generate_masks()를 dilate=0으로 만들 것).
    - postprocess_dmaps: 기본 3(remove-speckles+fill-gaps). 0이면 비활성.
    - resolution_level: 뎁스맵 계산 전 이미지 축소 단계(0=원본, 클수록 빠르지만
      점군 성김). None=OpenMVS 기본값.
    - number_views_fuse: 점 하나를 살리는 데 필요한 최소 동의 뷰 수. None=기본값(2).
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
    """OpenMVS dense PLY를 레코드 단위로 파싱한다.

    - 스키마: xyz+rgb+normal(고정 27바이트) + view_indices/view_weights(가변).
    - trimesh/open3d 라운드트립은 view 필드를 날리므로 원본 바이트를 직접 다룸.
    - Returns: (header 텍스트, body 바이트열, 레코드 경계 offset(N+1,), xyz(N,3)).
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
    """레코드 경계로 법선과 view_indices를 추가로 뽑는다.

    Returns: (법선(N,3), 점별 가변길이 view_indices 리스트(N개)).
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

    - occlusion 경계 flying pixel 노이즈 타깃 필터.
    - dense_ply_path: view_indices/view_weights/normal 필드 있는 dense PLY.
    - sparse_dir: undistort_for_dense() 출력(원본 sparse와 카메라 인덱스 다름).
    - min_score: 관측 뷰 평균 |cos(법선,시선방향)|이 이 미만이면 제거. 관측 정보
      없는 점은 유지.
    - Returns: (원본 점 개수, 유지된 점 개수).
    """
    header, body, offsets, xyz = _parse_dense_ply(dense_ply_path)
    normals, views = _extract_normals_and_views(body, offsets)
    n = len(xyz)

    recon = pycolmap.Reconstruction(str(sparse_dir))
    imgs_sorted = sorted(recon.images.items())
    camera_centers = np.array([img.projection_center() for _, img in imgs_sorted])

    scores = np.ones(n, dtype=np.float32)  # 관측 정보 없는 점="정면"(제거 안 함)
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
    """씬의 카메라 전부에 재투영해 마스크와 동의하는 비율이 낮은 점을 제거한다.

    - 단일 프레임 마스크 오분류에 낚이지 않고 다수 뷰 합의를 따르는 필터.
    - masks_dir: masking.generate_masks() 원본 마스크(<원본파일명>.png).
    - min_vote_ratio: 이 미만이면 제거. 관측 카메라 적은 부위(발등/발뒤꿈치)를
      통째로 지울 수 있어 기본 비권장.
    - Returns: (원본 점 개수, 유지된 점 개수).
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
    """dense 포인트클라우드에서 통계적 이상치를 제거한다 (DBSCAN 안 씀).

    - view_indices/view_weights는 원본 바이트 그대로 보존(_parse_dense_ply 참고).
    - prune_protrusions: 국소 밀도 기준 뿔/스파이크 후보 사전 제거. 실행 간
      편차로 판정이 흔들려 기본 OFF.
    - max_protrusion_ratio: prune_protrusions가 이 비율 넘게 지우려 하면 이번
      실행은 통째로 건너뜀(cleaning.py의 max_plane_ratio와 같은 패턴).
    - Returns: (원본 점 개수, 정리 후 점 개수).
    """
    header, body, offsets, xyz = _parse_dense_ply(dense_ply_path)
    n = len(xyz)

    inliers = filter_outlier_points(xyz, k=k, std_ratio=std_ratio)
    stat_kept_n = int(inliers.sum())
    print(f"[dense] 점단위 정리(통계적 이상치 제거): {n:,} -> {stat_kept_n:,} ({stat_kept_n/n:.1%} 유지)")

    if prune_protrusions:
        # 통계적 이상치 제거 후 남은 점 기준으로 판정(배경 노이즈 섞인 채로
        # 밀도를 재면 기준 자체가 흔들림).
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

    - smooth: 기본 0(끔) -- OpenMVS 기본값(2)은 형상 정교함을 깎음.
    - free_space_support/thickness_factor/quality_factor: 그래프컷 가중치
      튜닝. 기본값은 OpenMVS 자체 기본값 그대로.
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
    """OpenMVS 내장 가시성 기반 점군 필터. DensifyPointCloud를 필터 전용으로 재호출.

    - 주의: 출력 PLY는 view_indices/view_weights 없음 -- run_reconstruct_mesh()에
      바로 넘기면 크래시하니 restore_point_cloud_views()로 복원 후 넘길 것
      (run_dense_pipeline()은 이미 그렇게 엮여 있음).
    - point_cloud_ply: 필터링할 기존 점군(view 필드 있어야 함).
    - threshold: 음수만 필터 발동(절댓값 클수록 공격적).
    - Returns: 필터링된(view 필드 없는) 점군 경로.
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
    """filter_point_cloud_visibility()가 지운 view 필드를 좌표 매칭으로 복원한다.

    - 필터는 좌표를 안 건드리고 부분집합만 고르므로, 통과한 점을 좌표로
      view_source_ply에서 찾아 그 레코드를 그대로 복사.
    - view_source_ply: 필터 입력(view 필드 있음).
    - filtered_ply: filter_point_cloud_visibility()의 출력.
    - out_path: view 필드 복원 결과 -- run_reconstruct_mesh()에 넘길 것.
    - Returns: (필터링된 점 개수, 좌표 매칭 성공해 복원된 점 개수).
    """
    header, body, offsets, xyz = _parse_dense_ply(view_source_ply)
    xyz_to_idx = {(float(x), float(y), float(z)): i for i, (x, y, z) in enumerate(xyz)}

    filt_data = filtered_ply.read_bytes()
    filt_header_end = filt_data.index(b"end_header\n") + len(b"end_header\n")
    filt_header = filt_data[:filt_header_end].decode("ascii")
    filt_body = filt_data[filt_header_end:]
    filt_n = int([line for line in filt_header.splitlines() if line.startswith("element vertex")][0].split()[-1])

    FIXED_SIZE = 27  # xyz+rgb+normal만(view 필드 없는 필터 출력 스키마)
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
    """사진 광도일관성 기반으로 메쉬 정점 위치를 보정한다 (선택, 가장 느린 단계).

    - CPU 동작(--cuda-device 기본 -2). 파이프라인 소요시간 대부분을 차지 --
      빠른 반복 실험에서는 건너뛸 것.
    - decimate: refine 전 입력 메쉬 단순화 정도(0~1). 기본 1(단순화 끔,
      OpenMVS 기본값 0="auto"는 공격적으로 단순화함).
    - regularity_weight: photo-consistency 대 표면 정규화 가중치. None=OpenMVS
      기본값(0.2).
    - resolution_level: 계산 전 이미지 축소 단계(기본 0=원본). 소요시간에
      직접 영향.
    - scales: 다단계 최적화 반복 횟수(기본 2). 줄이면 빨라지지만 거칠어짐.
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


def _fibonacci_sphere(n: int) -> np.ndarray:
    """구 표면에 n개 방향을 고르게 뿌린다((n, 3) 단위벡터, 피보나치 나선)."""
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
    """접점이 가장 많이 몰리는 방향을 발바닥(아래) 방향으로 고른다.

    - length_axis 기준 exclude_cone_deg 이내(발끝/발목 쪽)는 후보 제외
      (발바닥은 길이축과 대략 수직이라는 가정).
    - surface_points: 중심 원점 이동된 표면적 기준 균등 샘플(정점 그대로 쓰면
      곡률 큰 돌기가 과대표집돼 접점 판정 왜곡됨).
    - length_axis: 발 길이 방향 단위벡터.
    - contact_band_ratio: 접점 허용 오차(bounding diagonal 대비 비율).
    - Returns: 아래(발바닥) 방향 단위벡터.
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

    # 접점은 뽑힌 방향의 "최솟값" 쪽 -> 발바닥 방향은 부호를 뒤집은 쪽.
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
    """PCA 주축을 좌표축에 맞추고, 발바닥까지 검출해 X=길이축,
    Y=높이축(발바닥 -Y), Z=너비축으로 맞춘다(중심 원점).

    find_sole_direction() 사용(표면적 균등 샘플로 삼각화 밀도 편향 회피).
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


def prune_far_fragments(
    mesh: trimesh.Trimesh,
    *,
    distance_percentile: float = 95.0,
    margin_mult: float = 1.4,
) -> trimesh.Trimesh:
    """`align_sole_down()` 이후(X=길이축, Z=너비축) 발 몸통에서 수평으로 멀리
    떨어진 정점(배경 파편이 얇은 다리로 이어져 위상적으로는 안 떨어지는
    경우)을 잘라내고 다시 최대 연결요소만 남긴다.

    실측(project_5): 발 몸통은 수평거리 90퍼센타일이 93mm인데 배경 파편은
    최대 147mm까지 뻗어있음 -- `keep_largest_component()`(위상 기반)로는 안
    떨어지지만(얇은 다리로 이어져 있어 같은 연결요소), 이 거리 기반 크롭은
    떨어진다. `_recover_holes_from_original()`이 되살린 배경 조각을 다시
    쳐내는 안전판 성격 -- 발 자체의 원형도 낮은 부위(발가락 벌어짐 등)까지
    치지 않도록 퍼센타일 기반(고정 mm 아님) + 넉넉한 마진(1.4배)을 쓴다.
    """
    v = mesh.vertices
    center = np.median(v[:, [0, 2]], axis=0)  # X=길이, Z=너비 (align_sole_down 직후 관례)
    dist = np.linalg.norm(v[:, [0, 2]] - center, axis=1)
    threshold = float(np.percentile(dist, distance_percentile)) * margin_mult
    keep = dist <= threshold
    if keep.all():
        return mesh

    face_mask = keep[mesh.faces].all(axis=1)
    out = mesh.copy()
    out.update_faces(face_mask)
    out.remove_unreferenced_vertices()

    out, faces_before, faces_after = keep_largest_component(out)
    if faces_after < faces_before or not keep.all():
        print(
            f"[정리] 발에서 수평으로 멀리 떨어진 파편 제거: 정점 {len(mesh.vertices):,} -> "
            f"{len(out.vertices):,}(거리 임계 {threshold:.4g})"
        )
    return out


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
    safety_height_ratio: float = 0.50,
    near_floor_ratio: float = 0.12,
    n_surface_samples: int = 20_000,
    rng: np.random.Generator | None = None,
) -> float | None:
    """align_sole_down() 직후(Y=높이축) 호출 전제 -- 다리 포함 스캔에서 발목
    높이를 찾는다.

    - 높이(Y) 구간별 단면 폭(구간 내 XZ 중심 반경 90퍼센타일)을 재면 발(넓음)
      -> 발목(잘록) -> 종아리(다시 굵어짐) 순 오목 패턴.
    - 발 쪽(첫 1/3)에서 폭 최댓값을 찾은 뒤 위로 훑어 "국소 최솟값 이후 다시
      확실히 굵어지는" 첫 지점을 찾음(발가락/뒤꿈치처럼 계속 가늘어지는
      경우와 구분).
    - 최솟값 지점을 바로 자르지 않고 반등 정점까지 올라감(복사뼈가 최솟값보다
      위에 튀어나와 있을 수 있어서) + extra_margin_bins만큼 안전 마진 추가.
    - exclude_top_ratio: 관측 경계 바로 아래는 후보 제외(카메라 프레임 절단
      노이즈 방지).
    - 폭 곡선 모양이 스캔마다 달라 절단 비율 편차가 커서(실측 0.37~0.70)
      max_length_ratio * 발_길이로 상한 추가. 발_길이는 바닥 근처
      (near_floor_ratio) 점들의 X축 범위로 근사.
    - safety_height_ratio: 반등 패턴을 못 찾아도(종아리가 완만하게만
      굵어지는 경우, 실측: project_6 -- 반등 1.09배로 rebound_ratio=1.15를
      살짝 못 넘김) 전체 높이가 발_길이의 이 비율을 넘으면 다리 포함으로
      보고 max_length_ratio 지점에서 그냥 자른다. rebound_ratio 자체는
      과거 여러 샘플로 튜닝된 값이라 건드리지 않고, 그 판정이 실패했을 때만
      작동하는 별도 안전판.
    - Returns: 자를 높이(Y). 패턴이 뚜렷하지 않으면 None(안전 우선 -- 정상
      발을 잘못 자르는 것보다 다리 케이스를 놓치는 쪽을 택함).
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

    # 폭 반등 패턴을 못 찾았어도(예: 종아리가 완만하게만 굵어져 rebound_ratio를
    # 못 넘김, 실측: project_6) 전체 높이가 발_길이 대비 명백히 비정상으로 크면
    # 다리가 찍힌 것으로 보고 안전 상한(max_length_ratio)에서 그냥 자른다.
    # rebound_ratio 자체는 과거 여러 샘플로 조정된 값이라 건드리지 않고, 이건
    # 그 판정이 실패했을 때만 작동하는 별도의 안전판.
    if foot_length is not None and foot_length > 0:
        total_height = float(y.max() - y.min())
        if total_height > safety_height_ratio * foot_length:
            return max_length_ratio * foot_length
    return None


def trim_leg_above_ankle(mesh: trimesh.Trimesh, **kwargs) -> trimesh.Trimesh:
    """align_sole_down() 직후(Y=높이축) 호출 전제 -- find_ankle_cut_height()로
    다리 패턴이 확인되면 발목 높이에서 잘라내고, 아니면 원본을 반환한다.
    """
    cut_y = find_ankle_cut_height(mesh, **kwargs)
    if cut_y is None:
        return mesh
    trimmed = mesh.slice_plane([0.0, cut_y, 0.0], [0.0, -1.0, 0.0], cap=True)
    if trimmed is None or len(trimmed.vertices) == 0:
        print("[trim] 다리 패턴이 감지됐지만 자르기 결과가 비어 원본을 유지합니다")
        return mesh
    # 절단면 근처에서 얇게만 이어져 있던 부분이 부유 조각으로 떨어질 수 있음.
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

    - 단일 최저 정점이 아닌 floor_percentile 백분위 기준(노이즈 스파이크
      하나에 전체 메쉬가 매달리는 문제 방지).
    - 표면적 기준 균등 샘플 사용(find_sole_direction()과 같은 이유).
    - align_sole_down()이 정한 좌표계(발바닥=-Y) 전제.
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

    대부분 3D 뷰어/슬라이서는 Z를 위로 가정.
    """
    rotated = mesh.copy()
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    rotated.vertices = np.stack([x, -z, y], axis=1)
    return rotated


def to_y_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """to_z_up()의 역변환 -- Z=높이를 Y=높이로 되돌린다(glTF Y-up 규약용)."""
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
    """쿼드릭 에지 축약으로 정점 수를 target_vertices 근방까지 줄인다
    (GNN 학습 데이터 해상도 맞춤용).

    - trimesh.simplify_quadric_decimation()은 면(face) 개수 기준 -- F≈2V
      근사로 첫 값을 잡고, 실제 결과 비율만큼 보정해 최대
      max_tuning_iterations번 재시도(매번 원본에서 재축약).
    - 축약은 뾰족한 아티팩트를 남기기 쉬워 기본으로 라플라시안 마감
      (finish_smooth_mesh()) 추가. smooth_iterations=5는 실측 근거 있음
      (volume_constraint 꺼둔 채 쓰므로 반복할수록 부피 수축 -- 20회는
      PCA 길이 ~2.5% 수축, 5회는 ~0.8%로 절충).
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
    """바닥에서 tolerance_mm 이내인 정점을 "접지 노드"로 표시한 불리언 배열
    (정점 순서 1:1 대응)을 반환한다.

    - rest_on_floor() + finalize_mesh() 이후(스케일 완료) 메쉬 전제.
    - up_axis 기본 2(Z, finalize_mesh(z_up=True) 기본값과 일치). Y=높이면 1.
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
    prune_far_fragments_enabled: bool = True,
) -> tuple[trimesh.Trimesh, float]:
    """run_dense_pipeline()이 만든 원본 메쉬를 축 정렬+스케일링+바닥 정착까지
    마친 최종 메쉬로 만든다. run_pipeline()과 run_dense_pipeline.py가 공유.

    - z_up: 기본 True -- 내부 계산은 Y=높이, 최종 결과만 to_z_up() 적용.
    - trim_leg(False): trim_leg_above_ankle()로 다리 절단(정렬 후, 패턴
      뚜렷할 때만).
    - target_vertices(None): 지정 시 decimate_mesh()로 단순화+마감 스무딩 후
      rest_on_floor() 재적용(접지 재보정).
    - prune_far_fragments_enabled(True): 정렬 후 발 몸통에서 수평으로 멀리
      떨어진 파편(`prune_far_fragments()`)을 잘라낸다.
    - Returns: (정렬/스케일/접지 완료된 메쉬, 적용된 스케일 배율)
    """
    mesh, faces_before, faces_after = keep_largest_component(mesh)
    if faces_after < faces_before:
        print(f"[정리] 몸통과 떨어진 부유 조각 제거: 면 {faces_before:,} -> {faces_after:,}")

    mesh = align_sole_down(mesh)
    if trim_leg:
        mesh = trim_leg_above_ankle(mesh)
    if prune_far_fragments_enabled:
        mesh = prune_far_fragments(mesh)

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
    # 40 -- stl_foot_extract 쪽(finishing.py)의 10과 값이 다른 건 의도된 것.
    # 거긴 이 스무딩 뒤에 decimate_mesh()가 축약+재스무딩을 한 번 더 해서
    # 반복횟수 영향이 최종 결과에 거의 안 남지만(실측 확인됨), target_vertices를
    # 안 쓰는 이 경로는 이게 진짜 마지막 스무딩이라 반복횟수에 비례해 실제로
    # 수축한다(40회=PCA 길이 약 -1.5%, 실측). 둘을 맞추려 하지 말 것.
    finish_smooth_iterations: int = 40,
    prune_protrusions: bool = False,
    target_vertices: int | None = None,
    keep_intermediates: bool = False,
) -> Path:
    """위 단계 전부를 엮는 오케스트레이션. 최종 메쉬 경로를 반환한다.

    - sparse_dir(필수): sparse 재구성 폴더(largest_sparse_dir() 참고).
    - masks_dir(필수): masking.generate_masks(..., dilate=0) 결과.
    - refine(False): RefineMesh(느림) 실행 여부.
    - target_vertices(None): 지정하면 스무딩 전에 decimate_mesh()로 먼저
      축약한다. smooth_high_curvature_regions()가 쓰는
      trimesh.curvature.discrete_mean_curvature_measure()가 정점 수에
      선형보다 훨씬 가파르게 느려져(145k개서 224초, 20k개서 4.9초) 스무딩을
      저해상도에서 먼저 하는 쪽이 압도적으로 빠르다.
    - densify_resolution_level/densify_number_views_fuse(None): 그대로 전달.
    - visibility_filter_threshold(None): 음수면 가시성 필터 활성화.
    - grazing_filter_min_score(None): grazing 필터, visibility보다 먼저 적용.
    - reprojection_consistency_min_vote(None): 배경 오염 필터, 비권장.
    - free_space_support/thickness_factor/quality_factor: run_reconstruct_mesh() 전달.
    - refine_decimate/refine_regularity_weight: refine=True일 때 전달.
      resolution_level/scales는 항상 0/2 고정(낮추면 폭 치수 불안정).
    - smooth_high_curvature(True): curvature_* 로 강도 조절.
    - fill_holes/sand_surface_enabled(True): 구멍 메움/사포질 후처리.
    - sand_min_neighbors/sand_max_neighbors/sand_iterations: sand_surface() 강도.
    - finish_smooth(True): 라플라시안 마감 스무딩(잔여 고주파 노이즈 정리).
    - prune_protrusions(False): 포인트클라우드 단계 뿔 프루닝.
    - keep_intermediates(False): 중간 산출물 보존 여부.
    """
    # OpenMVS 서브프로세스는 -w(workdir)를 cwd로 실행 -- 다른 입력 경로는
    # 전부 절대경로로 넘겨야 함.
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
    # 안 씀 -- clean_dense_point_cloud의 prune_protrusions(기본 꺼짐)를 대신
    # 사용. 배경/파편 제거+스무딩은 mesh_postprocess.postprocess_mesh()로 위임.
    mesh = trimesh.load(mesh_ply, process=False)
    if target_vertices is not None:
        # 스무딩(특히 discrete_mean_curvature_measure)이 정점 수에 훨씬
        # 가파르게 느려지므로 먼저 축약 -- 마감 스무딩은 postprocess_mesh가
        # 이어서 하니 여기선 끔.
        mesh = decimate_mesh(mesh, target_vertices=target_vertices, smooth_after=False)
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
    """mesh_ply를 <workdir>/mesh.ply로 옮기고 workdir의 나머지(중간 산출물)는
    전부 지운다. keep_intermediates=False(기본)일 때만 호출.
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
