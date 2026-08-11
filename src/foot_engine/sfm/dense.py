"""Sparse SfM 결과 기반 OpenMVS Dense 포인트클라우드/메쉬 복원 모듈.

메쉬 생성기 본체(2026-08-11부터, 템플릿 워프/SSM 대신 이 모듈이 만드는 dense
메쉬를 다듬고 경량화하는 쪽으로 결론남 — `archive/deformer_ssm_pipeline/`
참고). OpenMVS CLI 실행파일이 필요하며(`OPENMVS_BIN_DIR` 설정), CUDA 없이
CPU 빌드로 동작합니다.

파이프라인 흐름:
    undistort_for_dense() -> convert_masks_for_openmvs() -> run_interface_colmap()
    -> run_densify_point_cloud() -> clean_dense_point_cloud() -> run_reconstruct_mesh()
    -> run_refine_mesh() (선택) -> keep_largest_component()

주요 실측 결과 및 구현 규칙:
1. 마스크 처리: 배경 누출 방지를 위해 densify 이전에 마스크를 적용하며, dilate=0으로 설정합니다.
2. 노이즈 제거: DBSCAN은 발바닥 등 성긴 진짜 점을 삭제하므로 통계적 이상치 제거만 사용합니다.
   (가시성 필터/재투영 분류는 크래시 및 노이즈 필터링 한계로 사용하지 않음)
3. 메쉬 생성: 형태 보존을 위해 스무딩을 OFF(`--smooth 0`)하며, 크래시 방지를 위해 멀티스레드 수를 제한합니다.
   생성 후 부유 파편은 최대 연결 요소만 남겨 제거합니다.
4. 축 정렬: PCA 분산 순위 대신 상/하 평탄도 비대칭성을 측정하여 발바닥(-Y) 방향을 정렬합니다.

알려진 한계:
- 발바닥 미촬영 프로토콜 특성상 접지면에 큰 구멍(30%대)이 남습니다.
- 발목 부근의 뿔/스파이크 형태 돌출부 결함은 완전히 제거되지 않고 유지됩니다.
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np
import pycolmap
import trimesh
from scipy.spatial import cKDTree

from .geometry import pca_axes
from .reconstruction import filter_outlier_points

#: OpenMVS CLI 실행파일이 있는 폴더. 환경변수로 덮어쓸 것 -- 설치 방법은 README 참고.
DEFAULT_OPENMVS_BIN_DIR = os.environ.get("OPENMVS_BIN_DIR", "")

#: `DensifyPointCloud`의 간헐적 네이티브 크래시(원인 불명, 멀티스레드 경쟁
#: 상태로 추정) 재현을 막기 위한 기본 스레드 상한. 실측: 8로 낮추면 두 차례
#: 재현된 크래시(ACCESS_VIOLATION, 힙 손상)가 재발하지 않았다.
DEFAULT_MAX_THREADS = 8

#: `--postprocess-dmaps` 비트마스크: 1=remove-speckles, 2=fill-gaps.
#: 저텍스처 평면(발등/발바닥)의 깊이 추정 공백을 일부 메운다(실측: 점 13.8%↑).
DEFAULT_POSTPROCESS_DMAPS = 3


def _resolve_openmvs_bin(openmvs_bin: str | Path | None) -> Path:
    resolved = Path(openmvs_bin) if openmvs_bin else Path(DEFAULT_OPENMVS_BIN_DIR)
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"OpenMVS 실행파일 폴더를 찾을 수 없습니다: {resolved} -- "
            "OPENMVS_BIN_DIR 환경변수를 설정하거나 openmvs_bin 인자로 직접 넘기세요. "
            "설치 방법은 README의 'Dense MVS(선택)' 절 참고."
        )
    return resolved


def largest_sparse_dir(sparse_root: Path) -> Path:
    """`sparse/0`, `sparse/1`, ... 중 등록 이미지가 가장 많은 폴더를 찾는다.

    번호가 항상 크기순은 아니다(실측 확인: 어떤 촬영에서는 `sparse/1`이
    108장, `sparse/0`은 2장짜리 파편이었다) -- `sparse/0`을 무조건 가정하면
    안 된다.
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

    `pycolmap.undistort_images()`로 처리한다 -- 별도 `colmap.exe` 실행파일 없이도
    된다(실측 확인).
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
    output_name: str = "scene_dense.mvs",
) -> Path:
    """마스크 기반 dense 포인트클라우드를 만든다 (`scene_dense.ply` + `.mvs`).

    Args:
        masks_dir: `convert_masks_for_openmvs()`로 변환된(=`.mask.png`
            규칙) 마스크 폴더. 지정하면 배경 픽셀에서는 깊이 계산 자체를
            생략한다 -- 사후 필터링보다 근본적인 배경 제거(실측 확인:
            raw depth 계산량이 정확히 절반으로 줆). `masking.generate_masks()`를
            `dilate=0`으로 호출해 만들 것(모듈 docstring 2번 참고).
        postprocess_dmaps: 기본 3(remove-speckles+fill-gaps). 저텍스처 평면
            깊이 공백을 메운다(모듈 docstring 6번 참고). 0이면 비활성.
    """
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    args = [scene_mvs.name, "-o", output_name, "--max-threads", str(max_threads)]
    if postprocess_dmaps:
        args += ["--postprocess-dmaps", str(postprocess_dmaps)]
    if masks_dir is not None:
        args += ["--mask-path", str(Path(masks_dir).resolve()) + os.sep, "--ignore-mask-label", "0"]
    _run_openmvs("DensifyPointCloud.exe", args, workdir, bin_dir, "log_densify.txt")
    return workdir / output_name.replace(".mvs", ".ply")


def clean_dense_point_cloud(
    dense_ply_path: Path,
    out_path: Path,
    *,
    k: int = 8,
    std_ratio: float = 2.0,
    prune_protrusions: bool = False,
) -> tuple[int, int]:
    """dense 포인트클라우드에서 통계적 이상치를 제거한다 (DBSCAN 사용 안 함).

    OpenMVS의 dense PLY는 표준 필드(xyz/rgb/normal) 외에 커스텀 리스트
    속성(`view_indices`/`view_weights` -- 각 점을 관측한 카메라 가시성
    정보로, 이후 `ReconstructMesh`의 그래프컷 가중치 계산에 쓰인다)을 갖고
    있어 `trimesh`/`open3d`로 읽고 다시 쓰면 그 정보가 소실된다(둘 다
    조용히 무시함) -- 그 상태로 `ReconstructMesh -p`에 넘기면 크래시한다
    (실측 확인: ACCESS_VIOLATION). 그래서 원본 바이트를 그대로 보존한 채
    살아남는 점의 레코드만 골라 이어붙이는 방식으로 직접 파싱/저장한다.
    DBSCAN(최대 군집 유지)은 의도적으로 안 쓴다 -- 모듈 docstring 3번 참고
    Args:
        prune_protrusions: 위 경고 참고. 켜려면 `density_ratio`/`far_percentile`을
            대상 점군 규모에서 직접 검증한 뒤 켤 것 -- 기본은 꺼짐.

    Returns:
        (원본 점 개수, 정리 후 점 개수).
    """
    data = dense_ply_path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    body = data[header_end:]
    n = int([line for line in header.splitlines() if line.startswith("element vertex")][0].split()[-1])

    # 고정 필드 27바이트: x,y,z(f4)*3 + red,green,blue(u1)*3 + nx,ny,nz(f4)*3.
    # 그 뒤로 가변 길이 리스트 두 개(view_indices: u1 count + u4*count,
    # view_weights: u1 count + f4*count)가 이어진다 -- OpenMVS dense PLY 고정 스키마.
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
            f"{dense_ply_path} 파싱 실패 -- 예상 스키마(xyz+rgb+normal+view_indices"
            "+view_weights)와 다른 형식일 수 있습니다."
        )

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
        if n_protrusion:
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
        smooth: 기본 0(끔). OpenMVS 기본값(2)이 형상 정교함을 깎아낸다는
            육안 확인 있음(모듈 docstring 5번 참고) -- raw 메쉬가 더 정확했다.
        free_space_support / thickness_factor / quality_factor: 그래프컷
            가중치 튜닝(모듈 docstring 13번 참고) -- 기본값은 OpenMVS 자체
            기본값 그대로(꺼짐/1.0), 아직 실측 검증 전이라 파이프라인
            기본 경로는 안 건드린다.
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
    """OpenMVS 내장 가시성 기반 점군 필터(모듈 docstring 12번 참고, 아직 실측 검증 전).

    `DensifyPointCloud`를 densify 없이 필터 전용 모드로 재호출한다 -- 소스
    확인 결과 `--filter-point-cloud`는 `DenseReconstruction()`(실제 densify)
    *이전*에 체크되므로, densify와 같은 호출에 넣으면 아무 점군도 없는
    상태에서 필터가 실행돼 무효하다. 반드시 이미 만들어둔 점군을 `-p`로
    불러들이는 별도 2차 호출이어야 한다.

    Args:
        point_cloud_ply: 필터링할 기존 점군(`clean_dense_point_cloud()` 출력 등).
        threshold: 음수만 필터를 발동시킨다(0 이상이면 OpenMVS가 그냥
            건너뜀 -- 소스 확인). 절댓값이 클수록 더 공격적으로 제거.
            기본 -1은 보수적인 시작값 -- 실측 없이 더 큰 값을 기본으로
            두지 않는다.

    Returns:
        필터링된 점군 경로(OpenMVS가 `<output_name>_filtered.ply`로 저장).
    """
    if threshold >= 0:
        raise ValueError(f"threshold는 음수여야 필터가 발동합니다(소스 확인) -- 받은 값: {threshold}")
    print(
        "[dense][경고] filter_point_cloud_visibility()는 실측 결과(2026-08-11, test03) "
        "이 출력을 run_reconstruct_mesh()에 넘기면 ACCESS_VIOLATION으로 재현 가능하게 "
        "죽는 걸로 확인됐다(모듈 docstring 12번 참고) -- 원인 미해결. 계속 진행함."
    )
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    _run_openmvs(
        "DensifyPointCloud.exe",
        [scene_mvs.name, "-p", point_cloud_ply.name, "-o", output_name, "--filter-point-cloud", str(threshold)],
        workdir, bin_dir, "log_filter_point_cloud.txt",
    )
    return workdir / f"{output_name}_filtered.ply"


def run_refine_mesh(
    scene_mvs: Path,
    mesh_ply: Path,
    workdir: Path,
    *,
    openmvs_bin: str | Path | None = None,
    output_name: str = "scene_mesh_refined.mvs",
) -> Path:
    """사진 광도일관성 기반으로 메쉬 정점 위치를 보정한다 (선택적, 가장 느린 단계).

    CPU로도 동작한다(`--cuda-device` 기본값 -2). 노이즈/뿔 형태 결함을
    실제로 줄이는 효과가 육안 확인됐지만, 전체 파이프라인 소요시간의
    70~72%를 차지하는 압도적 병목이다(실측: 1.5~9분) -- 빠른 반복
    실험에서는 건너뛸 것.
    """
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    _run_openmvs(
        "RefineMesh.exe",
        [scene_mvs.name, "-m", mesh_ply.name, "-o", output_name],
        workdir, bin_dir, "log_refine_mesh.txt",
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
    """몸통에 이어져(fused) 붙은 뿔/스파이크를 통째로 잘라낸다(모듈 docstring 11번).

    `keep_largest_component()`는 이미 분리된 파편만 잡는다 -- 뿔은 몸통과
    같은 연결 요소라 무력하다. 거리 percentile 컷만으로는 뿔의 끝부분만
    잘리고 몸통에 이어진 뿌리는 남는다(실측 확인). 대신 "뿔은 국소 정점
    밀도가 몸통보다 뚜렷이 낮다"는 걸 핵심 단서로 쓴다 -- 얇고 길쭉한
    구조라 단위 부피당 정점 수가 적다:

        1. 각 정점의 반경 `density_radius_nn_mult * 전형적 10-최근접 간격`
           안 이웃 수(밀도) 계산 -- PCA 기반 `measured_length()`나 중심
           거리 같은 "물리적 크기" 척도는 안 쓴다. 전자는 뿔 자체가 축을
           왜곡시키는 순환 참조 문제가 있고(모듈 docstring 11번), 후자는
           실측해보니 점 밀도 분포(중심 근처에 몰림)에 좌우돼 반경이
           지나치게 작아지는 문제가 있었다(실측: median 정점간 거리
           기준으로 반경을 잡으니 밀도 median이 1.0으로 무너짐) --
           메쉬 해상도(정점 간격)에 비례하는 반경이라야 정점 수/전체
           크기가 달라져도 "밀도" 수치가 일관된 의미를 가진다.
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
    나머지는 배경에서 떨어져 나온 작은 파편이다(모듈 docstring 10번 참고).
    이미 공간적으로 분리된 덩어리 단위로만 자르므로, 3번 항목의 DBSCAN
    버그(성긴 진짜 부위를 노이즈로 오판)와 달리 안전하다 -- 다만 발 표면에
    이어져(fused) 붙은 경계 노이즈는 같은 덩어리라 이걸로 안 떨어진다.

    Returns:
        (필터링된 메쉬, 원본 face 수, 남은 face 수).
    """
    total_faces = len(mesh.faces)
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh, total_faces, total_faces
    largest = max(components, key=lambda c: len(c.faces))
    return largest, total_faces, len(largest.faces)


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


def _flatness(points: np.ndarray) -> float:
    """점 뭉치의 평탄도 -- PCA 고유값 중 가장 작은 것 / 가장 큰 것.

    0에 가까울수록 평평한 판 모양(발바닥 후보), 1에 가까울수록 등방적
    (구형)이거나 굴곡진 형태(발등/발목 후보)다.
    """
    c = points - points.mean(axis=0)
    cov = c.T @ c
    ev = np.linalg.eigvalsh(cov)
    return float(ev[0] / max(ev[-1], 1e-9))


def find_sole_direction(
    local_vertices: np.ndarray,
    *,
    candidate_axes: tuple[int, ...] = (1, 2),
    percentiles: tuple[float, ...] = (8.0, 15.0, 20.0),
) -> tuple[int, float]:
    """PCA 정렬된 로컬 좌표에서 발바닥(접지면) 축과 방향(부호)을 추정한다.

    핵심 가정: 발바닥은 지지면이라 상대적으로 평평하고, 반대쪽(발등/발목)은
    아치·발목 돌출 때문에 굴곡이 있다 -- 즉 진짜 "높이(위/아래)" 축은 한쪽
    끝은 평평하고 반대쪽 끝은 굴곡진 **비대칭**을 보이는 반면, "너비" 축은
    양쪽(안쪽/바깥쪽 복사뼈 라인)이 둘 다 어느 정도 굴곡져 있어 비대칭이
    약하다. PCA 분산 순위(축1=중간, 축2=최소)만으로 어느 게 높이인지
    가정하지 않는다 -- 실측 확인(test03, 2026-08-11): SfM 재구성 노이즈
    때문에 분산 순위가 실제 길이/너비/높이 순서와 안 맞는 경우가 있었다
    (축1/축2 고유값비가 1.6배로, 폭:높이 실측 비율(약 1.5~2배 표준편차,
    고유값으론 제곱이라 3배 이상 기대)보다 훨씬 덜 벌어짐). 대신 두 후보
    축 모두에 대해 이 평탄도 비대칭을 실제로 측정해, 비대칭이 더 크고
    percentile 크기에 걸쳐(8/15/20%) 더 일관된 쪽을 높이 축으로 뽑는다.

    Args:
        local_vertices: PCA 정렬 로컬 좌표(중심이 원점, 열 순서가 분산
            내림차순인 축과 일치해야 함) -- 보통 `align_sole_down()`이
            내부에서 만들어 넘긴다.
        candidate_axes: 높이 축 후보 열 인덱스. 기본 (1, 2) -- 축0(최대
            분산)은 발 길이 축이 거의 확실해 후보에서 제외.

    Returns:
        (height_axis_idx, sign) -- `local_vertices[:, height_axis_idx] * sign`이
        "위(발등 쪽)"가 되도록 하는 부호. 호출자는 이 축을 최종 Y로 보내고
        부호를 반전해(발바닥이 -Y) 정렬할 것.
    """
    best_axis, best_score, best_sign = candidate_axes[0], -1.0, 1.0
    for axis_idx in candidate_axes:
        h = local_vertices[:, axis_idx]
        asymmetries: list[float] = []
        votes: list[float] = []
        for pct in percentiles:
            lo = local_vertices[h <= np.percentile(h, pct)]
            hi = local_vertices[h >= np.percentile(h, 100.0 - pct)]
            if len(lo) < 10 or len(hi) < 10:
                continue
            f_lo, f_hi = _flatness(lo), _flatness(hi)
            asymmetries.append(abs(f_lo - f_hi))
            votes.append(1.0 if f_lo < f_hi else -1.0)  # lo가 더 평평하면 "위"는 +쪽
        if not asymmetries:
            continue
        score = float(np.mean(asymmetries))
        sign = 1.0 if sum(v > 0 for v in votes) >= len(votes) / 2 else -1.0
        if score > best_score:
            best_axis, best_score, best_sign = axis_idx, score, sign
    return best_axis, best_sign


def align_sole_down(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """`align_principal_axes()`의 다음 단계 -- 발바닥까지 검출해 최종 좌표계를
    X=길이축, Y=높이축(발바닥이 -Y), Z=너비축으로 맞춘다(중심은 원점).

    `find_sole_direction()`(모듈 docstring 참고)으로 높이 축과 부호를
    실측 기반 평탄도 비대칭으로 추정한다. **주의**: 지금 촬영 프로토콜은
    발바닥을 직접 못 찍어(모듈 상단 "알려진 한계" 참고) 그 부위 표면이
    그래프컷이 메운 결과일 수 있다 -- 그래도 실측 확인(test03,
    2026-08-11)해보니 발바닥 쪽이 발등/발목 쪽보다 뚜렷이 더 평평하게
    나와 이 휴리스틱이 신호를 잡아낸다. 다만 매 실행마다 검증된 건
    아니므로, 최종 산출물에 쓰기 전 실제 뷰어로 확인할 것.
    """
    centroid = mesh.vertices.mean(axis=0)
    c = mesh.vertices - centroid
    cov = c.T @ c
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    sorted_axes = eigvecs[:, order]
    local = c @ sorted_axes

    height_idx, sign = find_sole_direction(local)
    width_idx = 2 if height_idx == 1 else 1

    final_axes = np.stack([
        sorted_axes[:, 0],
        sorted_axes[:, height_idx] * sign,  # sign은 "축*sign=위(발등)" 정의(find_sole_direction 참고) -- 그대로 Y로
        sorted_axes[:, width_idx],
    ], axis=1)
    if np.linalg.det(final_axes) < 0:
        final_axes = final_axes.copy()
        final_axes[:, -1] *= -1.0  # 너비축(이미 결정 근거 없음) 부호만 뒤집어 순수 회전 유지

    aligned = mesh.copy()
    aligned.vertices = c @ final_axes
    return aligned


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
    visibility_filter_threshold: int | None = None,
    free_space_support: bool = False,
    thickness_factor: float = 1.0,
    quality_factor: float = 1.0,
) -> Path:
    """위 단계 전부를 엮는 오케스트레이션. 최종 메쉬 경로를 반환한다.

    Args:
        sparse_dir: `reconstruction.run_sparse_sfm()`이 만든 sparse 재구성
            폴더(예: `<workdir>/sparse/0`). 폴더 번호가 가장 큰 것이라는
            보장이 없으면 `largest_sparse_dir()`로 먼저 확인할 것.
        masks_dir: `masking.generate_masks(..., dilate=0)`로 만든 마스크
            폴더 (dilate 값이 커지면 배경 경계 노이즈가 늘어난다 -- 모듈
            docstring 2번 참고).
        refine: RefineMesh(사진 광도일관성 보정)까지 돌릴지. 기본 False --
            전체 시간의 70% 이상을 차지하는 병목이라(모듈 docstring 7번),
            빠른 확인이 필요하면 끄고 최종 산출물만 켤 것.
        visibility_filter_threshold: `filter_point_cloud_visibility()`를
            추가로 돌릴지(모듈 docstring 12번 참고). `None`(기본)이면 안
            돌림 -- 아직 실측 검증 전. 음수 값(예: -1)을 주면 활성화.
        free_space_support / thickness_factor / quality_factor:
            `run_reconstruct_mesh()`로 그대로 전달(모듈 docstring 13번 참고).
            아직 실측 검증 전이라 기본값은 OpenMVS 기본값 그대로.
    """
    # OpenMVS 서브프로세스는 -w(workdir)를 cwd로 실행되므로, 그 외 입력 경로는
    # 전부 절대경로로 넘겨야 한다 -- 상대경로를 그대로 두면 cwd가 바뀐 뒤
    # 엉뚱한 곳을 가리키게 된다(실측으로 확인된 버그, 조용히 실패하고 OpenMVS
    # 자체 로그도 안 남아 원인 파악이 어려웠다).
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
    )
    cleaned_ply = openmvs_dir / "scene_dense_cleaned.ply"
    clean_dense_point_cloud(dense_ply, cleaned_ply)

    mesh_input_ply = cleaned_ply
    if visibility_filter_threshold is not None:
        mesh_input_ply = filter_point_cloud_visibility(
            scene_mvs, cleaned_ply, openmvs_dir,
            threshold=visibility_filter_threshold, openmvs_bin=openmvs_bin,
        )

    mesh_ply = run_reconstruct_mesh(
        scene_mvs, mesh_input_ply, openmvs_dir, openmvs_bin=openmvs_bin,
        free_space_support=free_space_support, thickness_factor=thickness_factor,
        quality_factor=quality_factor,
    )

    if refine:
        mesh_ply = run_refine_mesh(scene_mvs, mesh_ply, openmvs_dir, openmvs_bin=openmvs_bin)

    # 뿔/스파이크 사후 프루닝(prune_thin_protrusions)은 메쉬 위상을 망가뜨려
    # 버렸다(모듈 docstring 11번) -- 점 단위 사전 제거(clean_dense_point_cloud
    # 의 prune_protrusions, 기본 꺼짐) 또는 위 visibility_filter_threshold 중
    # 하나를 실측으로 검증해 켤 것. 여기서는 이미 분리된 부유 파편만 정리한다.
    mesh = trimesh.load(mesh_ply, process=False)
    mesh, faces_before, faces_after = keep_largest_component(mesh)
    if faces_after < faces_before:
        print(f"[dense] 부유 파편 제거: {faces_before:,} -> {faces_after:,} faces")
        mesh.export(mesh_ply)

    print(f"[dense] 완료: {mesh_ply}")
    return mesh_ply
