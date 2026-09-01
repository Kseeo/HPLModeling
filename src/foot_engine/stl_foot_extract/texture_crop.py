"""색/텍스처가 있는 메쉬(GLB 등)에서 다중 가상 시점 렌더 + 피부 분할 투표로
발을 자동으로 크롭한다.

- crop.py/branch_cut.py/locate.py는 색 없는 STL 전제라 이 신호를 못 씀
  (sfm.masking의 MediaPipe 피부 분할은 색 의존). 텍스처 입력에서는 그
  모델을 재사용 -- 가상 카메라 여러 곳에서 렌더 후 정점별 "피부로 보인
  횟수"를 투표(multi-view semantic fusion), 사진/카메라 포즈 불필요.
- 한계: glTF/GLB는 UV 이음매에서 정점이 복제돼(같은 3D 위치, 다른 UV) 위상
  인접이 끊김 -- merge_vertices(merge_tex=True)로 합쳐야 위상 기반 후처리
  (keep_largest_component 등)가 정상 동작(_weld_uv_seams() 참고).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import trimesh
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from foot_engine.sfm.masking import load_skin_segmenter, skin_only_mask

from .crop import _remove_vertices
from .finishing import _hole_diag_and_circularity


def load_textured_mesh(path) -> trimesh.Trimesh:
    """Scene(멀티 지오메트리)이든 단일 메쉬든 텍스처를 보존한 채 하나의 Trimesh로 합친다."""
    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Scene):
        return trimesh.util.concatenate(list(scene.geometry.values()))
    return scene


def _camera_transform_on_sphere(center: np.ndarray, radius: float, index: int, n_views: int) -> np.ndarray:
    """구면 위에 골든각으로 고르게 뿌린 `index`번째 카메라의 world 변환(4x4)을 만든다."""
    golden = np.pi * (3 - np.sqrt(5))
    t = index / max(n_views - 1, 1)
    z = 1 - 2 * t
    r_xy = np.sqrt(max(0.0, 1 - z * z))
    theta = golden * index
    cam_dir = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), z])
    cam_pos = center + cam_dir * radius

    forward = center - cam_pos
    forward /= np.linalg.norm(forward)
    up_guess = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up_guess)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = up
    transform[:3, 2] = -forward  # trimesh 카메라 규약: -Z를 바라봄
    transform[:3, 3] = cam_pos
    return transform


def multiview_skin_vote(
    mesh: trimesh.Trimesh,
    *,
    n_views: int = 16,
    resolution: tuple[int, int] = (640, 480),
    erode: int = 4,
    facing_dot_min: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """가상 카메라 `n_views`곳에서 렌더 -> 피부 분할 -> 정점별 투표를 집계한다.

    가시성(occlusion)은 실제 z-buffer 대신 "정점 법선이 카메라를 향하는가"로
    근사한다(비용 절약) -- 자기 가림에 취약하지만 여러 뷰를 합치면 상쇄된다.

    Returns:
        (frac, seen) -- `frac[v]`는 관측된 뷰 중 피부로 판정된 비율(0~1),
        `seen[v]`는 한 번이라도 관측됐는지. 안 보인 정점의 `frac`은 0(무의미,
        `seen`으로 걸러 쓸 것).
    """
    v = mesh.vertices
    n = len(v)
    normals = mesh.vertex_normals
    center = mesh.bounds.mean(axis=0)
    radius = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) * 0.9

    segmenter = load_skin_segmenter()
    skin_votes = np.zeros(n)
    seen_counts = np.zeros(n)

    for i in range(n_views):
        cam_to_world = _camera_transform_on_sphere(center, radius, i, n_views)
        scene = mesh.scene()
        scene.camera.resolution = resolution
        scene.camera_transform = cam_to_world
        try:
            png = scene.save_image(resolution=resolution)
        except Exception:
            continue
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        mask = skin_only_mask(segmenter, img, erode=erode)

        K = scene.camera.K
        world_to_cam = np.linalg.inv(cam_to_world)
        v_h = np.hstack([v, np.ones((n, 1))])
        v_cam = (world_to_cam @ v_h.T).T[:, :3]

        cam_pos = cam_to_world[:3, 3]
        view_dir = cam_pos - v
        view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-12
        facing = np.einsum("ij,ij->i", normals, view_dir) > facing_dot_min

        z_cam = -v_cam[:, 2]
        in_front = z_cam > 1e-6
        x_ndc = v_cam[:, 0] / np.maximum(z_cam, 1e-6)
        y_ndc = v_cam[:, 1] / np.maximum(z_cam, 1e-6)
        u = K[0, 0] * x_ndc + K[0, 2]
        px_v = K[1, 2] - K[1, 1] * y_ndc
        w, h = resolution
        in_frame = (u >= 0) & (u < w) & (px_v >= 0) & (px_v < h)

        valid = facing & in_front & in_frame
        idxs = np.where(valid)[0]
        ui = u[idxs].astype(np.int32)
        vi = px_v[idxs].astype(np.int32)
        pixel_skin = mask[vi, ui] > 0
        skin_votes[idxs] += pixel_skin
        seen_counts[idxs] += 1

    seen = seen_counts > 0
    frac = np.zeros(n)
    frac[seen] = skin_votes[seen] / seen_counts[seen]
    return frac, seen


def _typical_spacing(points: np.ndarray, tree: cKDTree | None = None) -> float:
    tree = tree or cKDTree(points)
    nn_dist, _ = tree.query(points, k=min(6, len(points)))
    return float(np.median(nn_dist[:, 1:])) if len(points) > 1 else 1.0


def _close_gaps(points: np.ndarray, mask: np.ndarray, *, radius_mult: float = 15.0) -> np.ndarray:
    """`mask`를 공간적으로 닫힘(팽창 후 침식) 처리해 작은 빈틈만 메운다.

    다중뷰 투표는 "관측 안 됨/애매함"으로 생긴 진짜 빈 구멍(정점 자체가 한
    번도 피부로 안 잡힘)을 남기는데, `finishing.fill_small_holes()`는 메쉬
    경계 루프 기준이라 이런 넓은 미관측 영역은 못 메운다. 팽창-침식은 경계는
    거의 그대로 두면서 팽창 반경보다 좁은 구멍만 골라 메운다.

    "반경 이내에 반대쪽 점이 있는가"를 최근접 1점 거리로 판정한다(수학적으로
    `query_ball_point`로 반경 안 전체 이웃을 모아 판정하는 것과 동일) --
    이전엔 이웃 목록 전체를 모아 파이썬 루프로 판정해 큰 반경(기본 25배
    간격)에서 반경 안 점 개수에 비례해 느려졌다(project 175d98a727c1
    실측: 53.7초 중 31.9초, 전체의 60%). 최근접 거리 판정은 반경 안 점이
    몇 개든 트리 탐색 1회로 끝나 반경 크기와 거의 무관하다(같은 입력 32초
    -> 0.1초 미만으로 확인).
    """
    tree = cKDTree(points)
    radius = _typical_spacing(points, tree) * radius_mult

    fg_idx = np.where(mask)[0]
    if len(fg_idx) == 0:
        return mask

    fg_tree = cKDTree(points[fg_idx])
    dist_to_fg, _ = fg_tree.query(points, k=1, workers=-1)
    dilated = mask | (dist_to_fg <= radius)

    bg_idx = np.where(~dilated)[0]
    if len(bg_idx) == 0:
        return dilated  # 배경이 하나도 안 남았으면 침식 대상 없음(원본과 동일 동작)

    bg_tree = cKDTree(points[bg_idx])
    dilated_idx = np.where(dilated)[0]
    dist_to_bg, _ = bg_tree.query(points[dilated_idx], k=1, workers=-1)

    eroded = dilated.copy()
    eroded[dilated_idx[dist_to_bg <= radius]] = False
    return eroded


def _largest_spatial_cluster(points: np.ndarray, *, radius_mult: float = 10.0) -> np.ndarray:
    """점들을 공간적 반경으로 묶어 가장 큰 덩어리의 로컬 인덱스를 반환한다.

    텍스처 메쉬는 UV 이음매 때문에 위상(topology) 기반 연결요소 판정이
    실제보다 훨씬 잘게 쪼개진다(모듈 docstring 참고) -- 공간 반경 기반이라야
    맞는 결과가 나온다.
    """
    tree = cKDTree(points)
    nn_dist, _ = tree.query(points, k=min(6, len(points)))
    typical_spacing = float(np.median(nn_dist[:, 1:])) if len(points) > 1 else 1.0
    radius = max(typical_spacing * radius_mult, 1e-9)

    pairs = tree.query_pairs(radius, output_type="ndarray")
    n = len(points)
    if len(pairs) == 0:
        return np.array([0]) if n else np.array([], dtype=np.int64)
    graph = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = csg.connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    biggest_label = int(sizes.argmax())
    return np.where(labels == biggest_label)[0]


def _bbox_pad_mask(mesh: trimesh.Trimesh, keep_points: np.ndarray, *, pad_ratio: float) -> np.ndarray:
    """`keep_points`의 bbox를 각 변 `pad_ratio`만큼 부풀린 상자 안에 있는
    `mesh` 정점의 불리언 마스크. 2-pass 재프레이밍의 대략적인 관심영역 크롭용."""
    mn = keep_points.min(axis=0)
    mx = keep_points.max(axis=0)
    pad = (mx - mn) * pad_ratio
    mn = mn - pad
    mx = mx + pad
    v = mesh.vertices
    return np.all((v >= mn) & (v <= mx), axis=1)


def _recover_holes_from_original(
    mesh: trimesh.Trimesh,
    keep_mask: np.ndarray,
    *,
    min_circularity: float = 0.35,
    max_diag_ratio: float = 0.3,
    recover_radius_mult: float = 2.0,
) -> np.ndarray:
    """구멍 경계를 평평한 패치로 메우는 대신, 투표에서 탈락했던 **원본 메쉬의
    진짜 정점 조각**을 그 구멍 위치에서 다시 끌어와 채운다.

    `_close_gaps()`는 전역적으로 한 반경만큼 팽창-침식하는 방식이라, 그
    반경보다 큰 구멍은 못 메우고(project_5 실측: 반경 25로도 안 닫힘) 반경을
    무작정 키우면 배경/다리까지 같이 끌려온다(실측 확인). 이 함수는 대신:
    1) 현재 채택 영역(`keep_mask`)의 경계 루프를 찾고
    2) 발목 절단면처럼 원래 열려있어야 하는 크고 비원형인 루프는 제외하고
       (`finishing.fill_round_holes`와 같은 원형도 기준)
    3) 남은(작고 둥근) 구멍마다 그 위치 주변의 **원본 정점**(투표 통과 여부
       무관)을 다시 채택 영역에 포함시킨다 -- 평평한 삼각분할이 아니라 실제
       스캔된 표면 조각을 갖다 붙이는 것.

    Returns:
        갱신된 `keep_mask`(원본 메쉬 정점 기준).
    """
    face_mask = keep_mask[mesh.faces].all(axis=1)
    kept_faces = mesh.faces[face_mask]
    if len(kept_faces) == 0:
        return keep_mask

    edges = np.vstack([kept_faces[:, [0, 1]], kept_faces[:, [1, 2]], kept_faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    boundary_groups = trimesh.grouping.group_rows(edges_sorted, require_count=1)
    if len(boundary_groups) < 3:
        return keep_mask
    boundary = edges[boundary_groups]

    loops = nx.cycle_basis(nx.from_edgelist(boundary.tolist()))
    if not loops:
        return keep_mask

    mesh_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    tree = cKDTree(mesh.vertices)
    out_mask = keep_mask.copy()
    n_recovered_loops = 0
    n_recovered_verts = 0

    for loop in loops:
        diag, circularity = _hole_diag_and_circularity(mesh, loop)
        diag_ratio = diag / mesh_diag
        if diag_ratio > max_diag_ratio or circularity < min_circularity:
            continue  # 발목 절단면 등 원래 열려있어야 하는 큰/비원형 경계로 판단, 건드리지 않음

        centroid = mesh.vertices[loop].mean(axis=0)
        radius = max(diag * 0.5 * recover_radius_mult, 1e-9)
        nearby = tree.query_ball_point(centroid, radius)
        new_idx = [i for i in nearby if not out_mask[i]]
        if new_idx:
            out_mask[new_idx] = True
            n_recovered_loops += 1
            n_recovered_verts += len(new_idx)

    if n_recovered_loops:
        print(
            f"[texture_crop] 원본 조각 복구: 구멍 {n_recovered_loops}개에 원본 정점 "
            f"{n_recovered_verts:,}개를 다시 채워넣음(평평한 패치 아님)"
        )
    return out_mask


def _weld_uv_seams(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """UV/노멀이 달라 복제된, 같은 3D 위치의 정점들을 하나로 합친다.

    분류가 끝나 텍스처가 더 이상 필요 없는 시점에 불러 위상 기반 후처리
    (`finishing.postprocess_mesh` 등)가 정상 동작하게 만든다.
    """
    out = mesh.copy()
    out.merge_vertices(merge_tex=True, merge_norm=True)
    return out


def _coarse_reframe(
    mesh: trimesh.Trimesh,
    *,
    n_views: int,
    resolution: tuple[int, int],
    vote_threshold: float,
    close_gap_radius_mult: float,
    spatial_cluster_radius_mult: float,
    pad_ratio: float,
) -> trimesh.Trimesh:
    """`extract_by_skin_vote(two_pass=True)`의 1차(대략) 패스: 느슨한 투표로 발이
    있을 법한 영역만 찾아 그 주변 bbox로 미리 잘라낸다. 후보가 하나도 없으면
    원본 메쉬를 그대로 반환(안전 폴백, 본 패스가 평소처럼 전체 씬에서 다시 시도)."""
    frac0, seen0 = multiview_skin_vote(mesh, n_views=n_views, resolution=resolution)
    fg0 = seen0 & (frac0 >= vote_threshold)
    if not fg0.any():
        print("[texture_crop] 1차(대략) 패스: 피부 후보 없음 -- 재프레이밍 없이 본 패스로 진행")
        return mesh

    if close_gap_radius_mult > 0:
        fg0 = _close_gaps(mesh.vertices, fg0, radius_mult=close_gap_radius_mult)

    fg0_idx = np.where(fg0)[0]
    cluster_local = _largest_spatial_cluster(mesh.vertices[fg0_idx], radius_mult=spatial_cluster_radius_mult)
    keep0 = fg0_idx[cluster_local]

    crop_mask = _bbox_pad_mask(mesh, mesh.vertices[keep0], pad_ratio=pad_ratio)
    n_before = len(mesh.vertices)
    reframed = _remove_vertices(mesh, crop_mask)
    if len(reframed.vertices) == 0:
        print("[texture_crop] 1차(대략) 패스: 재프레이밍 결과가 비어 있음 -- 원본으로 폴백")
        return mesh

    print(
        f"[texture_crop] 1차(대략) 패스로 재프레이밍: 정점 {n_before:,} -> {len(reframed.vertices):,} "
        f"(발이 각 렌더 화면을 더 크게 채우도록 씬을 미리 좁힘)"
    )
    return reframed


def _reject_color_material_outliers(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    frac: np.ndarray,
    *,
    min_boundary_crease_ratio: float = 1.3,
    min_cluster_ratio: float = 0.05,
    min_frac_margin: float = 0.05,
) -> np.ndarray:
    """채택 영역을 색상 2-클러스터링해 소재가 다른 조각(의자 등)을 걸러낸다.

    사람이 이런 오염을 눈으로 잡아내는 근거는 색뿐 아니라 "거기서 표면이
    꺾인다(다른 물체가 맞닿은 이음매)"는 신호도 같이 본다 -- 색 분리만으로는
    발 자체의 음영차(빛 받는 발등 vs 그늘진 옆면)도 똑같이 갈라져서 못 쓴다
    (실측: project_228/230도 색분리도 0.55 안팎으로 오염된 project_5와
    구분 안 됨). 그래서 색 경계의 이면각(dihedral angle) 중앙값이 메쉬
    전체 전형값보다 확실히 꺾여있을 때만(`min_boundary_crease_ratio`배 이상)
    이음매로 인정한다(실측: project_5 경계/전체 1.48배 vs 정상 샘플 두 개
    1.07~1.12배 -- 뚜렷하게 갈림).

    진짜 이음매로 판정되면 두 클러스터 중 `multiview_skin_vote()`의 원래
    피부투표 frac 평균이 더 높은 쪽을 발로 보고 남긴다(발 모양 점수로
    골랐다가 project_5에서 점수차 0.02 수준의 노이즈로 반대쪽을 고르는
    실패를 실측으로 확인해 교체) -- frac 평균 차이가 `min_frac_margin`
    미만이면(둘 다 똑같이 애매해 신뢰 불가) 아예 건드리지 않는다. project_5
    자체는 이 마진 조건에 걸려 그냥 통과할 가능성이 높음(별도로 확인된 텍스처
    색편향 때문에 분류 신뢰도가 전체적으로 낮아 어느 쪽도 확신을 못 줌) --
    이 옵션은 분류기 신뢰도가 정상 범위인 스캔에서의 이음매용.
    """
    idx = np.where(mask)[0]
    if len(idx) < 50:
        return mask
    try:
        colors = mesh.visual.to_color().vertex_colors[idx][:, :3].astype(np.float64) / 255.0
    except Exception:
        return mask  # 색 정보 없음 -- 판정 불가, 안 건드림

    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit(colors).labels_
    counts = np.bincount(labels, minlength=2)
    if counts.min() / len(labels) < min_cluster_ratio:
        return mask  # 한쪽이 너무 작으면 노이즈로 보고 안 건드림

    vertex_label = np.full(len(mesh.vertices), -1, dtype=np.int8)
    vertex_label[idx] = labels

    face_mask = mask[mesh.faces].all(axis=1)
    face_idx = np.where(face_mask)[0]
    if len(face_idx) == 0:
        return mask
    face_vote = vertex_label[mesh.faces[face_idx]].sum(axis=1)
    face_label = np.full(len(mesh.faces), -1, dtype=np.int8)
    face_label[face_idx] = (face_vote >= 2).astype(np.int8)  # 3정점 중 다수결

    fa = mesh.face_adjacency
    angles = np.degrees(mesh.face_adjacency_angles)
    both_in = (face_label[fa[:, 0]] >= 0) & (face_label[fa[:, 1]] >= 0)
    if not both_in.any():
        return mask
    overall_med = float(np.median(angles[both_in]))
    is_boundary = both_in & (face_label[fa[:, 0]] != face_label[fa[:, 1]])
    if not is_boundary.any() or overall_med < 1e-6:
        return mask
    boundary_med = float(np.median(angles[is_boundary]))
    if boundary_med < min_boundary_crease_ratio * overall_med:
        return mask  # 경계가 전형적 곡률보다 안 꺾여있음 -- 음영차로 보고 안 건드림

    frac0 = float(frac[idx[labels == 0]].mean())
    frac1 = float(frac[idx[labels == 1]].mean())
    if abs(frac0 - frac1) < min_frac_margin:
        return mask  # 어느 쪽이 발인지 신뢰할 만큼 갈리지 않음 -- 안 건드림
    keep_label = 0 if frac0 >= frac1 else 1

    new_mask = mask.copy()
    new_mask[idx[labels != keep_label]] = False
    print(
        f"[texture_crop] 색+이면각 이음매 감지: 경계 {boundary_med:.1f}도"
        f"(전체 {overall_med:.1f}도의 {boundary_med / overall_med:.2f}배), "
        f"피부투표 frac {max(frac0, frac1):.2f} vs {min(frac0, frac1):.2f} -- "
        f"소재가 다른 조각 정점 {int(counts[1 - keep_label]):,}개 제외"
    )
    return new_mask


@dataclass(slots=True)
class TextureCropResult:
    """`extract_by_skin_vote()`의 산출물."""

    mesh: trimesh.Trimesh
    n_input_vertices: int
    n_voted_skin: int
    n_final_vertices: int


def extract_by_skin_vote(
    mesh: trimesh.Trimesh,
    *,
    n_views: int = 16,
    resolution: tuple[int, int] = (640, 480),
    vote_threshold: float = 0.5,
    close_gap_radius_mult: float = 25.0,
    spatial_cluster_radius_mult: float = 10.0,
    two_pass: bool = False,
    coarse_vote_threshold: float = 0.7,
    coarse_pad_ratio: float = 0.2,
    coarse_n_views: int | None = None,
    coarse_close_gap_radius_mult: float = 0.0,
    coarse_spatial_cluster_radius_mult: float = 5.0,
    recover_holes: bool = False,
    recover_min_circularity: float = 0.35,
    recover_max_diag_ratio: float = 0.3,
    recover_radius_mult: float = 2.0,
    reject_color_outliers: bool = False,
    reject_min_boundary_crease_ratio: float = 1.3,
    reject_min_cluster_ratio: float = 0.05,
) -> TextureCropResult:
    """`multiview_skin_vote()` + 빈틈 닫힘 + 공간 클러스터링 + UV 이음매 용접까지 엮은 진입점.

    Args:
        vote_threshold: 관측된 뷰 중 이 비율 이상 피부로 보인 정점만 채택.
        close_gap_radius_mult: 빈틈 닫힘(_close_gaps()) 반경. 0이면 끔. 기본값
            25는 실측 근거(12일 때 발뒤꿈치/발끝에 진짜 구멍이 남는 경우 확인,
            25로 대부분 해소) -- 일부 샘플엔 반경을 키워도 안 없어지는 잔여
            구멍이 남는데, 그건 카메라 각도상 진짜 미관측 부위로 판단.
        spatial_cluster_radius_mult: 공간 클러스터링 반경(전형적 정점 간격의 배수).
        two_pass: 켜면 먼저 대략적인 1차 투표(느슨한 `coarse_vote_threshold`,
            적은 뷰 수)로 발이 있는 대략적 영역만 찾은 뒤, 그 영역 주변으로
            메쉬를 미리 잘라내고 나서 본 투표(`n_views`/`vote_threshold`)를
            다시 돈다. 배경(부스/구조물 등)이 씬 대부분을 차지하는 입력에서
            전체 씬 기준으로 카메라를 프레이밍하면 발이 각 렌더에서 아주 작은
            조각으로만 찍혀 피부 분류기 정확도가 떨어지는 문제(실측: 정점
            63,495개 중 투표통과 1,306개까지 떨어짐 -> 2-pass로 55%까지 회복)를
            크게 개선하지만, 반대로 배경이 적어 잘 되던 입력에서는 1차 패스가
            엉뚱한 작은 덩어리에 락온돼 오히려 가느다란 실 모양 아티팩트를
            만드는 회귀를 실측으로 확인함(project_230) -- 그래서 기본값은
            꺼짐. 배경이 씬 대부분을 차지하는 케이스에서만 켜서 쓸 것, 매번
            결과를 렌더로 확인 권장. 1차 투표에서 후보가 하나도 없으면 크롭
            없이 원본 그대로 본 투표로 진행(안전 폴백).
        coarse_vote_threshold: 1차(대략) 패스의 투표 임계값 -- 본 패스보다
            오히려 높게(기본 0.7) 잡는다. 배경이 씬 대부분을 차지하는 입력은
            낮은 임계값에서 배경(포스터의 사람/발 사진 등)까지 오탐으로 잡혀
            영역이 안 좁혀지는 경우가 실측으로 확인됨 -- 높은 임계값 + 빈틈닫힘
            끔(`coarse_close_gap_radius_mult=0`) + 좁은 클러스터링 반경으로
            "확실히 피부로 보이는 덩어리"만 위치 힌트로 쓴다(발 경계 자체를
            정확히 그리는 목적이 아님, 본 패스가 그건 다시 함).
        coarse_pad_ratio: 1차 패스로 찾은 영역의 bbox를 각 변 이 비율만큼
            부풀려서 크롭(기본 0.2 = 20% 여유, 실제 발 경계를 잘라내지 않게).
        coarse_n_views: 1차 패스 카메라 개수. 기본(None)은 `n_views//2`(최소 8) --
            대략적 위치만 찾으면 되므로 본 패스보다 적어도 됨(속도).
        coarse_close_gap_radius_mult: 1차 패스의 빈틈닫힘 반경(기본 0=끔) --
            본 패스와 달리 여기서 닫아버리면 배경과 발이 하나로 이어져 영역이
            안 좁혀질 수 있어 기본적으로 끈다.
        coarse_spatial_cluster_radius_mult: 1차 패스의 공간 클러스터링 반경
            (기본 5, 본 패스의 10보다 좁게) -- 배경과 발이 공간적으로 이어져
            같은 덩어리로 묶이는 걸 줄인다.
        recover_holes: 켜면 최종 채택 영역의 구멍 경계 중 발목 절단면처럼 크고
            비원형인 것은 두고, 작고 둥근 구멍만 그 위치의 원본 정점(투표
            통과 여부 무관)을 다시 끌어와 채운다(`_recover_holes_from_original()`
            참고) -- 평평한 삼각분할 패치가 아니라 실제 스캔된 표면을 붙이는
            방식. `_close_gaps()`의 전역 반경으로는 못 닫히는 큰 구멍(실측:
            project_5 뒤꿈치)에는 유효하지만, 구멍이 많은 복잡한 씬에서는
            원형도 기준을 통과하는 배경 조각까지 대량으로 같이 끌려오는 심각한
            회귀를 실측으로 확인함(project_235: 구멍 50개/정점 15,556개가
            배경까지 복구돼 온갖 방향으로 뿔이 뻗은 형태가 됨) -- 기본 꺼짐,
            project_5류(배경이 씬 대부분이라 구멍 자체가 적은 케이스)에서만
            켜고 반드시 렌더로 확인할 것.
        recover_min_circularity / recover_max_diag_ratio / recover_radius_mult:
            `_recover_holes_from_original()`로 그대로 전달.
        reject_color_outliers: 켜면 최종 채택 영역을 색상 2-클러스터링해,
            색 경계가 메쉬 전형적 곡률보다 확실히 꺾여있는(실제 이음매로
            보이는) 경우에만 발 모양 점수가 낮은 쪽을 제외한다
            (`_reject_color_material_outliers()` 참고) -- 의자 등 소재가
            다른 배경 조각이 발과 공간적으로 붙어 위상/거리 기반 정리로는
            안 갈라지는 케이스에 유효할 수 있음(실측: project_5에서 색+
            이면각 신호가 뚜렷하게 갈림 확인). 다만 정점 색이 없는 입력이나
            클러스터링이 실패하는 경우엔 조용히 원본을 그대로 반환(안전
            폴백)하고, 아직 실전 케이스로 최종 확인된 옵션은 아니라 기본
            꺼짐 -- 켜서 쓸 때 결과를 반드시 렌더로 확인할 것.
        reject_min_boundary_crease_ratio / reject_min_cluster_ratio:
            `_reject_color_material_outliers()`로 그대로 전달.
    """
    if two_pass:
        mesh = _coarse_reframe(
            mesh, n_views=coarse_n_views or max(8, n_views // 2), resolution=resolution,
            vote_threshold=coarse_vote_threshold, close_gap_radius_mult=coarse_close_gap_radius_mult,
            spatial_cluster_radius_mult=coarse_spatial_cluster_radius_mult, pad_ratio=coarse_pad_ratio,
        )

    frac, seen = multiview_skin_vote(mesh, n_views=n_views, resolution=resolution)
    fg = seen & (frac >= vote_threshold)
    n_voted = int(fg.sum())
    if n_voted == 0:
        raise RuntimeError("피부로 판정된 정점이 하나도 없습니다 -- vote_threshold를 낮추거나 n_views를 늘려보세요.")

    n_closed = n_voted
    if close_gap_radius_mult > 0:
        fg = _close_gaps(mesh.vertices, fg, radius_mult=close_gap_radius_mult)
        n_closed = int(fg.sum())

    fg_idx = np.where(fg)[0]
    cluster_local = _largest_spatial_cluster(mesh.vertices[fg_idx], radius_mult=spatial_cluster_radius_mult)
    keep_idx = fg_idx[cluster_local]

    final_mask = np.zeros(len(mesh.vertices), dtype=bool)
    final_mask[keep_idx] = True
    if recover_holes:
        final_mask = _recover_holes_from_original(
            mesh, final_mask, min_circularity=recover_min_circularity,
            max_diag_ratio=recover_max_diag_ratio, recover_radius_mult=recover_radius_mult,
        )
    if reject_color_outliers:
        final_mask = _reject_color_material_outliers(
            mesh, final_mask, frac, min_boundary_crease_ratio=reject_min_boundary_crease_ratio,
            min_cluster_ratio=reject_min_cluster_ratio,
        )
    cropped = _remove_vertices(mesh, final_mask)
    welded = _weld_uv_seams(cropped)

    print(
        f"[texture_crop] 다중뷰 피부투표: 정점 {len(mesh.vertices):,} -> "
        f"투표통과 {n_voted:,} -> 빈틈닫힘 {n_closed:,} -> 최대덩어리+용접 {len(welded.vertices):,}"
    )
    return TextureCropResult(
        mesh=welded, n_input_vertices=len(mesh.vertices),
        n_voted_skin=n_voted, n_final_vertices=len(welded.vertices),
    )
