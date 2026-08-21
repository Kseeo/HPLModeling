"""색/텍스처가 있는 메쉬(GLB 등)에서 다중 가상 시점 렌더 + 피부 분할 투표로
발을 자동으로 크롭한다.

`crop.py`/`branch_cut.py`/`locate.py`는 색 없는 STL 전제라 이 신호를 못 쓴다
(`sfm.masking`의 MediaPipe 피부 분할은 색에 의존). 텍스처가 있는 입력에서는
이미 검증된 그 모델을 그대로 재사용할 수 있다 -- 가상 카메라 여러 곳에서
렌더링해 각 정점에 "몇 번 피부로 보였는지" 투표시키는 방식(multi-view
semantic fusion)이며, 사진/실제 카메라 포즈가 전혀 필요 없다(메쉬 하나만 입력).

한계: glTF/GLB는 UV 이음매에서 정점이 복제돼 있어(같은 3D 위치, 다른 UV) 위상
인접(topology)이 그 이음매에서 끊긴다 -- `merge_vertices(merge_tex=True)`로
복제를 합친 뒤에야 `keep_largest_component()` 등 위상 기반 후처리가 정상
동작한다(`_weld_uv_seams()` 참고).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import trimesh
from scipy.spatial import cKDTree

from foot_engine.sfm.masking import load_skin_segmenter, skin_only_mask

from .crop import _remove_vertices


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
    """
    tree = cKDTree(points)
    radius = _typical_spacing(points, tree) * radius_mult

    fg_idx = np.where(mask)[0]
    if len(fg_idx) == 0:
        return mask
    dilated = mask.copy()
    for idxs in tree.query_ball_point(points[fg_idx], radius):
        dilated[idxs] = True

    eroded = dilated.copy()
    dilated_idx = np.where(dilated)[0]
    for i, idxs in zip(dilated_idx, tree.query_ball_point(points[dilated_idx], radius)):
        if not dilated[idxs].all():
            eroded[i] = False
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


def _weld_uv_seams(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """UV/노멀이 달라 복제된, 같은 3D 위치의 정점들을 하나로 합친다.

    분류가 끝나 텍스처가 더 이상 필요 없는 시점에 불러 위상 기반 후처리
    (`finishing.postprocess_mesh` 등)가 정상 동작하게 만든다.
    """
    out = mesh.copy()
    out.merge_vertices(merge_tex=True, merge_norm=True)
    return out


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
) -> TextureCropResult:
    """`multiview_skin_vote()` + 빈틈 닫힘 + 공간 클러스터링 + UV 이음매 용접까지 엮은 진입점.

    Args:
        vote_threshold: 관측된 뷰 중 이 비율 이상 피부로 보인 정점만 채택.
        close_gap_radius_mult: 빈틈 닫힘(`_close_gaps()`) 반경. `0`이면 끈다. 기본값 25는
            실측(5샘플, 경계 루프 개수/크기 비교)으로 정함 -- 12(이전 기본값)에서는
            발뒤꿈치/발끝처럼 카메라 각도상 관측이 드문 부위에 진짜 구멍(정점 수백 개
            규모)이 남는 경우가 있었고, 25로 올리면 대부분 사라짐(228: 구멍 4개 -> 1개,
            그 1개도 다리 위쪽 관측 경계라 정상). 다만 모든 샘플에서 완전히 닫히는 건
            아님(229/232는 작은 잔여 구멍이 남음) -- 그 정도는 반경을 더 키워도 안
            없어져서, 카메라가 그 각도에서 아예 못 본(진짜 미관측) 부위일 가능성이 높음.
        spatial_cluster_radius_mult: 공간 클러스터링 반경(전형적 정점 간격의 배수).
    """
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
