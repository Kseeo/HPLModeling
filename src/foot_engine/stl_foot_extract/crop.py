"""사람이 좌표를 짚어(또는 자동 후보를 받아) 메쉬를 국소적으로 자르는 도구.

전부 순수 기하 연산 -- 사진/카메라 정보 불필요. `locate.py`의 자동 후보나
`picker.py`의 사람 클릭 좌표를 받아 실제로 잘라내는 실행 담당.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _remove_vertices(mesh: trimesh.Trimesh, keep_mask: np.ndarray) -> trimesh.Trimesh:
    """정점 부분집합만 남기고 나머지를 안전하게 잘라낸다.

    - Trimesh.update_vertices(keep_mask)를 직접 부르면 안 됨 -- 제거된
      정점의 face 인덱스를 0으로 재매핑해 가짜 스파이크 면이 생김(실측:
      최대 edge 길이가 median의 100배 이상 치솟음 확인).
    - update_faces()로 제거될 정점을 참조하는 face를 먼저 지운 뒤 정점 제거.
    """
    face_mask = keep_mask[mesh.faces].all(axis=1)
    out = mesh.copy()
    out.update_faces(face_mask)
    out.remove_unreferenced_vertices()
    return out


def crop_around_seed(
    mesh: trimesh.Trimesh,
    seed_point: tuple[float, float, float] | np.ndarray,
    radius: float,
) -> tuple[trimesh.Trimesh, int]:
    """씨앗점에서 유클리드 반지름 `radius` 밖의 정점을 전부 잘라낸다.

    반경 안 정점만 남기고 잘라내므로 절단면이 남을 수 있다 -- 이어서
    `finishing.keep_largest_component()`로 경계에서 떨어져나간 조각을 정리할 것.

    Args:
        seed_point: (x, y, z) -- 메쉬 자체 좌표계 기준(스케일링 전).
        radius: 이 반경보다 먼 정점은 제거.

    Returns:
        (크롭된 메쉬, 제거된 정점 수).
    """
    seed = np.asarray(seed_point, dtype=np.float64)
    dist = np.linalg.norm(mesh.vertices - seed, axis=1)
    remove_mask = dist > radius
    n_removed = int(remove_mask.sum())
    if n_removed == 0:
        return mesh, 0

    out = _remove_vertices(mesh, ~remove_mask)
    print(
        f"[crop] 씨앗점 크롭(반지름 {radius:g}): 정점 {n_removed:,}개 제거 "
        f"({len(mesh.vertices):,} -> {len(out.vertices):,})"
    )
    return out, n_removed


def remove_near_point(
    mesh: trimesh.Trimesh,
    point: tuple[float, float, float] | np.ndarray,
    radius: float,
) -> tuple[trimesh.Trimesh, int]:
    """`point` 반경 **안쪽** 정점만 지운다 -- `crop_around_seed()`의 반대(그쪽은
    반경 바깥을 지움).

    발과 물리적으로 떨어져 있는데도 자동 필터가 못 잡는 배경 덩어리를,
    사람이 그 위 지점을 직접 짚어 국소적으로 잘라낼 때 쓴다.

    Args:
        point: 지울 위치 중심(x, y, z).
        radius: 이 반경 안 정점을 전부 제거.

    Returns:
        (정리된 메쉬, 제거된 정점 수).
    """
    p = np.asarray(point, dtype=np.float64)
    dist = np.linalg.norm(mesh.vertices - p, axis=1)
    remove_mask = dist <= radius
    n_removed = int(remove_mask.sum())
    if n_removed == 0:
        return mesh, 0

    out = _remove_vertices(mesh, ~remove_mask)
    print(
        f"[crop] 국소 배경 제외(반지름 {radius:g}): 정점 {n_removed:,}개 제거 "
        f"({len(mesh.vertices):,} -> {len(out.vertices):,})"
    )
    return out, n_removed


def crop_to_region(
    mesh: trimesh.Trimesh,
    region_points: np.ndarray,
    margin: float,
) -> tuple[trimesh.Trimesh, int]:
    """`region_points`(공간적으로 뭉친 표면 샘플점 집합, `locate.find_dense_regions()`
    참고) 근방(거리 `margin` 이내)의 정점만 남긴다.

    `crop_around_seed()`(점 하나 + 구)와 달리 구역의 실제(구가 아닌) 모양을
    그대로 따라가며 자른다 -- 구역이 길쭉하거나 휘어 있어도 그 형태를 보존한다.

    Args:
        region_points: 남길 구역을 대표하는 표면 샘플점들.
        margin: 이 거리보다 먼 정점은 제거 -- 표면 샘플이 성긴 만큼(밀도 판정
            반경 정도) 여유를 둘 것.

    Returns:
        (크롭된 메쉬, 제거된 정점 수).
    """
    tree = cKDTree(region_points)
    dist, _ = tree.query(mesh.vertices)
    remove_mask = dist > margin
    n_removed = int(remove_mask.sum())
    if n_removed == 0:
        return mesh, 0

    out = _remove_vertices(mesh, ~remove_mask)
    print(
        f"[crop] 구역 기반 크롭(여유 {margin:g}): 정점 {n_removed:,}개 제거 "
        f"({len(mesh.vertices):,} -> {len(out.vertices):,})"
    )
    return out, n_removed


def trim_by_plane(
    mesh: trimesh.Trimesh,
    ankle_point: tuple[float, float, float] | np.ndarray,
    leg_direction_point: tuple[float, float, float] | np.ndarray,
) -> tuple[trimesh.Trimesh, int]:
    """`ankle_point`를 지나는 평면으로 메쉬를 자른다 -- `leg_direction_point`가
    있는 쪽(다리)을 버리고 `ankle_point` 쪽(발)만 남긴다.

    전역 PCA 자동 판정(다리 길이가 발 길이와 비슷한 스캔에서 엉뚱한 지점을
    자를 수 있음, 실측 확인)의 대안 -- 사람이 발목 지점과 다리 위쪽 아무
    지점을 순서대로 짚어 얻은 두 좌표로 자른다.

    Args:
        ankle_point: 자를 위치(발목). 이쪽이 남는다.
        leg_direction_point: 다리 위쪽 아무 지점 -- 평면 법선 방향만 정함.

    Returns:
        (잘린 메쉬, 제거된 정점 수).
    """
    ankle = np.asarray(ankle_point, dtype=np.float64)
    leg_dir_pt = np.asarray(leg_direction_point, dtype=np.float64)
    normal = ankle - leg_dir_pt
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError("ankle_point과 leg_direction_point가 너무 가까워 방향을 정할 수 없습니다.")
    normal = normal / norm

    n_before = len(mesh.vertices)
    trimmed = mesh.slice_plane(ankle, normal, cap=True)
    if trimmed is None or len(trimmed.vertices) == 0:
        print("[crop] 평면 크롭 결과가 비어(발목 지점이 메쉬 밖일 수 있음) 원본을 유지합니다")
        return mesh, 0
    n_removed = n_before - len(trimmed.vertices)
    print(f"[crop] 평면 크롭(발목 기준): 정점 {n_before:,} -> {len(trimmed.vertices):,}")
    return trimmed, n_removed
