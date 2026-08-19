"""말단(가지 끝)에서 몸통 쪽으로 걸어 들어가며 국소 진행 방향이 꺾이는 지점
(코너)을 찾아 그 자리에서 위상적으로 잘라 조각을 분리한다.

절대 방향(중력 등) 기준이 아니라 각 말단 자체의 진행 방향 변화만 본다 --
의자 다리처럼 발과 다른 각도로 이어진 부분을 자동으로 떼어내는 후보 생성기.
확정 판별기가 아니므로 나온 조각들은 사람이 확인해서 골라야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import dijkstra

from .crop import _remove_vertices
from .locate import _sphericity, _toe_cluster_score


def _geodesic_graph(mesh: trimesh.Trimesh) -> sp.csr_matrix:
    """엣지 길이 가중 인접 그래프(다익스트라 입력용)."""
    edges = mesh.edges_unique
    v = mesh.vertices
    elen = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    n = len(v)
    return sp.csr_matrix(
        (np.concatenate([elen, elen]),
         (np.concatenate([edges[:, 0], edges[:, 1]]), np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(n, n),
    )


@dataclass(slots=True)
class BranchTip:
    """말단 하나와, 그 지점 기준 전체 정점까지의 지오데식 거리."""

    vertex: int
    dist_from_tip: np.ndarray


def find_branch_tips(
    mesh: trimesh.Trimesh,
    *,
    top_k: int = 6,
    min_separation_ratio: float = 0.15,
) -> list[BranchTip]:
    """서로 지오데식으로 멀리 떨어진 말단들을 반복적 최원점 샘플링으로 찾는다.

    무게중심에 가장 가까운 정점에서 시작해, 지금까지 찾은 모든 지점으로부터
    가장 먼 정점을 하나씩 추가한다(farthest point sampling).
    """
    graph = _geodesic_graph(mesh)
    v = mesh.vertices
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    min_sep = diag * min_separation_ratio

    start = int(np.argmin(np.linalg.norm(v - v.mean(axis=0), axis=1)))
    tips: list[BranchTip] = []
    min_dist_to_chosen = dijkstra(graph, indices=start, directed=False)

    for _ in range(top_k):
        candidate = int(np.argmax(min_dist_to_chosen))
        if tips and min_dist_to_chosen[candidate] < min_sep:
            break
        d = dijkstra(graph, indices=candidate, directed=False)
        tips.append(BranchTip(vertex=candidate, dist_from_tip=d))
        min_dist_to_chosen = np.minimum(min_dist_to_chosen, d)

    return tips


@dataclass(slots=True)
class BendCut:
    """말단 하나에서 찾은 코너(꺾임) 지점."""

    tip_vertex: int
    cut_distance: float
    bend_angle_deg: float


def find_bend_cuts(
    mesh: trimesh.Trimesh,
    tips: list[BranchTip],
    *,
    band_width_edge_mult: float = 6.0,
    corner_angle_deg: float = 55.0,
    smoothing_bands: int = 2,
    max_search_ratio: float = 0.5,
    min_band_vertices: int = 4,
) -> list[BendCut]:
    """각 말단에서 몸통 쪽으로 걸어 들어가며 진행 방향이 처음 꺾이는 지점을 찾는다.

    코너를 못 찾은 말단은 건너뛴다(직선으로 곧게 이어진 경우 -- 자를 근거 없음).

    Args:
        band_width_edge_mult: 진행 방향을 구할 구간 폭(전형적 엣지 길이의 배수).
        corner_angle_deg: 직전 평균 방향과 이 각도 이상 벌어지면 코너로 판정.
        smoothing_bands: 직전 몇 구간의 평균 방향과 비교할지(노이즈 완화).
        max_search_ratio: 말단에서 이 거리(바운딩 대각선 비율)까지만 탐색.
    """
    v = mesh.vertices
    edge_len = np.linalg.norm(v[mesh.edges[:, 0]] - v[mesh.edges[:, 1]], axis=1)
    band_width = float(np.median(edge_len)) * band_width_edge_mult
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    max_dist = diag * max_search_ratio
    n_bands = max(int(max_dist / band_width), smoothing_bands + 2)

    cuts: list[BendCut] = []
    for tip in tips:
        d = tip.dist_from_tip
        centers: list[np.ndarray | None] = []
        for i in range(n_bands):
            band_mask = (d >= i * band_width) & (d < (i + 1) * band_width)
            centers.append(v[band_mask].mean(axis=0) if band_mask.sum() >= min_band_vertices else None)

        dirs: list[np.ndarray | None] = []
        for i in range(len(centers) - 1):
            if centers[i] is None or centers[i + 1] is None:
                dirs.append(None)
                continue
            delta = centers[i + 1] - centers[i]
            norm = np.linalg.norm(delta)
            dirs.append(delta / norm if norm > 1e-12 else None)

        for i in range(smoothing_bands, len(dirs)):
            if dirs[i] is None:
                continue
            prev = [dd for dd in dirs[i - smoothing_bands:i] if dd is not None]
            if not prev:
                continue
            avg_prev = np.mean(prev, axis=0)
            avg_norm = np.linalg.norm(avg_prev)
            if avg_norm < 1e-12:
                continue
            cos_angle = np.clip(np.dot(avg_prev / avg_norm, dirs[i]), -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cos_angle)))
            if angle >= corner_angle_deg:
                cuts.append(BendCut(tip_vertex=tip.vertex, cut_distance=i * band_width, bend_angle_deg=angle))
                break

    return cuts


def split_at_bends(
    mesh: trimesh.Trimesh,
    tips: list[BranchTip],
    cuts: list[BendCut],
    *,
    ring_width_edge_mult: float = 1.5,
) -> trimesh.Trimesh:
    """코너마다 그 지점의 얇은 정점 띠를 지워 위상적으로 끊는다.

    실제 분리는 이후 `mesh.split()`(예: `suggest_bend_components()`)이 한다 --
    이 함수는 연결을 끊기만 한다.
    """
    if not cuts:
        return mesh

    v = mesh.vertices
    edge_len = np.linalg.norm(v[mesh.edges[:, 0]] - v[mesh.edges[:, 1]], axis=1)
    ring_width = float(np.median(edge_len)) * ring_width_edge_mult

    tip_by_vertex = {t.vertex: t for t in tips}
    remove_mask = np.zeros(len(v), dtype=bool)
    for cut in cuts:
        d = tip_by_vertex[cut.tip_vertex].dist_from_tip
        remove_mask |= np.abs(d - cut.cut_distance) <= ring_width

    if not remove_mask.any():
        return mesh
    return _remove_vertices(mesh, ~remove_mask)


@dataclass(slots=True)
class BendComponent:
    """`suggest_bend_components()`가 반환하는 조각 하나 -- 사람이 고를 후보."""

    mesh: trimesh.Trimesh
    n_vertices: int
    bbox_size: np.ndarray  # (dx, dy, dz)
    sphericity_score: float
    toe_score: float


def suggest_bend_components(
    mesh: trimesh.Trimesh,
    *,
    tip_top_k: int = 6,
    tip_min_separation_ratio: float = 0.15,
    min_component_vertices: int = 30,
    **bend_kwargs,
) -> list[BendComponent]:
    """말단-코너 절단 전체를 엮어, 갈라진 조각들을 크기 내림차순으로 반환한다.

    `bend_kwargs`는 `find_bend_cuts()`에 그대로 전달된다(임계각 등 튜닝용).
    조각이 하나도 안 갈라지면(코너를 못 찾음) 원본 메쉬 하나만 반환한다.
    """
    tips = find_branch_tips(mesh, top_k=tip_top_k, min_separation_ratio=tip_min_separation_ratio)
    cuts = find_bend_cuts(mesh, tips, **bend_kwargs)
    cut_mesh = split_at_bends(mesh, tips, cuts)

    pieces = cut_mesh.split(only_watertight=False)
    if len(pieces) <= 1:
        pieces = [mesh]

    components: list[BendComponent] = []
    for piece in pieces:
        if len(piece.vertices) < min_component_vertices:
            continue
        pts = piece.vertices
        components.append(BendComponent(
            mesh=piece,
            n_vertices=len(pts),
            bbox_size=piece.bounds[1] - piece.bounds[0],
            sphericity_score=_sphericity(pts),
            toe_score=_toe_cluster_score(pts, pts.mean(axis=0)),
        ))
    components.sort(key=lambda c: c.n_vertices, reverse=True)
    return components
