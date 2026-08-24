"""메쉬 후처리(배경/파편 제거 + 스무딩) -- 입력은 메쉬 하나뿐, 사진/카메라/
마스크 정보 불필요.

- dense.py(사진->메쉬 생성) 출력에 이어 붙이던 후처리를 옮겨온 것 -- 이미
  가진 STL/PLY에도 독립 적용 가능(scripts/postprocess_mesh.py 참고).
- 흐름(postprocess_mesh()):
  keep_largest_component (부유 파편 제거)
  -> (선택) prune_thin_protrusions (뿔/스파이크 제거)
  -> (선택) fill_small_holes (핀홀만 메움)
  -> (선택) sand_surface (전체 표면 이차곡면 투영)
  -> (선택) smooth_high_curvature_regions (고곡률 국소 스무딩, 크레이터 완화)
  -> (선택) finish_smooth_mesh (라플라시안 마감 스무딩)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

#: smooth_high_curvature_regions() 기본 강도(곡률 백분위 임계값).
DEFAULT_CURVATURE_PERCENTILE = 60.0
DEFAULT_CURVATURE_MIN_RADIUS_MULT = 2.0
DEFAULT_CURVATURE_MAX_RADIUS_MULT = 25.0
#: Taubin(λ|μ) 반복 횟수.
DEFAULT_CURVATURE_ITERATIONS = 150
DEFAULT_CURVATURE_ALPHA = 0.7
DEFAULT_CURVATURE_MU = -0.75


def keep_largest_component(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int, int]:
    """면(face) 기준 가장 큰 연결 요소만 남기고 부유 파편을 지운다.

    - 공간적으로 분리된 덩어리 단위로만 잘라 DBSCAN보다 안전(성긴 진짜
      부위를 노이즈로 오판하지 않음). 단, 발 표면에 이어 붙은 경계
      노이즈는 같은 덩어리라 안 떨어짐.
    - Returns: (필터링된 메쉬, 원본 face 수, 남은 face 수).
    """
    total_faces = len(mesh.faces)
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh, total_faces, total_faces
    largest = max(components, key=lambda c: len(c.faces))
    return largest, total_faces, len(largest.faces)


def _protrusion_remove_mask(
    points: np.ndarray,
    *,
    density_radius_nn_mult: float,
    far_percentile: float,
    density_ratio: float,
    adjacency_edges: np.ndarray | None = None,
) -> np.ndarray:
    """뿔/스파이크에 해당하는 점 인덱스 마스크 계산 -- 메쉬 정점/raw
    포인트클라우드 공용 로직.

    adjacency_edges를 주면(메쉬) 그 인접성을 쓰고, None이면(raw 포인트클라우드)
    "얇음" 후보 점들끼리 반경 기반으로 인접 그래프를 새로 만든다.
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

    - keep_largest_component()는 분리된 파편만 잡음(뿔은 같은 연결 요소라
      무력) -- "뿔은 국소 정점 밀도가 몸통보다 낮다"는 걸 단서로 씀.
    - 한계: 메쉬 위상을 망가뜨릴 수 있어 dense.py 주 파이프라인은 점 단위
      사전 제거(clean_dense_point_cloud의 prune_protrusions)를 대신 씀.
      여기 남긴 건 완성 메쉬 사후 적용용.
    - Returns: (프루닝된 메쉬, 제거된 정점 수).
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


def fill_small_holes(
    mesh: trimesh.Trimesh,
    *,
    max_hole_diameter_ratio: float = 0.05,
    use_fan: bool = True,
) -> tuple[trimesh.Trimesh, int]:
    """작은 구멍(핀홀/관측 누락 조각)만 팬 삼각분할로 메운다.

    - 발바닥처럼 원래 안 찍은 큰 구멍은 그대로 둠(억지로 메우면 가짜
      뚜껑이 씌워져 형태 왜곡). 구멍 바운딩박스 대각선이 메쉬 전체 대각선의
      max_hole_diameter_ratio보다 작을 때만 메움.
    - trimesh.repair.fill_holes()와 같은 방식(경계 사이클+팬 삼각분할)이나
      크기 필터가 추가된 버전. 삼각형 추가만, 기존 정점/면은 그대로.

    Args:
        max_hole_diameter_ratio: 구멍 대각선/메쉬 전체 대각선 비율 상한(기본 5%).
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

    # winding 보정 -- 새 face 경계 edge가 기존 경계와 같은 방향이면 뒤집는다.
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

    - 공간(유클리드) 최근접이 아니라 표면을 따라간 위상 인접을 씀 -- 접힌/
      오목한 형태(발목 반대쪽, 발가락 사이 등)에서 유클리드 최근접은 국소
      곡면 피팅을 망가뜨림.
    - Returns: (idx_padded, mask) 둘 다 (n, max_neighbors). mask=False인
      자리의 idx_padded 값은 무의미(0으로 채워짐).
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

    - 정점 주변 위상 인접을 min_neighbors개 모아 국소 접평면(PCA) 구한 뒤
      h = a*u²+b*uv+c*v²+d*u+e*v+f 를 최소제곱 피팅, 예측 위치(f, 법선
      오프셋)로 이동. 이웃 평균으로 등방적으로 당기는 라플라시안과 달리
      법선 방향으로만 움직여 접평면 방향의 진짜 형태(2차 항)는 보존.
    - smooth_high_curvature_regions()와 달리 곡률 임계값 없이 전체 정점에
      균일 적용 -- 일반 노이즈 완화용. 정점/면 개수·위상은 그대로.
    - 이웃 좌표(u,w)는 국소 거리 스케일로 정규화(안 하면 정규방정식 조건수
      악화로 일부 정점 피팅이 극단값으로 튐). max_offset_ratio로 오프셋을
      국소 스케일 배수로 clamp.
    - 한계: 관측 부족 크레이터처럼 이웃 전체가 같은 방향으로 치우친 저주파
      왜곡은 이웃의 이차곡면 추세 자체가 왜곡돼 있어 못 없앰 --
      smooth_high_curvature_regions() 몫.

    Args:
        min_neighbors: 국소 곡면 피팅 최소 이웃 수(이차곡면 미지수 6개보다
            충분히 많아야 함). 1-ring 부족 시 2-ring, 3-ring...으로 확장.
        max_neighbors: 이웃 상한(배열 패딩 크기).
        iterations: 반복 횟수. 이웃 집합은 한 번만 계산(위상 기준, 메쉬
            불변), 매 반복 현재 위치로 재피팅.
        regularization: 정규방정식(AᵀA)에 더하는 상대적 대각 성분.
        max_offset_ratio: 오프셋을 국소 이웃 평균 거리의 이 배수로 제한.
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

        # u/w를 국소 스케일로 정규화 -- 상수항(f)은 (u,w)=(0,0) 값이라
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
    print(f"[postprocess] 사포질(전체 이차곡면 투영): 정점 {n:,}개, {iterations}회 반복")
    return out


def smooth_high_curvature_regions(
    mesh: trimesh.Trimesh,
    *,
    curvature_percentile: float = 90.0,
    min_radius_edge_mult: float = 2.0,
    max_radius_edge_mult: float = 25.0,
    iterations: int = 10,
    alpha: float = 0.6,
    mu: float = -0.65,
) -> trimesh.Trimesh:
    """곡률이 튀는 영역을 Taubin(λ|μ) 스무딩한다. 나머지 정점은 그대로 둔다.

    - 관측 부족 크레이터형 결함 완화용 -- 노이즈와 진짜 굴곡을 구분 못 해
      발가락 사이 등 진짜 디테일도 함께 뭉갬(트레이드오프 감수).
    - 확산 반경은 전체 고정이 아니라 코어 영역(연결된 고곡률 정점 덩어리)
      별로 그 영역 곡률 반경(1/|곡률|)에 비례해 부여(curvature-adaptive,
      bilateral mesh denoising과 같은 원리) -- 완만하고 넓은 결함은 넓게,
      좁고 조밀한 디테일은 좁게. 반경은 위상 그래프 다익스트라 실거리로 잼.
    - 평범한 라플라시안 단방향 반복은 체적이 체계적으로 줄어드는 편향이
      있어, alpha(양의 스텝)/mu(반대 방향 스텝, |mu|>alpha) 번갈아 적용하는
      Taubin 스무딩으로 상쇄.

    Args:
        curvature_percentile: 이 백분위 이상 |곡률|인 정점을 코어로 삼는다.
        min_radius_edge_mult/max_radius_edge_mult: 영역별 확산 반경을 전형적
            엣지 길이의 이 배수 범위로 clip(평평한 영역의 반경 폭주 방지).
        alpha: 라플라시안(수축) 스텝 크기.
        mu: 역방향(팽창) 스텝 크기(음수, |mu| > alpha).
    """
    v = mesh.vertices.copy()
    n = len(v)
    edge_len = np.linalg.norm(v[mesh.edges[:, 0]] - v[mesh.edges[:, 1]], axis=1)
    typical_edge = float(np.median(edge_len))
    curv = trimesh.curvature.discrete_mean_curvature_measure(mesh, v, typical_edge * 4)

    thresh = np.percentile(np.abs(curv), curvature_percentile)
    core_mask = np.abs(curv) > thresh

    neighbors = mesh.vertex_neighbors
    weight = np.zeros(n)

    if core_mask.any():
        # 코어를 위상 연결요소(=국소 결함 하나)별로 나눈다.
        visited = np.zeros(n, dtype=bool)
        clusters: list[np.ndarray] = []
        for start in np.where(core_mask)[0]:
            if visited[start]:
                continue
            stack, comp = [start], [start]
            visited[start] = True
            while stack:
                cur = stack.pop()
                for nb in neighbors[cur]:
                    if core_mask[nb] and not visited[nb]:
                        visited[nb] = True
                        comp.append(nb)
                        stack.append(nb)
            clusters.append(np.array(comp, dtype=np.int64))

        edges_u = mesh.edges_unique
        elen_u = np.linalg.norm(v[edges_u[:, 0]] - v[edges_u[:, 1]], axis=1)
        graph = sp.csr_matrix(
            (np.concatenate([elen_u, elen_u]),
             (np.concatenate([edges_u[:, 0], edges_u[:, 1]]), np.concatenate([edges_u[:, 1], edges_u[:, 0]]))),
            shape=(n, n),
        )
        min_radius = typical_edge * min_radius_edge_mult
        max_radius = typical_edge * max_radius_edge_mult

        for comp in clusters:
            local_radius = 1.0 / np.maximum(np.abs(curv[comp]), 1e-9)
            spread_radius = float(np.clip(np.median(local_radius), min_radius, max_radius))
            dist = dijkstra(graph, indices=comp, min_only=True, limit=spread_radius, directed=False)
            reached = np.isfinite(dist)
            w = np.clip(1.0 - dist[reached] / spread_radius, 0.0, 1.0)
            weight[reached] = np.maximum(weight[reached], w)
        weight[core_mask] = 1.0

    # 행별 정규화(각 행 합=1) 인접 행렬로 이웃 평균을 희소행렬 곱 한 번에
    # 계산(파이썬 for문보다 빠름). 이웃 없는 정점은 자기 자신에 1을 둠.
    rows = np.concatenate([np.full(len(neighbors[i]), i) for i in range(n)])
    cols = np.concatenate([np.asarray(neighbors[i], dtype=np.int64) for i in range(n)])
    deg = np.array([len(neighbors[i]) for i in range(n)])
    vals = np.concatenate([np.full(len(neighbors[i]), 1.0 / len(neighbors[i])) if len(neighbors[i]) else np.array([])
                            for i in range(n)])
    adj = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    isolated = deg == 0
    if isolated.any():
        adj = adj + sp.csr_matrix((np.ones(isolated.sum()), (np.where(isolated)[0], np.where(isolated)[0])), shape=(n, n))

    def laplacian_step(vv: np.ndarray, step: float) -> np.ndarray:
        avg = adj @ vv
        return vv + weight[:, None] * step * (avg - vv)

    new_v = v.copy()
    for _ in range(iterations):
        new_v = laplacian_step(new_v, alpha)
        new_v = laplacian_step(new_v, mu)

    out = mesh.copy()
    out.vertices = new_v
    print(
        f"[postprocess] 고곡률 국소 스무딩(Taubin λ|μ): 정점 {int((weight > 0).sum()):,}/{n:,}개 영향"
        f"(코어 {int(core_mask.sum()):,}개, curvature_percentile={curvature_percentile})"
    )
    return out


def finish_smooth_mesh(
    mesh: trimesh.Trimesh,
    *,
    lamb: float = 0.5,
    iterations: int = 40,
) -> trimesh.Trimesh:
    """전체 정점에 평범한(가중치 없는) 라플라시안 스무딩을 반복해 남은 고주파
    표면 노이즈를 마감 처리한다.

    - smooth_high_curvature_regions()(Taubin λ|μ)는 체적 수축 방지에 초점을
      둔 보수적 방식이라 자잘한 고주파 요철에는 약함 -- 방향 상쇄 없는
      평범한 라플라시안 반복으로 확실히 제거. 반복이 많으면 진짜 굴곡도
      뭉개지고 부피가 줄어드는 트레이드오프가 있어 마지막 마감 단계로만 사용.
    - trimesh.smoothing.filter_laplacian() 기본 volume_constraint=True는
      mesh.volume 비율로 정점을 재스케일해 부피 수축을 상쇄하는데, 이
      프로젝트 메쉬는 발바닥이 안 찍혀 항상 non-watertight라 mesh.volume이
      불안정(음수 -> (...)**(1/3) NaN -> 정점 전체 오염 확인됨) -- 그래서
      volume_constraint=False로 끄고 씀.
    - 정점 96,879개 기준 40회 반복 약 3.8초 -- 파이프라인 전체 대비 무시할 수준.

    Args:
        lamb: 반복당 이웃 평균 쪽으로 당기는 비율(0~1).
        iterations: 반복 횟수 -- 클수록 매끈해지지만 디테일도 더 죽는다.
    """
    out = mesh.copy()
    trimesh.smoothing.filter_laplacian(out, lamb=lamb, iterations=iterations, volume_constraint=False)
    if not np.isfinite(out.vertices).all():
        print("[postprocess][경고] 마감 스무딩 결과에 비정상 값(NaN/Inf)이 생겨 이번 단계는 건너뜁니다")
        return mesh
    print(f"[postprocess] 마감 스무딩(라플라시안 x{iterations}): 정점 {len(out.vertices):,}개")
    return out


@dataclass(slots=True)
class PostprocessStats:
    """postprocess_mesh()가 남기는 단계별 요약."""

    faces_before: int = 0
    faces_after_largest_component: int = 0
    n_protrusion_vertices_removed: int = 0
    n_holes_filled: int = 0
    steps_applied: list[str] = field(default_factory=list)


def postprocess_mesh(
    mesh: trimesh.Trimesh,
    *,
    keep_largest: bool = True,
    prune_protrusions: bool = False,
    protrusion_density_radius_nn_mult: float = 4.0,
    protrusion_far_percentile: float = 97.0,
    protrusion_density_ratio: float = 0.6,
    fill_holes: bool = True,
    fill_holes_max_diameter_ratio: float = 0.05,
    sand_surface_enabled: bool = True,
    sand_min_neighbors: int = 16,
    sand_max_neighbors: int = 32,
    sand_iterations: int = 3,
    smooth_high_curvature: bool = True,
    curvature_percentile: float = DEFAULT_CURVATURE_PERCENTILE,
    curvature_min_radius_mult: float = DEFAULT_CURVATURE_MIN_RADIUS_MULT,
    curvature_max_radius_mult: float = DEFAULT_CURVATURE_MAX_RADIUS_MULT,
    curvature_iterations: int = DEFAULT_CURVATURE_ITERATIONS,
    curvature_alpha: float = DEFAULT_CURVATURE_ALPHA,
    curvature_mu: float = DEFAULT_CURVATURE_MU,
    finish_smooth: bool = True,
    finish_smooth_lambda: float = 0.5,
    finish_smooth_iterations: int = 10,
) -> tuple[trimesh.Trimesh, PostprocessStats]:
    """배경/파편 제거 + 스무딩 단계 전부를 엮는다.

    - dense.run_dense_pipeline()의 메쉬 생성 이후 부분과 동일 순서, 메쉬
      하나만 입력받아 완성 STL 등에도 그대로 적용 가능(scripts/postprocess_mesh.py).

    Args:
        keep_largest(True): 가장 큰 연결 요소만 남기고 부유 파편 제거.
        prune_protrusions(False): 뿔/스파이크 사후 제거 -- 위상 손상 우려로 기본 꺼짐.
        fill_holes(True): 작은 구멍(핀홀)만 메움, 큰 구멍(발바닥 등)은 유지.
        sand_surface_enabled(True): 전체 정점 이차곡면 투영 노이즈 완화.
        smooth_high_curvature(True): 고곡률 영역 국소 Taubin 스무딩.
        finish_smooth(True): 라플라시안 마감 스무딩(가장 마지막 적용, 비용
            미미). finish_smooth_iterations 기본값 10은 실측 근거 -- 이 뒤
            dense.decimate_mesh()가 축약+마감 스무딩을 한 번 더 하므로,
            여기서 반복 횟수를 바꿔도 최종 결과 길이 변화가 ±0.8%p 안에서
            흔들려(단조 관계 아님) 영향이 미미함 -- 계산량만 줄이도록 40->10.

    Returns:
        (후처리된 메쉬, 단계별 통계).
    """
    stats = PostprocessStats(faces_before=len(mesh.faces))

    if keep_largest:
        mesh, faces_before, faces_after = keep_largest_component(mesh)
        stats.faces_after_largest_component = faces_after
        if faces_after < faces_before:
            print(f"[postprocess] 부유 파편 제거: {faces_before:,} -> {faces_after:,} faces")
            stats.steps_applied.append("keep_largest_component")
    else:
        stats.faces_after_largest_component = len(mesh.faces)

    if prune_protrusions:
        mesh, n_removed = prune_thin_protrusions(
            mesh, density_radius_nn_mult=protrusion_density_radius_nn_mult,
            far_percentile=protrusion_far_percentile, density_ratio=protrusion_density_ratio,
        )
        stats.n_protrusion_vertices_removed = n_removed
        if n_removed:
            print(f"[postprocess] 뿔/스파이크 제거: 정점 {n_removed:,}개")
            stats.steps_applied.append("prune_thin_protrusions")

    if fill_holes:
        mesh, n_filled = fill_small_holes(mesh, max_hole_diameter_ratio=fill_holes_max_diameter_ratio)
        stats.n_holes_filled = n_filled
        if n_filled:
            print(f"[postprocess] 작은 구멍 메움: {n_filled}개")
            stats.steps_applied.append("fill_small_holes")

    if sand_surface_enabled:
        mesh = sand_surface(
            mesh, min_neighbors=sand_min_neighbors, max_neighbors=sand_max_neighbors,
            iterations=sand_iterations,
        )
        stats.steps_applied.append("sand_surface")

    if smooth_high_curvature:
        mesh = smooth_high_curvature_regions(
            mesh, curvature_percentile=curvature_percentile,
            min_radius_edge_mult=curvature_min_radius_mult, max_radius_edge_mult=curvature_max_radius_mult,
            iterations=curvature_iterations, alpha=curvature_alpha, mu=curvature_mu,
        )
        stats.steps_applied.append("smooth_high_curvature_regions")

    if finish_smooth:
        mesh = finish_smooth_mesh(mesh, lamb=finish_smooth_lambda, iterations=finish_smooth_iterations)
        stats.steps_applied.append("finish_smooth_mesh")

    return mesh, stats
