"""발 후보 위치 자동 탐색 -- 카메라/사진 정보 없이 메쉬 형상만으로 "이게 발일
가능성이 높다"를 점수화하는 휴리스틱. 확정 판별기가 아니라 사람이 확인할
후보 순위를 매기는 보조 도구다(`picker.py`/`pipeline.py`에서 사용).

세 가지 국소 신호를 합성한다:
    1. 밀도(density) -- 실제로 관심 있게 스캔한 피사체(발)는 카메라를 가까이
       대고 찍어 배경(의자 등)보다 정점 밀도가 높은 경향이 있다(실측 확인:
       사람이 짚어 확인한 진짜 발 위치가 배경보다 밀도 상위 85 백분위).
       그 자체만으로는 완벽히 가르지 못해(스캐너가 장면 전체를 균일 해상도로
       찍었으면 무력화됨) 아래 두 신호와 합성해서만 쓴다.
    2. 구형성(sphericity) -- 발은 국소적으로 "통통한" 형태인 반면, 흔한 배경
       오염원(의자 다리=가늘고 김, 벽/좌판=넓고 납작함)은 국소 형상이 뚜렷이
       다르다. 공분산 고유값비(λ3/λ1)로 정량화.
    3. 말단 다지(多指) 구조(toe cluster) -- 발가락처럼 한쪽 끝이 여러 개의
       작은 둥근 돌기로 갈라지는 패턴은 의자/벽 등에는 거의 없는 독특한
       신호다. 후보 지역의 국소 주축을 구해 그 축 "끝" 쪽 정점들을 축에
       수직인 평면에 투영한 뒤 DBSCAN으로 몇 덩어리로 갈라지는지 센다.

한계: 경험적 가정이 여럿 겹친 휴리스틱이라 100% 신뢰 불가(예: 라운드 쿠션은
구형성에서, 손이 같이 찍히면 다지 구조에서 오작동할 수 있다). 후보 순위
도구로만 쓸 것 -- 최종 판단은 사람이 확인해야 한다(`picker.py` 참고).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


@dataclass(slots=True)
class FootCandidate:
    """`suggest_foot_regions()`가 반환하는 후보 하나."""

    point: np.ndarray
    score: float
    density_score: float
    sphericity_score: float
    toe_score: float
    suggested_radius: float


def _sphericity(points: np.ndarray) -> float:
    """점 집합의 공분산 고유값비(λ3/λ1, 오름차순 λ3<=λ2<=λ1) -- 0(막대/판)~1(구)."""
    if len(points) < 6:
        return 0.0
    cov = np.cov((points - points.mean(axis=0)).T)
    eigvals = np.linalg.eigvalsh(cov)
    l3, l1 = eigvals[0], eigvals[2]
    if l1 <= 1e-15:
        return 0.0
    return float(l3 / l1)


def _toe_cluster_score(
    points: np.ndarray,
    center: np.ndarray,
    *,
    tip_band_ratio: float = 0.35,
    dbscan_eps_ratio: float = 0.12,
    min_toe_clusters: int = 2,
    max_toe_clusters: int = 7,
) -> float:
    """후보 영역 안에서 "말단이 여러 갈래로 갈라지는" 패턴이 있으면 높은 점수.

    영역의 국소 주축(PCA 최장축)을 구해 점들을 그 축에 투영하고, 축 방향
    양쪽 끝(각각 `tip_band_ratio`만큼) 중 다지 구조가 더 뚜렷한 쪽을 택한다.
    그 끝 쪽 점들만 축에 수직인 평면에 투영해 DBSCAN으로 몇 덩어리로
    갈라지는지 센다 -- `min_toe_clusters`~`max_toe_clusters`개면 발가락
    패턴으로 보고 1.0에 가까운 점수, 그 밖이면(매끈한 끝=1개, 노이즈로
    너무 잘게 쪼개짐=너무 많음) 낮은 점수를 준다.

    Returns:
        0~1 점수 -- 다지 구조가 뚜렷할수록 높음.
    """
    if len(points) < 20:
        return 0.0
    c = points - points.mean(axis=0)
    cov = c.T @ c
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]  # 최대 고유값 -- 국소 주축(길이 방향)

    t = c @ axis
    span = t.max() - t.min()
    if span <= 1e-12:
        return 0.0

    perp_basis = eigvecs[:, :2]  # 축에 수직인 평면(중간/최소 고유값 방향)
    best = 0.0
    for sign in (1.0, -1.0):
        tip_mask = (sign * t) >= (sign * t).max() - span * tip_band_ratio
        tip_pts = c[tip_mask] @ perp_basis
        if len(tip_pts) < 10:
            continue
        eps = float(np.linalg.norm(tip_pts.max(axis=0) - tip_pts.min(axis=0))) * dbscan_eps_ratio
        if eps <= 1e-12:
            continue
        labels = DBSCAN(eps=eps, min_samples=4).fit_predict(tip_pts)
        n_clusters = len(set(labels.tolist()) - {-1})
        if min_toe_clusters <= n_clusters <= max_toe_clusters:
            # 클러스터 수가 이상적인 범위(3~5, 발가락 5개 근방) 중앙에 가까울수록 가점.
            ideal = 4.0
            closeness = 1.0 - min(abs(n_clusters - ideal) / ideal, 1.0)
            best = max(best, 0.6 + 0.4 * closeness)
    return best


def suggest_foot_regions(
    mesh: trimesh.Trimesh,
    *,
    n_samples: int = 15_000,
    neighborhood_radius_ratio: float = 0.045,
    density_prefilter_ratio: float = 0.5,
    top_k: int = 5,
    min_separation_ratio: float = 0.12,
    min_neighbors: int = 10,
    density_weight: float = 1.0,
    sphericity_weight: float = 1.0,
    toe_weight: float = 1.5,
    rng: np.random.Generator | None = None,
) -> list[FootCandidate]:
    """메쉬 표면을 샘플링해 밀도+구형성+다지구조 합성 점수로 발 후보 위치 순위를 매긴다.

    Args:
        neighborhood_radius_ratio: 국소 판정 반경(바운딩 대각선 비율) -- 발
            단면 크기와 비슷한 스케일이어야 의미가 있다.
        density_prefilter_ratio: 전체 샘플 중 밀도 상위 이 비율만 구형성/다지구조
            계산 대상으로 남긴다(속도용 -- 배경 대부분은 밀도로 먼저 걸러짐).
        top_k: 반환할 후보 개수.
        min_separation_ratio: 후보끼리 최소 이격 거리(대각선 비율, 비최대 억제).
        density_weight/sphericity_weight/toe_weight: 세 점수(각 0~1 근방) 합성 가중치.

    Returns:
        `FootCandidate` 리스트, `score` 내림차순.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    surface_points, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=rng)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    radius = diag * neighborhood_radius_ratio

    tree = cKDTree(surface_points)
    neighbor_lists = tree.query_ball_point(surface_points, radius)
    density = np.array([len(idxs) for idxs in neighbor_lists], dtype=np.float64)

    # 1단계: 밀도로 후보 풀을 줄인다(비싼 구형성/다지구조 계산 대상 축소).
    density_thresh = np.percentile(density, (1 - density_prefilter_ratio) * 100)
    pool_idx = np.where(density >= density_thresh)[0]
    if len(pool_idx) == 0:
        return []
    density_norm = density / max(density.max(), 1.0)

    candidates_raw: list[tuple[int, float, float, float, float]] = []
    for i in pool_idx:
        idxs = neighbor_lists[i]
        if len(idxs) < min_neighbors:
            continue
        pts = surface_points[idxs]
        sph = _sphericity(pts)
        toe = _toe_cluster_score(pts, surface_points[i])
        d = float(density_norm[i])
        score = density_weight * d + sphericity_weight * sph + toe_weight * toe
        candidates_raw.append((i, score, d, sph, toe))

    candidates_raw.sort(key=lambda c: c[1], reverse=True)

    min_sep = diag * min_separation_ratio
    chosen: list[FootCandidate] = []
    for i, score, d, sph, toe in candidates_raw:
        p = surface_points[i]
        if any(np.linalg.norm(p - fc.point) < min_sep for fc in chosen):
            continue
        chosen.append(FootCandidate(
            point=p, score=float(score), density_score=d, sphericity_score=sph,
            toe_score=toe, suggested_radius=radius,
        ))
        if len(chosen) >= top_k:
            break
    return chosen


@dataclass(slots=True)
class MeshComponent:
    """`list_components()`가 반환하는, 이미 공간적으로 분리된 연결 요소 하나."""

    index: int
    mesh: trimesh.Trimesh
    n_vertices: int
    bbox_size: np.ndarray
    centroid: np.ndarray
    sphericity_score: float
    toe_score: float


def list_components(mesh: trimesh.Trimesh, *, min_vertices: int = 30) -> list[MeshComponent]:
    """공간적으로 이미 분리된 연결 요소를 정점 수 내림차순으로 나열한다.

    `finishing.keep_largest_component()`는 이 중 1번(가장 큰 것)만 자동으로
    선택하는데, 배경 물체가 발보다 더 크면(예: 의자 좌판) 잘못 선택될 수
    있다 -- 사람이 `picker.py`로 직접 확인해서 고를 때 쓴다.
    """
    pieces = sorted(mesh.split(only_watertight=False), key=lambda p: len(p.vertices), reverse=True)
    out: list[MeshComponent] = []
    for piece in pieces:
        if len(piece.vertices) < min_vertices:
            continue
        pts = piece.vertices
        centroid = pts.mean(axis=0)
        out.append(MeshComponent(
            index=len(out), mesh=piece, n_vertices=len(pts),
            bbox_size=piece.bounds[1] - piece.bounds[0], centroid=centroid,
            sphericity_score=_sphericity(pts), toe_score=_toe_cluster_score(pts, centroid),
        ))
    return out


@dataclass(slots=True)
class DenseRegion:
    """`find_dense_regions()`가 반환하는 "구역" 하나 -- 점 하나가 아니라
    공간적으로 뭉친 표면 샘플점 집합이라, 크롭할 때 그 실제(구가 아닌) 모양을
    그대로 따라갈 수 있다(`crop.crop_to_region()`)."""

    points: np.ndarray
    centroid: np.ndarray
    score: float
    sphericity_score: float
    toe_score: float
    n_points: int
    density_radius: float
    extent_radius: float


def find_dense_regions(
    mesh: trimesh.Trimesh,
    *,
    n_samples: int = 20_000,
    density_radius_ratio: float = 0.02,
    density_top_ratio: float = 0.35,
    cluster_eps_mult: float = 1.0,
    min_cluster_size: int = 80,
    top_k: int = 5,
    rng: np.random.Generator | None = None,
) -> list[DenseRegion]:
    """표면 샘플점 중 국소 밀도 상위 `density_top_ratio`만 남겨 공간적으로
    군집화한다 -- "정점이 빽빽하게 몰린 구역"을 직접 찾는 방식(점 하나 +
    고정 배수 반경으로 구를 씌우던 이전 방식의 대안).

    실측 확인: 스캐너가 피사체(발)를 가까이서 찍어 배경(의자 등)보다 촘촘하게
    담는 경향이 있다는 전제 -- 배경이 완전히 분리돼 있으면(밀도 낮은 다리로만
    위상 연결) 잘 갈라지지만, **발과 배경이 진짜로 fuse돼 밀도 차이 없이
    이어져 있으면(예: project_223) 군집화 파라미터를 아무리 조여도 안
    갈라진다** -- 이건 데이터 자체에 분리 근거가 없는 것이라 알고리즘
    한계가 아니라 원리적 한계다(실측: eps를 0.8~1.3배로 바꿔봐도 최대
    구역 크기가 거의 안 변함). 그런 케이스는 `crop.py`의 사람 개입 도구
    (`picker.py`로 좌표 확인 후 `crop_around_seed`/`remove_near_point`)를 쓸 것.

    Args:
        density_radius_ratio: 밀도 판정 반경(바운딩 대각선 비율).
        density_top_ratio: 전체 샘플 중 밀도 상위 이 비율만 군집화 대상으로 남긴다.
        cluster_eps_mult: DBSCAN 반경 = `density_radius_ratio`가 만든 반경의 이 배수.
            너무 크면 서로 다른 물체가 하나로 뭉쳐 잡힌다(과소 분리), 너무 작으면
            같은 물체가 여러 조각으로 쪼개진다(과다 분리).
        min_cluster_size: 이보다 작은 군집은 노이즈로 버린다.
        top_k: 반환할 구역 개수(점 개수 기준 큰 순서로 상위 후보 풀을 만든 뒤
            그 안에서 발 형상 점수로 재정렬).

    Returns:
        `DenseRegion` 리스트, 형상 점수(`score`) 내림차순.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    surface_points, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=rng)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    density_radius = diag * density_radius_ratio

    tree = cKDTree(surface_points)
    density = np.array(tree.query_ball_point(surface_points, density_radius, return_length=True), dtype=np.float64)
    thresh = np.percentile(density, (1 - density_top_ratio) * 100)
    pool = surface_points[density >= thresh]
    if len(pool) < min_cluster_size:
        return []

    labels = DBSCAN(eps=density_radius * cluster_eps_mult, min_samples=6).fit_predict(pool)

    raw_regions = []
    for lbl in set(labels.tolist()) - {-1}:
        member = pool[labels == lbl]
        if len(member) < min_cluster_size:
            continue
        raw_regions.append(member)
    # 점 개수 기준 상위 후보 풀을 넉넉히(top_k의 3배) 잡은 뒤 형상 점수로 재정렬 --
    # 배경 구역이 우연히 발보다 커도 형상 점수로 순위가 바뀔 여지를 남긴다.
    raw_regions.sort(key=len, reverse=True)
    raw_regions = raw_regions[: max(top_k * 3, top_k)]

    regions: list[DenseRegion] = []
    for member in raw_regions:
        sph = _sphericity(member)
        toe = _toe_cluster_score(member, member.mean(axis=0))
        score = sph + 1.5 * toe
        centroid = member.mean(axis=0)
        extent_radius = float(np.linalg.norm(member - centroid, axis=1).max())
        regions.append(DenseRegion(
            points=member, centroid=centroid, score=float(score),
            sphericity_score=sph, toe_score=toe, n_points=len(member),
            density_radius=density_radius, extent_radius=extent_radius,
        ))
    regions.sort(key=lambda r: r.score, reverse=True)
    return regions[:top_k]
