"""SfM sparse 포인트클라우드에서 배경을 걷어내고 피사체(발)만 남긴다.

마스크 기반(`masks_dir` 지정 시, 권장): 각 3D점의 관측 픽셀이 마스크
안쪽에 찍히는 비율로 발/배경 분류. 평면 가정이 없어 발을 가로지르는
오분류가 없다.

마스크 없을 때(폴백): 평면 제거(RANSAC) + 공통 초점 거리 필터로 기하학적
추정. `max_plane_ratio`(기본 40%) 넘게 먹는 "평면"은 버린다.

`intersect=True`(실험적, 기본 비권장): 마스크+기하학적 교집합만 제거 —
발이 바닥에 거의 붙어있으면 발 자체가 깎여나갈 수 있다.

`cluster=True`(DBSCAN): 마스크 정리 이후에만 켤 것 — 원본 점군에 바로
걸면 발이 조각날 수 있다.

권장 조합: `clean_point_cloud(..., masks_dir=..., cluster=True)`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pycolmap
from sklearn.cluster import DBSCAN

from .reconstruction import filter_outlier_points


def classify_by_mask(
    recon: pycolmap.Reconstruction,
    point_ids: list[int],
    masks_dir: Path,
    *,
    min_observation_ratio: float = 0.5,
) -> np.ndarray:
    """각 3D점을, 그 점을 관측한 사진들의 마스크 안쪽에 찍히는 비율로 발/배경 분류.

    - 평면 가정 없음 -- 기하학적 방식의 실패 유형이 구조적으로 불가능.
    - 재구성 결과 점을 사후 분류만 하므로 등록률에 영향 없음.
    - Returns: point_ids와 같은 순서의 (N,) bool 배열(True=발).
    """
    mask_cache: dict[str, np.ndarray | None] = {}

    def get_mask(name: str) -> np.ndarray | None:
        if name not in mask_cache:
            path = masks_dir / f"{name}.png"
            mask_cache[name] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
        return mask_cache[name]

    keep = np.zeros(len(point_ids), dtype=bool)
    for i, pid in enumerate(point_ids):
        votes = total = 0
        for el in recon.points3D[pid].track.elements:
            mask = get_mask(recon.images[el.image_id].name)
            if mask is None:
                continue
            x, y = recon.images[el.image_id].points2D[el.point2D_idx].xy
            xi, yi = int(round(x)), int(round(y))
            h, w = mask.shape
            if 0 <= yi < h and 0 <= xi < w:
                total += 1
                votes += mask[yi, xi] > 0
        keep[i] = total > 0 and votes / total >= min_observation_ratio

    return keep


def plane_removal_keep_mask(
    points: np.ndarray,
    *,
    n_planes: int,
    distance_threshold: float,
    min_plane_ratio: float = 0.05,
    max_plane_ratio: float = 0.4,
    rng_seed: int = 0,
) -> np.ndarray:
    """RANSAC으로 지배적인 평면을 최대 n_planes개 찾아 그 근처 점을 제거한다.

    max_plane_ratio: 찾은 "평면"이 전체 점의 이 비율을 넘게 먹으면 진짜
        평면이 아니라 임계값이 잘못 잡혀 장면을 가로지르는 슬라이스일
        가능성이 높다고 보고 버린다.

    Returns:
        `points`와 같은 순서의 (N,) bool 배열 — True면 평면에 속하지 않아 남음.
    """
    rng = np.random.default_rng(rng_seed)
    total_n = len(points)
    alive_idx = np.arange(total_n)
    remaining = points.copy()

    for _ in range(n_planes):
        n = len(remaining)
        if n < 50:
            break

        best_inliers = None
        n_iterations = 200
        for _ in range(n_iterations):
            idx = rng.choice(n, size=3, replace=False)
            p1, p2, p3 = remaining[idx]
            normal = np.cross(p2 - p1, p3 - p1)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal = normal / norm
            dist = np.abs((remaining - p1) @ normal)
            inliers = dist < distance_threshold
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers

        if best_inliers is None or best_inliers.sum() < n * min_plane_ratio:
            break  # 더 이상 뚜렷한 평면이 없음
        if best_inliers.sum() > total_n * max_plane_ratio:
            print(
                f"[평면 제거] 후보 평면이 전체의 {best_inliers.sum() / total_n:.1%}를 차지해 "
                f"건너뜀(임계값이 너무 크게 잡혔을 가능성 — plane_threshold를 직접 줄여보세요)."
            )
            break

        keep_local = ~best_inliers
        remaining = remaining[keep_local]
        alive_idx = alive_idx[keep_local]

    keep_mask = np.zeros(total_n, dtype=bool)
    keep_mask[alive_idx] = True
    return keep_mask


def estimate_focus_point(recon: pycolmap.Reconstruction) -> np.ndarray:
    """카메라들의 시선이 모이는 공통 지점을 추정한다.

    피사체를 중심으로 돌며 촬영했다면, 각 카메라의 시선은 대략 피사체를 향한다.
    "여러 직선에 가장 가까운 한 점"을 최소제곱으로 구하는 고전적인 문제로 푼다
    (각 광선에 수직인 투영 성분의 제곱합을 최소화).
    """
    origins = np.array([img.projection_center() for img in recon.images.values()])
    dirs = np.array([img.viewing_direction() for img in recon.images.values()])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    a = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        proj = np.eye(3) - np.outer(d, d)  # 광선에 수직인 방향으로 투영
        a += proj
        b += proj @ o
    return np.linalg.solve(a, b)


def geometric_keep_mask(
    points: np.ndarray,
    recon: pycolmap.Reconstruction,
    *,
    n_planes: int,
    plane_threshold: float,
    max_plane_ratio: float,
    focus_percentile: float,
) -> np.ndarray:
    """평면 제거 + 공통 초점 거리 필터를 순서대로 적용한 최종 keep-mask.

    두 단계 다 `points`에 대해 정렬된 (N,) bool 배열로 합성한다 — 마스크 기반
    분류와 같은 인덱스 공간이라 바로 논리 AND(교집합)를 할 수 있다.
    """
    stage1 = plane_removal_keep_mask(
        points, n_planes=n_planes, distance_threshold=plane_threshold, max_plane_ratio=max_plane_ratio
    )
    print(f"[평면 제거] {len(points):,} -> {int(stage1.sum()):,}개 (임계값 {plane_threshold:.4f})")

    focus = estimate_focus_point(recon)
    dist = np.linalg.norm(points - focus, axis=1)
    # 원래 로직과 동일하게, 백분위수는 평면 제거를 통과한 점들만 기준으로 계산한다.
    threshold = np.percentile(dist[stage1], focus_percentile)
    stage2 = dist <= threshold
    combined = stage1 & stage2
    print(
        f"[초점 거리 필터] {int(stage1.sum()):,} -> {int(combined.sum()):,}개 "
        f"(focus={focus.round(3).tolist()})"
    )
    return combined


def keep_largest_cluster(points: np.ndarray, *, eps: float, min_samples: int = 5) -> np.ndarray:
    """DBSCAN으로 군집화한 뒤 가장 큰 군집만 남긴다."""
    if len(points) < min_samples:
        return points
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return points  # 군집이 하나도 안 잡히면 원본 유지
    largest_label = np.bincount(valid).argmax()
    return points[labels == largest_label]


def clean_point_cloud(
    recon: pycolmap.Reconstruction,
    *,
    masks_dir: Path | None = None,
    intersect: bool = False,
    mask_min_ratio: float = 0.5,
    n_planes: int = 2,
    plane_threshold: float | None = None,
    max_plane_ratio: float = 0.4,
    focus_percentile: float = 80.0,
    cluster: bool = False,
    cluster_eps: float | None = None,
) -> np.ndarray:
    """복원 결과에서 배경을 제거하고 정리된 3D 점 배열을 반환한다.

    `masks_dir`가 있으면 마스크 기반 분류, 없으면 기하학적 폴백을 쓴다. 
    `intersect=True`는 실험적 옵션, 기본으로 켜지 말 것

    Returns:
        정리된 (N,3) 점 배열.
    """
    point_ids = list(recon.points3D.keys())
    points = np.array([recon.points3D[pid].xyz for pid in point_ids])
    print(f"[입력] 점 {len(points):,}개, 카메라 {len(recon.images)}개")

    if intersect and masks_dir is None:
        raise ValueError("intersect=True는 masks_dir와 같이 써야 합니다.")

    scale = float(np.median(np.abs(points - np.median(points, axis=0))))  # MAD 기반(이상치에 덜 민감)
    resolved_plane_threshold = plane_threshold or scale * 0.04

    mask_keep = geom_keep = None
    if masks_dir is not None:
        mask_keep = classify_by_mask(recon, point_ids, masks_dir, min_observation_ratio=mask_min_ratio)
        print(
            f"[마스크 분류] {len(points):,} -> {int(mask_keep.sum()):,}개 "
            f"(관측 중 마스크 안쪽 비율 >= {mask_min_ratio:.0%})"
        )
    if masks_dir is None or intersect:
        geom_keep = geometric_keep_mask(
            points, recon, n_planes=n_planes, plane_threshold=resolved_plane_threshold,
            max_plane_ratio=max_plane_ratio, focus_percentile=focus_percentile,
        )

    if intersect:
        combined_keep = mask_keep & geom_keep
        print(f"[교집합] 마스크 {int(mask_keep.sum()):,}개 ∩ 기하학적 {int(geom_keep.sum()):,}개 -> {int(combined_keep.sum()):,}개")
    else:
        combined_keep = mask_keep if mask_keep is not None else geom_keep

    step1 = points[combined_keep]
    # 어느 경로든 결과에 남는 잘못 삼각측량된 점(노이즈)은 통계적 이상치 제거로 마저 정리.
    inliers = filter_outlier_points(step1)
    step2 = step1[inliers]
    print(f"[이상치 제거] {len(step1):,} -> {len(step2):,}개")

    if not cluster or len(step2) < 20:
        step3 = step2
    else:
        # 군집화 반경은 필터링된(피사체 규모의) 점군 기준으로 다시 추정한다 —
        # 원본 전체 장면 기준 스케일을 쓰면 sparse한 피사체가 조각날 수 있다.
        local_scale = float(np.linalg.norm(step2.std(axis=0)))
        resolved_eps = cluster_eps or local_scale * 0.25
        step3 = keep_largest_cluster(step2, eps=resolved_eps)
        print(f"[군집화] {len(step2):,} -> {len(step3):,}개 (eps={resolved_eps:.4f})")

    print(f"[결과] 원본 대비 {len(step3) / len(points):.1%} 유지")
    return step3
