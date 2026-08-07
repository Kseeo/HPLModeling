"""공통 템플릿 → 개별 스캔 비강체 정합(non-rigid registration).

목적: 모든 스캔이 템플릿과 **같은 정점 개수·순서(위상)** 를 갖도록 만들어서,
"정점 i는 항상 같은 해부학적 지점"이라는 전제 아래 PCA를 돌릴 수 있게 한다.
서로 다른 정점 수/삼각분할을 가진 raw 스캔들 사이에 이런 대응을 만드는 게
SSM 구축에서 가장 손이 많이 가는 단계다.

알고리즘: RBF 변위장 기반 반복 최근접점 정합 — 비강체 ICP(Amberg et al.)의
단순화 버전이다.

    for iteration:
        1. 현재 변형된 템플릿의 각 정점에서 타겟 표면의 최근접점을 찾는다.
        2. 너무 먼 대응(outlier_distance_mm 초과)은 제외한다 — 특히 스캔이
           템플릿보다 짧게 잘려 있거나(열린 경계) 노이즈가 남아있을 때 중요하다.
        3. RBF로 "정점 → 최근접점" 변위장을 보간해 전체 정점에 적용한다.
        4. smoothing 강도를 반복마다 줄여, 처음엔 전역적으로 크게 맞추고
           뒤로 갈수록 국소적으로 세밀하게 맞춘다(coarse-to-fine).

`deformer.py`의 `_solve_displacement_field()`와 같은 RBFInterpolator 메커니즘을
재사용하지만, 여기서는 대응점을 사람이 지정한 랜드마크가 아니라 최근접점 탐색으로
자동으로 찾는다는 점이 다르다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.interpolate import RBFInterpolator

from .preprocessing import pca_axes


def rigid_prealign(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    icp_samples: int = 2000,
    rng_seed: int = 0,
) -> np.ndarray:
    """비강체 정합 전에 source를 target과 대략 같은 위치·방향으로 맞춘다.

    실제 스캔은 사진 기반 재구성이라 파일마다 좌표축 관례(어느 축이 길이/너비/
    높이인지, 부호가 어느 쪽인지)가 다를 수 있다 — 템플릿처럼 항상 X=길이축
    이라고 가정할 수 없다. PCA로 두 메쉬의 주축을 먼저 맞추고, 남은 4가지
    부호 조합(반사는 제외, 좌우 반전은 `mirror_side()`로 별도 처리) 중 ICP
    비용이 가장 낮은 것을 고른다.

    Returns:
        source와 같은 개수의 (V,3) 정점 배열 — target 좌표계로 옮겨진 것.

    Raises:
        ValueError: 모든 부호 조합에서 ICP가 실패한 경우.
    """
    rng = np.random.default_rng(rng_seed)
    src_centroid = source.vertices.mean(axis=0)
    tgt_centroid = target.vertices.mean(axis=0)
    src_axes = pca_axes(source.vertices)
    tgt_axes = pca_axes(target.vertices)
    src_local = (source.vertices - src_centroid) @ src_axes

    best_verts: np.ndarray | None = None
    best_cost = float("inf")
    # sz = sx*sy 로 고정해 반사(det=-1, 좌우 뒤집힘)는 배제하고 회전만 허용한다
    # — 좌우 반전은 side 메타데이터 기반 mirror_side()가 명시적으로 처리해야 한다.
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        sz = sx * sy
        candidate_world = (src_local * np.array([sx, sy, sz])) @ tgt_axes.T + tgt_centroid

        sample = candidate_world
        if len(sample) > icp_samples:
            sample = sample[rng.choice(len(sample), icp_samples, replace=False)]

        try:
            # scale/reflection 은 반드시 꺼야 한다 — trimesh.registration.icp 가 내부적으로
            # 쓰는 procrustes() 의 기본값이 scale=True, reflection=True라서, 꺼두지 않으면
            # 스캔마다 제멋대로 크기를 재조정하거나 좌우를 뒤집어버린다. 축척은 이미
            # self_normalize()로, 좌우는 mirror_side()로 명시적으로 통제하고 있으므로
            # 여기서는 회전+이동만 허용한다.
            matrix, _, cost = trimesh.registration.icp(
                sample, target, max_iterations=20, scale=False, reflection=False
            )
        except Exception:
            continue
        if cost < best_cost:
            best_cost = cost
            best_verts = trimesh.transform_points(candidate_world, matrix)

    if best_verts is None:
        raise ValueError("강체 사전 정렬에 실패했습니다(모든 방향 조합에서 ICP 실패).")
    return best_verts


def _kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """중심을 맞춘 두 대응점 집합 사이, 반사 없는 최적 회전행렬을 구한다(Kabsch).

    `source @ R.T` 가 `target` 에 최소자승 최적합이 되는 R(3,3)을 반환한다.
    """
    h = source.T @ target
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    return vt.T @ correction @ u.T


def generalized_procrustes_align(
    vertex_stack: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    max_iterations: int = 20,
    tol: float = 1e-4,
) -> np.ndarray:
    """정합된 여러 스캔을 하나의 공통 좌표계로 강체 정렬한다(GPA).

    `register_template_to_target()`의 출력은 각 **타겟 스캔 자신의** 원본
    좌표계(위치·회전)에 놓여 있다 — 실제 스캔 파일마다 절대 위치/회전 관례가
    제각각이라(스케일만 `self_normalize()`로 통일했을 뿐), 이 상태로 바로 PCA를
    돌리면 주성분 상위 항목이 "발 모양 차이"가 아니라 "스캔별 자세(회전+이동)
    차이"를 먼저 흡수해버린다 — 성분 점수 스케일이 발 크기(수백mm)보다 훨씬
    큰 값(수만mm)으로 나오는 게 그 증거다. 정점 대응(위상)이 이미 확립돼 있으므로
    최근접점 탐색 없이 Kabsch 알고리즘으로 정확한 회전만 정렬한다(스케일·반사는
    이미 self_normalize/mirror_side로 고정했으므로 여기서 다시 건드리지 않는다).

    Args:
        vertex_stack: (N, V, 3) — 같은 위상을 공유하는 정합된 정점 배열들.
        reference: 정렬 기준으로 쓸 고정 형상, (V,3). 생략하면 데이터 자신의
            평균 형상에 반복 수렴시키는 고전적 GPA를 쓰지만, 그 결과 좌표계는
            **임의의 방향**이 된다(어느 축이 길이/너비/높이인지 보장 없음) —
            `mesh_utils.FootFrame`/`measure_foot()`처럼 X=길이·Z=높이를 가정하는
            코드와 맞지 않게 된다. 정합에 쓴 템플릿(`template_norm.vertices`,
            이미 절차적으로 그 관례를 따름)을 넘기면 한 번의 Kabsch 정렬로
            그 관례를 그대로 물려받는다 — 실전에서는 항상 이 방식을 쓸 것.
        max_iterations: `reference`가 없을 때, 평균 형상이 수렴할 때까지
            반복할 최대 횟수(reference가 있으면 반복 없이 1회로 충분해 무시됨).
        tol: 평균 형상의 이동량(mm)이 이보다 작아지면 수렴으로 보고 멈춘다.

    Returns:
        (N, V, 3) — 공통 좌표계로 정렬된 정점 배열(각 스캔은 자신의 중심으로
        평행이동된 뒤 회전만 적용됨).
    """
    aligned = vertex_stack - vertex_stack.mean(axis=1, keepdims=True)

    if reference is not None:
        target_shape = reference - reference.mean(axis=0)
        rotated = np.empty_like(aligned)
        for i in range(len(aligned)):
            r = _kabsch_rotation(aligned[i], target_shape)
            rotated[i] = aligned[i] @ r.T
        return rotated

    mean_shape = aligned.mean(axis=0)
    for _ in range(max_iterations):
        rotated = np.empty_like(aligned)
        for i in range(len(aligned)):
            r = _kabsch_rotation(aligned[i], mean_shape)
            rotated[i] = aligned[i] @ r.T
        new_mean = rotated.mean(axis=0)
        shift = float(np.linalg.norm(new_mean - mean_shape))
        aligned, mean_shape = rotated, new_mean
        if shift < tol:
            break

    return aligned


@dataclass(slots=True)
class RegistrationResult:
    """정합 결과 — 정점 배열은 템플릿과 동일한 개수/순서를 유지한다."""

    vertices: np.ndarray  # (V, 3) — 템플릿과 같은 정점 수, 타겟 표면에 맞춰짐
    rms_error_mm: float  # 마지막 반복의 평균 최근접거리 — 정합 품질 지표
    iterations_used: int
    inlier_ratio: float  # 마지막 반복에서 대응으로 인정된 정점 비율


def _default_smoothing_schedule(n_iterations: int, *, floor: float = 3.0) -> tuple[float, ...]:
    """반복마다 지수적으로 감소하는 RBF smoothing 강도(coarse-to-fine), 바닥값 있음.

    바닥(`floor`)이 없으면 마지막 반복의 smoothing이 0에 가까워져(예: 8회 기준
    0.03) 각 스캔의 국소 노이즈·미세한 개인차까지 정점이 그대로 따라간다 — 개별
    스캔 하나하나는 자기 표면에 잘 맞지만("정합 RMS는 좋음"), 그 대가로 스캔마다
    "같은 정점 인덱스"가 조금씩 다른 해부학적 위치에 안착해버린다(대응 일관성
    저하). 이 상태로 PCA 평균을 내면 스캔 간 대응 어긋남이 서로 상쇄되며 세부
    형태(발볼 굴곡, 뒤꿈치 곡률 등)가 뭉개진 완만한 쐐기 모양으로 수렴한다 —
    개별 등록 결과는 발 모양이 잘 살아있는데 평균만 뭉개지는 게 그 증거였다.
    바닥값을 두면 마지막까지도 어느 정도 전역적인(smooth) 변형만 허용해,
    개별 정밀도를 약간 희생하는 대신 집단 전체의 대응 일관성을 지킨다.
    """
    return tuple(max(20.0 * (0.4**i), floor) for i in range(n_iterations))


def register_template_to_target(
    template: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    n_iterations: int = 8,
    smoothing_schedule: tuple[float, ...] | None = None,
    outlier_distance_mm: float = 15.0,
    min_inliers: int = 10,
    max_control_points: int = 800,
    rng_seed: int = 0,
    prealign: bool = True,
) -> RegistrationResult:
    """템플릿 정점을 타겟 표면에 맞춰 비강체 정합한다(템플릿 위상은 유지).

    Args:
        template: 정합시킬 공통 템플릿(모든 스캔에 재사용).
        target: 이 스캔에 맞출 대상 표면. 정점 개수가 템플릿과 달라도 무방하다
            (대응점 탐색에 표면으로서만 쓰이므로).
        prealign: True면 비강체 정합 전에 `rigid_prealign()`으로 먼저 위치·
            방향을 맞춘다. 실제 스캔처럼 좌표축 관례가 파일마다 다를 수 있는
            데이터에서는 필수에 가깝다(끄면 템플릿과 타겟이 애초에 다른
            곳을 보고 있어 대응점 자체를 못 찾고 실패하기 쉽다). 이미 같은
            좌표계로 정렬된 데이터(합성 테스트 등)에서만 꺼도 안전하다.
        n_iterations: 반복 횟수.
        smoothing_schedule: 반복마다 쓸 RBF smoothing 강도. None 이면 기본
            coarse-to-fine 스케줄(`_default_smoothing_schedule`) 사용.
        outlier_distance_mm: 이보다 먼 최근접점은 대응에서 제외한다 — 스캔이
            템플릿 대비 더 짧게 잘려 있는 부위(열린 경계 등)에서 엉뚱하게 먼
            점에 붙는 것을 막는다.
        min_inliers: 유효 대응점이 이보다 적으면 그 시점에서 반복을 멈춘다.
        max_control_points: RBF의 중심점(대응점) 개수 상한. RBF solve는 점
            개수의 세제곱에 비례해 느려지므로(예: 정점 7,682개를 그대로 쓰면
            건당 수십 초), 매끄러운 변위장을 표현하는 데 충분한 수만 무작위로
            골라 쓴다 — `deformer.py`가 조밀한 정점 대신 소수의 제어점+격자만
            RBF에 쓰는 것과 같은 이유다. 변위장은 골라낸 점들로 fit 하고,
            **평가는 항상 전체 정점에** 적용하므로 결과 위상은 그대로 유지된다.
        rng_seed: 대응점 서브샘플링에 쓰는 난수 시드(재현성).

    Returns:
        `RegistrationResult` — 실패해도 예외를 던지지 않고 그때까지의
        최선 결과를 반환한다(호출자가 `rms_error_mm`로 품질을 판단).
    """
    schedule = smoothing_schedule or _default_smoothing_schedule(n_iterations)
    rng = np.random.default_rng(rng_seed)

    if prealign:
        verts = rigid_prealign(template, target, rng_seed=rng_seed)
    else:
        verts = np.array(template.vertices, dtype=float, copy=True)
    proximity = trimesh.proximity.ProximityQuery(target)

    rms = float("inf")
    inlier_ratio = 0.0
    used = 0

    for i, smoothing in enumerate(schedule):
        closest, distance, _ = proximity.on_surface(verts)
        inlier = np.flatnonzero(distance < outlier_distance_mm)
        if len(inlier) < min_inliers:
            break

        # RBF 중심점은 대응점 중 무작위 서브샘플로 제한한다(위 max_control_points 설명 참고).
        centers = inlier if len(inlier) <= max_control_points else rng.choice(
            inlier, size=max_control_points, replace=False
        )

        try:
            rbf = RBFInterpolator(
                verts[centers],
                closest[centers] - verts[centers],
                kernel="thin_plate_spline",
                smoothing=smoothing,
            )
            displacement = rbf(verts)
        except Exception:
            break

        if not np.isfinite(displacement).all():
            break

        verts = verts + displacement
        rms = float(np.sqrt(np.mean(distance[inlier] ** 2)))
        inlier_ratio = len(inlier) / len(distance)
        used = i + 1

    return RegistrationResult(
        vertices=verts, rms_error_mm=rms, iterations_used=used, inlier_ratio=inlier_ratio
    )
