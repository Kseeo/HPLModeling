"""SfM sparse 포인트클라우드에서 계측치를 뽑아 `FootMeshDeformer`로 3D 모델을 만든다.

SSM(PCA 기반) 경로는 스캔 간 대응점 불일치로 뭉개진 형상만 나온다는 게 확인됐다
대신 이 모듈은 이미 검증된 `deformer.FootMeshDeformer`를 SfM 점군에 연결한다:

    1. 점군을 템플릿의 좌표계로 강체 정렬(회전+이동)한다 — PCA로 축을 맞추고
       ICP로 다듬는다. 스케일은 점군 자신의 PCA 길이를 250mm 기준으로 임시
       정규화한다(진짜 크기는 자기신고 사이즈로 추후 보정 필요).
    2. 정렬된 점들에 `mesh_utils.measure_foot()`와 같은 정의(u/v/w 밴드, 창)를
       적용해 발 길이·발볼너비·뒤꿈치너비·아치높이 등을 직접 잰다. 메쉬가 아니라
       점군이라 법선 기반 바닥면 판정 대신 높이 기준(w<0.25) 대체 로직을 쓴다.
    3. 그 계측치를 `FootMeshDeformer.deform_from_measurements()`에 넣어 실제
       발 템플릿을 변형한 메쉬를 만든다

좌우(카이랄리티) 주의: 발은 카이랄 형상이라 회전만으로는 좌우가 다른
템플릿에 정렬할 수 없다. 
입력 점군과 템플릿의 실제 좌우를 알고 있다면 반드시 `side`/`template_side`를 넘길 것 —
다르면 `mirror_points()`로 정렬 전에 명시적으로 반전한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .. import config as cfg
from ..deformer import FootMeshDeformer
from ..schemas import FootMeasurements
from ..ssm import pca_axes
from ..ssm.preprocessing import DEFAULT_REFERENCE_LENGTH_MM
from ..template_factory import save_reference_template


def measured_length(points: np.ndarray) -> float:
    """점군 자신의 PCA 최장축(추정 길이) 크기."""
    axes = pca_axes(points)
    projected = (points - points.mean(axis=0)) @ axes[:, 0]
    return float(projected.max() - projected.min())


def mirror_points(points: np.ndarray) -> np.ndarray:
    """점군의 좌우를 반전한다(왼발 <-> 오른발).

    `ssm/preprocessing.py`의 `mirror_side()`와 같은 원리(PCA 너비축 반전, 특정
    좌표축 관례에 의존하지 않음)를 raw 점 배열에 적용한 버전 — 메쉬가 아니라
    점군이라 face winding 뒤집기가 필요 없다.
    """
    centroid = points.mean(axis=0)
    axes = pca_axes(points)
    local = (points - centroid) @ axes
    local[:, 1] *= -1.0  # 너비 축 반전
    return local @ axes.T + centroid


def rigid_prealign_points(
    source_points: np.ndarray, target_points: np.ndarray, *, icp_samples: int = 2000, rng_seed: int = 0
) -> np.ndarray:
    """`ssm/registration.py`의 `rigid_prealign()`과 같은 알고리즘, raw 점군 타겟용.

    PCA 축의 부호 모호성(4가지 조합)만 탐색한다 — 좌우(카이랄리티)가 다른 경우는
    이걸로 해결되지 않는다. 필요하면 이 함수를 부르기 전에 `mirror_points()`로
    먼저 반전할 것(`side`/`template_side` 인자를 쓰는 `fit_point_cloud_to_template()`
    가 이를 자동으로 처리한다).
    """
    rng = np.random.default_rng(rng_seed)
    src_centroid = source_points.mean(axis=0)
    tgt_centroid = target_points.mean(axis=0)
    src_axes = pca_axes(source_points)
    tgt_axes = pca_axes(target_points)
    src_local = (source_points - src_centroid) @ src_axes

    best_points: np.ndarray | None = None
    best_cost = float("inf")
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        sz = sx * sy
        candidate = (src_local * np.array([sx, sy, sz])) @ tgt_axes.T + tgt_centroid
        sample = candidate
        if len(sample) > icp_samples:
            sample = sample[rng.choice(len(sample), icp_samples, replace=False)]
        try:
            matrix, _, cost = trimesh.registration.icp(
                sample, target_points, max_iterations=20, scale=False, reflection=False
            )
        except Exception:
            continue
        if cost < best_cost:
            best_cost = cost
            best_points = trimesh.transform_points(candidate, matrix)

    if best_points is None:
        raise ValueError("점군 강체 정렬에 실패했습니다(모든 방향 조합에서 ICP 실패).")
    return best_points


def measure_point_cloud(points: np.ndarray, engine: FootMeshDeformer) -> FootMeasurements:
    """정렬된 점군에 `mesh_utils.measure_foot()`와 같은 정의를 적용해 계측한다.

    메쉬가 아니라 점이라 법선이 없다 — 바닥면(plantar) 판정은 높이 기준
    (w < 0.25, `arch_apex()`가 법선 판정 실패 시 쓰는 것과 같은 대체 규칙)으로
    대신한다.
    """
    frame = engine.frame
    uvw = frame.to_uvw(points)
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    y, z = points[:, 1], points[:, 2]

    def in_u(window: tuple[float, float]) -> np.ndarray:
        return (u >= window[0]) & (u <= window[1])

    # 퍼센타일 기반 폭/높이: 메쉬 정점과 달리 점군은 마스킹을 뚫고 남은 배경
    # 잔여 노이즈가 섞여 있어(예: 발볼 구간에서 폭이 발 하나 폭의 2배 이상으로
    # 튀는 것을 확인) 단순 min/max는 노이즈 한두 점에도 크게 흔들린다.
    # 5~95 퍼센타일로 완충한다.
    _LO, _HI = 5, 95

    def y_span(mask: np.ndarray, fallback: float, min_points: int = 5) -> float:
        if mask.sum() < min_points:
            return fallback
        lo, hi = np.percentile(y[mask], [_LO, _HI])
        return float(hi - lo)

    def z_top(mask: np.ndarray, fallback: float, min_points: int = 5) -> float:
        if mask.sum() < min_points:
            return fallback
        return float(np.percentile(z[mask], _HI) - frame.z_floor)

    m = FootMeasurements()
    # frame.length(템플릿 자신의 길이)에 기대지 않고 정렬된 점군 자신의 X축
    # extent를 직접 잰다 — 스케일 단계에서 250mm 근처로 맞췄지만 템플릿과 정확히
    # 같은 값이 될 이유는 없다(형태가 다르면 X extent도 다르다). 길이는 폭/높이와
    # 달리 진짜 끝점(뒤꿈치·발끝)이 중요하므로 퍼센타일로 깎지 않고 실제 min/max를
    # 쓴다 — 애초에 스케일 정규화 자체가 이 PCA 기반 min/max 길이 기준이었다.
    m.foot_length_mm = float(points[:, 0].max() - points[:, 0].min())

    m.heel_width_mm = y_span(in_u(cfg.HEEL_WINDOW_U), engine.template_measurements.heel_width_mm)
    m.ball_width_mm = y_span(in_u(cfg.BALL_WINDOW_U), engine.template_measurements.ball_width_mm)

    ankle_band = in_u(cfg.REAR_WINDOW_U) & (np.abs(w - cfg.ANKLE_HEIGHT_W_FRACTION) < cfg.ANKLE_BAND_W)
    m.ankle_width_mm = y_span(ankle_band, m.heel_width_mm)

    instep_mask = in_u(cfg.INSTEP_WINDOW_U)
    m.instep_height_mm = z_top(instep_mask, engine.template_measurements.instep_height_mm)
    m.ankle_height_mm = float(frame.extents[2] * cfg.ANKLE_HEIGHT_W_FRACTION)

    band_lo, band_hi = cfg.ARCH_MEDIAL_BAND_V
    win_lo, win_hi = cfg.ARCH_WINDOW_U
    arch_region = (u >= win_lo) & (u <= win_hi) & (v >= band_lo) & (v <= band_hi) & (w < 0.25)
    m.arch_height_mm = z_top(arch_region, engine.template_measurements.arch_height_mm)

    for name in m.field_names():
        m.sources[name] = ["sfm-pointcloud"]
    return m


def default_template_path(root: Path) -> Path:
    """템플릿을 안 넘겼을 때 쓸 기본 절차적 템플릿 경로(없으면 생성)."""
    template_path = root / "data" / "templates" / "base_foot_template.stl"
    if not template_path.is_file():
        save_reference_template(template_path, length_mm=250.0, side="right")
    return template_path


def fit_point_cloud_to_template(
    points: np.ndarray,
    engine: FootMeshDeformer,
    *,
    side: str | None = None,
    template_side: str = "right",
    rng_seed: int = 0,
) -> tuple[trimesh.Trimesh, FootMeasurements]:
    """점군 -> (스케일 정규화 -> 좌우 정렬 -> 강체 정렬 -> 계측 -> 템플릿 워프) 한 번에.

    Args:
        points: raw SfM 점군(임의 축척, 임의 좌표계).
        engine: 로드된 `FootMeshDeformer`(템플릿).
        side: 입력 점군(영상)이 실제로 어느 쪽 발인지. `template_side`와 다르면
            ICP 정렬 전에 점군을 미러링한다. `None`이면 좌우 확인을 건너뛴다
            (주의: 좌우가 다른데 생략하면 발이 뒤틀린 채로 정렬됨 — 실측 확인된
            실패 모드, `lr-chirality-fix-and-heel-bug-isolation` 메모리 참고).
        template_side: 템플릿의 실제 발 좌우.
        rng_seed: 강체 정렬 ICP 후보 샘플링 시드.

    Returns:
        (변형된 메쉬, 점군에서 뽑은 계측치). 상세 리포트는
        `engine.last_report`로 확인한다.
    """
    own_length = measured_length(points)
    scale = DEFAULT_REFERENCE_LENGTH_MM / own_length
    scaled_points = points * scale
    print(
        f"[스케일] 점군 자체 PCA 길이 {own_length:.3f}(SfM 임의 단위) -> "
        f"{DEFAULT_REFERENCE_LENGTH_MM:.0f}mm 기준(x{scale:.4f}) "
        "— 절대 축척 아님, 형태 비교용 임시값"
    )

    if side is not None and side != template_side:
        print(
            f"[좌우] 입력={side} != 템플릿={template_side} -> 점군 미러링 "
            "(발은 카이랄 형상이라 회전만으로는 좌우가 다른 템플릿에 정렬 불가"
            " — ICP는 proper-rotation 4종만 탐색하므로 반사를 직접 적용해야 함)"
        )
        scaled_points = mirror_points(scaled_points)
    elif side is None:
        print(
            "[좌우] side 미지정 — 점군/템플릿 좌우 일치 여부를 확인하지 않음. "
            "템플릿이 실제 발과 좌우가 다르면 정렬이 뒤틀릴 수 있음."
        )

    aligned = rigid_prealign_points(scaled_points, engine.vertices, rng_seed=rng_seed)

    measured = measure_point_cloud(aligned, engine)
    print("\n[점군에서 뽑은 계측치(mm)]")
    for name in measured.field_names():
        print(f"  {name:<20} {getattr(measured, name):.1f}")

    deformed = engine.deform_from_measurements(measured, image_count=0)
    return deformed, measured
