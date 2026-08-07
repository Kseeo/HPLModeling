"""2D 이미지 랜드마크 → 3D 변형을 구동할 실측 계측치(mm) 변환.

핵심 아이디어
-------------
* 이미지 장수는 **동적**이다. 각 뷰(top/medial/lateral/rear)가 제공할 수 있는
  계측 항목을 규칙 테이블(`EXTRACTION_RULES`)로 선언해 두고, 들어온 이미지에
  존재하는 랜드마크만 골라 계산한다.
* 픽셀 → mm 변환(캘리브레이션)은 2-pass 로 처리한다.
    pass 1: `scale_mm_per_px` 또는 `reference_length_mm` 가 있는 이미지 처리
    pass 2: pass 1 에서 확정된 발 길이(mm)를 기준자로 삼아 나머지 이미지 캘리브레이션
* 같은 계측 항목을 여러 뷰가 제공하면 뷰별 신뢰도 가중 평균을 취한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .exceptions import LandmarkValidationError
from .schemas import FootMeasurements, LandmarkImage, LandmarkPayload

# ---------------------------------------------------------------------------
# 랜드마크 이름 별칭 — 클라이언트마다 명명이 달라도 흡수한다.
# ---------------------------------------------------------------------------

Names = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExtractionRule:
    """'이 뷰에서 이 두 랜드마크의 거리는 이 계측치다' 라는 선언.

    Attributes:
        measurement: `FootMeasurements` 의 필드명.
        a_names: 첫 번째 랜드마크 후보 이름들(앞쪽이 우선).
        b_names: 두 번째 랜드마크 후보 이름들.
        mode: 'dist'(유클리드) | 'dx'(가로 성분) | 'dy'(세로 성분).
        weight: 여러 뷰가 같은 항목을 줄 때의 신뢰도 가중치.
    """

    measurement: str
    a_names: Names
    b_names: Names
    mode: str = "dist"
    weight: float = 1.0


#: 이미지 y축은 아래로 증가하지만 모든 계산에 절댓값을 쓰므로 부호는 무관하다.
_GROUND: Names = ("ground_ref", "ground", "floor_ref", "heel_bottom")

EXTRACTION_RULES: dict[str, tuple[ExtractionRule, ...]] = {
    # ---- 위(또는 아래)에서 본 뷰: 길이와 너비에 가장 신뢰도가 높다 -------------
    "top": (
        ExtractionRule("foot_length_mm", ("heel_center", "heel_back_center", "heel_back"),
                       ("toe_tip", "longest_toe_tip", "hallux_tip"), "dist", 1.0),
        ExtractionRule("ball_width_mm", ("mtp1_medial", "ball_medial", "mtp1"),
                       ("mtp5_lateral", "ball_lateral", "mtp5"), "dist", 1.0),
        ExtractionRule("heel_width_mm", ("heel_medial",), ("heel_lateral",), "dist", 1.0),
    ),
    # ---- 내측(medial) 측면 뷰: 아치·발등·복사뼈 높이의 기준 -------------------
    "medial": (
        ExtractionRule("foot_length_mm", ("heel_back", "heel_back_center"),
                       ("toe_tip", "hallux_tip"), "dx", 0.7),
        ExtractionRule("arch_height_mm", ("arch_apex", "navicular", "arch_top"),
                       _GROUND, "dy", 1.0),
        ExtractionRule("instep_height_mm", ("instep_top", "dorsum_top"), _GROUND, "dy", 1.0),
        ExtractionRule("ankle_height_mm", ("medial_malleolus", "malleolus"), _GROUND, "dy", 1.0),
    ),
    # ---- 외측(lateral) 측면 뷰: 보조 지표 -----------------------------------
    "lateral": (
        ExtractionRule("foot_length_mm", ("heel_back", "heel_back_center"),
                       ("toe_tip", "fifth_toe_tip"), "dx", 0.7),
        ExtractionRule("instep_height_mm", ("instep_top", "dorsum_top"), _GROUND, "dy", 0.6),
        # 외측 복사뼈는 내측보다 낮으므로 가중치를 낮춘다.
        ExtractionRule("ankle_height_mm", ("lateral_malleolus", "malleolus"), _GROUND, "dy", 0.6),
    ),
    # ---- 후면 뷰: 발목/뒤꿈치 폭 --------------------------------------------
    "rear": (
        ExtractionRule("heel_width_mm", ("heel_medial",), ("heel_lateral",), "dx", 0.9),
        ExtractionRule("ankle_width_mm", ("medial_malleolus",), ("lateral_malleolus",), "dx", 1.0),
        ExtractionRule("ankle_height_mm", ("medial_malleolus", "malleolus"), _GROUND, "dy", 0.8),
    ),
    # ---- 전면 뷰: 볼 너비 보조 ----------------------------------------------
    "front": (
        ExtractionRule("ball_width_mm", ("mtp1_medial", "ball_medial"),
                       ("mtp5_lateral", "ball_lateral"), "dx", 0.7),
        ExtractionRule("instep_height_mm", ("instep_top", "dorsum_top"), _GROUND, "dy", 0.5),
    ),
}

#: 뷰별 '발 길이' 산출 규칙 — 캘리브레이션(기준자)용으로도 재사용한다.
_LENGTH_RULE_BY_VIEW: dict[str, ExtractionRule] = {
    view: next((r for r in rules if r.measurement == "foot_length_mm"), None)  # type: ignore[misc]
    for view, rules in EXTRACTION_RULES.items()
}


# ---------------------------------------------------------------------------
# 기본 연산
# ---------------------------------------------------------------------------


def _resolve(image: LandmarkImage, names: Names) -> tuple[float, float] | None:
    """후보 이름들 중 이미지에 실제로 존재하는 첫 랜드마크를 반환."""
    for name in names:
        point = image.get(name)
        if point is not None:
            return point
    return None


def _pixel_distance(
    a: tuple[float, float], b: tuple[float, float], mode: str
) -> float:
    """두 픽셀 좌표 사이의 거리(px). mode 에 따라 성분만 취한다."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if mode == "dx":
        return dx
    if mode == "dy":
        return dy
    if mode == "dist":
        return math.hypot(dx, dy)
    raise ValueError(f"알 수 없는 거리 모드: {mode}")


def _pixel_foot_length(image: LandmarkImage) -> float | None:
    """해당 이미지에서 발 길이를 픽셀 단위로 잰다. 불가능하면 None."""
    rule = _LENGTH_RULE_BY_VIEW.get(image.view)
    if rule is None:
        return None
    a = _resolve(image, rule.a_names)
    b = _resolve(image, rule.b_names)
    if a is None or b is None:
        return None
    length_px = _pixel_distance(a, b, rule.mode)
    return length_px if length_px > 1e-6 else None


def resolve_scale(
    image: LandmarkImage, *, foot_length_mm: float | None = None
) -> float | None:
    """이미지의 mm/px 스케일을 결정한다.

    우선순위:
        1) 명시된 `scale_mm_per_px`
        2) `reference_length_mm` + `reference_pair` (촬영 시 함께 찍은 기준물)
        3) 이미 확정된 발 길이(mm) ÷ 이 이미지의 발 길이(px)

    Returns:
        mm/px 값. 어떤 방법으로도 결정할 수 없으면 None.
    """
    if image.scale_mm_per_px:
        return image.scale_mm_per_px

    if image.reference_length_mm and image.reference_pair:
        a = image.get(image.reference_pair[0])
        b = image.get(image.reference_pair[1])
        if a is None or b is None:
            raise LandmarkValidationError(
                f"view '{image.view}': reference_pair 로 지정한 랜드마크"
                f"{list(image.reference_pair)} 를 찾을 수 없습니다."
            )
        ref_px = _pixel_distance(a, b, "dist")
        if ref_px <= 1e-6:
            raise LandmarkValidationError(
                f"view '{image.view}': reference_pair 두 점이 동일합니다."
            )
        return image.reference_length_mm / ref_px

    if foot_length_mm:
        length_px = _pixel_foot_length(image)
        if length_px:
            return foot_length_mm / length_px

    return None


def _extract_from_image(
    image: LandmarkImage, scale_mm_per_px: float
) -> list[tuple[str, float, float, str]]:
    """이미지 1장에서 계산 가능한 모든 계측치를 뽑는다.

    Returns:
        (계측항목, 값(mm), 가중치, 출처설명) 튜플 리스트.
    """
    results: list[tuple[str, float, float, str]] = []
    for rule in EXTRACTION_RULES.get(image.view, ()):  # 미지원 뷰는 조용히 skip
        a = _resolve(image, rule.a_names)
        b = _resolve(image, rule.b_names)
        if a is None or b is None:
            continue  # 이 뷰에 해당 랜드마크가 없으면 그냥 건너뛴다(동적 입력 대응)
        value_px = _pixel_distance(a, b, rule.mode)
        if value_px <= 1e-6:
            continue
        results.append(
            (
                rule.measurement,
                value_px * scale_mm_per_px,
                rule.weight,
                f"{image.view}:{rule.a_names[0]}~{rule.b_names[0]}",
            )
        )
    return results


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def extract_measurements(
    payload: LandmarkPayload,
) -> tuple[FootMeasurements, list[str]]:
    """랜드마크 payload 로부터 계측치를 산출한다.

    Args:
        payload: `schemas.parse_payload()` 결과.

    Returns:
        (FootMeasurements, 경고 메시지 리스트)

    Raises:
        LandmarkValidationError: 어떤 이미지도 캘리브레이션할 수 없고,
            직접 입력된 계측치도 없는 경우.
    """
    warnings: list[str] = []
    #: 계측항목 → [(값, 가중치, 출처)]
    collected: dict[str, list[tuple[float, float, str]]] = {}

    def add(items: list[tuple[str, float, float, str]]) -> None:
        for name, value, weight, source in items:
            collected.setdefault(name, []).append((value, weight, source))

    # --- pass 1: 자체 스케일을 가진 이미지 -----------------------------------
    pending: list[LandmarkImage] = []
    for image in payload.images:
        scale = resolve_scale(image)
        if scale is None:
            pending.append(image)
            continue
        add(_extract_from_image(image, scale))

    # --- 기준 발 길이 확정 ----------------------------------------------------
    # 직접 입력된 값이 있으면 그것을 최우선 기준자로 삼는다.
    ruler_length = payload.explicit_measurements.get("foot_length_mm")
    if ruler_length is None and "foot_length_mm" in collected:
        ruler_length = _weighted_mean(collected["foot_length_mm"])

    # --- pass 2: 발 길이를 기준자로 삼아 나머지 이미지 캘리브레이션 -------------
    for image in pending:
        scale = resolve_scale(image, foot_length_mm=ruler_length)
        if scale is None:
            warnings.append(
                f"view '{image.view}' 는 mm/px 스케일을 결정할 수 없어 무시했습니다. "
                f"(scale_mm_per_px, reference_length_mm, 또는 발 길이 기준자 필요)"
            )
            continue
        add(_extract_from_image(image, scale))

    if not collected and not payload.explicit_measurements:
        raise LandmarkValidationError(
            "어떤 이미지에서도 계측치를 추출하지 못했습니다. "
            "랜드마크 이름과 스케일 정보를 확인하세요.",
            detail={"views": [img.view for img in payload.images]},
        )

    # --- 집계: 가중 평균 → 직접 입력값으로 덮어쓰기 ----------------------------
    measurements = FootMeasurements()
    for name, entries in collected.items():
        value = _weighted_mean(entries)
        setattr(measurements, name, value)
        measurements.sources[name] = [src for _, _, src in entries]
        # 같은 항목의 뷰 간 편차가 크면 촬영/라벨링 오류일 가능성이 높다.
        # (내·외측 복사뼈 높이 차처럼 해부학적으로 정상인 편차가 있으므로 임계값은 15%)
        spread = _relative_spread(entries)
        if spread > 0.15:
            warnings.append(
                f"'{name}' 의 뷰 간 편차가 {spread * 100:.1f}% 입니다. "
                f"랜드마크 라벨링을 확인하세요. (출처: {measurements.sources[name]})"
            )

    for name, value in payload.explicit_measurements.items():
        setattr(measurements, name, float(value))
        measurements.sources.setdefault(name, []).insert(0, "explicit")

    measurements.validate_positive()

    missing = measurements.missing()
    if missing:
        warnings.append(
            f"측정되지 않은 항목은 템플릿 값을 유지합니다: {missing}"
        )

    return measurements, warnings


def _weighted_mean(entries: list[tuple[float, float, str]]) -> float:
    """(값, 가중치, 출처) 리스트의 가중 평균."""
    total_w = sum(w for _, w, _ in entries)
    if total_w <= 0:
        return sum(v for v, _, _ in entries) / len(entries)
    return sum(v * w for v, w, _ in entries) / total_w


def _relative_spread(entries: list[tuple[float, float, str]]) -> float:
    """같은 항목을 여러 뷰에서 잰 값들의 상대 산포도 (0 이면 완전 일치)."""
    if len(entries) < 2:
        return 0.0
    values = [v for v, _, _ in entries]
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    return (max(values) - min(values)) / mean
