"""입력/출력 데이터 구조 정의.

의존성을 가볍게 유지하기 위해 표준 `dataclass` 로 작성했다.
FastAPI 로 확장할 때는 아래 dataclass 들을 그대로 Pydantic `BaseModel` 로
바꾸거나, `pydantic.dataclasses.dataclass` 데코레이터만 교체하면 된다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Iterable

from .exceptions import LandmarkValidationError

# ---------------------------------------------------------------------------
# 계측치(Measurement)
# ---------------------------------------------------------------------------

#: 변형을 구동하는 계측 항목. (이름 → 사람이 읽는 설명)
MEASUREMENT_LABELS: dict[str, str] = {
    "foot_length_mm": "발 길이 (뒤꿈치~최장 발가락)",
    "ball_width_mm": "볼 너비 (MTP1~MTP5)",
    "heel_width_mm": "뒤꿈치 너비",
    "instep_height_mm": "발등 높이",
    "ankle_height_mm": "복사뼈 높이 (바닥 기준)",
    "ankle_width_mm": "발목 너비 (내·외측 복사뼈)",
    "arch_height_mm": "내측 아치 높이",
}


@dataclass(slots=True)
class FootMeasurements:
    """발 계측치(mm). 모든 필드는 선택적이며 None 은 '측정되지 않음'을 뜻한다.

    측정되지 않은 항목은 변형 시 템플릿 값을 그대로 사용(scale=1.0)한다.
    덕분에 입력 이미지 개수가 1장이든 4장이든 동일한 코드 경로로 동작한다.
    """

    foot_length_mm: float | None = None
    ball_width_mm: float | None = None
    heel_width_mm: float | None = None
    instep_height_mm: float | None = None
    ankle_height_mm: float | None = None
    ankle_width_mm: float | None = None
    arch_height_mm: float | None = None

    # 각 계측치가 어느 이미지/랜드마크에서 왔는지 추적 (디버깅·리포트용)
    sources: dict[str, list[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """`sources` 를 제외한 계측 필드 이름 목록."""
        return tuple(f.name for f in fields(cls) if f.name != "sources")

    def to_dict(self, *, drop_none: bool = True) -> dict[str, float]:
        """계측치만 dict 로. (`sources` 제외)"""
        out = {k: getattr(self, k) for k in self.field_names()}
        return {k: v for k, v in out.items() if v is not None} if drop_none else out

    def filled_with(self, fallback: "FootMeasurements") -> "FootMeasurements":
        """None 인 항목을 `fallback`(보통 템플릿 계측치) 값으로 채운 새 객체를 반환."""
        merged = FootMeasurements(sources=dict(self.sources))
        for name in self.field_names():
            value = getattr(self, name)
            if value is None:
                value = getattr(fallback, name)
                merged.sources.setdefault(name, []).append("template-fallback")
            setattr(merged, name, value)
        return merged

    def missing(self) -> list[str]:
        """값이 없는 계측 항목 목록."""
        return [n for n in self.field_names() if getattr(self, n) is None]

    def validate_positive(self) -> None:
        """계측치가 물리적으로 말이 되는 값인지 확인."""
        for name in self.field_names():
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or value != value:  # NaN 포함
                raise LandmarkValidationError(
                    f"계측치 '{name}' 가 숫자가 아닙니다: {value!r}"
                )
            if value <= 0:
                raise LandmarkValidationError(
                    f"계측치 '{name}' 는 0보다 커야 합니다: {value}"
                )


# ---------------------------------------------------------------------------
# 입력 payload
# ---------------------------------------------------------------------------

#: 지원하는 촬영 뷰. 'top' 은 위(또는 아래)에서, 'medial/lateral' 은 측면,
#: 'rear' 는 뒤에서 촬영한 이미지를 뜻한다.
SUPPORTED_VIEWS: frozenset[str] = frozenset({"top", "medial", "lateral", "rear", "front"})


@dataclass(slots=True)
class LandmarkImage:
    """이미지 1장에 대한 2D 랜드마크 묶음."""

    view: str
    landmarks: dict[str, tuple[float, float]]
    image_size_px: tuple[int, int] | None = None
    scale_mm_per_px: float | None = None
    #: 캘리브레이션용 참조 길이(예: 촬영 시 함께 찍은 A4 용지 변 길이)
    reference_length_mm: float | None = None
    #: 참조 길이에 대응하는 두 랜드마크 이름
    reference_pair: tuple[str, str] | None = None

    def get(self, name: str) -> tuple[float, float] | None:
        return self.landmarks.get(name)

    def require(self, *names: str) -> bool:
        """주어진 랜드마크가 모두 존재하는지 여부."""
        return all(n in self.landmarks for n in names)


@dataclass(slots=True)
class LandmarkPayload:
    """`deform_mesh()` 가 받는 입력 전체."""

    images: list[LandmarkImage] = field(default_factory=list)
    #: 사진 계측을 건너뛰고 직접 넣는 계측치 (이미지에서 얻은 값보다 우선)
    explicit_measurements: dict[str, float] = field(default_factory=dict)
    #: 'right' | 'left'
    side: str = "right"
    #: 제어점 이름 → [dx, dy, dz] (mm). 특정 부위만 미세 조정할 때 사용.
    control_point_offsets_mm: dict[str, tuple[float, float, float]] = field(
        default_factory=dict
    )
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 파싱 & 검증
# ---------------------------------------------------------------------------


def _as_xy(value: Any, *, where: str) -> tuple[float, float]:
    """[x, y] 또는 {'x':..,'y':..} 형태를 (x, y) 튜플로 정규화."""
    if isinstance(value, dict):
        if "x" not in value or "y" not in value:
            raise LandmarkValidationError(
                f"{where}: 랜드마크 dict 에는 'x','y' 키가 있어야 합니다.", detail=value
            )
        value = (value["x"], value["y"])
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise LandmarkValidationError(
            f"{where}: 랜드마크는 [x, y] 형식이어야 합니다.", detail=value
        )
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise LandmarkValidationError(
            f"{where}: 랜드마크 좌표를 float 으로 변환할 수 없습니다.", detail=value
        ) from exc


def parse_payload(raw: dict) -> LandmarkPayload:
    """사용자 입력 dict(=JSON) 를 검증하여 `LandmarkPayload` 로 변환한다.

    허용하는 형태::

        {
          "meta":   {"subject_id": "...", "side": "right"},
          "measurements": {"foot_length_mm": 268.0},        # 선택
          "images": [                                        # 개수 자유
            {"view": "top",
             "image_size_px": [1200, 1600],
             "scale_mm_per_px": 0.2,
             "landmarks": {"heel_center": [600, 1500], ...}},
            ...
          ],
          "control_point_offsets_mm": {"instep_top": [0, 0, 2.0]}   # 선택
        }

    Raises:
        LandmarkValidationError: 스키마가 맞지 않을 때.
    """
    if not isinstance(raw, dict):
        raise LandmarkValidationError(
            "landmarks_data 는 dict 여야 합니다.", detail=type(raw).__name__
        )

    meta = raw.get("meta") or {}
    if not isinstance(meta, dict):
        raise LandmarkValidationError("'meta' 는 dict 여야 합니다.", detail=meta)

    side = str(raw.get("side") or meta.get("side") or "right").lower()
    if side not in {"right", "left"}:
        raise LandmarkValidationError(
            "'side' 는 'right' 또는 'left' 여야 합니다.", detail=side
        )

    raw_images = raw.get("images", [])
    if not isinstance(raw_images, list):
        raise LandmarkValidationError("'images' 는 list 여야 합니다.", detail=raw_images)

    images: list[LandmarkImage] = []
    for idx, item in enumerate(raw_images):
        where = f"images[{idx}]"
        if not isinstance(item, dict):
            raise LandmarkValidationError(f"{where}: dict 여야 합니다.", detail=item)

        view = str(item.get("view", "")).lower()
        if view not in SUPPORTED_VIEWS:
            raise LandmarkValidationError(
                f"{where}: 지원하지 않는 view '{view}'. "
                f"가능한 값: {sorted(SUPPORTED_VIEWS)}"
            )

        raw_lms = item.get("landmarks")
        if not isinstance(raw_lms, dict) or not raw_lms:
            raise LandmarkValidationError(
                f"{where}: 비어있지 않은 'landmarks' dict 가 필요합니다."
            )
        landmarks = {
            str(name): _as_xy(pt, where=f"{where}.landmarks.{name}")
            for name, pt in raw_lms.items()
        }

        size = item.get("image_size_px")
        image_size = (int(size[0]), int(size[1])) if size else None

        scale = item.get("scale_mm_per_px")
        if scale is not None:
            scale = float(scale)
            if scale <= 0:
                raise LandmarkValidationError(
                    f"{where}: scale_mm_per_px 는 0보다 커야 합니다.", detail=scale
                )

        ref_pair = item.get("reference_pair")
        if ref_pair is not None:
            if not isinstance(ref_pair, (list, tuple)) or len(ref_pair) != 2:
                raise LandmarkValidationError(
                    f"{where}: reference_pair 는 랜드마크 이름 2개여야 합니다.",
                    detail=ref_pair,
                )
            ref_pair = (str(ref_pair[0]), str(ref_pair[1]))

        images.append(
            LandmarkImage(
                view=view,
                landmarks=landmarks,
                image_size_px=image_size,
                scale_mm_per_px=scale,
                reference_length_mm=(
                    float(item["reference_length_mm"])
                    if item.get("reference_length_mm") is not None
                    else None
                ),
                reference_pair=ref_pair,
            )
        )

    explicit = raw.get("measurements") or {}
    if not isinstance(explicit, dict):
        raise LandmarkValidationError("'measurements' 는 dict 여야 합니다.", detail=explicit)
    unknown = set(explicit) - set(MEASUREMENT_LABELS)
    if unknown:
        raise LandmarkValidationError(
            f"알 수 없는 계측 항목: {sorted(unknown)}. "
            f"가능한 항목: {sorted(MEASUREMENT_LABELS)}"
        )
    explicit = {k: float(v) for k, v in explicit.items() if v is not None}

    offsets_raw = raw.get("control_point_offsets_mm") or {}
    if not isinstance(offsets_raw, dict):
        raise LandmarkValidationError(
            "'control_point_offsets_mm' 는 dict 여야 합니다.", detail=offsets_raw
        )
    offsets: dict[str, tuple[float, float, float]] = {}
    for name, vec in offsets_raw.items():
        if not isinstance(vec, (list, tuple)) or len(vec) != 3:
            raise LandmarkValidationError(
                f"control_point_offsets_mm['{name}'] 는 [dx, dy, dz] 여야 합니다.",
                detail=vec,
            )
        offsets[str(name)] = (float(vec[0]), float(vec[1]), float(vec[2]))

    if not images and not explicit:
        raise LandmarkValidationError(
            "변형을 구동할 정보가 없습니다. 'images' 또는 'measurements' 중 "
            "최소 하나는 제공해야 합니다."
        )

    return LandmarkPayload(
        images=images,
        explicit_measurements=explicit,
        side=side,
        control_point_offsets_mm=offsets,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QualityReport:
    """메쉬 품질 점검 결과."""

    is_watertight: bool = False
    is_winding_consistent: bool = False
    is_volume_positive: bool = False
    euler_number: int = 0
    volume_mm3: float = 0.0
    vertex_count: int = 0
    face_count: int = 0
    holes_filled: int = 0
    degenerate_faces_removed: int = 0
    flipped_face_ratio: float = 0.0
    repaired: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return (
            self.is_watertight
            and self.is_winding_consistent
            and self.is_volume_positive
            and not self.warnings
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class DeformationReport:
    """한 번의 `deform_mesh()` 호출에 대한 요약. API 응답 body 로 그대로 사용 가능."""

    side: str = "right"
    image_count: int = 0
    template_measurements: dict[str, float] = field(default_factory=dict)
    target_measurements: dict[str, float] = field(default_factory=dict)
    achieved_measurements: dict[str, float] = field(default_factory=dict)
    scale_factors: dict[str, float] = field(default_factory=dict)
    measurement_error_mm: dict[str, float] = field(default_factory=dict)
    control_point_count: int = 0
    max_displacement_mm: float = 0.0
    arch_iterations_used: int = 0
    quality: QualityReport = field(default_factory=QualityReport)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["quality"] = self.quality.to_dict()
        return data

    def summary_lines(self) -> list[str]:
        """CLI 출력용 요약 표."""
        lines = [
            f"{'계측 항목':<22}{'템플릿':>10}{'목표':>10}{'결과':>10}{'오차':>9}",
            "-" * 61,
        ]
        for name in FootMeasurements.field_names():
            t = self.template_measurements.get(name)
            g = self.target_measurements.get(name)
            a = self.achieved_measurements.get(name)
            e = self.measurement_error_mm.get(name)
            lines.append(
                f"{name:<22}"
                f"{'-' if t is None else format(t, '10.1f')}"
                f"{'-' if g is None else format(g, '10.1f')}"
                f"{'-' if a is None else format(a, '10.1f')}"
                f"{'-' if e is None else format(e, '9.2f')}"
            )
        return lines
