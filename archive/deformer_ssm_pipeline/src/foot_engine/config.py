"""엔진 전역 설정값과 '해부학적 규약(convention)' 정의.

여기서 정의한 규약은 템플릿 메쉬와 변형 알고리즘이 공유하는 계약이다.
실제 스캔 기반 템플릿으로 교체할 때는 이 파일의 상수만 조정하면 된다.

--------------------------------------------------------------------------
정규 좌표계 (Canonical Frame)
--------------------------------------------------------------------------
    X : 발 길이 방향. x_min = 뒤꿈치(heel) 끝,  x_max = 발가락(toe) 끝
    Y : 발 너비 방향. 오른발 기준 y_min = 내측(medial, 엄지쪽),
                                  y_max = 외측(lateral, 새끼쪽)
    Z : 높이 방향.   z_min = 바닥(sole),        z_max = 발목 절단면(leg cut)
    단위: mm

정규화 좌표 (u, v, w) 는 Bounding Box 기준 0~1 값이다.
    u = (x - x_min) / L,  v = (y - y_min) / W,  w = (z - z_min) / H
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# 1) 계측 구간(Region Window) 규약  — 모두 정규화 u(길이방향) 기준
# ---------------------------------------------------------------------------

#: 뒤꿈치 너비를 재는 구간
HEEL_WINDOW_U: tuple[float, float] = (0.03, 0.20)
#: 볼(ball, MTP1~MTP5) 너비를 재는 구간
BALL_WINDOW_U: tuple[float, float] = (0.60, 0.80)
#: 발등(instep) 높이를 재는 구간
INSTEP_WINDOW_U: tuple[float, float] = (0.45, 0.62)
#: 발목/뒤꿈치 기둥 구간 (발목 둘레·너비 계측용)
REAR_WINDOW_U: tuple[float, float] = (0.02, 0.30)
#: 아치(arch) 곡률을 재고 변형하는 구간
ARCH_WINDOW_U: tuple[float, float] = (0.28, 0.58)
#: 아치 정점(apex)을 읽는 내측 밴드 — 측면 사진의 실루엣 정의와 일치시킨다.
ARCH_MEDIAL_BAND_V: tuple[float, float] = (0.00, 0.30)

#: 템플릿의 '복사뼈(malleolus) 높이'를 정의하는 규약값.
#: 기하학적으로 자동 검출이 어려우므로 제어점 규약(ankle_medial 의 w)과 동일하게 고정한다.
ANKLE_HEIGHT_W_FRACTION: float = 0.62
#: 발목 너비를 재는 높이 밴드 (w = ANKLE_HEIGHT_W_FRACTION ± ANKLE_BAND_W)
ANKLE_BAND_W: float = 0.08

#: 바닥면(plantar surface) 판정용 법선 임계값 (vertex normal 의 z 성분)
SOLE_NORMAL_Z_MAX: float = -0.25


# ---------------------------------------------------------------------------
# 2) 제어점(Control Point) 규약
# ---------------------------------------------------------------------------

#: 해부학적 제어점의 정규화 좌표 (u, v, w).
#: RBF 의 source point 로 사용되며, 이름은 landmarks_data 의
#: `control_point_offsets_mm` 로 개별 미세 조정할 때의 key 이기도 하다.
ANATOMICAL_CONTROL_POINTS: dict[str, tuple[float, float, float]] = {
    # --- 뒤꿈치 / 발목 ---
    "heel_back_center":   (0.00, 0.50, 0.28),
    "heel_bottom_center": (0.06, 0.50, 0.00),
    "heel_medial":        (0.10, 0.00, 0.25),
    "heel_lateral":       (0.10, 1.00, 0.25),
    "ankle_medial":       (0.16, 0.00, ANKLE_HEIGHT_W_FRACTION),
    "ankle_lateral":      (0.16, 1.00, ANKLE_HEIGHT_W_FRACTION - 0.04),
    "ankle_top_center":   (0.08, 0.50, 1.00),
    "ankle_front_center": (0.30, 0.50, 0.72),
    # --- 중족부 / 아치 ---
    "midfoot_medial":     (0.42, 0.00, 0.18),
    "midfoot_lateral":    (0.42, 1.00, 0.12),
    "arch_apex":          (0.42, 0.15, 0.05),
    "sole_mid_center":    (0.50, 0.50, 0.02),
    "instep_top":         (0.52, 0.50, 0.55),
    # --- 전족부 / 발가락 ---
    "ball_medial_mtp1":   (0.70, 0.00, 0.10),
    "ball_lateral_mtp5":  (0.66, 1.00, 0.09),
    "ball_bottom_center": (0.70, 0.50, 0.00),
    "toe_dorsum":         (0.85, 0.45, 0.18),
    "hallux_tip":         (0.98, 0.12, 0.10),
    "fifth_toe_tip":      (0.92, 0.90, 0.07),
    "toe_tip_center":     (1.00, 0.45, 0.10),
}


# ---------------------------------------------------------------------------
# 3) 변형 파라미터
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DeformConfig:
    """변형 엔진 튜닝 파라미터.

    FastAPI 에서 요청별로 다른 설정을 주고 싶으면 이 dataclass 를
    Pydantic 모델로 감싸서 그대로 넘기면 된다.
    """

    # --- RBF / TPS ---
    #: scipy.interpolate.RBFInterpolator 커널.
    #: 'thin_plate_spline' 이 3D 형상 보간의 표준(TPS). 'multiquadric' 도 사용 가능.
    rbf_kernel: str = "thin_plate_spline"
    #: 0 이면 제어점을 정확히 통과(보간), >0 이면 완만하게 근사(스무딩)
    rbf_smoothing: float = 0.0
    #: 부가 다항식 차수. TPS 는 최소 1 이상이어야 far-field 가 안정적이다.
    rbf_degree: int = 1
    #: 정점 수가 많을 때 메모리 폭주를 막기 위한 평가 청크 크기
    rbf_chunk_size: int = 20_000

    # --- 원거리장(far-field) 고정용 격자 앵커 ---
    #: Bounding Box 를 이 비율만큼 확장한 위치에 격자 제어점을 배치
    lattice_padding: float = 0.25
    #: 격자 해상도(축당 점 개수). 3 이면 3x3x3 = 27점
    lattice_resolution: int = 3

    # --- 아치(arch) 국소 변형 ---
    #: 아치 정점의 길이방향 중심(u)
    arch_center_u: float = 0.43
    #: 길이방향 가우시안 표준편차(u 단위)
    arch_sigma_u: float = 0.13
    #: 바닥에서 이 비율(w) 이상 높이는 아치 변형의 영향을 받지 않음
    arch_z_falloff_w: float = 0.35
    #: 내측 가중치 하한 (외측 끝단에서의 가중치)
    arch_lateral_floor: float = 0.35
    #: 목표 아치 높이 수렴을 위한 반복 횟수
    arch_iterations: int = 3
    #: 아치 높이 수렴 허용 오차(mm)
    arch_tolerance_mm: float = 0.3

    # --- 품질 보장 ---
    #: trimesh.repair 기반 자동 복구 수행 여부
    auto_repair: bool = True
    #: 뒤집힌 면(inverted face) 허용 비율. 초과 시 경고(strict 면 예외)
    max_flipped_face_ratio: float = 0.01
    #: True 면 품질 미달 시 MeshQualityError 를 발생시킨다.
    strict_quality: bool = False
    #: 목표 계측치 대비 결과 오차 경고 임계값(%)
    measurement_tolerance_pct: float = 2.0

    # --- 안전장치 ---
    #: 변위가 발 길이의 이 비율을 넘으면 발산으로 간주하고 예외
    max_displacement_ratio: float = 0.6

    def validate(self) -> None:
        """설정값 sanity check."""
        if self.rbf_degree < 1 and self.rbf_kernel == "thin_plate_spline":
            raise ValueError("thin_plate_spline 커널은 rbf_degree >= 1 이어야 합니다.")
        if not (0.0 < self.arch_center_u < 1.0):
            raise ValueError("arch_center_u 는 0~1 사이여야 합니다.")
        if self.lattice_resolution < 2:
            raise ValueError("lattice_resolution 은 2 이상이어야 합니다.")


# ---------------------------------------------------------------------------
# 4) 스케일 프로파일 노드
#    길이방향 위치(u)에 따라 폭/높이 스케일을 어떻게 섞을지 정의한다.
# ---------------------------------------------------------------------------

#: 폭(Y) 스케일 보간 노드 — [뒤꿈치, 뒤꿈치, 중족(혼합), 볼, 볼]
WIDTH_PROFILE_U: np.ndarray = np.array([0.00, 0.18, 0.40, 0.70, 1.00])
#: 높이(Z) 스케일 보간 노드 — [발목, 발목, 발등, 발등, 발등]
HEIGHT_PROFILE_U: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75, 1.00])

#: '발목 너비' 스케일이 지배하기 시작/완료하는 구간.
#: 복사뼈 계측 밴드(u<=0.20, w≈ANKLE_HEIGHT_W_FRACTION)에서 가중치가 1.0 이 되도록
#: 잡아야 목표 발목 너비가 그대로 재현된다.
ANKLE_BLEND_U_RANGE: tuple[float, float] = (0.20, 0.35)  # u<=0.20 → 1.0, u>=0.35 → 0.0
ANKLE_BLEND_W_RANGE: tuple[float, float] = (0.42, 0.62)  # w<=0.42 → 0.0, w>=0.62 → 1.0


def medial_arch_weight(
    v: np.ndarray,
    *,
    floor: float = 0.35,
    span: float = 0.70,
) -> np.ndarray:
    """내측(medial) → 외측(lateral) 아치 가중치 프로파일.

    v = 0 (내측 끝)에서 1.0, v >= span 에서 `floor` 로 수렴한다.
    템플릿 생성기와 변형기가 **동일한 프로파일**을 써야 아치 높이 계측이
    자기일관적(self-consistent)이 되므로 공용 함수로 둔다.

    Args:
        v: 정규화 폭 좌표 (0=내측, 1=외측). shape 자유.
        floor: 외측 끝단 가중치 하한.
        span: 가중치가 floor 로 떨어지는 구간 폭.

    Returns:
        v 와 같은 shape 의 가중치 배열 (floor ~ 1.0).
    """
    v = np.asarray(v, dtype=float)
    medialness = np.clip((span - v) / span, 0.0, 1.0)
    return floor + (1.0 - floor) * medialness


def smoothstep(t: np.ndarray) -> np.ndarray:
    """0~1 구간의 3차 스무스스텝. 국소 변형의 경계를 부드럽게 만든다."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# 5) 기본 경로 (프로젝트 루트 기준)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Paths:
    """프로젝트 기본 경로 모음. 배포 시 환경변수로 덮어쓰기 쉽게 분리."""

    template_dir: str = "data/templates"
    sample_dir: str = "data/samples"
    output_dir: str = "data/output"
    default_template_name: str = "base_foot_template.stl"


DEFAULT_PATHS = Paths()

#: export 가 지원하는 확장자
SUPPORTED_EXPORT_SUFFIXES: frozenset[str] = frozenset({".stl", ".glb", ".ply", ".obj"})
