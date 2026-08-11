"""세그멘테이션 실루엣(마스크) → 2D 랜드마크 픽셀 좌표 자동 추출.

`landmarks.py`는 "사진 속 이름 붙은 랜드마크의 픽셀 좌표"를 입력으로 받아 계측치를
뽑는다 — 그 픽셀 좌표를 사람이 손으로 찍지 않고, 이미 갖고 있는 배경 제거 마스크
(`generate_foot_masks.py`, rembg 기반)의 윤곽선에서 기하학적으로 직접 추출하는 게
이 모듈의 역할이다.

    마스크(0/255 이진 이미지)
        └─ 가장 큰 윤곽선 추출
              └─ PCA로 발의 길이축·너비축을 찾는다(사진이 어느 각도로 찍혔든 무관)
                    └─ 길이축을 따라 0(뒤꿈치)~1(발끝)로 정규화 — `config.py`의
                       HEEL_WINDOW_U/BALL_WINDOW_U 등 3D 계측과 **같은 구간 규약**을 그대로 씀
                          └─ 구간별 폭 최댓값 지점 = 뒤꿈치/발볼 내외측 랜드마크
                          └─ 중족부 오목한 정도로 내측(medial) 쪽 판별
                                └─ 이름 붙은 랜드마크 dict(px 좌표)로 반환

전제(중요): 이 모듈은 **위에서 내려다본(top-down) 발 실루엣 하나**만 다룬다. 아치
높이·발등 높이처럼 옆에서 봐야 하는 계측치는 실루엣만으로는 원리적으로 알 수 없다
(2D 윤곽선에는 높이 정보가 없다) — 그 항목들은 계속 템플릿 기본값을 쓰거나, 별도
측면 사진에서 사람이 보조 입력하는 경로가 필요하다.

발끝/뒤꿈치 자동 판별은 발끝 쪽이 뒤꿈치 쪽보다 대체로 더 넓다(발볼이 있으므로)는
경험적 규칙에 의존한다 — 완벽하지 않으므로 `toe_end` 로 강제 지정할 수 있게 열어둔다.
"""

from __future__ import annotations

import numpy as np

from . import config as cfg
from .exceptions import LandmarkValidationError

try:
    import cv2
except ImportError as exc:  # pragma: no cover - 환경에 opencv 없을 때만
    raise ImportError(
        "silhouette_landmarks 모듈은 opencv-python(cv2)이 필요합니다."
    ) from exc


def _clean_mask(mask: np.ndarray, *, open_kernel_frac: float = 0.03) -> np.ndarray:
    """가는 연결부(발-가구 다리처럼)를 끊어내는 morphological opening.

    실 서비스에서는 사용자가 초점이 나가거나 엉뚱한 프레임(다른 사람이 잡히는 등)은
    직접 골라내지만, 발 주변 바닥·가구 일부가 마스크에 살짝 같이 붙는 것까지는
    막기 어렵다. 그런 배경 조각이 발과 **가는 다리 하나로만** 이어져 있으면(예: 의자
    다리), opening(침식→팽창)으로 그 연결을 끊어 별개 윤곽선으로 분리할 수 있다 —
    각자 두꺼운 덩어리는 살아남고, 얇은 다리만 사라진다.
    """
    binary = (mask > 127).astype(np.uint8) * 255
    kernel_size = max(3, int(round(min(mask.shape) * open_kernel_frac)) | 1)  # 홀수로
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    """이진 마스크에서 가장 큰 윤곽선의 점들을 (N,2) px 좌표로 반환.

    Opening으로 가는 배경 연결을 끊은 뒤 면적이 가장 큰 덩어리를 고른다 — 발이
    화면에서 가장 큰 피사체라는 전제(실 서비스에서는 사용자가 프레임을 그렇게
    고른다는 전제)에 의존한다.
    """
    cleaned = _clean_mask(mask)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise LandmarkValidationError("마스크에서 윤곽선을 찾을 수 없습니다(빈 마스크?).")
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(float)  # (N, 2) — (x, y) px


def _pca_axes_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2D 점들의 주축(분산 내림차순)과 중심을 반환. axes[:,0]=길이축, axes[:,1]=너비축."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order], centroid


def _prepare_length_axis(
    contour: np.ndarray, toe_end: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """윤곽선에서 PCA 길이축을 찾고 u(0=뒤꿈치~1=발끝)·직교축 좌표를 계산한다.

    top/side 뷰 추출기가 공통으로 쓰는 전처리 — 발끝/뒤꿈치 판별 방식만 다르므로
    `toe_end`를 강제 지정할 수 있게 열어둔다(자동 판별 규칙은 top 뷰의 "발볼이 더
    넓다"는 경험칙 하나뿐이라, side 뷰에서는 지정을 권장).

    Returns:
        (axes, centroid, u, cross) — cross는 아직 부호(내측/외측 또는 바닥/발등)가
        정해지지 않은 직교축 로컬 좌표.
    """
    axes, centroid = _pca_axes_2d(contour)
    local = (contour - centroid) @ axes
    length_min, length_max = local[:, 0].min(), local[:, 0].max()
    span = length_max - length_min
    if span < 1e-6:
        raise LandmarkValidationError("실루엣이 퇴화했습니다(길이 방향 폭이 0).")

    def _width_near(end_value: float, inward_sign: float, frac: float = 0.10) -> float:
        probe = end_value + inward_sign * span * frac
        band = np.abs(local[:, 0] - probe) < span * 0.03
        if not band.any():
            return 0.0
        return float(local[band, 1].max() - local[band, 1].min())

    if toe_end is None:
        toe_sign = 1.0 if _width_near(length_max, -1.0) >= _width_near(length_min, 1.0) else -1.0
    else:
        toe_sign = 1.0 if toe_end == "positive" else -1.0

    signed_length = local[:, 0] * toe_sign
    u = (signed_length - signed_length.min()) / (signed_length.max() - signed_length.min())
    return axes, centroid, u, local[:, 1]


def extract_top_view_landmarks(
    mask: np.ndarray, *, toe_end: str | None = None
) -> dict[str, tuple[float, float]]:
    """위에서 본 발 실루엣 마스크에서 이름 붙은 랜드마크의 px 좌표를 뽑는다.

    Args:
        mask: (H, W) 그레이스케일 마스크. 0=배경, 그 외=발.
        toe_end: "positive" | "negative" | None. 길이축(PCA 1주성분)의 어느 방향이
            발끝인지 강제 지정. None이면 폭이 더 넓은 쪽을 발끝으로 자동 판별한다
            (발볼이 뒤꿈치보다 넓다는 경험칙 — 촬영 각도가 애매하면 틀릴 수 있으니
            중요한 용도에서는 지정을 권장).

    Returns:
        `landmarks.py`의 "top" 뷰가 기대하는 이름으로 키가 붙은 {name: (x_px, y_px)}.
        heel_center, toe_tip, heel_medial, heel_lateral, mtp1_medial, mtp5_lateral.

    Raises:
        LandmarkValidationError: 윤곽선을 찾을 수 없거나 형태가 퇴화한 경우.
    """
    contour = _largest_contour(mask)
    axes, centroid, u, width_coord = _prepare_length_axis(contour, toe_end)

    def _extreme_point(u_target: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(u - u_target)))
        return contour[idx]

    def _window_edges(window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """구간 내 폭 최솟값/최댓값 지점의 원본 px 좌표. (min_side_point, max_side_point)."""
        mask_u = (u >= window[0]) & (u <= window[1])
        if not mask_u.any():
            raise LandmarkValidationError(f"구간 {window} 안에 윤곽선 점이 없습니다.")
        idx_lo = np.flatnonzero(mask_u)[np.argmin(width_coord[mask_u])]
        idx_hi = np.flatnonzero(mask_u)[np.argmax(width_coord[mask_u])]
        return contour[idx_lo], contour[idx_hi]

    heel_center_px = _extreme_point(0.0)
    toe_tip_px = _extreme_point(1.0)
    heel_lo_px, heel_hi_px = _window_edges(cfg.HEEL_WINDOW_U)
    ball_lo_px, ball_hi_px = _window_edges(cfg.BALL_WINDOW_U)

    # --- 내측(medial) 판별: 중족부(아치 구간)에서 더 오목하게 들어간 쪽이 내측 ------
    arch_mask = (u >= cfg.ARCH_WINDOW_U[0]) & (u <= cfg.ARCH_WINDOW_U[1])
    if arch_mask.any():
        # 뒤꿈치~발볼 폭-최솟값측 지점을 잇는 직선 대비, 아치 구간에서 안쪽으로
        # 얼마나 들어와 있는지를 양쪽(low/high width side)에 대해 각각 잰다.
        heel_u, ball_u = 0.5 * sum(cfg.HEEL_WINDOW_U), 0.5 * sum(cfg.BALL_WINDOW_U)

        def _concavity(side: str) -> float:
            coord = width_coord if side == "lo" else -width_coord
            ref_heel = heel_lo_px if side == "lo" else heel_hi_px
            ref_ball = ball_lo_px if side == "lo" else ball_hi_px
            # 두 기준점을 로컬(길이,폭) 좌표로 변환해 직선 보간
            ref_heel_local = (ref_heel - centroid) @ axes
            ref_ball_local = (ref_ball - centroid) @ axes
            ref_heel_w = ref_heel_local[1] if side == "lo" else -ref_heel_local[1]
            ref_ball_w = ref_ball_local[1] if side == "lo" else -ref_ball_local[1]
            arch_idx = np.flatnonzero(arch_mask)
            interp_w = np.interp(u[arch_idx], [heel_u, ball_u], [ref_heel_w, ref_ball_w])
            actual_w = coord[arch_idx]
            # 직선보다 안쪽(작은 값)으로 들어간 정도의 최댓값 = 오목함 크기
            return float(np.max(interp_w - actual_w))

        medial_is_lo = _concavity("lo") >= _concavity("hi")
    else:
        medial_is_lo = True  # 판별 불가 시 기본값(추정 실패는 경고 대상으로 상위에서 처리)

    if medial_is_lo:
        heel_medial_px, heel_lateral_px = heel_lo_px, heel_hi_px
        mtp1_medial_px, mtp5_lateral_px = ball_lo_px, ball_hi_px
    else:
        heel_medial_px, heel_lateral_px = heel_hi_px, heel_lo_px
        mtp1_medial_px, mtp5_lateral_px = ball_hi_px, ball_lo_px

    return {
        "heel_center": (float(heel_center_px[0]), float(heel_center_px[1])),
        "toe_tip": (float(toe_tip_px[0]), float(toe_tip_px[1])),
        "heel_medial": (float(heel_medial_px[0]), float(heel_medial_px[1])),
        "heel_lateral": (float(heel_lateral_px[0]), float(heel_lateral_px[1])),
        "mtp1_medial": (float(mtp1_medial_px[0]), float(mtp1_medial_px[1])),
        "mtp5_lateral": (float(mtp5_lateral_px[0]), float(mtp5_lateral_px[1])),
    }


def _binned_profiles(
    u: np.ndarray, cross: np.ndarray, contour: np.ndarray, n_bins: int = 48
) -> list[tuple[float, int, int]]:
    """u를 n_bins 구간으로 나눠, 구간별 (u중심, cross최솟값 인덱스, cross최댓값 인덱스)를 낸다.

    옆에서 본 실루엣은 닫힌 윤곽선 하나지만, 매 길이 위치(u)마다 "바닥쪽 가장자리"와
    "발등쪽 가장자리" 두 개의 경계가 있다 — 그 구간별 최솟값/최댓값이 각각의 경계다.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        in_bin = (u >= edges[i]) & (u < edges[i + 1] if i < n_bins - 1 else u <= edges[i + 1])
        if not in_bin.any():
            continue
        idx = np.flatnonzero(in_bin)
        lo_idx = idx[np.argmin(cross[idx])]
        hi_idx = idx[np.argmax(cross[idx])]
        out.append((float((edges[i] + edges[i + 1]) / 2), int(lo_idx), int(hi_idx)))
    return out


def extract_side_view_landmarks(
    mask: np.ndarray, *, toe_end: str | None = None
) -> dict[str, tuple[float, float]]:
    """옆에서 본(medial/lateral) 발 실루엣 마스크에서 랜드마크 px 좌표를 뽑는다.

    Top view와 달리 이 축(교차축)은 폭이 아니라 **사진 속 높이**를 뜻한다 — 바닥에
    닿는 발바닥 쪽 경계와, 발등·발목이 있는 위쪽 경계, 두 개로 나뉜다. 어느 쪽이
    바닥인지는 "발바닥 쪽이 발등 쪽보다 평평하다"(굴곡이 적다)는 경험칙으로
    판별한다 — 발등 쪽은 발목 돌기·발등 융기 때문에 굴곡이 뚜렷하다.

    Args:
        mask: (H, W) 그레이스케일 마스크.
        toe_end: `extract_top_view_landmarks`와 동일. 옆모습은 발볼 폭 경험칙을 못
            쓰므로(폭이 아니라 높이 축이라) 자동판별 신뢰도가 낮다 — 지정을 권장.

    Returns:
        {name: (x_px, y_px)}. heel_back, toe_tip, arch_apex, instep_top,
        medial_malleolus(또는 lateral_malleolus — 어느 쪽 옆모습인지는 알 수 없으므로
        `landmarks.py` 규칙에 맞춰 호출자가 이름을 붙여야 함, 기본은 medial 이름 사용),
        ground_ref.
    """
    contour = _largest_contour(mask)
    axes, centroid, u, cross = _prepare_length_axis(contour, toe_end)

    def _extreme_point(u_target: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(u - u_target)))
        return contour[idx]

    heel_back_px = _extreme_point(0.0)
    toe_tip_px = _extreme_point(1.0)

    profiles = _binned_profiles(u, cross, contour)
    if len(profiles) < 6:
        raise LandmarkValidationError("실루엣이 너무 작거나 얇아 옆모습 랜드마크를 뽑을 수 없습니다.")

    # --- 바닥(sole) 쪽 판별: 중앙 구간(발끝/뒤꿈치 끝 제외)에서 더 평평한(분산이
    # 작은) 쪽이 바닥이다 — 발등 쪽은 발목 돌기·발등 융기로 굴곡이 크다.
    central = [(uc, lo, hi) for uc, lo, hi in profiles if 0.12 <= uc <= 0.88]
    lo_values = np.array([cross[lo] for _, lo, _ in central])
    hi_values = np.array([cross[hi] for _, _, hi in central])
    sole_is_lo = lo_values.std() <= hi_values.std()

    def _profile_arrays(is_lo: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx_col = 1 if is_lo else 2
        us = np.array([p[0] for p in profiles])
        idxs = np.array([p[idx_col] for p in profiles])
        values = cross[idxs]
        return us, idxs, values

    sole_u, sole_idx, sole_val = _profile_arrays(sole_is_lo)
    dorsum_u, dorsum_idx, dorsum_val = _profile_arrays(not sole_is_lo)

    # 바닥 기준선: 뒤꿈치·발볼 구간(체중이 실려 바닥에 닿는 곳)의 바닥 프로필 중앙값.
    ground_band = (sole_u >= cfg.HEEL_WINDOW_U[0]) & (sole_u <= cfg.BALL_WINDOW_U[1])
    ground_level = float(np.median(sole_val[ground_band])) if ground_band.any() else float(np.median(sole_val))
    ground_ref_px = contour[sole_idx[np.argmin(np.abs(sole_u - 0.5 * sum(cfg.HEEL_WINDOW_U)))]]

    # 바닥이 lo쪽이면 "바닥에서 들뜬 정도"는 sole_val이 ground_level보다 커질수록 크다.
    # (hi쪽이 바닥이면 반대로 작아질수록 들뜬 것.)
    lift_sign = 1.0 if sole_is_lo else -1.0

    def _pick_in_window(values: np.ndarray, us: np.ndarray, idxs: np.ndarray, window: tuple[float, float], sign: float) -> np.ndarray:
        band = (us >= window[0]) & (us <= window[1])
        if not band.any():
            raise LandmarkValidationError(f"구간 {window} 안에 옆모습 윤곽선 점이 없습니다.")
        local_idx = np.flatnonzero(band)
        best = local_idx[np.argmax(sign * (values[local_idx] - ground_level))]
        return contour[idxs[best]]

    arch_apex_px = _pick_in_window(sole_val, sole_u, sole_idx, cfg.ARCH_WINDOW_U, lift_sign)
    instep_top_px = _pick_in_window(dorsum_val, dorsum_u, dorsum_idx, cfg.INSTEP_WINDOW_U, -lift_sign)
    # 복사뼈 돌기: 발등 쪽, 뒤꿈치 바로 앞 구간에서 가장 높이 들린 점으로 근사.
    malleolus_px = _pick_in_window(dorsum_val, dorsum_u, dorsum_idx, cfg.REAR_WINDOW_U, -lift_sign)

    return {
        "heel_back": (float(heel_back_px[0]), float(heel_back_px[1])),
        "toe_tip": (float(toe_tip_px[0]), float(toe_tip_px[1])),
        "arch_apex": (float(arch_apex_px[0]), float(arch_apex_px[1])),
        "instep_top": (float(instep_top_px[0]), float(instep_top_px[1])),
        "medial_malleolus": (float(malleolus_px[0]), float(malleolus_px[1])),
        "ground_ref": (float(ground_ref_px[0]), float(ground_ref_px[1])),
    }
