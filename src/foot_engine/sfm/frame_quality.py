"""SfM에 넘기기 전, 프레임 1장 단위로 "쓸 만한가"를 판별하는 절대기준 게이트.

`reconstruction.filter_blurry_frames()`는 배치 **상대** 순위(선명도 상위
keep_ratio%)로 골라내는 도구라 배치 전체가 흐리면 그 안에서만 상위를 뽑아
결국 나쁜 프레임을 통과시킨다. `reconstruction.report_unregistered_frames()`는
SfM을 이미 다 돌린 **뒤** 실패 원인을 보여주는 사후 진단이라, 애초에 못 쓸
프레임에 특징점 추출/매칭 비용을 쓴 다음에야 알 수 있다.

이 모듈은 그 사이 빈틈 — SfM을 돌리기 **전**, 각 프레임이 절대 기준을
통과하는지 미리 판별해 걸러내는 단계 — 를 채운다. 여기서 거르는 건 카메라/
피사체에 무관하게 보편적으로 "못 쓰는" 프레임(파일 손상, 렌즈 캡, 완전
날아간 노출 등)이지, 이 세트 안에서 상대적으로 나은/나쁜 걸 가리는 게
아니다 — 그건 여전히 `filter_blurry_frames()`의 역할이다.

**주의 — 블러 절대 임계값(`min_sharpness`)은 기본 비활성(None)**: 이 파일의
다른 임계값(노출, 해상도)은 0~255 강도 스케일 기준이라 카메라 기종에
크게 좌우되지 않지만, 라플라시안 분산(선명도 점수)의 절대 스케일은 해상도·
피사체 질감·조명에 따라 달라진다. 이 프로젝트의 다른 기본값들은 전부
실측 A/B로 검증된 숫자인데(모듈 상단 주석들 참고), 이 값만 근거 없이 지어
넣으면 멀쩡한 프레임을 잘못 버릴 수 있다. 실제 촬영 영상으로
`scripts/inspect_frame_quality.py`를 돌려 선명도 분포를 보고 임계값을 정한
뒤 `min_sharpness`로 넘길 것.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .reconstruction import IMAGE_EXTS, compute_sharpness

#: 그레이스케일 강도가 이 값 미만/초과인 픽셀을 "클리핑(정보 손실)"으로 센다.
_CLIP_LOW = 5
_CLIP_HIGH = 250

# 노출/해상도 기본값은 카메라 기종에 크게 좌우되지 않는 보편적 안전선이다
# (렌즈 캡·완전 암전·완전 화이트아웃·손상 파일처럼 "명백히 못 쓰는" 경우만
# 잡도록 느슨하게 잡았다 — 정상 범위의 그림자/하이라이트는 통과시킨다).
DEFAULT_MIN_WIDTH = 240
DEFAULT_MIN_HEIGHT = 240
DEFAULT_MIN_MEAN_BRIGHTNESS = 15.0
DEFAULT_MAX_MEAN_BRIGHTNESS = 240.0
DEFAULT_MAX_CLIP_FRACTION = 0.98


@dataclass(slots=True)
class FrameQualityResult:
    """프레임 1장의 QC 판정 결과."""

    name: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    sharpness: float = 0.0
    mean_brightness: float = 0.0
    low_clip_frac: float = 0.0
    high_clip_frac: float = 0.0


def _exposure_stats(gray: np.ndarray) -> tuple[float, float, float]:
    """(평균 밝기, 저(암부) 클리핑 비율, 고(하이라이트) 클리핑 비율)."""
    mean_brightness = float(gray.mean())
    low_clip = float((gray < _CLIP_LOW).mean())
    high_clip = float((gray > _CLIP_HIGH).mean())
    return mean_brightness, low_clip, high_clip


def assess_frame(
    path: Path,
    *,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
    min_sharpness: float | None = None,
    min_mean_brightness: float = DEFAULT_MIN_MEAN_BRIGHTNESS,
    max_mean_brightness: float = DEFAULT_MAX_MEAN_BRIGHTNESS,
    max_clip_fraction: float = DEFAULT_MAX_CLIP_FRACTION,
) -> FrameQualityResult:
    """프레임 1장을 절대 기준으로 판별한다.

    Args:
        min_sharpness: 라플라시안 분산 절대 임계값. None(기본)이면 이 검사를
            건너뛴다 — 모듈 docstring 참고, 실측 없이 기본값을 넣지 않았다.
        max_clip_fraction: 그레이스케일 픽셀 중 (거의) 순검정/순백인 비율이
            이 값을 넘으면 렌즈 캡/완전 화이트아웃으로 보고 탈락시킨다.
    """
    reasons: list[str] = []

    img = cv2.imread(str(path))
    if img is None:
        return FrameQualityResult(name=path.name, passed=False, reasons=["unreadable"])

    height, width = img.shape[:2]
    if width < min_width or height < min_height:
        reasons.append(f"resolution_too_low({width}x{height})")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness, low_clip, high_clip = _exposure_stats(gray)
    if mean_brightness < min_mean_brightness:
        reasons.append(f"too_dark(mean={mean_brightness:.1f})")
    elif mean_brightness > max_mean_brightness:
        reasons.append(f"too_bright(mean={mean_brightness:.1f})")
    clip_frac = low_clip + high_clip
    if clip_frac > max_clip_fraction:
        reasons.append(f"blown_out(clip_frac={clip_frac:.2f})")

    sharpness = compute_sharpness(path)
    if min_sharpness is not None and sharpness < min_sharpness:
        reasons.append(f"too_blurry(sharpness={sharpness:.1f}<{min_sharpness:.1f})")

    return FrameQualityResult(
        name=path.name,
        passed=not reasons,
        reasons=reasons,
        width=width,
        height=height,
        sharpness=sharpness,
        mean_brightness=mean_brightness,
        low_clip_frac=low_clip,
        high_clip_frac=high_clip,
    )


def assess_frames(
    images_dir: Path,
    names: list[str] | None = None,
    **thresholds,
) -> list[FrameQualityResult]:
    """이미지 폴더(또는 그 안의 일부 파일 목록)를 일괄 판별한다.

    Args:
        names: 지정하면 이 파일들만(다른 게이트를 이미 통과한 후보 목록 등).
            None이면 `images_dir`의 전체 이미지 파일.
        **thresholds: `assess_frame()`에 그대로 전달되는 임계값 키워드.
    """
    if names is not None:
        paths = [images_dir / name for name in names]
    else:
        paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [assess_frame(p, **thresholds) for p in paths]


def passed_names(results: list[FrameQualityResult]) -> list[str]:
    """통과한 프레임 파일명만 정렬해서 반환."""
    return sorted(r.name for r in results if r.passed)


def summarize(results: list[FrameQualityResult]) -> dict:
    """사유별 탈락 수 집계 — 로그/리포트용."""
    reason_counts: dict[str, int] = {}
    for r in results:
        for reason in r.reasons:
            key = reason.split("(")[0]  # 수치 붙은 상세 사유는 종류만 집계
            reason_counts[key] = reason_counts.get(key, 0) + 1
    passed = sum(1 for r in results if r.passed)
    return {
        "total": len(results),
        "passed": passed,
        "rejected": len(results) - passed,
        "reject_reasons": reason_counts,
    }


def print_summary(results: list[FrameQualityResult]) -> None:
    stats = summarize(results)
    print(
        f"[QC] {stats['total']}장 중 {stats['passed']}장 통과, "
        f"{stats['rejected']}장 탈락"
    )
    if stats["reject_reasons"]:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(stats["reject_reasons"].items()))
        print(f"[QC] 탈락 사유: {detail}")
