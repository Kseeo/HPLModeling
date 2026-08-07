"""프레임별 QC 지표(선명도/밝기/클리핑)를 표로 뽑아 임계값을 정하기 위한 진단 CLI.

`frame_quality.py`의 블러 절대 임계값(`min_sharpness`)은 실측 없이 기본값을
넣지 않았다(모듈 docstring 참고) — 실제로 촬영한 영상/사진 폴더에 대해 이
스크립트를 돌려 선명도 분포(특히 등록 성공/실패 프레임의 경계)를 보고 값을
정할 것.

사용 예::

    python scripts/inspect_frame_quality.py data/output/sfm_prototype/test02_run/images
    python scripts/inspect_frame_quality.py --video data/samples/test00.mp4 --workdir data/output/qc_test00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

import numpy as np  # noqa: E402

from foot_engine.sfm import frame_quality, reconstruction  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="프레임별 QC 지표를 표로 출력(임계값 보정용)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("images_dir", type=Path, nargs="?", help="이미 추출된 이미지 폴더")
    src.add_argument("--video", type=Path, help="순서대로 촬영된 영상 파일(먼저 프레임을 추출)")
    parser.add_argument("--workdir", type=Path, default=None, help="--video일 때 프레임을 뽑을 폴더")
    parser.add_argument("--interval", type=float, default=0.5, help="--video일 때 추출 간격(초)")
    parser.add_argument("--top", type=int, default=20, help="선명도 낮은 순으로 몇 장 보여줄지(기본 20)")
    args = parser.parse_args(argv)

    if args.video is not None:
        if args.workdir is None:
            print("[error] --video와 함께는 --workdir이 필요합니다.", file=sys.stderr)
            return 1
        images_dir = reconstruction.extract_frames(args.video, args.workdir / "images", args.interval)
    else:
        images_dir = args.images_dir
        if not images_dir.is_dir():
            print(f"[error] 이미지 폴더가 없습니다: {images_dir}", file=sys.stderr)
            return 1

    # min_sharpness=None: 여기서는 아무것도 블러로 걸러내지 않고 지표만 뽑는다.
    results = frame_quality.assess_frames(images_dir)
    if not results:
        print("[error] 이미지가 없습니다.", file=sys.stderr)
        return 1

    sharpness = np.array([r.sharpness for r in results])
    print(f"[선명도 분포] n={len(sharpness)}  "
          f"min={sharpness.min():.1f}  p10={np.percentile(sharpness, 10):.1f}  "
          f"p50={np.percentile(sharpness, 50):.1f}  p90={np.percentile(sharpness, 90):.1f}  "
          f"max={sharpness.max():.1f}")
    print(
        "[안내] SfM을 실제로 돌려 report_unregistered_frames()가 지목한 미등록 프레임들의 "
        "선명도가 몰려 있는 구간을 확인한 뒤, 그 경계값을 --min-sharpness로 넘기세요."
    )

    print(f"\n[하위 {min(args.top, len(results))}장 — 선명도 낮은 순]")
    print(f"{'파일명':<20}{'선명도':>10}{'밝기':>8}{'저클리핑':>10}{'고클리핑':>10}{'해상도':>12}  사유")
    for r in sorted(results, key=lambda r: r.sharpness)[: args.top]:
        reason = ", ".join(r.reasons) if r.reasons else "-"
        print(
            f"{r.name:<20}{r.sharpness:>10.1f}{r.mean_brightness:>8.1f}"
            f"{r.low_clip_frac:>10.3f}{r.high_clip_frac:>10.3f}"
            f"{f'{r.width}x{r.height}':>12}  {reason}"
        )

    frame_quality.print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
