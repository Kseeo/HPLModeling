"""사진 폴더 전체의 발/피부 마스크를 생성하는 CLI (rembg + MediaPipe 피부 정제).

실제 로직은 `foot_engine.sfm.masking`에 있다 — 이 파일은 인자 파싱과 출력
포맷만 담당하는 얇은 CLI 래퍼다.

사용 예::

    python scripts/generate_foot_masks.py data/output/sfm_prototype/test02_run/images `
        --out data/output/sfm_prototype/test02_run/masks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

# cp949 등 비-UTF8 콘솔에서 한글 출력이 깨지거나 죽는 문제 방지
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

from foot_engine.sfm import masking  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="발/피부 세그멘테이션 마스크 생성 (rembg + MediaPipe)")
    parser.add_argument("images_dir", type=Path, help="원본 이미지 폴더")
    parser.add_argument("--out", type=Path, required=True, help="마스크 저장 폴더")
    parser.add_argument(
        "--model", default="u2net_human_seg",
        help="rembg 모델 이름 (기본 u2net_human_seg — 사람 피부 인식에 특화)",
    )
    parser.add_argument(
        "--dilate", type=int, default=15,
        help="마스크 경계를 이만큼(px) 팽창시켜 안전 여유를 둔다(기본 15px)",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=0.03,
        help="마스크가 전체 프레임의 이 비율(기본 3%%) 미만이면 세그멘테이션 실패로 "
             "보고 그 프레임은 마스크 없이(원본 전체 사용) 처리한다",
    )
    parser.add_argument(
        "--reject-coverage", type=float, default=0.005,
        help="마스크가 전체 프레임의 이 비율(기본 0.5%%) 미만이면 세그멘테이션 실패가 "
             "아니라 발이 프레임에 없다고 보고 해당 프레임을 아예 제외한다 "
             "(--min-coverage보다 훨씬 낮게 잡을 것).",
    )
    parser.add_argument(
        "--skin-refine", dest="skin_refine", action="store_true", default=True,
        help="MediaPipe Selfie Multiclass로 옷/장신구를 추가 제외한다(기본 on).",
    )
    parser.add_argument(
        "--no-skin-refine", dest="skin_refine", action="store_false",
        help="피부 정제 단계를 끄고 기존 rembg 단독 마스크만 쓴다.",
    )
    parser.add_argument(
        "--skin-model", type=Path, default=masking.DEFAULT_SKIN_MODEL_PATH,
        help="MediaPipe Selfie Multiclass 모델(.tflite) 경로.",
    )
    parser.add_argument(
        "--skin-erode", type=int, default=8,
        help="피부 마스크 경계를 이만큼(px) 깎는다(기본 8px) — 옷과의 경계선 자체가 "
             "고리 모양으로 삼각측량되는 걸 막기 위한 안전 여유.",
    )
    args = parser.parse_args(argv)

    if not args.images_dir.is_dir():
        print(f"[error] 이미지 폴더가 없습니다: {args.images_dir}", file=sys.stderr)
        return 1

    stats = masking.generate_masks(
        args.images_dir, args.out,
        model=args.model, dilate=args.dilate,
        min_coverage=args.min_coverage, reject_coverage=args.reject_coverage,
        skin_refine=args.skin_refine, skin_model=args.skin_model, skin_erode=args.skin_erode,
    )

    print(f"[완료] {stats['total']}장 처리, 마스크 저장: {args.out}")
    if stats["rejected"]:
        r = stats["rejected_reasons"]
        print(
            f"[안내] {stats['rejected']}장을 후보에서 제외했습니다 "
            f"(발 미검출 {r['no_foot']}, 세그멘테이션 애매 {r['low_coverage']}, "
            f"피부 정제 붕괴 {r['skin_refine_collapsed']}): {stats['rejected_names']}"
        )
    if args.skin_refine:
        print(f"[안내] {stats['refined']}장은 피부 정제(옷/장신구 제외)가 적용되었습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
