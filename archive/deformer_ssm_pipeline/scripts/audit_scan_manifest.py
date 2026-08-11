"""스캔 매니페스트(CSV) 검증 + 감사 리포트 CLI.

`foot_engine.scan_dataset.load_manifest()`로 매니페스트를 읽어 스키마를 검증하고,
카테고리/좌우 분포를 출력한다. SSM 학습을 시작하기 전에 데이터가 준비됐는지
확인하는 첫 관문이다.

사용 예::

    python scripts/audit_scan_manifest.py data/scans/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

from foot_engine import ScanDatasetError, category_counts, load_manifest, side_counts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="스캔 매니페스트 검증 + 감사 리포트")
    parser.add_argument("manifest", type=Path, help="매니페스트 CSV 경로")
    parser.add_argument(
        "--stl-root", type=Path, default=None,
        help="stl_path 가 상대경로일 때 기준 폴더 (생략하면 매니페스트 위치 기준)",
    )
    parser.add_argument(
        "--min-category-count", type=int, default=20,
        help="이 개수 미만인 카테고리는 경고 표시 (기본 20 — SSM 학습에 부족할 수 있는 기준)",
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="아직 값이 안 채워졌거나 STL이 없는 행은 에러 대신 건너뛴다 "
             "(데이터를 아직 정리하는 중일 때 사용)",
    )
    args = parser.parse_args(argv)

    try:
        records, skip_warnings = load_manifest(
            args.manifest, stl_root=args.stl_root, skip_incomplete=args.allow_incomplete
        )
    except ScanDatasetError as exc:
        print(f"[error] {exc.message}", file=sys.stderr)
        if exc.detail:
            print(f"        detail: {exc.detail}", file=sys.stderr)
        return 1

    print(f"[ok] {len(records)}건 로드 + 검증 통과")
    if skip_warnings:
        print(f"[skip] {len(skip_warnings)}건은 미완성이라 건너뜀 (--allow-incomplete)")
    print()

    print("[카테고리별 개수]")
    category_warnings: list[str] = []
    for category, count in sorted(category_counts(records).items()):
        flag = ""
        if count == 0:
            flag = "  <- 데이터 없음"
        elif count < args.min_category_count:
            flag = f"  <- {args.min_category_count}건 미만, SSM 학습에 부족할 수 있음"
            category_warnings.append(f"'{category}' 카테고리가 {count}건뿐입니다.")
        print(f"  {category:<20}{count:>5}{flag}")

    print("\n[좌/우 개수]")
    for side, count in sorted(side_counts(records).items()):
        print(f"  {side:<20}{count:>5}")

    subject_ids = {r.subject_id for r in records}
    missing_length = sum(1 for r in records if r.self_reported_length_mm is None)
    print(f"\n[피험자 수] {len(subject_ids)}명 (스캔 {len(records)}건)")
    if missing_length:
        print(f"[참고] 자기신고 길이가 없는 스캔 {missing_length}건 "
              f"(형상 학습엔 지장 없음, 최종 크기 기준자로만 못 씀)")

    if category_warnings:
        print("\n[경고]")
        for w in category_warnings:
            print(f"  - {w}")

    if skip_warnings:
        shown = skip_warnings[:10]
        print(f"\n[건너뛴 행 {len(skip_warnings)}건 중 {len(shown)}건 예시]")
        for w in shown:
            print(f"  - {w}")
        if len(skip_warnings) > len(shown):
            print(f"  ... 외 {len(skip_warnings) - len(shown)}건 더")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
