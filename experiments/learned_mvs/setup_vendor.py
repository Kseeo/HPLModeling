"""PatchmatchNet 원본 저장소를 고정 커밋으로 `_vendor/PatchmatchNet`에 clone한다.

1회 실행용. `_vendor/`는 .gitignore 처리되어 있으므로 이 저장소를 지우면(또는 이
`experiments/learned_mvs/` 폴더 전체를 지우면) 흔적 없이 원상복구된다.

원본: https://github.com/FangjinhuaWang/PatchmatchNet (MIT 라이선스,
사전학습 체크포인트 `checkpoints/params_000007.ckpt` 내장 -- 별도 다운로드 불필요).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/FangjinhuaWang/PatchmatchNet.git"
# 2026-08-25 확인 시점 최신 커밋에 고정 -- API가 바뀌어도 run_prototype.py가
# 갑자기 깨지지 않도록.
PINNED_COMMIT = "8dc6cb40bdb7053e856598b378425c76a9dcf5e0"

VENDOR_DIR = Path(__file__).parent / "_vendor" / "PatchmatchNet"


def main() -> int:
    if VENDOR_DIR.exists():
        print(f"이미 존재: {VENDOR_DIR} -- 다시 받으려면 폴더를 지우고 재실행하세요.")
        return 0

    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO_URL, str(VENDOR_DIR)], check=True)
    subprocess.run(["git", "-C", str(VENDOR_DIR), "checkout", PINNED_COMMIT], check=True)

    ckpt = VENDOR_DIR / "checkpoints" / "params_000007.ckpt"
    if not ckpt.exists():
        print(f"[경고] 사전학습 체크포인트가 없습니다: {ckpt}", file=sys.stderr)
        return 1

    print(f"완료: {VENDOR_DIR} (체크포인트 확인됨: {ckpt})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
