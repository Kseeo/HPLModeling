"""scripts/*.py 공용 부트스트랩.

import되는 순간(부작용) 콘솔을 UTF-8로 고정하고 `src/`를 import 경로에 추가한다.
각 스크립트 맨 위, 다른 `foot_engine` import보다 먼저 `import _cli_common`으로 쓴다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

# cp949 등 비-UTF8 콘솔에서 한글 출력이 깨지거나 죽는 문제 방지.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
