"""exe 배포용 진입점 -- Flask 서버를 띄우고 기본 브라우저로 마법사 화면을 연다.

app.py를 직접 실행하는 것과 차이:
  - 창을 새로 안 만들고(pywebview 등) 시스템 기본 브라우저 탭을 연다 -- Windows
    WebView2 런타임 의존성 없이 훨씬 가볍고 안정적으로 패키징된다.
  - 서버가 뜨는 걸 기다렸다가 열어야(안 그러면 "연결 거부" 화면) 헬스체크로 대기.
  - PyInstaller가 이 파일을 진입점으로 그대로 얼린다(build_exe.py 참고).
"""
from __future__ import annotations

import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

HOST, PORT = "127.0.0.1", 5050
URL = f"http://{HOST}:{PORT}/"


def _wait_and_open_browser() -> None:
    for _ in range(120):  # 최대 60초 대기(모델 로딩 등으로 첫 기동이 느릴 수 있음)
        try:
            urllib.request.urlopen(URL, timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    webbrowser.open(URL)


def main() -> int:
    # 얼린(frozen) exe엔 별도 python.exe가 없어 app.py가 원래 하던
    # `subprocess.run([sys.executable, "stage1_worker.py", ...])` 방식이 안
    # 통한다(그 자체가 exe라 스크립트를 인자로 못 받음) -- 대신 이 exe를
    # 특수 플래그로 재귀 호출해 워커로 동작하게 분기한다(app.py의 프로즌
    # 분기 참고). pyglet 렌더링 크래시를 별도 프로세스로 격리하는 목적은
    # 그대로 유지된다.
    if len(sys.argv) > 1 and sys.argv[1] == "--stage1-worker":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        import stage1_worker
        return stage1_worker.main()
    if len(sys.argv) > 1 and sys.argv[1] == "--stage1-orientation-worker":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        import stage1_orientation_worker
        return stage1_orientation_worker.main()

    import app  # noqa: E402  (frozen 상태에서 안전하게 임포트되도록 여기서)

    threading.Thread(target=_wait_and_open_browser, daemon=True).start()
    app.app.run(host=HOST, port=PORT, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
