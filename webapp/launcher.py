"""exe 배포용 진입점 -- Flask 서버를 띄우고 기본 브라우저로 마법사 화면을 연다.

app.py를 직접 실행하는 것과 차이:
  - 창을 새로 안 만들고(pywebview 등) 시스템 기본 브라우저 탭을 연다 -- Windows
    WebView2 런타임 의존성 없이 훨씬 가볍고 안정적으로 패키징된다.
  - 서버가 뜨는 걸 기다렸다가 열어야(안 그러면 "연결 거부" 화면) 헬스체크로 대기.
  - PyInstaller가 이 파일을 진입점으로 그대로 얼린다.

얼린 콘솔 exe에서 print()가 화면에 안 보이고 쌓이기만 하는 버퍼링 문제가 흔해서
(실측: 사용자가 "창만 뜨고 아무것도 안 보임" 보고) 맨 위에서 줄단위 버퍼링으로
강제 전환한다 -- 그래도 문제가 재발하면 최소한 무슨 단계에서 멈췄는지는 보여야
진단이 되므로 매 단계 진행 메시지를 flush=True로 찍는다.
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        # encoding도 같이 안 주면 콘솔 기본 코드페이지(cp949 등)로 남아 맨 처음
        # 몇 줄의 한글이 깨진다(실측 확인 -- line_buffering만 바꿨을 땐 그
        # 이후 어디선가(Flask 등) 인코딩이 다시 utf-8로 바뀌어 뒷줄만 정상).
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


def _log(msg: str) -> None:
    print(msg, flush=True)


import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import urllib.request  # noqa: E402
import webbrowser  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

HOST, PORT = "127.0.0.1", 5050
URL = f"http://{HOST}:{PORT}/"


def _wait_and_open_browser() -> None:
    _log(f"[대기] 서버가 뜰 때까지 기다리는 중... ({URL})")
    for i in range(120):  # 최대 60초 대기(모델 로딩 등으로 첫 기동이 느릴 수 있음)
        try:
            urllib.request.urlopen(URL, timeout=1)
            _log("[준비완료] 브라우저를 엽니다.")
            webbrowser.open(URL)
            return
        except Exception:
            time.sleep(0.5)
    _log(
        f"[안내] 60초가 지나도 서버 응답이 없어 자동으로 못 열었습니다 -- "
        f"이 주소를 직접 브라우저에 입력해 보세요: {URL}"
    )


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

    _log("=" * 50)
    _log("발 추출 마법사를 시작합니다...")
    _log("(피부 분류 모델 로딩 등으로 처음 한 번은 몇 초~수십 초 걸릴 수 있습니다)")
    _log("=" * 50)

    import app  # noqa: E402  (frozen 상태에서 안전하게 임포트되도록 여기서)

    threading.Thread(target=_wait_and_open_browser, daemon=True).start()
    _log(f"[서버] {URL} 에서 대기 시작...")
    app.app.run(host=HOST, port=PORT, debug=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        # --console 빌드라 창은 안 닫히지만, 원인 없이 그냥 멈춘 것처럼 보이지
        # 않게 에러를 확실히 찍고(줄단위 버퍼링이라 바로 보임) 눌러야 닫히게 한다.
        traceback.print_exc()
        input("\n오류가 발생했습니다. 위 내용을 확인해 주세요. Enter를 누르면 창이 닫힙니다...")
        raise SystemExit(1)
