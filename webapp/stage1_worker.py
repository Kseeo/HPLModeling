"""1단계(크롭+정렬+발목절단)를 별도 프로세스로 격리해서 실행하는 워커.

GLB 다중뷰 렌더링(pyglet)이 서버 프로세스 안에서 여러 번 반복 호출되면
크래시(트레이스백도 안 남기고 프로세스가 죽음)하는 걸 실측으로 확인했다 --
그래서 app.py가 이 스크립트를 매 요청마다 새 subprocess로 띄운다. 여기서
죽어도 subprocess만 죽고 Flask 서버 본체는 살아있다.

결과는 stdout에 JSON 한 줄로만 출력(로그는 stderr로).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# cp949 등 비-UTF8 콘솔에서 한글/em-dash 출력이 깨지거나 죽는 문제 방지.
# 로그는 stderr로만 보내고(app.py가 로그로 표시), stdout은 JSON 결과 전용으로 남긴다.
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
sys.stdout = sys.stderr  # process_glb_to_foot()의 print()들이 stdout JSON을 오염시키지 않도록

from foot_engine.stl_foot_extract.postprocess_pipeline import export_result, process_glb_to_foot  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target_vertices", type=int, default=18119)
    p.add_argument("--trim_leg", type=int, default=1)
    p.add_argument("--reference_length_mm", type=float, default=None)
    p.add_argument("--two_pass", type=int, default=0)
    p.add_argument("--result_json", required=True)
    args = p.parse_args()

    result = process_glb_to_foot(
        args.input,
        postprocess=False,
        align=True,
        trim_leg=bool(args.trim_leg),
        target_vertices=args.target_vertices,
        reference_length_mm=args.reference_length_mm,
        floor_contact_tolerance_mm=2.0,
        z_up=False,
        two_pass=bool(args.two_pass),
    )
    out_path = Path(args.output)
    export_result(result, out_path)

    info = {
        "file": out_path.name,
        "n_vertices": len(result.mesh.vertices),
        "n_faces": len(result.mesh.faces),
    }
    Path(args.result_json).write_text(json.dumps(info), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
