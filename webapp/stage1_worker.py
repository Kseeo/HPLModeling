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

import trimesh  # noqa: E402

from foot_engine.stl_foot_extract.postprocess_pipeline import (  # noqa: E402
    crop_foot_mesh,
    export_result,
    finish_foot_mesh,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--trim_leg", type=int, default=1)
    p.add_argument("--reference_length_mm", type=float, default=None)
    p.add_argument("--two_pass", type=int, default=0)
    p.add_argument("--prune_neck_fragments", type=int, default=1)
    p.add_argument("--recover_holes", type=int, default=0)
    p.add_argument("--reject_color_outliers", type=int, default=0)
    # 발목 절단 폭곡선 반등 탐지가 노이즈 많은 스캔에서 엉뚱한 높이를 "다리"로
    # 오판해 뒤꿈치까지 잘려나가는 사례 확인(project_5) -- 켜면 반등 탐지를
    # 사실상 끄고 고정 비율(max_length_ratio) 절단만 쓴다. 기본 꺼짐(정상
    # 스캔은 반등 탐지가 더 정확).
    p.add_argument("--trim_leg_no_rebound", type=int, default=0)
    # 방향 후보 픽커(stage1_orientation_worker.py)가 이미 크롭해둔 결과가 있으면
    # 그걸 재사용하고(다중뷰 렌더링 재실행 안 함), 사용자가 고른 방향으로 정렬한다.
    p.add_argument("--cropped_input", default=None)
    p.add_argument("--down_direction", default=None, help="'x,y,z' 콤마 구분")
    p.add_argument("--result_json", required=True)
    args = p.parse_args()

    down_direction = None
    if args.down_direction:
        down_direction = [float(v) for v in args.down_direction.split(",")]

    if args.cropped_input:
        cropped_mesh = trimesh.load(args.cropped_input, force="mesh", process=False)
        n_input = len(cropped_mesh.vertices)
    else:
        cropped_mesh, n_input = crop_foot_mesh(
            args.input, two_pass=bool(args.two_pass),
            recover_holes=bool(args.recover_holes),
            reject_color_outliers=bool(args.reject_color_outliers),
        )

    result = finish_foot_mesh(
        cropped_mesh,
        n_input_vertices=n_input,
        postprocess=False,
        align=True,
        trim_leg=bool(args.trim_leg),
        # target_vertices는 넘기지 않는다(축약 안 함) -- 해상도 맞춤은 3단계
        # (build_dataset.py --target_faces)에서 한다. 1단계에서 미리 축약해야
        # 할 이유가 없고, 오히려 1·2단계 산출물의 디테일만 깎였다(2026-09-01).
        reference_length_mm=args.reference_length_mm,
        floor_contact_tolerance_mm=2.0,
        z_up=False,
        down_direction=down_direction,
        trim_leg_kwargs={"rebound_ratio": 999.0} if args.trim_leg_no_rebound else None,
        prune_neck_fragments=bool(args.prune_neck_fragments),
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
