"""발바닥 방향 후보를 뽑아 각각 미리보기 이미지로 저장하는 워커.

크롭(다중뷰 렌더링)이 느려서(실측 20초 안팎) 여기서 한 번 크롭한 결과를
파일로 캐싱해두고, stage1_worker.py가 사용자가 고른 방향으로 그 캐시를
재사용해 마무리한다(재크롭 안 함). stage1_worker.py처럼 매 요청마다 새
subprocess로 띄운다(다중뷰 렌더링 크래시 격리, 그 모듈 docstring 참고).

결과는 stdout에 JSON 한 줄로만 출력(로그는 stderr로).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
sys.stdout = sys.stderr

import trimesh  # noqa: E402

from foot_engine.sfm.dense import align_sole_down, keep_largest_component, sole_direction_candidates_for_mesh  # noqa: E402
from foot_engine.stl_foot_extract.postprocess_pipeline import crop_foot_mesh  # noqa: E402


def _render_preview(mesh: trimesh.Trimesh, out_path: Path) -> None:
    """방향 확인용 빠른 미리보기 한 장(발바닥이 아래로 가게 정렬된 상태에서 옆에서 봄)."""
    scene = mesh.scene()
    scene.set_camera(angles=[0, 0, 0], distance=mesh.scale * 0.9, center=mesh.centroid)
    png = scene.save_image(resolution=(360, 360))
    out_path.write_bytes(png)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output_cropped", required=True)
    p.add_argument("--job_dir", required=True)
    # 10개를 한 번에 뽑아둔다(웹앱은 처음엔 5개만 보여주고 "더 보기"로 재크롭
    # 없이 나머지를 마저 보여줌) -- 크롭(다중뷰 렌더링, ~20초)은 한 번만 하고
    # 후보 계산+썸네일 렌더만 늘어나는 거라 비용이 적다.
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--result_json", required=True)
    args = p.parse_args()

    cropped_mesh, _ = crop_foot_mesh(args.input)
    out_cropped = Path(args.output_cropped)
    cropped_mesh.export(out_cropped)

    job_dir = Path(args.job_dir)
    # sole_direction_candidates_for_mesh()는 내부적으로 keep_largest_component()를
    # 거친 뒤 후보를 뽑는다 -- 미리보기 정렬도 같은 전처리를 거쳐야 좌표계가 맞는다.
    largest, _, _ = keep_largest_component(cropped_mesh)
    candidates = sole_direction_candidates_for_mesh(cropped_mesh, k=args.k)

    info = {"cropped_file": out_cropped.name, "candidates": []}
    for i, cand in enumerate(candidates):
        aligned = align_sole_down(largest, down_direction=cand.direction)
        thumb_name = f"1a_cand{i}.png"
        _render_preview(aligned, job_dir / thumb_name)
        info["candidates"].append({
            "index": i,
            "score": cand.score,
            "direction": cand.direction.tolist(),
            "thumbnail": thumb_name,
        })

    Path(args.result_json).write_text(json.dumps(info), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
