"""전체 파이프라인 데모 CLI.

    2D 랜드마크 JSON  →  템플릿 로드  →  변형  →  품질 검사  →  STL/GLB 저장

사용 예::

    python scripts/run_deform_demo.py                                  # 4뷰 샘플
    python scripts/run_deform_demo.py --landmarks data/samples/landmarks_2views.json
    python scripts/run_deform_demo.py --out data/output/foot.glb --json-report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

from foot_engine import FootEngineError, FootMeshDeformer  # noqa: E402
from foot_engine.template_factory import save_reference_template  # noqa: E402

DEFAULT_TEMPLATE = ROOT / "data" / "templates" / "base_foot_template.stl"
DEFAULT_LANDMARKS = ROOT / "data" / "samples" / "landmarks_4views.json"
DEFAULT_OUTPUT = ROOT / "data" / "output" / "output_deformed_foot.stl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="발 메쉬 파라메트릭 변형 데모")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help=".stl / .glb / .ply / .obj")
    parser.add_argument("--template-length", type=float, default=250.0,
                        help="템플릿이 없을 때 자동 생성할 발 길이(mm)")
    parser.add_argument("--strict", action="store_true",
                        help="품질 기준 미달 시 예외를 던진다")
    parser.add_argument("--json-report", action="store_true",
                        help="리포트를 JSON 으로 출력")
    args = parser.parse_args(argv)

    # --- 0) 템플릿 준비 ---------------------------------------------------
    if not args.template.is_file():
        print(f"[setup] 템플릿이 없어 기준 템플릿을 생성합니다: {args.template}")
        save_reference_template(args.template, length_mm=args.template_length)

    # --- 1) 랜드마크 로드 -------------------------------------------------
    if not args.landmarks.is_file():
        print(f"[error] 랜드마크 파일을 찾을 수 없습니다: {args.landmarks}", file=sys.stderr)
        return 2
    landmarks_data = json.loads(args.landmarks.read_text(encoding="utf-8"))

    # --- 2) 변형 ----------------------------------------------------------
    from foot_engine.config import DeformConfig

    conf = DeformConfig(strict_quality=args.strict)
    try:
        deformer = FootMeshDeformer(args.template, conf=conf)
        print(f"[template] {args.template.name}  "
              f"정점 {len(deformer.vertices):,} / 면 {len(deformer.faces):,}  "
              f"내측={deformer.frame.medial_side}  제어점 {len(deformer.control_points)}개")
        for note in deformer.setup_notes:
            print(f"[template] {note}")

        deformer.deform_mesh(landmarks_data)
        saved = deformer.export_mesh(args.out)
    except FootEngineError as exc:
        print(f"[error] {type(exc).__name__}: {exc.message}", file=sys.stderr)
        if exc.detail:
            print(f"        detail: {exc.detail}", file=sys.stderr)
        return 1

    report = deformer.last_report
    assert report is not None

    # --- 3) 리포트 --------------------------------------------------------
    if args.json_report:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"\n[입력] 이미지 {report.image_count}장 / {args.landmarks.name}")
    print("\n".join(report.summary_lines()))
    print(f"\n[변형] 최대 변위 {report.max_displacement_mm:.2f}mm, "
          f"아치 보정 {report.arch_iterations_used}회")
    q = report.quality
    print(f"[품질] watertight={q.is_watertight}  winding={q.is_winding_consistent}  "
          f"체적={q.volume_mm3:,.0f}mm³  뒤집힌 면={q.flipped_face_ratio:.3%}")
    print(f"[저장] {saved}")
    for warning in report.warnings:
        print(f"[warn] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
