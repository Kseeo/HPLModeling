"""SfM sparse 포인트클라우드에서 계측치를 뽑아 `FootMeshDeformer`로 3D 모델을 만드는 CLI.

실제 로직은 `foot_engine.sfm.fitting`에 있다 — 이 파일은 인자 파싱과 출력
포맷만 담당하는 얇은 CLI 래퍼다.

사용 예::

    python scripts/fit_deformer_to_pointcloud.py `
        data/output/sfm_prototype/test02_run/sparse_points.ply `
        --out data/output/test02_deformer_fit.stl
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

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from foot_engine.deformer import FootMeshDeformer  # noqa: E402
from foot_engine.sfm.fitting import default_template_path, fit_point_cloud_to_template  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SfM 점군 → 계측치 → deformer.py 3D 모델")
    parser.add_argument("points", type=Path, help="sparse_sfm_prototype.py/clean_point_cloud.py가 만든 .ply")
    parser.add_argument("--out", type=Path, required=True, help="변형된 메쉬 저장 경로(.stl)")
    parser.add_argument("--template", type=Path, default=None, help="기준 템플릿 STL(생략 시 절차적 생성)")
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default=None,
        help=(
            "입력 점군(영상)이 실제로 어느 쪽 발인지. --template-side와 다르면 "
            "ICP 정렬 전에 점군을 미러링한다. 생략하면 좌우 확인을 건너뛴다 "
            "(주의: 좌우가 다른데 생략하면 발이 뒤틀린 채로 정렬됨 — 2026-08-07 "
            "test00+S0001 조합에서 실측 확인된 실패 모드)."
        ),
    )
    parser.add_argument(
        "--template-side",
        choices=["left", "right"],
        default="right",
        help="템플릿의 실제 발 좌우(기본 right — 절차적 템플릿 기본값과 동일).",
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args(argv)

    template_path = args.template or default_template_path(ROOT)
    engine = FootMeshDeformer(template_path)

    cloud = trimesh.load(args.points)
    raw_points = np.asarray(cloud.vertices)
    print(f"[입력] 점군 {len(raw_points):,}개")

    deformed, measured = fit_point_cloud_to_template(
        raw_points, engine, side=args.side, template_side=args.template_side, rng_seed=args.rng_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    deformed.export(args.out)

    report = engine.last_report
    assert report is not None
    print("\n" + "\n".join(report.summary_lines()))
    print(f"\n저장: {args.out}")
    for warning in report.warnings:
        print(f"[warn] {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
