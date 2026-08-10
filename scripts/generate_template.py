"""기준 발 템플릿(base_foot_template.stl) 생성 CLI.

사용 예::

    python scripts/generate_template.py                      # 250mm 오른발
    python scripts/generate_template.py --length 265 --side left
    python scripts/generate_template.py --out data/templates/foot_L280.stl --length 280
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

from foot_engine import mesh_utils as mu  # noqa: E402
from foot_engine.template_factory import save_reference_template  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="절차적 기준 발 템플릿 생성")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data" / "templates" / "base_foot_template.stl",
                        help="출력 STL 경로")
    parser.add_argument("--length", type=float, default=250.0, help="발 길이(mm)")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--arch", type=float, default=24.0,
                        help="아치 형상 파라미터(mm). 계측되는 아치 높이는 약 75%%")
    parser.add_argument("--stations", type=int, default=120, help="길이방향 단면 수")
    parser.add_argument("--ring", type=int, default=64, help="단면당 정점 수")
    args = parser.parse_args(argv)

    path, mesh = save_reference_template(
        args.out,
        length_mm=args.length,
        side=args.side,
        arch_shape_mm=args.arch,
        n_stations=args.stations,
        n_ring=args.ring,
    )

    measurements = mu.measure_foot(mesh)
    print(f"저장: {path}")
    print(f"정점 {len(mesh.vertices):,} / 면 {len(mesh.faces):,}")
    print(f"watertight: {mesh.is_watertight} | 체적: {mesh.volume:,.0f} mm³")
    print("계측치(mm):")
    for name, value in measurements.to_dict().items():
        print(f"  {name:<20}{value:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
