"""STL 두 개(예: raw vs smoothing 후)를 비교하는 진단 CLI.

파일 크기 차이가 실제 형상 차이 때문인지, 단순 포맷(ASCII/바이너리) 차이 때문인지
구분하기 위해 정점/면 개수, watertight 여부, 부피, bounding box를 나란히 보여준다.

사용 예::

    python scripts/compare_stl_pair.py raw.stl smoothed.stl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh


def _is_ascii_stl(path: Path) -> bool:
    """STL이 ASCII 포맷인지 확인(바이너리는 헤더 80바이트 뒤 삼각형 개수가 온다)."""
    with path.open("rb") as f:
        head = f.read(5)
    return head == b"solid"


def _summarize(path: Path) -> dict:
    mesh = trimesh.load(path, force="mesh")
    return {
        "path": path,
        "file_size_kb": path.stat().st_size / 1024,
        "format": "ASCII" if _is_ascii_stl(path) else "binary",
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": mesh.is_watertight,
        "volume_mm3": mesh.volume if mesh.is_watertight else None,
        "bbox_size": (mesh.bounds[1] - mesh.bounds[0]).round(2).tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STL 두 개 비교 (raw vs smoothing 등)")
    parser.add_argument("file_a", type=Path, help="예: raw.stl")
    parser.add_argument("file_b", type=Path, help="예: smoothed.stl")
    args = parser.parse_args(argv)

    a = _summarize(args.file_a)
    b = _summarize(args.file_b)

    label_a, label_b = "A", "B"
    print(f"{'항목':<16}{label_a:>20}{label_b:>20}")
    print("-" * 56)
    print(f"{'경로':<16}{a['path'].name:>20}{b['path'].name:>20}")
    print(f"{'파일 크기(KB)':<16}{a['file_size_kb']:>20.1f}{b['file_size_kb']:>20.1f}")
    print(f"{'포맷':<16}{a['format']:>20}{b['format']:>20}")
    print(f"{'정점 수':<16}{a['vertices']:>20,}{b['vertices']:>20,}")
    print(f"{'면 수':<16}{a['faces']:>20,}{b['faces']:>20,}")
    print(f"{'watertight':<16}{str(a['watertight']):>20}{str(b['watertight']):>20}")
    vol_a = f"{a['volume_mm3']:.0f}" if a["volume_mm3"] else "-"
    vol_b = f"{b['volume_mm3']:.0f}" if b["volume_mm3"] else "-"
    print(f"{'부피(mm3)':<16}{vol_a:>20}{vol_b:>20}")
    print(f"{'bbox 크기':<16}{str(a['bbox_size']):>20}{str(b['bbox_size']):>20}")

    face_ratio = b["faces"] / a["faces"] if a["faces"] else float("inf")
    print(f"\n[해석 힌트] 면 개수 비율(B/A) = {face_ratio:.2f}배")
    if a["format"] != b["format"]:
        print("  -> 포맷이 다릅니다(ASCII/바이너리) — 파일 크기 차이 상당 부분은 이걸로 설명될 수 있습니다.")
    if face_ratio > 3:
        print("  -> 면 개수가 3배 넘게 늘었습니다 — 형상이 실제로 바뀌었는지, "
              "단순 세분화로 불필요하게 부풀려진 건 아닌지 확인해보세요.")
    if a["volume_mm3"] and b["volume_mm3"]:
        vol_diff_pct = abs(b["volume_mm3"] - a["volume_mm3"]) / a["volume_mm3"] * 100
        print(f"  -> 부피 차이: {vol_diff_pct:.1f}% (형상이 실제로 달라졌는지의 가장 직접적인 지표)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
