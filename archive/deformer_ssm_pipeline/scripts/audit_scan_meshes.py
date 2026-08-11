"""매니페스트에 실린 STL 파일들의 지오메트리 품질을 감사하는 CLI.

`audit_scan_manifest.py`가 메타데이터(카테고리/좌우/사이즈)를 검증한다면, 이건
그 뒤에 이어지는 단계로 **파일 내용물 자체**를 점검한다: 정점/면 개수, watertight
여부, 열린 경계(구멍) 크기, 스파이크성 노이즈, bounding box 크기. 전처리·정합에
들어가기 전에 문제 있는 스캔을 미리 걸러내거나, 유독 무거운 스캔이 있어서
정합이 느려지는 원인을 찾는 데 쓴다.

사용 예::

    python scripts/audit_scan_meshes.py data/samples/scan_manifest_template.csv --stl-root data --allow-incomplete
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from foot_engine import ScanDatasetError, load_manifest  # noqa: E402
from foot_engine.mesh_utils import _open_edge_count  # noqa: E402


@dataclass(slots=True)
class MeshAuditRow:
    scan_id: str
    category: str
    vertices: int
    faces: int
    watertight: bool
    open_edges: int
    bbox_mm: tuple[float, float, float]
    spike_count: int
    file_size_kb: float


def _spike_count(mesh: trimesh.Trimesh, *, ratio_threshold: float = 8.0) -> int:
    """중앙값 대비 유별나게 긴 edge 개수 — 스파이크성 노이즈의 대용 지표.

    스캐너 노이즈로 뾰족하게 튀어나온 점은 주변 정점과의 거리(edge 길이)가
    비정상적으로 길다는 성질을 이용한다.
    """
    lengths = mesh.edges_unique_length
    if len(lengths) == 0:
        return 0
    median = np.median(lengths)
    if median <= 0:
        return 0
    return int((lengths > median * ratio_threshold).sum())


def audit_one(scan_id: str, category: str, stl_path: Path) -> MeshAuditRow:
    mesh = trimesh.load(stl_path, force="mesh", process=False)
    mesh.merge_vertices()
    bbox = mesh.bounds[1] - mesh.bounds[0]
    return MeshAuditRow(
        scan_id=scan_id,
        category=category,
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        watertight=bool(mesh.is_watertight),
        open_edges=_open_edge_count(mesh),
        bbox_mm=tuple(round(float(x), 1) for x in bbox),
        spike_count=_spike_count(mesh),
        file_size_kb=stl_path.stat().st_size / 1024,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="스캔 STL 지오메트리 품질 감사")
    parser.add_argument("manifest", type=Path, help="매니페스트 CSV 경로")
    parser.add_argument("--stl-root", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--max-spike", type=int, default=5,
        help="이보다 스파이크가 많으면 노이즈 의심으로 표시(기본 5)",
    )
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 N건만 감사(시험용)")
    args = parser.parse_args(argv)

    try:
        records, skip_warnings = load_manifest(
            args.manifest, stl_root=args.stl_root, skip_incomplete=args.allow_incomplete
        )
    except ScanDatasetError as exc:
        print(f"[error] {exc.message}", file=sys.stderr)
        return 1

    if skip_warnings:
        print(f"[manifest] {len(skip_warnings)}건 건너뜀(미완성)")

    if args.limit:
        records = records[: args.limit]

    print(f"[감사 시작] {len(records)}건\n")
    rows: list[MeshAuditRow] = []
    failures: list[str] = []
    for record in records:
        try:
            row = audit_one(record.scan_id, record.category, record.stl_path)
        except Exception as exc:  # noqa: BLE001 — 파일 하나 실패로 전체가 멈추면 안 됨
            failures.append(f"{record.scan_id}: 로딩 실패 ({exc})")
            continue
        rows.append(row)

    if not rows:
        print("[error] 감사할 수 있는 스캔이 없습니다.", file=sys.stderr)
        return 1

    vcounts = np.array([r.vertices for r in rows])
    print("[정점 수 분포]")
    print(f"  최소 {vcounts.min():,} / 중앙값 {int(np.median(vcounts)):,} / "
          f"평균 {vcounts.mean():,.0f} / 최대 {vcounts.max():,}")

    n_not_watertight = sum(1 for r in rows if not r.watertight)
    print(f"\n[watertight] {len(rows) - n_not_watertight}/{len(rows)}건 watertight "
          f"(스캔이 발목에서 열려 있는 설계라 대부분 False인 게 정상)")

    heavy_threshold = float(np.median(vcounts)) * 3
    heavy = [r for r in rows if r.vertices > heavy_threshold]
    if heavy:
        print(f"\n[정점 수 이상치] 중앙값의 3배({heavy_threshold:,.0f}) 초과 {len(heavy)}건 "
              f"— 정합이 유독 느렸다면 이 스캔들일 가능성이 높습니다:")
        for r in sorted(heavy, key=lambda r: -r.vertices)[:10]:
            print(f"  {r.scan_id} [{r.category}]: 정점 {r.vertices:,}개")

    noisy = [r for r in rows if r.spike_count > args.max_spike]
    print(f"\n[노이즈 의심] 스파이크 {args.max_spike}개 초과: {len(noisy)}건")
    for r in sorted(noisy, key=lambda r: -r.spike_count)[:10]:
        print(f"  {r.scan_id} [{r.category}]: 스파이크 {r.spike_count}개")

    bboxes = np.array([r.bbox_mm for r in rows])
    print(f"\n[bbox 크기(mm) 분포] X: {bboxes[:,0].min():.0f}~{bboxes[:,0].max():.0f}, "
          f"Y: {bboxes[:,1].min():.0f}~{bboxes[:,1].max():.0f}, "
          f"Z: {bboxes[:,2].min():.0f}~{bboxes[:,2].max():.0f}")
    tiny_or_huge = [
        r for r in rows if not (50 < r.bbox_mm[0] < 500 and 20 < r.bbox_mm[1] < 300)
    ]
    if tiny_or_huge:
        print(f"\n[크기 이상치] 발 치수로 보기 힘든 bbox {len(tiny_or_huge)}건 "
              f"(단위 오류나 손상된 파일일 수 있음):")
        for r in tiny_or_huge[:10]:
            print(f"  {r.scan_id}: bbox={r.bbox_mm}")

    if failures:
        print(f"\n[로딩 실패] {len(failures)}건:")
        for msg in failures:
            print(f"  - {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
