"""`foot_engine.stl_foot_extract` CLI -- `scripts/`의 공용 유틸(`_cli_common` 등)에
의존하지 않는 독립 진입점.

사용 예::

    # 씨앗점 피커 열기(자동 후보 힌트 포함)
    python -m foot_engine.stl_foot_extract.cli pick project_229.stl

    # 사람이 피커에서 확인한 좌표로 크롭 + 정리
    python -m foot_engine.stl_foot_extract.cli extract project_229.stl \\
        --seed-point 0.01,0.08,0.09 --crop-radius 0.03 --out project_229_clean.stl

    # 자동 후보 1위로 바로 크롭(휴리스틱, 결과 확인 필수) + 정리
    python -m foot_engine.stl_foot_extract.cli extract project_229.stl --auto --out project_229_clean.stl

    # 후보만 출력(크롭 없이)
    python -m foot_engine.stl_foot_extract.cli suggest project_229.stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trimesh

from .locate import find_dense_regions
from .picker import open_picker
from .pipeline import extract_foot

# cp949 등 비-UTF8 콘솔에서 한글 출력이 깨지거나 죽는 문제 방지 -- scripts/_cli_common.py와
# 같은 부작용이지만, 이 패키지는 scripts/에 의존하지 않아야 해서 여기 자체적으로 둔다.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def _parse_point(text: str) -> tuple[float, float, float]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"'x,y,z' 형식이어야 합니다: {text!r}")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"숫자 파싱 실패: {text!r} ({e})") from e


def _add_common_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-postprocess", dest="postprocess", action="store_false",
                         help="배경 파편 정리 + 스무딩(finishing.postprocess_mesh)을 끈다. 기본 켜짐.")
    parser.add_argument("--sand-iterations", type=int, default=3, help="사포질 반복 횟수(기본 3).")
    parser.add_argument("--curvature-iterations", type=int, default=150, help="고곡률 스무딩 반복 횟수(기본 150).")
    parser.add_argument("--finish-smooth-iterations", type=int, default=40, help="마감 라플라시안 반복 횟수(기본 40).")


def cmd_suggest(args: argparse.Namespace) -> int:
    mesh = trimesh.load(args.mesh, process=True)
    regions = find_dense_regions(mesh, top_k=args.top_k)
    if not regions:
        print("구역을 하나도 못 찾았습니다.")
        return 1
    for i, r in enumerate(regions):
        print(
            f"#{i+1} score={r.score:.3f} (구형성={r.sphericity_score:.2f} 다지구조={r.toe_score:.2f}) "
            f"점개수={r.n_points} 중심={r.centroid.round(6).tolist()} "
            f"권장여유={r.density_radius * 2.0:.5g}(판정반경*2.0)"
        )
    return 0


def cmd_pick(args: argparse.Namespace) -> int:
    open_picker(args.mesh, n_regions=args.top_k, auto_open=not args.no_open)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    mesh = trimesh.load(args.mesh, process=True)
    print(f"[extract] 입력: {args.mesh} (정점 {len(mesh.vertices):,}개, 면 {len(mesh.faces):,}개)")

    seed_point = _parse_point(args.seed_point) if args.seed_point else None
    result = extract_foot(
        mesh,
        seed_point=seed_point,
        crop_radius=args.crop_radius,
        auto=args.auto,
        region_index=args.region_index,
        postprocess=args.postprocess,
        postprocess_kwargs=dict(
            sand_iterations=args.sand_iterations,
            curvature_iterations=args.curvature_iterations,
            finish_smooth_iterations=args.finish_smooth_iterations,
        ),
    )
    out_path = args.out or args.mesh.with_name(f"{args.mesh.stem}_extracted{args.mesh.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(out_path)
    print(
        f"[extract] 저장: {out_path} (정점 {len(result.mesh.vertices):,}개, "
        f"면 {len(result.mesh.faces):,}개)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m foot_engine.stl_foot_extract.cli",
        description="외부 STL(스캔 품질은 좋으나 배경/다른 물체가 같이 찍힌 경우)에서 발 부위 추출",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_suggest = sub.add_parser("suggest", help="발 구역 후보를 점수와 함께 출력(크롭 없음)")
    p_suggest.add_argument("mesh", type=Path)
    p_suggest.add_argument("--top-k", type=int, default=5)
    p_suggest.set_defaults(func=cmd_suggest)

    p_pick = sub.add_parser("pick", help="씨앗점 피커(로컬 HTML, 자동 구역 힌트 포함)를 연다")
    p_pick.add_argument("mesh", type=Path)
    p_pick.add_argument("--top-k", type=int, default=5, help="같이 표시할 자동 구역 개수(기본 5, 0=끔)")
    p_pick.add_argument("--no-open", action="store_true", help="생성 후 브라우저로 자동으로 열지 않는다")
    p_pick.set_defaults(func=cmd_pick)

    p_extract = sub.add_parser("extract", help="발 부위를 크롭 + 정리해 저장")
    p_extract.add_argument("mesh", type=Path)
    p_extract.add_argument("--out", type=Path, default=None)
    p_extract.add_argument("--seed-point", type=str, default=None, help="'x,y,z' -- pick으로 확인한 좌표")
    p_extract.add_argument("--crop-radius", type=float, default=None, help="--seed-point 크롭 반지름")
    p_extract.add_argument("--auto", action="store_true",
                            help="자동 구역으로 바로 크롭(휴리스틱, 결과 확인 필수) -- --seed-point 대안")
    p_extract.add_argument("--region-index", type=int, default=0, help="--auto일 때 몇 번째 구역을 쓸지(기본 0=1위)")
    _add_common_postprocess_args(p_extract)
    p_extract.set_defaults(func=cmd_extract)

    args = parser.parse_args(argv)
    if args.command == "extract" and not args.auto and args.seed_point is None:
        parser.error("extract는 --seed-point(+--crop-radius) 또는 --auto 중 하나가 필요합니다")
    if args.command == "extract" and (args.seed_point is None) != (args.crop_radius is None):
        parser.error("--seed-point와 --crop-radius는 같이 지정해야 합니다")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
