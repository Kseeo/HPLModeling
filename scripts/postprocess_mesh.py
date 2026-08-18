"""완성된 메쉬(STL 등)에 배경/파편 제거 + 스무딩만 독립적으로 적용한다.

`foot_engine.sfm.mesh_postprocess.postprocess_mesh()`의 얇은 CLI 래퍼 --
사진/카메라/마스크가 전혀 필요 없다(이미 갖고 있는 STL 파일에 바로 쓸 수
있음). 사진 -> 3D 메쉬 생성(`run_sfm_pipeline.py`/`run_dense_pipeline.py`)과는
완전히 분리된 별도 단계다.

사용 예::

    # 파일 하나
    python scripts/postprocess_mesh.py data/output/test03_fit.stl --out data/output/test03_clean.stl

    # 폴더 전체 일괄 처리(각 파일 옆에 <stem>_clean.stl로 저장)
    python scripts/postprocess_mesh.py data/output --glob "*.stl"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

import _cli_common  # noqa: F401  -- sys.path 설정 + 콘솔 UTF-8 고정(부작용 import)

from foot_engine.sfm.mesh_postprocess import (  # noqa: E402
    DEFAULT_CURVATURE_ALPHA,
    DEFAULT_CURVATURE_ITERATIONS,
    DEFAULT_CURVATURE_MAX_RADIUS_MULT,
    DEFAULT_CURVATURE_MIN_RADIUS_MULT,
    DEFAULT_CURVATURE_MU,
    DEFAULT_CURVATURE_PERCENTILE,
    postprocess_mesh,
)


def add_postprocess_args(parser: argparse.ArgumentParser) -> None:
    """`postprocess_mesh()` 튜닝 인자 전부를 `parser`에 추가한다."""
    parser.add_argument(
        "--no-keep-largest", dest="keep_largest", action="store_false",
        help="가장 큰 연결 요소만 남기는 부유 파편 제거를 끈다. 기본 켜짐.",
    )
    parser.add_argument(
        "--prune-protrusions", action="store_true",
        help="몸통에 이어붙은 뿔/스파이크를 국소 밀도 기준으로 사후 제거한다"
             "(prune_thin_protrusions). 메쉬 위상을 망가뜨릴 수 있어 검증 전 -- 기본 꺼짐.",
    )
    parser.add_argument(
        "--no-fill-holes", dest="fill_holes", action="store_false",
        help="작은 구멍(핀홀)만 크기 필터로 골라 메우는 fill_small_holes를 끈다. "
             "기본 켜짐 -- 발바닥 등 큰 구멍은 원래 안 건드린다.",
    )
    parser.add_argument(
        "--no-sand-surface", dest="sand_surface_enabled", action="store_false",
        help="전체 정점을 국소 이차곡면에 투영해 다듬는 sand_surface를 끈다. "
             "기본 켜짐 -- 정점/면 개수는 그대로다.",
    )
    parser.add_argument(
        "--sand-min-neighbors", type=int, default=16,
        help="sand_surface 국소 곡면 피팅에 쓸 최소 이웃 수(기본 16). 키우면 더 "
             "매끈해지지만 디테일도 죽는다.",
    )
    parser.add_argument("--sand-max-neighbors", type=int, default=32, help="sand_surface 이웃 상한(기본 32).")
    parser.add_argument("--sand-iterations", type=int, default=3, help="sand_surface 반복 횟수(기본 3).")
    parser.add_argument(
        "--no-finish-smooth", dest="finish_smooth", action="store_false",
        help="마감 라플라시안 스무딩(finish_smooth_mesh)을 끈다. 기본 켜짐 -- "
             "sand_surface/smooth_high_curvature로 안 빠지는 잔여 고주파 표면 노이즈를 "
             "정리한다. 비용 미미(정점 10만개 기준 수 초).",
    )
    parser.add_argument(
        "--finish-smooth-lambda", type=float, default=0.5,
        help="마감 스무딩 반복당 이웃 평균 쪽으로 당기는 비율(0~1, 기본 0.5).",
    )
    parser.add_argument(
        "--finish-smooth-iterations", type=int, default=40,
        help="마감 스무딩 반복 횟수(기본 40). 키울수록 매끈해지지만 디테일도 더 죽는다.",
    )
    parser.add_argument(
        "--no-smooth-high-curvature", dest="smooth_high_curvature", action="store_false",
        help="고곡률 국소 스무딩(smooth_high_curvature_regions)을 끈다. 기본 켜짐 -- "
             "관측 부족 크레이터를 완화하지만 발가락 사이 등 디테일도 함께 뭉갠다.",
    )
    parser.add_argument(
        "--curvature-percentile", type=float, default=DEFAULT_CURVATURE_PERCENTILE,
        help=f"고곡률 스무딩 코어 판정 기준(이 백분위 이상 |곡률|, 기본 {DEFAULT_CURVATURE_PERCENTILE}). "
             "낮출수록 더 넓은 영역이 스무딩된다.",
    )
    parser.add_argument(
        "--curvature-min-radius-mult", type=float, default=DEFAULT_CURVATURE_MIN_RADIUS_MULT,
        help=f"영역별 확산 반경 하한(전형적 엣지 길이의 배수, 기본 {DEFAULT_CURVATURE_MIN_RADIUS_MULT}).",
    )
    parser.add_argument(
        "--curvature-max-radius-mult", type=float, default=DEFAULT_CURVATURE_MAX_RADIUS_MULT,
        help=f"영역별 확산 반경 상한(기본 {DEFAULT_CURVATURE_MAX_RADIUS_MULT}).",
    )
    parser.add_argument(
        "--curvature-iterations", type=int, default=DEFAULT_CURVATURE_ITERATIONS,
        help=f"라플라시안 반복 횟수(기본 {DEFAULT_CURVATURE_ITERATIONS}). 늘릴수록 더 매끄러워짐.",
    )
    parser.add_argument(
        "--curvature-alpha", type=float, default=DEFAULT_CURVATURE_ALPHA,
        help=f"반복당 라플라시안(수축) 스텝 크기(0~1, 기본 {DEFAULT_CURVATURE_ALPHA}).",
    )
    parser.add_argument(
        "--curvature-mu", type=float, default=DEFAULT_CURVATURE_MU,
        help=f"반복당 역방향(팽창) 스텝 크기(음수, 기본 {DEFAULT_CURVATURE_MU}).",
    )


def _run_one(src: Path, dst: Path, args: argparse.Namespace) -> None:
    # process=True로 로드 -- 외부 STL은 대개 face마다 정점이 중복 저장돼(공유 안 됨)
    # 있어, 병합 없이 그대로 쓰면 위상 인접이 전혀 안 잡혀
    # keep_largest_component()가 메쉬를 face 하나짜리 조각들로 오판한다.
    mesh = trimesh.load(src, process=True)
    print(f"[postprocess] 입력: {src} (정점 {len(mesh.vertices):,}개, 면 {len(mesh.faces):,}개)")
    mesh, stats = postprocess_mesh(
        mesh,
        keep_largest=args.keep_largest,
        prune_protrusions=args.prune_protrusions,
        fill_holes=args.fill_holes,
        sand_surface_enabled=args.sand_surface_enabled,
        sand_min_neighbors=args.sand_min_neighbors,
        sand_max_neighbors=args.sand_max_neighbors,
        sand_iterations=args.sand_iterations,
        smooth_high_curvature=args.smooth_high_curvature,
        curvature_percentile=args.curvature_percentile,
        curvature_min_radius_mult=args.curvature_min_radius_mult,
        curvature_max_radius_mult=args.curvature_max_radius_mult,
        curvature_iterations=args.curvature_iterations,
        curvature_alpha=args.curvature_alpha,
        curvature_mu=args.curvature_mu,
        finish_smooth=args.finish_smooth,
        finish_smooth_lambda=args.finish_smooth_lambda,
        finish_smooth_iterations=args.finish_smooth_iterations,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)
    print(
        f"[postprocess] 저장: {dst} (정점 {len(mesh.vertices):,}개, 면 {len(mesh.faces):,}개, "
        f"적용 단계: {', '.join(stats.steps_applied) or '없음'})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="완성된 메쉬(STL 등)에 배경/파편 제거 + 스무딩만 독립적으로 적용 (사진/카메라 불필요)"
    )
    parser.add_argument("input", type=Path, help="메쉬 파일 하나 또는 --glob으로 훑을 폴더")
    parser.add_argument(
        "--glob", type=str, default=None,
        help="input이 폴더일 때 매칭할 패턴(예: '*.stl'). 지정하면 폴더 안 매칭 파일 전부 처리.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="출력 경로(파일 입력 하나일 때만). 생략하면 <stem>_clean<확장자>에 저장.",
    )
    parser.add_argument(
        "--suffix", type=str, default="_clean",
        help="폴더 일괄 처리 시 각 출력 파일명에 붙일 접미사(기본 '_clean').",
    )
    add_postprocess_args(parser)
    args = parser.parse_args(argv)

    if args.glob is not None:
        if not args.input.is_dir():
            parser.error("--glob을 쓰려면 input이 폴더여야 합니다")
        targets = sorted(args.input.glob(args.glob))
        if not targets:
            parser.error(f"{args.input}에서 '{args.glob}'에 매칭되는 파일이 없습니다")
        for src in targets:
            dst = src.with_name(f"{src.stem}{args.suffix}{src.suffix}")
            _run_one(src, dst, args)
    else:
        if not args.input.is_file():
            parser.error(f"파일이 아닙니다(폴더 일괄 처리는 --glob 사용): {args.input}")
        dst = args.out or args.input.with_name(f"{args.input.stem}{args.suffix}{args.input.suffix}")
        _run_one(args.input, dst, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
