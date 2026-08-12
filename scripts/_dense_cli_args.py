"""`run_sfm_pipeline.py`/`run_dense_pipeline.py`가 공유하는 dense MVS 튜닝 CLI 인자.

두 스크립트 모두 `dense.run_dense_pipeline()`(또는 그걸 감싸는
`pipeline.run_pipeline()`)에 그대로 전달되는 동일한 옵션 세트를 쓴다 — 한쪽만
고치고 다른 쪽을 깜빡하는 걸 막기 위해 여기 하나로 모았다. 튜닝 근거는
`dense.py` 모듈 docstring 참고.
"""

from __future__ import annotations

import argparse

import _cli_common  # noqa: F401  -- sys.path 설정(부작용 import)

from foot_engine.sfm import dense  # noqa: E402


def add_dense_args(
    parser: argparse.ArgumentParser,
    *,
    thread_flag: str = "--max-threads",
    thread_dest: str = "max_threads",
) -> None:
    """dense MVS 튜닝 인자 전부를 `parser`에 추가한다.

    `thread_flag`/`thread_dest`만 호출부마다 다르다 — `dense.run_dense_pipeline()`은
    파라미터명이 `max_threads`, 그걸 감싸는 `pipeline.run_pipeline()`은
    `dense_max_threads`라서 CLI 플래그도 그에 맞춰 다르게 노출해왔다. 나머지는
    두 CLI가 완전히 동일하다.
    """
    parser.add_argument(
        "--openmvs-bin", type=str, default=None,
        help="OpenMVS 실행파일 폴더(생략 시 OPENMVS_BIN_DIR 환경변수)",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="RefineMesh(사진 광도일관성 보정)까지 실행. 전체 소요시간의 70%%+ 를 "
             "차지하는 병목이라(실측) 기본은 끔 — 최종 산출물에만 켤 것.",
    )
    parser.add_argument(
        "--no-gapfill", dest="postprocess_dmaps", action="store_const", const=0,
        default=dense.DEFAULT_POSTPROCESS_DMAPS,
        help="저텍스처 평면 공백 메우기(--postprocess-dmaps)를 끈다.",
    )
    parser.add_argument(
        thread_flag, dest=thread_dest, type=int, default=dense.DEFAULT_MAX_THREADS,
        help=f"DensifyPointCloud 스레드 상한(기본 {dense.DEFAULT_MAX_THREADS}) — "
             "원인불명 간헐적 크래시 방지용(실측 근거 dense.py 참고).",
    )
    parser.add_argument(
        "--visibility-filter-threshold", type=int, default=None,
        help="OpenMVS 내장 가시성 필터(--filter-point-cloud) 임계값(음수, 예: -1). "
             "생략(기본)하면 안 돌림 — dense.py 모듈 docstring 5번 참고, 발바닥 보존은 "
             "확인됐지만 경계 노이즈 제거 효과는 육안 검증 필요.",
    )
    parser.add_argument(
        "--grazing-filter-min-score", type=float, default=None,
        help="법선-시선 grazing-angle 필터(filter_grazing_points) 임계값(0~1, 예: 0.3). "
             "생략(기본)하면 안 돌림 — dense.py의 filter_grazing_points() docstring 참고, "
             "발바닥 보존은 확인됐지만 경계 노이즈 제거 효과는 육안 검증 필요.",
    )
    parser.add_argument(
        "--reprojection-consistency-min-vote", type=float, default=None,
        help="전체 카메라 재투영 다수결 필터(filter_by_reprojection_consistency) 임계값(0~1, "
             "예: 0.6). 생략(기본)하면 안 돌림 — 실측(test03, 배경 오염): 0.6에서 오염 후보 "
             "44%% 제거/발 오제거 11%%.",
    )
    parser.add_argument(
        "--free-space-support", action="store_true",
        help="ReconstructMesh --free-space-support 켬 — 실측 확인: 메쉬가 뾰족하게 뒤틀리는 "
             "부작용이 있어 권장 안 함(dense_mvs_results/README.md 참고).",
    )
    parser.add_argument(
        "--thickness-factor", type=float, default=1.0,
        help="ReconstructMesh --thickness-factor(기본 1.0=OpenMVS 기본값). "
             "실측 확인: 2.0에서도 위 free-space-support와 같은 부작용 발생.",
    )
    parser.add_argument(
        "--quality-factor", type=float, default=1.0,
        help="ReconstructMesh --quality-factor(기본 1.0=OpenMVS 기본값).",
    )
    parser.add_argument(
        "--refine-decimate", type=float, default=1.0,
        help="RefineMesh --decimate(0~1, 기본 1=단순화 끔·해상도 보존). `--refine` 켰을 때만 적용.",
    )
    parser.add_argument(
        "--refine-regularity-weight", type=float, default=None,
        help="RefineMesh --regularity-weight(생략 시 OpenMVS 기본값 0.2). `--refine` 켰을 때만 적용.",
    )
    parser.add_argument(
        "--no-smooth-high-curvature", dest="smooth_high_curvature", action="store_false",
        help="고곡률 국소 스무딩(smooth_high_curvature_regions)을 끈다. 기본 켜짐 — "
             "관측 부족 크레이터 완화 효과 실측 확인, 발가락 사이 등 디테일도 함께 "
             "뭉개지는 트레이드오프는 감수하기로 결정됨.",
    )
    parser.add_argument(
        "--fill-holes", action="store_true",
        help="작은 구멍(핀홀)만 크기 필터로 골라 팬 삼각분할로 메운다(fill_small_holes). "
             "발바닥 등 큰 구멍은 그대로 둔다. 기본 꺼짐 — 아직 실측 검증 전. "
             "폴리곤은 추가만 되고 기존 정점/면은 그대로다.",
    )
    parser.add_argument(
        "--sand-surface", dest="sand_surface_enabled", action="store_true",
        help="전체 정점을 국소 이차곡면(quadric)에 투영해 다듬는다(sand_surface) — "
             "곡률 임계값 없이 발 전체에 균일 적용, --no-smooth-high-curvature와 달리 "
             "크레이터 전용이 아닌 일반 노이즈 완화. 기본 꺼짐 — 아직 실측 검증 전. "
             "정점/면 개수는 그대로다.",
    )
    parser.add_argument(
        "--keep-intermediates", action="store_true",
        help="완료 후 중간 산출물을 지우지 않고 그대로 둔다(이 스크립트가 만든 범위 — "
             "run_dense_pipeline.py는 undistort 워크스페이스/depth map/OpenMVS 로그, "
             "run_sfm_pipeline.py는 그거 포함 프레임/마스크/DB/sparse 복원까지). 기본은 "
             "끔(정리) — 최종 메쉬만 남는다. 디버깅/로깅 목적이거나, run_dense_pipeline.py로 "
             "dense 파라미터를 나중에 따로 튜닝하려면 켤 것.",
    )
