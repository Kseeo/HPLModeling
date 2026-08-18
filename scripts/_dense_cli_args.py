"""`run_sfm_pipeline.py`/`run_dense_pipeline.py`가 공유하는 dense MVS 튜닝 CLI 인자.

두 스크립트가 같은 옵션 세트를 `dense.run_dense_pipeline()`에 전달하므로
여기 하나로 모았다. 튜닝 근거는 `dense.py` 모듈 docstring 참고.
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

    `thread_flag`/`thread_dest`만 호출부마다 다르다(파라미터명이 `max_threads`
    vs `dense_max_threads`) — 나머지는 두 CLI가 동일하다.
    """
    parser.add_argument(
        "--openmvs-bin", type=str, default=None,
        help="OpenMVS 실행파일 폴더(생략 시 OPENMVS_BIN_DIR 환경변수)",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="RefineMesh(사진 광도일관성 보정)까지 실행. 시간이 오래 걸리는 병목이라 "
             "기본은 끔 — 최종 산출물에만 켤 것.",
    )
    parser.add_argument(
        "--no-gapfill", dest="postprocess_dmaps", action="store_const", const=0,
        default=dense.DEFAULT_POSTPROCESS_DMAPS,
        help="저텍스처 평면 공백 메우기(--postprocess-dmaps)를 끈다.",
    )
    parser.add_argument(
        thread_flag, dest=thread_dest, type=int, default=dense.DEFAULT_MAX_THREADS,
        help=f"DensifyPointCloud 스레드 상한(기본 {dense.DEFAULT_MAX_THREADS}) — "
             "간헐적 크래시 방지용.",
    )
    parser.add_argument(
        "--densify-resolution-level", type=int, default=None,
        help="DensifyPointCloud --resolution-level(0=원본 해상도, 숫자가 클수록 축소돼 "
             "빠르지만 점군이 성겨진다). 생략(기본)하면 OpenMVS 자체 기본값. "
             "정밀도를 올리려면 0으로 줄 것 -- 시간은 늘어난다.",
    )
    parser.add_argument(
        "--densify-number-views-fuse", type=int, default=None,
        help="DensifyPointCloud --number-views-fuse(점 하나를 살릴 최소 동의 뷰 개수). "
             "생략(기본)하면 OpenMVS 자체 기본값(2).",
    )
    parser.add_argument(
        "--visibility-filter-threshold", type=int, default=None,
        help="OpenMVS 내장 가시성 필터(--filter-point-cloud) 임계값(음수, 예: -1). "
             "생략(기본)하면 안 돌림.",
    )
    parser.add_argument(
        "--grazing-filter-min-score", type=float, default=None,
        help="법선-시선 grazing-angle 필터(filter_grazing_points) 임계값(0~1, 예: 0.3). "
             "생략(기본)하면 안 돌림.",
    )
    parser.add_argument(
        "--reprojection-consistency-min-vote", type=float, default=None,
        help="전체 카메라 재투영 다수결 필터(filter_by_reprojection_consistency) 임계값(0~1). "
             "생략(기본)하면 안 돌림 — 켜지 말 것을 권장(dense.py 모듈 docstring 6번 참고).",
    )
    parser.add_argument(
        "--free-space-support", action="store_true",
        help="ReconstructMesh --free-space-support 켬 — 메쉬가 뾰족하게 뒤틀리는 "
             "부작용이 있어 권장 안 함.",
    )
    parser.add_argument(
        "--thickness-factor", type=float, default=1.0,
        help="ReconstructMesh --thickness-factor(기본 1.0=OpenMVS 기본값). "
             "free-space-support와 같은 부작용이 있다.",
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
             "관측 부족 크레이터를 완화하지만 발가락 사이 등 디테일도 함께 뭉갠다.",
    )
    parser.add_argument(
        "--curvature-percentile", type=float, default=dense.DEFAULT_CURVATURE_PERCENTILE,
        help=f"고곡률 스무딩 코어 판정 기준(이 백분위 이상 |곡률|, 기본 "
             f"{dense.DEFAULT_CURVATURE_PERCENTILE}). 낮출수록 더 넓은 영역이 스무딩된다.",
    )
    parser.add_argument(
        "--curvature-min-radius-mult", type=float, default=dense.DEFAULT_CURVATURE_MIN_RADIUS_MULT,
        help=f"영역(연결된 고곡률 정점 덩어리)별 확산 반경의 하한(전형적 엣지 길이의 배수, 기본 "
             f"{dense.DEFAULT_CURVATURE_MIN_RADIUS_MULT}). 실제 반경은 그 영역의 곡률 반경(1/|곡률|)에서 "
             "정하되 이 범위로 clip한다.",
    )
    parser.add_argument(
        "--curvature-max-radius-mult", type=float, default=dense.DEFAULT_CURVATURE_MAX_RADIUS_MULT,
        help=f"위 확산 반경의 상한(기본 {dense.DEFAULT_CURVATURE_MAX_RADIUS_MULT}) -- 거의 평평한(곡률 "
             "0에 가까운) 영역이 반경 폭주하는 것을 막는 안전장치.",
    )
    parser.add_argument(
        "--curvature-iterations", type=int, default=dense.DEFAULT_CURVATURE_ITERATIONS,
        help=f"라플라시안 반복 횟수(기본 {dense.DEFAULT_CURVATURE_ITERATIONS}). 늘릴수록 더 매끄러워짐.",
    )
    parser.add_argument(
        "--curvature-alpha", type=float, default=dense.DEFAULT_CURVATURE_ALPHA,
        help=f"반복당 라플라시안(수축) 스텝 크기(0~1, 기본 {dense.DEFAULT_CURVATURE_ALPHA}).",
    )
    parser.add_argument(
        "--curvature-mu", type=float, default=dense.DEFAULT_CURVATURE_MU,
        help=f"반복당 역방향(팽창) 스텝 크기(음수, 절댓값이 curvature-alpha보다 커야 함, 기본 "
             f"{dense.DEFAULT_CURVATURE_MU}). alpha와 번갈아 적용하는 Taubin 스무딩으로 체적 "
             "수축 편향을 상쇄한다.",
    )
    parser.add_argument(
        "--no-fill-holes", dest="fill_holes", action="store_false",
        help="작은 구멍(핀홀)만 크기 필터로 골라 메우는 fill_small_holes를 끈다. "
             "기본 켜짐 — 발바닥 등 큰 구멍은 원래 안 건드린다.",
    )
    parser.add_argument(
        "--no-sand-surface", dest="sand_surface_enabled", action="store_false",
        help="전체 정점을 국소 이차곡면에 투영해 다듬는 sand_surface를 끈다. "
             "기본 켜짐 — 정점/면 개수는 그대로다.",
    )
    parser.add_argument(
        "--sand-min-neighbors", type=int, default=16,
        help="sand_surface 국소 곡면 피팅에 쓸 최소 이웃 수(기본 16). 키우면 더 "
             "매끈해지지만 디테일도 죽는다.",
    )
    parser.add_argument(
        "--sand-max-neighbors", type=int, default=32,
        help="sand_surface 이웃 상한(기본 32).",
    )
    parser.add_argument(
        "--sand-iterations", type=int, default=3,
        help="sand_surface 반복 횟수(기본 3).",
    )
    parser.add_argument(
        "--no-finish-smooth", dest="finish_smooth", action="store_false",
        help="마감 라플라시안 스무딩(finish_smooth_mesh)을 끈다. 기본 켜짐 — "
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
        "--prune-protrusions", action="store_true",
        help="포인트클라우드 단계(메싱 전)에서 국소 밀도 기준으로 뿔/스파이크 후보 점을 "
             "미리 제거한다(clean_dense_point_cloud prune_protrusions). 발목 부근 뿔 결함을 "
             "겨냥한 대책. 기본 꺼짐 — 실행마다 결과가 흔들려 검증 전.",
    )
    parser.add_argument(
        "--trim-leg", action="store_true",
        help="발목 위 다리까지 찍혀 축 정렬이 다리 쪽으로 쏠리는 케이스를 겨냥해, 잘록해졌다가"
             "(발목) 다시 굵어진 채 안 가늘어지는(다리) 패턴이 뚜렷할 때만 잘라내고 정렬한다"
             "(finalize_mesh trim_leg). 애매하면 자르지 않음. 기본 꺼짐 — 소수 사례로만 검증됨.",
    )
    parser.add_argument(
        "--keep-intermediates", action="store_true",
        help="완료 후 중간 산출물을 지우지 않고 그대로 둔다(이 스크립트가 만든 범위 — "
             "run_dense_pipeline.py는 undistort 워크스페이스/depth map/OpenMVS 로그, "
             "run_sfm_pipeline.py는 그거 포함 프레임/마스크/DB/sparse 복원까지). 기본은 "
             "끔(정리) — 최종 메쉬만 남는다. 디버깅/로깅 목적이거나, run_dense_pipeline.py로 "
             "dense 파라미터를 나중에 따로 튜닝하려면 켤 것.",
    )
