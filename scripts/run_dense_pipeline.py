"""Sparse SfM 결과 -> OpenMVS dense 포인트클라우드/메쉬. `foot_engine.sfm.dense` CLI.

실제 로직은 `foot_engine.sfm.dense`에 있다 — 이 파일은 인자 파싱과 출력
포맷만 담당하는 얇은 CLI 래퍼다. 튜닝 근거(마스크 dilate=0, DBSCAN 미사용,
smooth=0 등)는 `dense.py` 모듈 docstring 참고.

**선택적 단계**다 — `fitting.py` 경로(스칼라 계측치만 필요)는 이 스크립트
없이도 그대로 동작한다. 실제 3D 메쉬(시각화/QA/향후 고정밀 활용)가 필요할
때만 쓴다.

**OpenMVS 별도 설치 필요** (`InterfaceCOLMAP`/`DensifyPointCloud`/
`ReconstructMesh`/`RefineMesh`, CPU 빌드로 충분): 이 스크립트를 돌리기 전
`OPENMVS_BIN_DIR` 환경변수를 그 실행파일들이 있는 폴더로 설정하거나
`--openmvs-bin`으로 직접 넘길 것. 설치 방법은 README 참고.

사용 예::

    # 1) 먼저 sparse SfM + 마스크(dilate=0)를 만들어 둔다
    python scripts/run_sfm_pipeline.py --video data/samples/test02.mp4 `
        --workdir data/output/sfm_pipeline/test02_run --out data/output/test02_fit.stl

    # 2) 그 결과로 dense 메쉬 생성 (RefineMesh 제외, 빠름)
    python scripts/run_dense_pipeline.py data/output/sfm_pipeline/test02_run

    # 3) 최종 품질까지 원하면 RefineMesh 포함 (느림, 전체 시간의 70%+ 차지)
    python scripts/run_dense_pipeline.py data/output/sfm_pipeline/test02_run --refine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

from foot_engine.sfm.dense import largest_sparse_dir, run_dense_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="sparse SfM 결과 -> OpenMVS dense 포인트클라우드/메쉬 (선택적 단계)"
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="run_sfm_pipeline.py의 --workdir (images/, masks/, sparse/ 를 포함하는 폴더)",
    )
    parser.add_argument(
        "--sparse-dir", type=Path, default=None,
        help="sparse 재구성 폴더를 직접 지정(생략 시 <run_dir>/sparse 아래에서 등록 이미지가 "
             "가장 많은 폴더를 자동으로 찾는다 — 번호가 항상 크기순은 아니다).",
    )
    parser.add_argument(
        "--masks-dir", type=Path, default=None,
        help="마스크 폴더(생략 시 <run_dir>/masks_dense를 dilate=0으로 새로 생성).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="산출물 폴더(기본 <run_dir>/dense_mvs)")
    parser.add_argument("--openmvs-bin", type=str, default=None, help="OpenMVS 실행파일 폴더(생략 시 OPENMVS_BIN_DIR 환경변수)")
    parser.add_argument(
        "--refine", action="store_true",
        help="RefineMesh(사진 광도일관성 보정)까지 실행. 전체 소요시간의 70%%+ 를 "
             "차지하는 병목이라(실측) 기본은 끔 — 최종 산출물에만 켤 것.",
    )
    parser.add_argument("--no-gapfill", dest="postprocess_dmaps", action="store_const", const=0, default=3,
                         help="저텍스처 평면 공백 메우기(--postprocess-dmaps)를 끈다.")
    parser.add_argument("--max-threads", type=int, default=8,
                         help="DensifyPointCloud 스레드 상한(기본 8) — 원인불명 간헐적 크래시 방지용(실측 근거 dense.py 참고).")
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
    parser.add_argument("--thickness-factor", type=float, default=1.0,
                         help="ReconstructMesh --thickness-factor(기본 1.0=OpenMVS 기본값). "
                              "실측 확인: 2.0에서도 위 free-space-support와 같은 부작용 발생.")
    parser.add_argument("--quality-factor", type=float, default=1.0,
                         help="ReconstructMesh --quality-factor(기본 1.0=OpenMVS 기본값).")
    parser.add_argument("--refine-decimate", type=float, default=1.0,
                         help="RefineMesh --decimate(0~1, 기본 1=단순화 끔·해상도 보존). "
                              "`--refine` 켰을 때만 적용.")
    parser.add_argument("--refine-regularity-weight", type=float, default=None,
                         help="RefineMesh --regularity-weight(생략 시 OpenMVS 기본값 0.2). "
                              "`--refine` 켰을 때만 적용.")
    parser.add_argument(
        "--no-smooth-high-curvature", dest="smooth_high_curvature", action="store_false",
        help="고곡률 국소 스무딩(smooth_high_curvature_regions)을 끈다. 기본 켜짐 — "
             "관측 부족 크레이터 완화 효과 실측 확인, 발가락 사이 등 디테일도 함께 "
             "뭉개지는 트레이드오프는 감수하기로 결정됨.",
    )
    parser.add_argument(
        "--keep-intermediates", action="store_true",
        help="완료 후 undistort 워크스페이스/depth map/OpenMVS 로그 등 중간 산출물을 "
             "지우지 않고 그대로 둔다. 기본은 끔(정리) — <out-dir>/mesh.ply 하나만 남는다. "
             "디버깅/로깅 목적일 때만 켤 것.",
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    sparse_dir = args.sparse_dir or largest_sparse_dir(run_dir / "sparse")
    print(f"[dense] sparse 폴더: {sparse_dir}")

    masks_dir = args.masks_dir
    if masks_dir is None:
        from foot_engine.sfm.masking import generate_masks
        masks_dir = run_dir / "masks_dense"
        stats = generate_masks(run_dir / "images", masks_dir, dilate=0)
        print(f"[dense] 마스크(dilate=0) 새로 생성: {stats}")

    out_dir = args.out_dir or (run_dir / "dense_mvs")
    mesh_path = run_dense_pipeline(
        sparse_dir=sparse_dir,
        images_dir=run_dir / "images",
        masks_dir=masks_dir,
        workdir=out_dir,
        openmvs_bin=args.openmvs_bin,
        refine=args.refine,
        postprocess_dmaps=args.postprocess_dmaps,
        max_threads=args.max_threads,
        visibility_filter_threshold=args.visibility_filter_threshold,
        grazing_filter_min_score=args.grazing_filter_min_score,
        reprojection_consistency_min_vote=args.reprojection_consistency_min_vote,
        free_space_support=args.free_space_support,
        thickness_factor=args.thickness_factor,
        quality_factor=args.quality_factor,
        refine_decimate=args.refine_decimate,
        refine_regularity_weight=args.refine_regularity_weight,
        smooth_high_curvature=args.smooth_high_curvature,
        keep_intermediates=args.keep_intermediates,
    )
    print(f"\n최종 메쉬: {mesh_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
