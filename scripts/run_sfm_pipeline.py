"""영상/사진 한 벌 -> dense MVS 발 메쉬. `foot_engine.sfm` 전체를 한 번에 돌리는 CLI.

내부적으로 아래 단계를 순서대로 실행한다(각 단계를 따로 돌리고 싶으면
`sparse_sfm_prototype.py` / `generate_foot_masks.py` / `clean_point_cloud.py` /
`run_dense_pipeline.py`를 개별적으로 쓸 것 — 중간 결과를 실제 뷰어로 확인하고
싶을 때 유용하다):

    1. (영상이면) 프레임 추출
    2. 발/피부 마스크 생성 (rembg + MediaPipe 피부 정제)
    3. Sparse SfM 복원 (pycolmap)
    4. Dense MVS + 메싱 (OpenMVS, 별도 설치 필요 — README 참고)
    5. 스케일 보정(자기신고 발길이 기준, 없으면 placeholder)

2026-08-10부로 `FootMeshDeformer` 템플릿 워프 경로를 대체했다 —
`sfm/pipeline.py` 모듈 docstring 참고.

**OpenMVS 별도 설치 필요**: `OPENMVS_BIN_DIR` 환경변수를 실행파일 폴더로
설정하거나 `--openmvs-bin`으로 직접 넘길 것. 설치 방법은 README 참고.

사용 예::

    python scripts/run_sfm_pipeline.py --video data/samples/test00.mp4 `
        --reference-length-mm 255 `
        --out data/output/test00_pipeline_fit.ply `
        --workdir data/output/sfm_pipeline/test00_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

# cp949 등 비-UTF8 콘솔에서 한글 출력이 깨지거나 죽는 문제 방지(실측 확인, 2026-08-07).
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

from foot_engine.sfm import dense  # noqa: E402
from foot_engine.sfm.pipeline import run_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="영상/사진 -> dense 발 메쉬, 파이프라인 전체 실행")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="순서대로 촬영된 영상 파일")
    src.add_argument("--images-dir", type=Path, help="이미 추출된, 순서가 보존된 이미지 폴더")
    parser.add_argument("--workdir", type=Path, required=True, help="중간 산출물 저장 폴더")
    parser.add_argument("--out", type=Path, required=True, help="최종 메쉬 저장 경로(.ply 등)")
    parser.add_argument(
        "--reference-length-mm", type=float, default=None,
        help="자기신고 발길이(mm). 있으면 최종 메쉬를 이 길이에 맞춰 스케일링한다"
             "(SfM은 절대 축척이 없음). 생략하면 250mm placeholder로 스케일링하고 경고 출력.",
    )
    parser.add_argument("--interval", type=float, default=0.5, help="영상에서 프레임을 뽑을 간격(초)")
    parser.add_argument("--start-time", type=float, default=0.0, help="사용할 구간의 시작 시각(초)")
    parser.add_argument("--end-time", type=float, default=None, help="사용할 구간의 끝 시각(초)")
    parser.add_argument(
        "--no-quality-gate", dest="quality_gate", action="store_false", default=True,
        help="SfM 전 프레임별 절대기준 QC(파일 손상/해상도/노출)를 끈다(기본 on).",
    )
    parser.add_argument(
        "--min-sharpness", type=float, default=None,
        help="QC 블러 절대 임계값(라플라시안 분산). 기본 None=검사 안 함 — "
             "scripts/inspect_frame_quality.py로 실제 영상의 선명도 분포를 보고 정할 것.",
    )
    parser.add_argument(
        "--min-frames", type=int, default=8,
        help="QC/마스킹 게이트 통과 후 최소로 남아야 하는 프레임 수(기본 8).",
    )
    parser.add_argument(
        "--blur-keep-ratio", type=float, default=1.0,
        help="선명도 상위 몇 %%만 쓸지(0~1, 기본 1.0=전체 사용).",
    )
    parser.add_argument(
        "--mask-during-extraction", action="store_true",
        help="특징점 추출 단계에서도 마스크를 적용한다. 배경이 복잡/혼재할 때만 — "
             "텍스처 있는 배경이 피사체와 함께 고정된 촬영에서는 오히려 손해였다(실측 확인).",
    )
    parser.add_argument(
        "--no-skin-refine", dest="skin_refine", action="store_false", default=True,
        help="MediaPipe 피부 정제(옷/장신구 제외)를 끈다.",
    )
    parser.add_argument(
        "--no-cluster", dest="cluster", action="store_false", default=True,
        help="QA용 cleaned_points.ply에 DBSCAN 군집화를 적용하지 않는다(기본은 적용). "
             "최종 dense 메쉬에는 영향 없음.",
    )
    parser.add_argument("--openmvs-bin", type=str, default=None, help="OpenMVS 실행파일 폴더(생략 시 OPENMVS_BIN_DIR 환경변수)")
    parser.add_argument(
        "--refine", action="store_true",
        help="RefineMesh(사진 광도일관성 보정)까지 실행. 전체 소요시간의 70%%+ 를 "
             "차지하는 병목이라(실측) 기본은 끔 — 최종 산출물에만 켤 것.",
    )
    parser.add_argument("--no-gapfill", dest="postprocess_dmaps", action="store_const", const=0,
                         default=dense.DEFAULT_POSTPROCESS_DMAPS,
                         help="저텍스처 평면 공백 메우기(--postprocess-dmaps)를 끈다.")
    parser.add_argument("--dense-max-threads", type=int, default=dense.DEFAULT_MAX_THREADS,
                         help=f"DensifyPointCloud 스레드 상한(기본 {dense.DEFAULT_MAX_THREADS}) — "
                              "원인불명 간헐적 크래시 방지용(실측 근거 dense.py 참고).")
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
    args = parser.parse_args(argv)

    result = run_pipeline(
        workdir=args.workdir,
        out_mesh=args.out,
        video=args.video,
        images_dir=args.images_dir,
        reference_length_mm=args.reference_length_mm,
        interval=args.interval,
        start_time=args.start_time,
        end_time=args.end_time,
        quality_gate=args.quality_gate,
        min_sharpness=args.min_sharpness,
        min_frames=args.min_frames,
        blur_keep_ratio=args.blur_keep_ratio,
        mask_during_extraction=args.mask_during_extraction,
        skin_refine=args.skin_refine,
        cluster=args.cluster,
        openmvs_bin=args.openmvs_bin,
        refine=args.refine,
        postprocess_dmaps=args.postprocess_dmaps,
        dense_max_threads=args.dense_max_threads,
        visibility_filter_threshold=args.visibility_filter_threshold,
        grazing_filter_min_score=args.grazing_filter_min_score,
        reprojection_consistency_min_vote=args.reprojection_consistency_min_vote,
        free_space_support=args.free_space_support,
        thickness_factor=args.thickness_factor,
        quality_factor=args.quality_factor,
        refine_decimate=args.refine_decimate,
        refine_regularity_weight=args.refine_regularity_weight,
        smooth_high_curvature=args.smooth_high_curvature,
    )

    print("\n[파이프라인 요약]")
    if result.quality_stats:
        print(f"  QC: {result.quality_stats}")
    print(f"  등록된 이미지: {result.n_points_registered_images} / {result.n_points_total_images}")
    print(f"  sparse 점: {result.n_points_raw:,}개 -> 정리 후(QA용): {result.n_points_cleaned:,}개")
    print(f"  마스크: {result.mask_stats}")
    print(
        f"  최종 메쉬: 정점 {result.n_mesh_vertices:,}개, 면 {result.n_mesh_faces:,}개, "
        f"스케일 x{result.scale_factor:.4f}"
    )
    print(f"  저장 경로: {result.output_mesh_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
