"""영상/사진 한 벌 -> dense MVS 발 메쉬. `foot_engine.sfm` 전체를 한 번에 돌리는 CLI.

단계: 프레임 추출(영상이면) -> 발/피부 마스크 -> sparse SfM -> dense MVS
+ 메싱 -> 스케일 보정. 각 단계를 따로 돌리려면 `sparse_sfm_prototype.py` /
`generate_foot_masks.py` / `clean_point_cloud.py` / `run_dense_pipeline.py`를
개별적으로 쓸 것.

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
from pathlib import Path

import _cli_common  # noqa: F401  -- sys.path 설정 + 콘솔 UTF-8 고정(부작용 import)
from _dense_cli_args import add_dense_args  # noqa: E402

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
        help="자기신고 발길이(mm), 스케일 기준(SfM은 절대 축척 없음). "
             "생략하면 250mm placeholder 사용.",
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
             "scripts/inspect_frame_quality.py로 선명도 분포 보고 정할 것.",
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
             "텍스처 있는 배경이 피사체와 함께 고정된 촬영에서는 오히려 손해다.",
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
    parser.add_argument(
        "--target-vertices", type=int, default=None,
        help="지정하면 최종 메쉬를 이 정점 수 근방까지 단순화한 뒤 마감 스무딩한다 "
             "(다른 데이터셋/모델의 메쉬 해상도에 맞출 때). 스무딩 전에 먼저 축약되므로 "
             "고곡률 스무딩이 훨씬 빨라진다.",
    )
    add_dense_args(parser, thread_flag="--dense-max-threads", thread_dest="dense_max_threads")
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
        densify_resolution_level=args.densify_resolution_level,
        densify_number_views_fuse=args.densify_number_views_fuse,
        visibility_filter_threshold=args.visibility_filter_threshold,
        grazing_filter_min_score=args.grazing_filter_min_score,
        reprojection_consistency_min_vote=args.reprojection_consistency_min_vote,
        free_space_support=args.free_space_support,
        thickness_factor=args.thickness_factor,
        quality_factor=args.quality_factor,
        refine_decimate=args.refine_decimate,
        refine_regularity_weight=args.refine_regularity_weight,
        smooth_high_curvature=args.smooth_high_curvature,
        curvature_percentile=args.curvature_percentile,
        curvature_min_radius_mult=args.curvature_min_radius_mult,
        curvature_max_radius_mult=args.curvature_max_radius_mult,
        curvature_iterations=args.curvature_iterations,
        curvature_alpha=args.curvature_alpha,
        curvature_mu=args.curvature_mu,
        fill_holes=args.fill_holes,
        sand_surface_enabled=args.sand_surface_enabled,
        sand_min_neighbors=args.sand_min_neighbors, sand_max_neighbors=args.sand_max_neighbors,
        sand_iterations=args.sand_iterations,
        finish_smooth=args.finish_smooth, finish_smooth_lambda=args.finish_smooth_lambda,
        finish_smooth_iterations=args.finish_smooth_iterations,
        prune_protrusions=args.prune_protrusions,
        target_vertices=args.target_vertices,
        trim_leg=args.trim_leg,
        keep_intermediates=args.keep_intermediates,
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
