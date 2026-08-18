"""SfM(Structure-from-Motion) 기반 2D 사진/영상 -> dense MVS 3D 발 메쉬 파이프라인.

`pipeline.run_pipeline()`이 최상위 진입점:

    frame_quality.assess_frames() — 프레임 QC
        └─ masking.generate_masks() — 발/피부 마스크 생성, 프레임 필터링
              └─ reconstruction.run_sparse_sfm() — sparse 포인트클라우드
                    └─ cleaning.clean_point_cloud() — QA용 정리(메쉬엔 안 씀)
                    └─ dense.run_dense_pipeline() — dense 메쉬 생성

메쉬 생성기는 dense MVS 하나(템플릿 워프/SSM은 `archive/
deformer_ssm_pipeline/`). `geometry.py`는 스케일 추정용 유틸.

촬영 조건(코드로 못 고침): 발 고정, 선명한 사진, 소매/바지단 최대한 안
보이게, 손으로 잡지 않기.
"""

from __future__ import annotations

from .cleaning import (
    classify_by_mask,
    clean_point_cloud,
    estimate_focus_point,
    geometric_keep_mask,
    keep_largest_cluster,
    plane_removal_keep_mask,
)
from .dense import (
    align_principal_axes,
    clean_dense_point_cloud,
    convert_masks_for_openmvs,
    largest_sparse_dir,
    run_dense_pipeline,
    run_densify_point_cloud,
    run_interface_colmap,
    run_reconstruct_mesh,
    run_refine_mesh,
    undistort_for_dense,
)
from .frame_quality import (
    FrameQualityResult,
    assess_frame,
    assess_frames,
    passed_names,
    print_summary,
    summarize,
)
from .geometry import measured_length, pca_axes
from .masking import generate_masks, load_skin_segmenter, skin_only_mask
from .mesh_postprocess import keep_largest_component, postprocess_mesh
from .pipeline import PipelineResult, run_pipeline
from .reconstruction import (
    compute_sharpness,
    extract_frames,
    filter_blurry_frames,
    filter_outlier_points,
    report_unregistered_frames,
    run_sparse_sfm,
)

__all__ = [
    # frame_quality
    "FrameQualityResult",
    "assess_frame",
    "assess_frames",
    "passed_names",
    "summarize",
    "print_summary",
    # reconstruction
    "extract_frames",
    "compute_sharpness",
    "filter_blurry_frames",
    "filter_outlier_points",
    "run_sparse_sfm",
    "report_unregistered_frames",
    # masking
    "generate_masks",
    "load_skin_segmenter",
    "skin_only_mask",
    # cleaning
    "classify_by_mask",
    "plane_removal_keep_mask",
    "estimate_focus_point",
    "geometric_keep_mask",
    "keep_largest_cluster",
    "clean_point_cloud",
    # geometry
    "measured_length",
    "pca_axes",
    # pipeline
    "run_pipeline",
    "PipelineResult",
    # dense (선택적 dense MVS)
    "largest_sparse_dir",
    "undistort_for_dense",
    "convert_masks_for_openmvs",
    "run_interface_colmap",
    "run_densify_point_cloud",
    "clean_dense_point_cloud",
    "run_reconstruct_mesh",
    "run_refine_mesh",
    "keep_largest_component",
    "align_principal_axes",
    "run_dense_pipeline",
    # mesh_postprocess (배경/파편 제거 + 스무딩, 사진/카메라 불필요 -- 완성된 메쉬에도 독립 적용)
    "postprocess_mesh",
]
