"""SfM(Structure-from-Motion) 기반 2D 사진/영상 -> 3D 발 메쉬 파이프라인.

외부 포토그래메트리를 대체하는 in-house 경로다. 6단계로 구성된다.

    frame_quality.assess_frames()     — SfM 전, 프레임별 절대기준 QC(파일 손상/해상도/노출)
        └─ reconstruction.run_sparse_sfm()  — 사진/영상 -> 카메라 포즈 + sparse 포인트클라우드
              └─ masking.generate_masks()   — 사진별 발/피부 마스크, 프레임 필터링
                    └─ cleaning.clean_point_cloud()  — 마스크 기반 배경 제거 + 이상치 제거 + 군집화
                          └─ fitting.fit_point_cloud_to_template()  — 좌우 정렬 + 계측 + 템플릿 워프
                                └─ pipeline.run_pipeline()  — 위 다섯 단계를 한 번에 엮는 오케스트레이션

`fitting.py` 경로는 스칼라 계측치 몇 개만 있으면 되므로 위 다섯 단계가 전부다.
실제 3D 메쉬(시각화/QA용)가 필요하면 `dense.py`(OpenMVS 기반 dense MVS,
별도 CLI 설치 필요, `run_dense_pipeline()`)를 선택적으로 이어붙인다 —
`sparse_dir`(run_sparse_sfm 결과)와 `masks_dir`(generate_masks, dilate=0)만
있으면 된다. `dense.py` 모듈 docstring에 실측 검증된 튜닝 근거가 정리돼 있다.

알려진 한계(코드로 못 고치는 촬영 조건):
    - 촬영 내내 발이 완전히 고정돼야 한다(의자 위에 올려두는 등)
    - 사진이 선명해야 한다 — 영상에서 캡쳐하는 방식을 사용 중, 일정 간격으로 사진을 찍는 방식으로 대체 예정
    - 발목 위로 소매/바지단이 최대한 보이지 않도록 한다
    - 발을 손으로 잡고 있으면 안 된다(손이 발 점군에 섞인다)
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
    keep_largest_component,
    largest_sparse_dir,
    run_dense_pipeline,
    run_densify_point_cloud,
    run_interface_colmap,
    run_reconstruct_mesh,
    run_refine_mesh,
    undistort_for_dense,
)
from .fitting import (
    fit_point_cloud_to_template,
    measure_point_cloud,
    measured_length,
    mirror_points,
    rigid_prealign_points,
)
from .frame_quality import (
    FrameQualityResult,
    assess_frame,
    assess_frames,
    passed_names,
    print_summary,
    summarize,
)
from .masking import generate_masks, load_skin_segmenter, skin_only_mask
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
    # fitting
    "measured_length",
    "mirror_points",
    "rigid_prealign_points",
    "measure_point_cloud",
    "fit_point_cloud_to_template",
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
]
