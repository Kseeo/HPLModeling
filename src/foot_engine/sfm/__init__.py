"""SfM(Structure-from-Motion) 기반 2D 사진/영상 -> dense MVS 3D 발 메쉬 파이프라인.

외부 포토그래메트리(VRIN)를 대체하는 in-house 경로다. `pipeline.run_pipeline()`이
아래를 이 순서로 엮는 최상위 진입점이다:

    frame_quality.assess_frames()  — SfM 전, 프레임별 절대기준 QC(파일 손상/해상도/노출)
        └─ masking.generate_masks()  — 사진별 발/피부 마스크(sparse+dense 두 벌을
              한 번에 생성 — extra_dilations, 추론 중복 방지), 프레임 필터링
                └─ reconstruction.run_sparse_sfm()  — 카메라 포즈 + sparse 포인트클라우드
                      └─ cleaning.clean_point_cloud()  — 마스크 기반 배경 제거 + 이상치 제거 + 군집화(QA용, 최종 메쉬엔 안 씀)
                      └─ dense.run_dense_pipeline()  — OpenMVS densify + 메싱 + 파편 제거

메쉬 생성기는 dense MVS 하나다(2026-08-11, 템플릿 워프/SSM 경로는
`archive/deformer_ssm_pipeline/`로 옮김 — 대응점 노이즈로 SSM이 무산된 뒤
템플릿 워프까지 유지할 이유가 약해져, dense 메쉬 자체를 다듬고
경량화하는 쪽으로 결론남). `geometry.py`는 그 결정 이후에도 남은 유일한
의존성(스케일 추정용 `pca_axes`/`measured_length`)을 담는다.

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
]
