"""SfM(Structure-from-Motion) 기반 2D 사진/영상 -> 3D 발 메쉬 파이프라인.

VRIN(외부 유료 포토그래메트리)을 대체하는 in-house 경로다. 6단계로 구성된다.

    frame_quality.assess_frames()     — SfM 전, 프레임별 절대기준 QC(파일 손상/해상도/노출)
        └─ reconstruction.run_sparse_sfm()  — 사진/영상 -> 카메라 포즈 + sparse 포인트클라우드
                                          (pycolmap, exhaustive 매칭 + incremental mapping)
              └─ masking.generate_masks()   — 사진별 발/피부 마스크(rembg + MediaPipe 피부 정제),
                                                발이 아예 안 보이는 프레임은 여기서 추가로 제외
                    └─ cleaning.clean_point_cloud()  — 마스크 기반 배경 제거 + 이상치 제거 + 군집화
                          └─ fitting.fit_point_cloud_to_template()  — 좌우 정렬 + 계측 + 템플릿 워프
                                └─ pipeline.run_pipeline()  — 위 다섯 단계를 한 번에 엮는 오케스트레이션

각 모듈은 2026-08-07 세션에서 test00/02/03 세 영상에 대해 A/B 테스트로 검증된
기본값을 쓴다 — 근거와 실측 수치는 각 함수 docstring 및 프로젝트 메모리
(`sfm-prototype-robustness-upgrade`, `test00-sleeve-cuff-contamination`,
`skin-refine-masking-shipped`, `lr-chirality-fix-and-heel-bug-isolation`)에
있다. SSM(PCA 기반 통계형상모델) 경로는 대응점 노이즈 문제로 폐기됐고
(`ssm-pipeline-build-notes` 참고), `fitting.py`는 대신 이미 검증된
`FootMeshDeformer`(측정치 기반 템플릿 워프)를 그대로 재사용한다.

알려진 한계(코드로 못 고치는 촬영 조건):
    - 촬영 내내 발이 완전히 고정돼야 한다(자세 유지 X, 바닥/유리판에 얹을 것).
    - 사진이 선명해야 한다 — 흔들리면 매칭 자체가 안 된다.
    - 발목 위로 소매/바지단이 보이면 안 된다 — 일반 인물 세그멘테이션은 옷을
      피부와 구분 못하고, 피부-옷 경계선 자체가 SIFT에 강한 특징점이라 완전한
      후처리 제거가 어렵다(`skin-refine-masking-shipped` 참고).
    - 발을 손으로 잡고 있으면 안 된다 — 같은 이유로 손이 발 점군에 섞인다.
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
]
