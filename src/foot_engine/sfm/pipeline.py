"""사진/영상 한 벌 -> 변형된 발 메쉬. 앞선 다섯 모듈을 한 번에 엮는 오케스트레이션.

    frame_quality.assess_frames()   — SfM 전 프레임별 절대기준 QC
        └─ reconstruction.run_sparse_sfm()
              └─ masking.generate_masks()  — 여기서 발 미검출 프레임을 추가로 제외
                    └─ cleaning.clean_point_cloud()
                          └─ fitting.fit_point_cloud_to_template()

각 단계의 튜닝 근거/기본값은 해당 모듈 docstring 참고. 이 함수는 그것들을
엮기만 하고 새 로직은 추가하지 않는다 — 단계별로 따로 돌리고 싶으면(중간
결과를 실제 뷰어로 확인하는 등) 각 모듈을 직접 불러도 된다, 이 함수는
편의용이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from ..deformer import FootMeshDeformer
from ..exceptions import CaptureQualityError
from ..schemas import FootMeasurements
from . import cleaning, frame_quality, masking, reconstruction
from .fitting import fit_point_cloud_to_template

#: SfM에 넘길 최소 프레임 수 — `reconstruction.extract_frames()`의 "최소 8장
#: 권장" 가이드와 맞춘다. QC 게이트 통과 후 이보다 적게 남으면 비싼 SfM을
#: 돌리기 전에 명확한 에러로 끊는다.
MIN_CANDIDATE_FRAMES = 8


@dataclass(slots=True)
class PipelineResult:
    """`run_pipeline()`의 산출물 경로 + 요약 통계."""

    images_dir: Path
    masks_dir: Path
    sparse_points_path: Path
    cleaned_points_path: Path
    output_mesh_path: Path
    measurements: FootMeasurements
    n_points_registered_images: int
    n_points_total_images: int
    n_points_raw: int
    n_points_cleaned: int
    quality_stats: dict
    mask_stats: dict


def run_pipeline(
    *,
    workdir: Path,
    template_path: Path,
    out_mesh: Path,
    video: Path | None = None,
    images_dir: Path | None = None,
    side: str | None = None,
    template_side: str = "right",
    interval: float = 0.5,
    start_time: float = 0.0,
    end_time: float | None = None,
    quality_gate: bool = True,
    min_sharpness: float | None = None,
    min_frames: int = MIN_CANDIDATE_FRAMES,
    blur_keep_ratio: float = 1.0,
    max_features: int = reconstruction.DEFAULT_MAX_FEATURES,
    peak_threshold: float = reconstruction.DEFAULT_PEAK_THRESHOLD,
    ransac_max_error: float = reconstruction.DEFAULT_RANSAC_MAX_ERROR,
    mask_during_extraction: bool = False,
    skin_refine: bool = True,
    skin_erode: int = 8,
    mask_dilate: int = 15,
    cluster: bool = True,
    rng_seed: int = 0,
) -> PipelineResult:
    """영상/사진 -> 발/피부 마스크 -> sparse SfM -> 배경 제거 -> 템플릿 워프.

    Args:
        workdir: 중간 산출물(프레임, 마스크, DB, sparse 복원, 점군)을 저장할 폴더.
        template_path: 기준 발 템플릿 STL.
        out_mesh: 최종 변형된 메쉬 저장 경로(.stl).
        video / images_dir: 둘 중 하나만 지정. `video`면 프레임을 먼저 추출한다.
        side / template_side: 좌우(카이랄리티) — `fitting.fit_point_cloud_to_template()`
            참고. 실제 발과 템플릿의 좌우를 알고 있다면 반드시 `side`를 넘길 것.
        mask_during_extraction: 특징점 추출 단계에서도 마스크를 적용할지.
            기본 False — 배경이 피사체와 함께 고정된 텍스처 있는 촬영에서는
            오히려 등록률을 깎는 게 실측으로 확인됐다(`reconstruction.
            run_sparse_sfm` docstring 참고). 배경이 복잡/혼재할 때만 True로.
        quality_gate: SfM 전에 프레임별 절대기준 QC(`frame_quality.py` — 파일
            손상/해상도/노출)를 적용할지. 기본 True.
        min_sharpness: QC의 블러 절대 임계값. None(기본)이면 블러 검사는
            건너뛴다 — `frame_quality.py` docstring 참고, 카메라마다 다른
            선명도 스케일을 실측 없이 임의로 정하지 않았다.
        min_frames: QC/마스킹 게이트를 통과해야 하는 최소 프레임 수. 이보다
            적게 남으면 SfM을 돌리지 않고 `CaptureQualityError`를 던진다.
        cluster: 정리된 점군에 DBSCAN 군집화(가장 큰 덩어리만 유지)를 추가로
            적용할지. 마스크 정리 이후 적용은 안전하다고 실측 확인됨(기본 True).

    Returns:
        산출물 경로와 요약 통계를 담은 `PipelineResult`.
    """
    if (video is None) == (images_dir is None):
        raise ValueError("video와 images_dir 중 정확히 하나만 지정해야 합니다.")

    workdir.mkdir(parents=True, exist_ok=True)

    if video is not None:
        resolved_images_dir = reconstruction.extract_frames(
            video, workdir / "images", interval, start_time=start_time, end_time=end_time
        )
    else:
        resolved_images_dir = images_dir
        if not resolved_images_dir.is_dir():
            raise FileNotFoundError(f"이미지 폴더가 없습니다: {resolved_images_dir}")

    all_names = sorted(
        p.name for p in resolved_images_dir.iterdir() if p.suffix.lower() in reconstruction.IMAGE_EXTS
    )

    quality_stats: dict = {}
    if quality_gate:
        quality_results = frame_quality.assess_frames(
            resolved_images_dir, all_names, min_sharpness=min_sharpness
        )
        frame_quality.print_summary(quality_results)
        quality_stats = frame_quality.summarize(quality_results)
        survivors = frame_quality.passed_names(quality_results)
        if len(survivors) < min_frames:
            raise CaptureQualityError(
                f"품질 게이트를 통과한 프레임이 {len(survivors)}장뿐입니다"
                f"(최소 {min_frames}장 필요) — 초점/노출/촬영 구도를 확인하세요.",
                detail=quality_stats,
            )
    else:
        survivors = all_names

    candidate_names = survivors
    if blur_keep_ratio < 1.0:
        candidate_names = reconstruction.filter_blurry_frames(
            resolved_images_dir, blur_keep_ratio, names=survivors
        )

    masks_dir = workdir / "masks"
    mask_stats = masking.generate_masks(
        resolved_images_dir, masks_dir, names=candidate_names,
        dilate=mask_dilate, skin_refine=skin_refine, skin_erode=skin_erode,
    )
    print(
        f"[마스크] {mask_stats['total']}장 처리 "
        f"(원본 전체 폴백 {mask_stats['fallback']}장, 피부 정제 적용 {mask_stats['refined']}장, "
        f"발 미검출 제외 {mask_stats['rejected']}장)"
    )
    if mask_stats["rejected"]:
        rejected = set(mask_stats["rejected_names"])
        candidate_names = [name for name in candidate_names if name not in rejected]
        if len(candidate_names) < min_frames:
            raise CaptureQualityError(
                f"마스킹까지 마친 후 남은 프레임이 {len(candidate_names)}장뿐입니다"
                f"(최소 {min_frames}장 필요) — 촬영 구도(발이 프레임 중앙에 있는지)를 확인하세요.",
                detail=mask_stats,
            )

    # candidate_names가 all_names 전부와 같더라도 항상 명시적으로 넘긴다 — None으로
    # 넘기면 pycolmap이 `images_dir` 안의 파일을 전부 다시 스캔해 QC/마스킹에서
    # 제외된 프레임까지 도로 섞여 들어간다.
    recon = reconstruction.run_sparse_sfm(
        resolved_images_dir, workdir, image_names=candidate_names,
        masks_dir=masks_dir if mask_during_extraction else None,
        max_features=max_features, peak_threshold=peak_threshold, ransac_max_error=ransac_max_error,
    )
    reconstruction.report_unregistered_frames(resolved_images_dir, recon, candidate_names)

    sparse_points_path = workdir / "sparse_points.ply"
    raw_points, _removed = reconstruction.export_point_cloud(recon, sparse_points_path)
    print(
        f"\n[SfM 결과] 등록 {recon.num_reg_images()}/{len(candidate_names)}장, "
        f"3D점 {len(raw_points):,}개, 평균 재투영 오차 {recon.compute_mean_reprojection_error():.3f}px"
    )

    cleaned_points = cleaning.clean_point_cloud(recon, masks_dir=masks_dir, cluster=cluster)
    cleaned_points_path = workdir / "cleaned_points.ply"
    cleaned_points_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(cleaned_points).export(cleaned_points_path)

    engine = FootMeshDeformer(template_path)
    deformed, measured = fit_point_cloud_to_template(
        cleaned_points, engine, side=side, template_side=template_side, rng_seed=rng_seed,
    )
    out_mesh.parent.mkdir(parents=True, exist_ok=True)
    deformed.export(out_mesh)

    report = engine.last_report
    if report is not None:
        print("\n" + "\n".join(report.summary_lines()))
        for warning in report.warnings:
            print(f"[warn] {warning}")
    print(f"\n저장: {out_mesh}")

    return PipelineResult(
        images_dir=resolved_images_dir,
        masks_dir=masks_dir,
        sparse_points_path=sparse_points_path,
        cleaned_points_path=cleaned_points_path,
        output_mesh_path=out_mesh,
        measurements=measured,
        n_points_registered_images=recon.num_reg_images(),
        n_points_total_images=len(candidate_names),
        n_points_raw=len(raw_points),
        n_points_cleaned=len(cleaned_points),
        quality_stats=quality_stats,
        mask_stats=mask_stats,
    )
