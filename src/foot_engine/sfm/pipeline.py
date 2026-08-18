"""사진/영상 한 벌 -> dense MVS 발 메쉬. 앞선 모듈들을 엮는 오케스트레이션.

    frame_quality.assess_frames() — 프레임 QC
        └─ masking.generate_masks() — 마스크 생성 + 발 미검출 프레임 제외
              └─ reconstruction.run_sparse_sfm() — sparse SfM
                    └─ cleaning.clean_point_cloud() — QA용 정리(메쉬엔 안 씀)
                    └─ dense.run_dense_pipeline() — dense 메쉬 생성 + 스케일 보정
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import trimesh

from ..exceptions import CaptureQualityError
from . import cleaning, dense, frame_quality, masking, reconstruction
from .dense import DEFAULT_REFERENCE_LENGTH_MM

#: SfM에 넘길 최소 프레임 수 — `reconstruction.extract_frames()`의 "최소 8장
#: 권장" 가이드와 맞춘다. QC 게이트 통과 후 이보다 적게 남으면 비싼 SfM을
#: 돌리기 전에 명확한 에러로 끊는다.
MIN_CANDIDATE_FRAMES = 8


@dataclass(slots=True)
class PipelineResult:
    """`run_pipeline()`의 산출물 경로 + 요약 통계."""

    images_dir: Path
    masks_dir: Path
    dense_masks_dir: Path
    sparse_points_path: Path
    cleaned_points_path: Path
    output_mesh_path: Path
    n_points_registered_images: int
    n_points_total_images: int
    n_points_raw: int
    n_points_cleaned: int
    n_mesh_vertices: int
    n_mesh_faces: int
    scale_factor: float
    reference_length_mm: float | None
    quality_stats: dict
    mask_stats: dict
    dense_mask_stats: dict


def run_pipeline(
    *,
    workdir: Path,
    out_mesh: Path,
    video: Path | None = None,
    images_dir: Path | None = None,
    reference_length_mm: float | None = None,
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
    openmvs_bin: str | Path | None = None,
    refine: bool = False,
    postprocess_dmaps: int = dense.DEFAULT_POSTPROCESS_DMAPS,
    dense_max_threads: int = dense.DEFAULT_MAX_THREADS,
    densify_resolution_level: int | None = None,
    densify_number_views_fuse: int | None = None,
    visibility_filter_threshold: int | None = None,
    grazing_filter_min_score: float | None = None,
    reprojection_consistency_min_vote: float | None = None,
    free_space_support: bool = False,
    thickness_factor: float = 1.0,
    quality_factor: float = 1.0,
    refine_decimate: float = 1.0,
    refine_regularity_weight: float | None = None,
    smooth_high_curvature: bool = True,
    curvature_percentile: float = dense.DEFAULT_CURVATURE_PERCENTILE,
    curvature_min_radius_mult: float = dense.DEFAULT_CURVATURE_MIN_RADIUS_MULT,
    curvature_max_radius_mult: float = dense.DEFAULT_CURVATURE_MAX_RADIUS_MULT,
    curvature_iterations: int = dense.DEFAULT_CURVATURE_ITERATIONS,
    curvature_alpha: float = dense.DEFAULT_CURVATURE_ALPHA,
    curvature_mu: float = dense.DEFAULT_CURVATURE_MU,
    fill_holes: bool = True,
    sand_surface_enabled: bool = True,
    sand_min_neighbors: int = 16,
    sand_max_neighbors: int = 32,
    sand_iterations: int = 3,
    finish_smooth: bool = True,
    finish_smooth_lambda: float = 0.5,
    finish_smooth_iterations: int = 40,
    prune_protrusions: bool = False,
    trim_leg: bool = False,
    keep_intermediates: bool = False,
) -> PipelineResult:
    """영상/사진 -> 발/피부 마스크 -> sparse SfM -> dense MVS -> 스케일 보정 메쉬.

    Args:
        workdir(필수): 중간 산출물 저장 폴더.
        out_mesh(필수): 최종 메쉬 저장 경로.
        video / images_dir(필수, 택1): video면 프레임 먼저 추출.
        reference_length_mm(None): 자기신고 발길이(mm), 스케일 기준. 없으면
            `DEFAULT_REFERENCE_LENGTH_MM` placeholder 사용.
        interval(0.5): 영상 프레임 추출 간격(초).
        start_time/end_time(0.0/None): 사용할 영상 구간.
        quality_gate(True): SfM 전 프레임별 절대기준 QC 적용 여부.
        min_sharpness(None): QC 블러 절대 임계값, 없으면 검사 생략.
        min_frames(8): 이보다 적게 남으면 `CaptureQualityError`.
        blur_keep_ratio(1.0): 선명도 상위 몇 %만 쓸지.
        max_features/peak_threshold/ransac_max_error: SIFT/RANSAC 튜닝값.
        mask_during_extraction(False): 특징점 추출 단계에도 마스크 적용할지.
        skin_refine(True): MediaPipe 피부 정제(옷/장신구 제외) 여부.
        skin_erode(8): 피부 마스크 경계 침식 폭(px).
        mask_dilate(15): sparse용 마스크 팽창 폭(px).
        cluster(True): QA용 `cleaned_points.ply`에 DBSCAN 적용 여부.
        openmvs_bin(None): OpenMVS 실행파일 폴더.
        refine(False): RefineMesh(느림) 실행 여부.
        densify_resolution_level(None)/densify_number_views_fuse(None): 점군 밀도
            튜닝, `dense.run_dense_pipeline()`으로 그대로 전달.
        postprocess_dmaps/dense_max_threads/visibility_filter_threshold/
        grazing_filter_min_score/reprojection_consistency_min_vote/
        free_space_support/thickness_factor/quality_factor/refine_decimate/
        refine_regularity_weight/smooth_high_curvature/curvature_percentile/
        curvature_min_radius_mult/curvature_max_radius_mult/curvature_iterations/
        curvature_alpha/curvature_mu/fill_holes/
        sand_surface_enabled/finish_smooth/finish_smooth_lambda/
        finish_smooth_iterations/prune_protrusions: `dense.run_dense_pipeline()`으로
            그대로 전달(각 인자 설명은 그쪽 docstring 참고).
        trim_leg(False): `dense.finalize_mesh()`로 그대로 전달 -- 다리 포함
            케이스 자동 트림, 그쪽 docstring 참고.
        keep_intermediates(False): 성공 후 중간 산출물 정리 여부. `True`면
            `run_dense_pipeline.py`로 dense 파라미터 재튜닝 가능.

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

    # dense용 마스크(dilate=0)도 여기서 같이 만든다 — rembg/피부 정제 추론(비싼
    # 부분)은 판정에 dilate가 영향을 주지 않으므로 한 번만 돌리고, 팽창 폭만
    # 다르게 두 벌 저장한다(masking.generate_masks() docstring 참고). 예전에는
    # SfM 이후 dense_masks_dir을 위해 같은 추론을 통째로 다시 돌렸었다.
    masks_dir = workdir / "masks"
    dense_masks_dir = workdir / "masks_dense"
    mask_stats = masking.generate_masks(
        resolved_images_dir, masks_dir, names=candidate_names,
        dilate=mask_dilate, skin_refine=skin_refine, skin_erode=skin_erode,
        extra_dilations=[(dense_masks_dir, 0)],
    )
    dense_mask_stats = mask_stats  # 같은 판정을 공유(위 참고) -- 별도로 다시 돌리지 않음
    print(
        f"[마스크] {mask_stats['total']}장 처리 "
        f"(피부 정제 적용 {mask_stats['refined']}장, 제외 {mask_stats['rejected']}장 "
        f"— 발 미검출 {mask_stats['rejected_reasons']['no_foot']}, "
        f"세그멘테이션 애매 {mask_stats['rejected_reasons']['low_coverage']}, "
        f"피부 정제 붕괴 {mask_stats['rejected_reasons']['skin_refine_collapsed']})"
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

    # QA용 — 최종 dense 메쉬 생성에는 안 쓰인다(마스크 기반 배경 제거는
    # dense.run_dense_pipeline() 내부에서 densify 이전에 따로 적용된다).
    cleaned_points = cleaning.clean_point_cloud(recon, masks_dir=masks_dir, cluster=cluster)
    cleaned_points_path = workdir / "cleaned_points.ply"
    cleaned_points_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(cleaned_points).export(cleaned_points_path)

    sparse_dir = dense.largest_sparse_dir(workdir / "sparse")
    dense_workdir = workdir / "dense_mvs"
    mesh_ply = dense.run_dense_pipeline(
        sparse_dir=sparse_dir, images_dir=resolved_images_dir, masks_dir=dense_masks_dir,
        workdir=dense_workdir, openmvs_bin=openmvs_bin, refine=refine,
        postprocess_dmaps=postprocess_dmaps, max_threads=dense_max_threads,
        densify_resolution_level=densify_resolution_level,
        densify_number_views_fuse=densify_number_views_fuse,
        visibility_filter_threshold=visibility_filter_threshold,
        grazing_filter_min_score=grazing_filter_min_score,
        reprojection_consistency_min_vote=reprojection_consistency_min_vote,
        free_space_support=free_space_support, thickness_factor=thickness_factor,
        quality_factor=quality_factor, refine_decimate=refine_decimate,
        refine_regularity_weight=refine_regularity_weight,
        smooth_high_curvature=smooth_high_curvature,
        curvature_percentile=curvature_percentile,
        curvature_min_radius_mult=curvature_min_radius_mult, curvature_max_radius_mult=curvature_max_radius_mult,
        curvature_iterations=curvature_iterations, curvature_alpha=curvature_alpha,
        curvature_mu=curvature_mu,
        fill_holes=fill_holes, sand_surface_enabled=sand_surface_enabled,
        sand_min_neighbors=sand_min_neighbors, sand_max_neighbors=sand_max_neighbors,
        sand_iterations=sand_iterations,
        finish_smooth=finish_smooth, finish_smooth_lambda=finish_smooth_lambda,
        finish_smooth_iterations=finish_smooth_iterations,
        prune_protrusions=prune_protrusions,
        keep_intermediates=keep_intermediates,
    )

    # 부유 파편 제거(keep_largest_component)는 dense.run_dense_pipeline() 안에서
    # 이미 적용되어 mesh_ply에 반영돼 있다 — 여기서 다시 호출하면 이미 단일
    # 덩어리인 메쉬에 대한 무의미한 재호출이 된다.
    mesh = trimesh.load(mesh_ply, process=False)

    # 축 정렬(X=길이, Y=높이·발바닥은 -Y) + 스케일링 + 바닥 정착.
    # `run_dense_pipeline.py`도 같은 후처리를 쓴다(`dense.finalize_mesh()` 참고).
    mesh, scale_factor = dense.finalize_mesh(
        mesh, reference_length_mm=reference_length_mm, trim_leg=trim_leg,
    )

    out_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_mesh)
    print(f"\n저장: {out_mesh} (정점 {len(mesh.vertices):,}개, 면 {len(mesh.faces):,}개)")

    if not keep_intermediates:
        # out_mesh(위에서 이미 저장됨)만 남기고 workdir 스크래치는 전부 지운다.
        # video로 받은 프레임(workdir/images)만 지우고, images_dir로 사용자가
        # 직접 넘긴 외부 폴더는 우리 소유가 아니므로 건드리지 않는다.
        cleanup_targets = [
            workdir / "database.db", workdir / "sparse", masks_dir, dense_masks_dir,
            sparse_points_path, cleaned_points_path, dense_workdir,
        ]
        if video is not None:
            cleanup_targets.append(resolved_images_dir)
        for target in cleanup_targets:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.is_file():
                target.unlink()
        print(f"[정리] keep_intermediates=False -- {workdir} 중간 산출물 삭제 (최종 결과는 {out_mesh})")

    return PipelineResult(
        images_dir=resolved_images_dir,
        masks_dir=masks_dir,
        dense_masks_dir=dense_masks_dir,
        sparse_points_path=sparse_points_path,
        cleaned_points_path=cleaned_points_path,
        output_mesh_path=out_mesh,
        n_points_registered_images=recon.num_reg_images(),
        n_points_total_images=len(candidate_names),
        n_points_raw=len(raw_points),
        n_points_cleaned=len(cleaned_points),
        n_mesh_vertices=len(mesh.vertices),
        n_mesh_faces=len(mesh.faces),
        scale_factor=scale_factor,
        reference_length_mm=reference_length_mm,
        quality_stats=quality_stats,
        mask_stats=mask_stats,
        dense_mask_stats=dense_mask_stats,
    )
