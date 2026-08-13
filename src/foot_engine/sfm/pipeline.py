"""사진/영상 한 벌 -> dense MVS 발 메쉬. 앞선 모듈들을 한 번에 엮는 오케스트레이션.

    frame_quality.assess_frames()  — SfM 전 프레임별 절대기준 QC
        └─ masking.generate_masks()  — sparse용(dilate=15)+dense용(dilate=0)
              마스크를 한 번에 생성(추론 중복 방지, extra_dilations 참고).
              발 미검출 프레임을 여기서 추가로 제외.
                └─ reconstruction.run_sparse_sfm()
                      └─ cleaning.clean_point_cloud()  — QA용 sparse 정리(최종 메쉬엔 안 씀)
                      └─ dense.run_dense_pipeline()  — OpenMVS densify + 메싱 + 파편 제거
                            └─ (선택) 스케일 보정 — geometry.measured_length() 기준
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
    curvature_rings: int = dense.DEFAULT_CURVATURE_RINGS,
    curvature_iterations: int = dense.DEFAULT_CURVATURE_ITERATIONS,
    curvature_alpha: float = dense.DEFAULT_CURVATURE_ALPHA,
    fill_holes: bool = True,
    sand_surface_enabled: bool = True,
    prune_protrusions: bool = False,
    keep_intermediates: bool = False,
) -> PipelineResult:
    """영상/사진 -> 발/피부 마스크 -> sparse SfM -> dense MVS -> 스케일 보정 메쉬.

    Args:
        workdir: 중간 산출물(프레임, 마스크, DB, sparse/dense 복원)을 저장할 폴더.
        out_mesh: 최종 dense 메쉬 저장 경로(.ply/.stl 등 trimesh가 지원하는 형식).
        video / images_dir: 둘 중 하나만 지정. `video`면 프레임을 먼저 추출한다.
        reference_length_mm: 자기신고 발길이(mm). 있으면 최종 메쉬를 이 길이에
            맞춰 스케일링한다(SfM은 절대 축척이 없다). 없으면
            `DEFAULT_REFERENCE_LENGTH_MM` placeholder로 스케일링하고 경고를
            출력한다 — 진짜 mm 크기가 아니므로 실사용 전 반드시 확인할 것.
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
        cluster: `cleaned_points.ply`(QA용 sparse 정리 산출물)에 DBSCAN
            군집화를 추가로 적용할지. 최종 dense 메쉬에는 영향 없다 — dense
            경로는 DBSCAN을 의도적으로 안 쓴다(`dense.py` docstring 3번 참고).
        openmvs_bin / refine / postprocess_dmaps / dense_max_threads /
        visibility_filter_threshold / grazing_filter_min_score /
        reprojection_consistency_min_vote / free_space_support /
        thickness_factor / quality_factor / refine_decimate /
        refine_regularity_weight / smooth_high_curvature /
        curvature_percentile / curvature_rings / curvature_iterations /
        curvature_alpha / fill_holes / sand_surface_enabled /
        prune_protrusions:
            `dense.run_dense_pipeline()`으로 그대로 전달. 튜닝 근거는
            `dense.py` 모듈 docstring 및 `dense_mvs_results/README.md` 참고.
            `refine_decimate`(기본 1=해상도 보존), `smooth_high_curvature`/
            `curvature_*`(기본 켜짐, 2026-08-12 강화), `fill_holes`/
            `sand_surface_enabled`(기본 켜짐)는 실측 검증된 기본값이다.
            `reprojection_consistency_min_vote`(배경 오염 제거)와 `refine`
            (느림)은 다른 촬영본에서도 안전한지 아직 test03 1건만 검증돼
            기본 꺼짐. free_space_support/thickness_factor는 실측에서
            부작용(메쉬 뒤틀림)만 확인돼 기본값(꺼짐/1.0)을 건드리지 말 것.
            `prune_protrusions`은 실행 간 편차가 커 기본 꺼짐(dense.py 참고).
        keep_intermediates: `False`(기본)이면 성공 후 `workdir` 안의 모든
            중간 산출물(추출 프레임, 마스크, DB, sparse 재구성, QA용 sparse
            점군, dense MVS 스크래치)을 지운다 — 남는 건 `out_mesh` 하나뿐.
            `run_dense_pipeline.py`로 dense 파라미터만 다시 튜닝하려면
            이 폴더의 `images/`/`masks_dense/`/`sparse/`가 남아있어야 하므로
            `True`로 켤 것.

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
        visibility_filter_threshold=visibility_filter_threshold,
        grazing_filter_min_score=grazing_filter_min_score,
        reprojection_consistency_min_vote=reprojection_consistency_min_vote,
        free_space_support=free_space_support, thickness_factor=thickness_factor,
        quality_factor=quality_factor, refine_decimate=refine_decimate,
        refine_regularity_weight=refine_regularity_weight,
        smooth_high_curvature=smooth_high_curvature,
        curvature_percentile=curvature_percentile, curvature_rings=curvature_rings,
        curvature_iterations=curvature_iterations, curvature_alpha=curvature_alpha,
        fill_holes=fill_holes, sand_surface_enabled=sand_surface_enabled,
        prune_protrusions=prune_protrusions,
        keep_intermediates=keep_intermediates,
    )

    # 부유 파편 제거(keep_largest_component)는 dense.run_dense_pipeline() 안에서
    # 이미 적용되어 mesh_ply에 반영돼 있다 — 여기서 다시 호출하면 이미 단일
    # 덩어리인 메쉬에 대한 무의미한 재호출이 된다.
    mesh = trimesh.load(mesh_ply, process=False)

    # 축 정렬(X=길이) + 발바닥 검출(Y=높이, 발바닥이 -Y, 접지) + 스케일링 —
    # 템플릿이 없어져 앞/뒤(발끝 방향)는 여전히 못 정하지만, 평탄도 비대칭
    # 휴리스틱으로 위/아래는 결정한다(`dense.align_sole_down()` docstring
    # 참고, 2026-08-11 실측: test03에서 신호 확인됨 — 매 실행 검증된 건
    # 아니라 실사용 전 뷰어로 확인할 것). `dense.run_dense_pipeline.py`
    # 단독 재실행 스크립트도 같은 후처리를 쓴다(`dense.finalize_mesh()` 참고).
    mesh, scale_factor = dense.finalize_mesh(mesh, reference_length_mm=reference_length_mm)

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
