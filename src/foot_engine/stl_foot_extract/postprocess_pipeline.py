"""GLB(색/텍스처 있는 원본) 하나로 "발 검출 → 정리/스무딩 → 정렬 → 발목 절단 →
해상도 맞춤 → 접지 노드 계산"까지 잇는 후처리 파이프라인 한곳에 모음.

CLI(`cli.py`의 `texture-extract` 서브커맨드)와 다른 스크립트/노트북에서 똑같이
쓸 수 있도록 argparse와 분리해뒀다 -- `process_glb_to_foot()` 하나만 부르면 된다.

단계 요약(각 단계 구현은 다른 모듈에 있음, 여기는 순서만 고정):
    1. `texture_crop.extract_by_skin_vote()` -- 다중뷰 피부분할 투표로 발 부위만 크롭
    2. `finishing.postprocess_mesh()` -- 배경 파편 제거 + 스무딩 + 구멍 메움
    3. `sfm.dense.finalize_mesh()` -- 부유 조각 정리 + 축 정렬 + (선택)발목 절단 +
       스케일 + (선택)해상도 맞춤 다운샘플링+스무딩 + 바닥 접지
    4. `sfm.dense.find_floor_contact_mask()` -- (선택) 접지 노드 마스크
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from foot_engine.sfm.dense import finalize_mesh, find_floor_contact_mask

from .branch_cut import pick_most_foot_like, suggest_bend_components, suggest_neck_components
from .finishing import postprocess_mesh
from .texture_crop import extract_by_skin_vote, load_textured_mesh


@dataclass(slots=True)
class FootPipelineResult:
    """`process_glb_to_foot()` 결과."""

    mesh: trimesh.Trimesh
    n_input_vertices: int
    scale_factor: float | None  #: `align=False`면 None(스케일 안 함).
    up_axis: str  #: "Z" 또는 "Y" -- `floor_contact_mask`가 어느 축 기준인지.
    floor_contact_mask: np.ndarray | None  #: `mesh.vertices`와 1:1 대응하는 불리언 배열.


def crop_foot_mesh(
    mesh_path: str | Path,
    *,
    n_views: int = 16,
    resolution: int = 640,
    vote_threshold: float = 0.5,
    close_gap_radius_mult: float = 25.0,
    two_pass: bool = False,
    coarse_vote_threshold: float = 0.7,
    coarse_pad_ratio: float = 0.2,
    recover_holes: bool = False,
    recover_min_circularity: float = 0.35,
    recover_max_diag_ratio: float = 0.3,
    recover_radius_mult: float = 2.0,
    reject_color_outliers: bool = False,
    reject_min_boundary_crease_ratio: float = 1.3,
    reject_min_cluster_ratio: float = 0.05,
    prune_bent_branches: bool = False,
    prune_bent_branches_kwargs: dict | None = None,
) -> tuple[trimesh.Trimesh, int]:
    """`process_glb_to_foot()`의 1단계(발 검출 크롭)만 떼어낸 것.

    `finish_foot_mesh()`와 짝지어 쓴다 -- 크롭은 다중뷰 렌더링 때문에 느린데
    (실측 20초 안팎), 발바닥 방향 후보 픽커처럼 "크롭 결과는 그대로 두고
    정렬 방향만 바꿔 다시 마무리"가 필요한 경우 크롭을 다시 돌리지 않아도
    되게 분리해뒀다. 인자 설명은 `process_glb_to_foot()` 참고.

    Returns:
        (크롭된 메쉬, 입력 메쉬 정점 수)
    """
    mesh_path = Path(mesh_path)
    mesh = load_textured_mesh(mesh_path)
    n_input = len(mesh.vertices)
    print(f"[foot-pipeline] 입력: {mesh_path} (정점 {n_input:,}개)")

    result = extract_by_skin_vote(
        mesh, n_views=n_views, resolution=(resolution, int(resolution * 0.75)),
        vote_threshold=vote_threshold, close_gap_radius_mult=close_gap_radius_mult,
        two_pass=two_pass, coarse_vote_threshold=coarse_vote_threshold, coarse_pad_ratio=coarse_pad_ratio,
        recover_holes=recover_holes, recover_min_circularity=recover_min_circularity,
        recover_max_diag_ratio=recover_max_diag_ratio, recover_radius_mult=recover_radius_mult,
        reject_color_outliers=reject_color_outliers,
        reject_min_boundary_crease_ratio=reject_min_boundary_crease_ratio,
        reject_min_cluster_ratio=reject_min_cluster_ratio,
    )
    out_mesh = result.mesh

    if prune_bent_branches:
        n_before = len(out_mesh.vertices)
        components = suggest_bend_components(out_mesh, **(prune_bent_branches_kwargs or {}))
        if len(components) > 1:
            chosen = pick_most_foot_like(components)
            out_mesh = chosen.mesh
            print(
                f"[foot-pipeline] 꺾임 감지로 배경/파편 조각 분리: {len(components)}개 중 "
                f"발 모양 점수로 채택(정점 {n_before:,} -> {len(out_mesh.vertices):,}, "
                f"구형성 {chosen.sphericity_score:.2f} + 발가락점수 {chosen.toe_score:.2f})"
            )

    return out_mesh, n_input


def finish_foot_mesh(
    cropped_mesh: trimesh.Trimesh,
    *,
    n_input_vertices: int = 0,
    # 2) 크롭 직후 정리/스무딩
    postprocess: bool = True,
    sand_iterations: int = 3,
    curvature_iterations: int = 150,
    finish_smooth_iterations: int = 10,
    fill_holes_max_diameter_ratio: float = 0.05,
    fill_round_holes_enabled: bool = False,
    fill_round_holes_min_circularity: float = 0.7,
    # 3) 정렬 + 발목 절단 + 해상도 맞춤
    align: bool = True,
    reference_length_mm: float | None = None,
    trim_leg: bool = False,
    z_up: bool = True,
    target_vertices: int | None = None,
    decimate_smooth_after: bool = True,
    down_direction: np.ndarray | None = None,
    prune_neck_fragments: bool = False,
    prune_neck_fragments_kwargs: dict | None = None,
    # 4) 접지 노드
    floor_contact_tolerance_mm: float | None = None,
) -> FootPipelineResult:
    """`process_glb_to_foot()`의 2~4단계(정리 → 정렬 → 접지 노드)만 떼어낸 것.

    `crop_foot_mesh()`가 만든(또는 캐시해둔) 크롭 메쉬를 받아 이어서 처리한다.
    `down_direction`을 지정하면 `finalize_mesh()`의 자동 발바닥 방향 탐색을
    건너뛰고 그 방향을 그대로 쓴다 -- `sole_direction_candidates_for_mesh()`가
    준 후보 중 사람이 고른 걸 넣는 용도(자동 1등이 가끔 틀려서 발이 옆으로
    누운 채 정렬되는 사례 확인, `dense.align_sole_down` 참고). 나머지 인자
    설명은 `process_glb_to_foot()` 참고.
    """
    out_mesh = cropped_mesh

    if postprocess:
        out_mesh, _ = postprocess_mesh(
            out_mesh,
            sand_iterations=sand_iterations,
            curvature_iterations=curvature_iterations,
            finish_smooth_iterations=finish_smooth_iterations,
            fill_holes_max_diameter_ratio=fill_holes_max_diameter_ratio,
            fill_round_holes_enabled=fill_round_holes_enabled,
            fill_round_holes_min_circularity=fill_round_holes_min_circularity,
        )

    scale_factor: float | None = None
    floor_contact_mask: np.ndarray | None = None
    up_axis = "Z" if z_up else "Y"

    if align:
        out_mesh, scale_factor = finalize_mesh(
            out_mesh,
            reference_length_mm=reference_length_mm,
            trim_leg=trim_leg,
            z_up=z_up,
            target_vertices=target_vertices,
            decimate_smooth_after=decimate_smooth_after,
            down_direction=down_direction,
        )
        print(f"[foot-pipeline] 정렬 완료(x{scale_factor:.4f} 스케일, {up_axis}-up, 발바닥={up_axis}0)")

        if prune_neck_fragments:
            n_before = len(out_mesh.vertices)
            components = suggest_neck_components(out_mesh, **(prune_neck_fragments_kwargs or {}))
            if len(components) > 1:
                chosen = pick_most_foot_like(components)
                out_mesh = chosen.mesh
                print(
                    f"[foot-pipeline] 목(neck) 감지로 파편 조각 분리: {len(components)}개 중 "
                    f"발 모양 점수로 채택(정점 {n_before:,} -> {len(out_mesh.vertices):,}, "
                    f"구형성 {chosen.sphericity_score:.2f} + 발가락점수 {chosen.toe_score:.2f})"
                )

        if floor_contact_tolerance_mm is not None:
            floor_contact_mask = find_floor_contact_mask(
                out_mesh, tolerance_mm=floor_contact_tolerance_mm, up_axis=2 if z_up else 1,
            )

    return FootPipelineResult(
        mesh=out_mesh,
        n_input_vertices=n_input_vertices,
        scale_factor=scale_factor,
        up_axis=up_axis,
        floor_contact_mask=floor_contact_mask,
    )


def process_glb_to_foot(
    mesh_path: str | Path,
    *,
    # 1) 발 검출(다중뷰 피부투표 크롭)
    n_views: int = 16,
    resolution: int = 640,
    vote_threshold: float = 0.5,
    close_gap_radius_mult: float = 25.0,
    two_pass: bool = False,
    coarse_vote_threshold: float = 0.7,
    coarse_pad_ratio: float = 0.2,
    recover_holes: bool = False,
    recover_min_circularity: float = 0.35,
    recover_max_diag_ratio: float = 0.3,
    recover_radius_mult: float = 2.0,
    reject_color_outliers: bool = False,
    reject_min_boundary_crease_ratio: float = 1.3,
    reject_min_cluster_ratio: float = 0.05,
    prune_bent_branches: bool = False,
    prune_bent_branches_kwargs: dict | None = None,
    prune_neck_fragments: bool = False,
    prune_neck_fragments_kwargs: dict | None = None,
    # 2) 크롭 직후 정리/스무딩
    postprocess: bool = True,
    sand_iterations: int = 3,
    curvature_iterations: int = 150,
    finish_smooth_iterations: int = 10,
    fill_holes_max_diameter_ratio: float = 0.05,
    fill_round_holes_enabled: bool = False,
    fill_round_holes_min_circularity: float = 0.7,
    # 3) 정렬 + 발목 절단 + 해상도 맞춤
    align: bool = True,
    reference_length_mm: float | None = None,
    trim_leg: bool = False,
    z_up: bool = True,
    target_vertices: int | None = None,
    decimate_smooth_after: bool = True,
    down_direction: np.ndarray | None = None,
    # 4) 접지 노드
    floor_contact_tolerance_mm: float | None = None,
) -> FootPipelineResult:
    """색/텍스처 있는 원본 메쉬(GLB 등) 경로 하나를 받아 발 부위만 검출+정리+정렬까지
    끝낸 최종 메쉬를 반환한다. `crop_foot_mesh()` + `finish_foot_mesh()`를 그대로
    이어 부른 것 -- 각 단계는 독립적으로 끌 수 있다(`postprocess`, `align`,
    `trim_leg`, `target_vertices`, `floor_contact_tolerance_mm`).

    Args:
        two_pass: 배경이 씬 대부분을 차지하는 입력에서 발이 각 렌더 화면의
            일부로만 찍혀 피부투표가 실패하는 문제를 완화하지만, 배경이 적은
            정상 입력에서는 오히려 회귀(가느다란 실 아티팩트)를 만드는 게
            실측으로 확인돼 기본 꺼짐(`extract_by_skin_vote` 참고) -- 배경이
            씬 대부분인 케이스에서만 켜서 쓸 것.
        reject_color_outliers: 색 경계+이면각(꺾임) 신호가 같이 있을 때만
            소재가 다른 조각(의자 등)을 제외한다(`texture_crop.
            _reject_color_material_outliers` 참고) -- 위상/거리 기반
            정리로는 안 갈라지는 케이스를 노릴 수 있지만 아직 실전 검증
            부족, 기본 꺼짐.
        prune_bent_branches: 크롭 직후, 말단에서 진행방향이 꺾이는 지점
            (`branch_cut.suggest_bend_components`)을 찾아 위상적으로 자르고
            발 모양 점수(`branch_cut.pick_most_foot_like`)로 고른 한 조각만
            남긴다 -- 얇은 다리로 이어져 `close_gap`/거리 기반
            정리로는 안 떨어지는 배경 파편에 유효(실측: project_5). 발가락처럼
            진짜 발의 일부까지 갈라낼 위험이 있어(이 기법 과거 다른 용도로
            실패 이력 있음, `branch_cut.py` 참고) 기본 꺼짐 -- 결과를 반드시
            확인하며 쓸 것.
        prune_neck_fragments: `align` 완료 후, 국소 단면 폭이 잘록해졌다가
            다시 넓어지는 지점(`branch_cut.suggest_neck_components`)을 찾아
            그 자리에서 잘라내고 발 모양 점수로 고른 한 조각만 남긴다(2026-09-01부터
            크기 1등 대신 -- 배경 조각이 발보다 커지면 크기 기준 선택은 그걸
            채택하는 구조적 허점이 있어 선제 수정, project_5의 의자 오검출이
            정확히 이 경로였는지는 미확인). 발목 위 다리가
            아니면서(`trim_leg`로도 못 잡음) `prune_far_fragments`(거리
            기반, 발 자체의 길이축 길쭉함과 구분 못 함)로도 안 걸리는
            가늘고 긴 파편에 유효(실측: project 8aba7fd31e6c/46ba7863f4f8,
            발가락처럼 단조롭게 넓어지는 정상 케이스 4개는 오탐 없음 확인).
            기본 꺼짐 -- `prune_bent_branches`처럼 결과를 반드시 확인하며 쓸 것.
        trim_leg: 발목 위 다리까지 찍힌 패턴이 뚜렷하면 잘라낸다(`finalize_mesh`
            참고). 기본 꺼짐.
        target_vertices: 지정하면 이 정점 수 근방까지 단순화+스무딩한다(다른
            데이터셋/모델의 메쉬 해상도에 맞출 때). `align=True`일 때만 적용됨.
        decimate_smooth_after: `finalize_mesh()`로 그대로 전달 -- 축약 직후
            마감 스무딩 여부. `postprocess=True`처럼 이 함수 안에서 이미
            스무딩을 예정해뒀거나, 호출부가 이 결과물에 스무딩 단계를 뒤이어
            돌릴 계획이면 False로 꺼서 중복을 피할 것.
        down_direction: `finish_foot_mesh()`로 그대로 전달 -- 지정하면 발바닥
            방향 자동 탐색 대신 이 방향을 쓴다.
        floor_contact_tolerance_mm: 지정하면 바닥에서 이 거리(mm) 이내 정점을
            표시하는 불리언 마스크를 같이 계산한다. `align=True`일 때만 의미
            있음(스케일이 mm 단위로 맞춰진 뒤라야 거리 임계값이 유효).

    Returns:
        `FootPipelineResult` -- 최종 메쉬 + 부가 정보.
    """
    out_mesh, n_input = crop_foot_mesh(
        mesh_path, n_views=n_views, resolution=resolution, vote_threshold=vote_threshold,
        close_gap_radius_mult=close_gap_radius_mult, two_pass=two_pass,
        coarse_vote_threshold=coarse_vote_threshold, coarse_pad_ratio=coarse_pad_ratio,
        recover_holes=recover_holes, recover_min_circularity=recover_min_circularity,
        recover_max_diag_ratio=recover_max_diag_ratio, recover_radius_mult=recover_radius_mult,
        reject_color_outliers=reject_color_outliers,
        reject_min_boundary_crease_ratio=reject_min_boundary_crease_ratio,
        reject_min_cluster_ratio=reject_min_cluster_ratio,
        prune_bent_branches=prune_bent_branches, prune_bent_branches_kwargs=prune_bent_branches_kwargs,
    )
    return finish_foot_mesh(
        out_mesh, n_input_vertices=n_input,
        postprocess=postprocess, sand_iterations=sand_iterations,
        curvature_iterations=curvature_iterations, finish_smooth_iterations=finish_smooth_iterations,
        fill_holes_max_diameter_ratio=fill_holes_max_diameter_ratio,
        fill_round_holes_enabled=fill_round_holes_enabled,
        fill_round_holes_min_circularity=fill_round_holes_min_circularity,
        align=align, reference_length_mm=reference_length_mm, trim_leg=trim_leg, z_up=z_up,
        target_vertices=target_vertices, decimate_smooth_after=decimate_smooth_after,
        down_direction=down_direction,
        prune_neck_fragments=prune_neck_fragments, prune_neck_fragments_kwargs=prune_neck_fragments_kwargs,
        floor_contact_tolerance_mm=floor_contact_tolerance_mm,
    )


def export_result(result: FootPipelineResult, out_path: str | Path) -> Path:
    """`FootPipelineResult`를 `out_path`에 저장한다. `floor_contact_mask`가 있으면
    `<out_path stem>_floor_contact.npy`로 같이 저장(정점 순서와 1:1 대응).

    Returns:
        메쉬가 저장된 경로(`out_path`).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(out_path)
    print(f"[foot-pipeline] 저장: {out_path} "
          f"(정점 {len(result.mesh.vertices):,}개, 면 {len(result.mesh.faces):,}개)")

    if result.floor_contact_mask is not None:
        mask_path = out_path.with_name(f"{out_path.stem}_floor_contact.npy")
        np.save(mask_path, result.floor_contact_mask)
        n = int(result.floor_contact_mask.sum())
        print(f"[foot-pipeline] 접지 노드 {n:,}/{len(result.floor_contact_mask):,}개 저장: {mask_path}")

    return out_path
