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


def process_glb_to_foot(
    mesh_path: str | Path,
    *,
    # 1) 발 검출(다중뷰 피부투표 크롭)
    n_views: int = 16,
    resolution: int = 640,
    vote_threshold: float = 0.5,
    close_gap_radius_mult: float = 12.0,
    # 2) 크롭 직후 정리/스무딩
    postprocess: bool = True,
    sand_iterations: int = 3,
    curvature_iterations: int = 150,
    finish_smooth_iterations: int = 40,
    fill_holes_max_diameter_ratio: float = 0.05,
    fill_round_holes_enabled: bool = False,
    fill_round_holes_min_circularity: float = 0.7,
    # 3) 정렬 + 발목 절단 + 해상도 맞춤
    align: bool = True,
    reference_length_mm: float | None = None,
    trim_leg: bool = False,
    z_up: bool = True,
    target_vertices: int | None = None,
    # 4) 접지 노드
    floor_contact_tolerance_mm: float | None = None,
) -> FootPipelineResult:
    """색/텍스처 있는 원본 메쉬(GLB 등) 경로 하나를 받아 발 부위만 검출+정리+정렬까지
    끝낸 최종 메쉬를 반환한다. 각 단계는 독립적으로 끌 수 있다(`postprocess`,
    `align`, `trim_leg`, `target_vertices`, `floor_contact_tolerance_mm`).

    Args:
        trim_leg: 발목 위 다리까지 찍힌 패턴이 뚜렷하면 잘라낸다(`finalize_mesh`
            참고). 기본 꺼짐.
        target_vertices: 지정하면 이 정점 수 근방까지 단순화+스무딩한다(다른
            데이터셋/모델의 메쉬 해상도에 맞출 때). `align=True`일 때만 적용됨.
        floor_contact_tolerance_mm: 지정하면 바닥에서 이 거리(mm) 이내 정점을
            표시하는 불리언 마스크를 같이 계산한다. `align=True`일 때만 의미
            있음(스케일이 mm 단위로 맞춰진 뒤라야 거리 임계값이 유효).

    Returns:
        `FootPipelineResult` -- 최종 메쉬 + 부가 정보.
    """
    mesh_path = Path(mesh_path)
    mesh = load_textured_mesh(mesh_path)
    n_input = len(mesh.vertices)
    print(f"[foot-pipeline] 입력: {mesh_path} (정점 {n_input:,}개)")

    result = extract_by_skin_vote(
        mesh, n_views=n_views, resolution=(resolution, int(resolution * 0.75)),
        vote_threshold=vote_threshold, close_gap_radius_mult=close_gap_radius_mult,
    )
    out_mesh = result.mesh

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
        )
        print(f"[foot-pipeline] 정렬 완료(x{scale_factor:.4f} 스케일, {up_axis}-up, 발바닥={up_axis}0)")

        if floor_contact_tolerance_mm is not None:
            floor_contact_mask = find_floor_contact_mask(
                out_mesh, tolerance_mm=floor_contact_tolerance_mm, up_axis=2 if z_up else 1,
            )

    return FootPipelineResult(
        mesh=out_mesh,
        n_input_vertices=n_input,
        scale_factor=scale_factor,
        up_axis=up_axis,
        floor_contact_mask=floor_contact_mask,
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
