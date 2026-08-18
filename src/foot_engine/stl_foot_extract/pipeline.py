"""`locate.py`(자동 후보/구역 탐색) + `crop.py`(자르기) + `finishing.py`(정리+스무딩)를
엮는 최상위 진입점.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .crop import crop_around_seed
from .finishing import PostprocessStats, postprocess_mesh
from .locate import DenseRegion, find_dense_regions


@dataclass(slots=True)
class ExtractResult:
    """`extract_foot()`의 산출물."""

    mesh: trimesh.Trimesh
    regions: list[DenseRegion]
    chosen: DenseRegion | None
    postprocess_stats: PostprocessStats | None


def extract_foot(
    mesh: trimesh.Trimesh,
    *,
    seed_point: tuple[float, float, float] | np.ndarray | None = None,
    crop_radius: float | None = None,
    auto: bool = False,
    region_index: int = 0,
    crop_expansion_mult: float = 1.6,
    n_regions: int = 5,
    postprocess: bool = True,
    postprocess_kwargs: dict | None = None,
    locate_kwargs: dict | None = None,
) -> ExtractResult:
    """오염된 STL에서 발 부위를 찾아 잘라내고(선택) 정리한다.

    세 가지 모드:
        1. `seed_point`/`crop_radius`를 직접 주면(사람이 `picker.py`로 확인한
           좌표) 그걸로 그대로 크롭한다 -- 가장 신뢰할 수 있는 경로.
        2. `auto=True`면 `find_dense_regions()`(정점 밀도가 높은 표면 샘플점을
           공간적으로 군집화한 "구역", 점 하나+구가 아니라 그 실제 모양을
           따라감) 중 형상 점수 최상위(`region_index`로 다른 구역 선택
           가능)를 자동으로 크롭한다 -- 확정 아닌 휴리스틱 기반이니 결과를
           반드시 눈으로 확인할 것.
        3. 둘 다 안 주면 크롭 없이 구역 목록만 계산해 반환한다(사람이 고르게,
           `picker.py`로 좌표를 직접 짚는 것도 가능).

    **알려진 한계**: 발과 배경이 진짜로 fuse돼 밀도 차이 없이 이어져 있는
    케이스(실측: project_223)는 `auto=True`로도 안 갈라진다 -- 데이터 자체에
    분리 근거가 없는 원리적 한계다. 그런 경우 `picker.py`로 좌표를 직접
    확인해 `seed_point`/`crop_radius`(+`crop.remove_near_point()`로 국소
    보정)를 쓸 것.

    Args:
        crop_expansion_mult: `auto=True`일 때 구역의 `extent_radius`(구역
            중심에서 구역 표본점까지의 최대 거리 -- 구역이 실제로 퍼진 범위)에
            곱해 크롭 반경을 정한다. 구역 자체가 이미 밀도 상위 표본만 모은
            것이라 발등/뒤꿈치처럼 국소 밀도가 살짝 낮게 잡힌 부위가 표본에서
            빠질 수 있는데, `density_radius`(밀도 판정 반경, 훨씬 작음)만큼만
            여유를 두면 그런 부위가 잘려나간다(실측 확인) -- `extent_radius`
            기준으로 넉넉히 키워야 구역 전체(발 전체)가 살아남는다.
        postprocess(True): 크롭 후 `finishing.postprocess_mesh()`까지 적용할지.
        locate_kwargs: `find_dense_regions()`에 그대로 전달할 추가 인자.

    Returns:
        `ExtractResult` -- 크롭 안 했으면 `mesh`는 입력 그대로, `chosen`은 `None`.
    """
    if (seed_point is None) != (crop_radius is None):
        raise ValueError("seed_point와 crop_radius는 둘 다 지정하거나 둘 다 생략해야 합니다.")

    # seed_point를 직접 받은 경로(모드 1)는 구역 탐색 자체가 필요 없다 -- 비용 큰
    # find_dense_regions() 호출을 생략한다.
    regions: list[DenseRegion] = []
    if seed_point is None:
        regions = find_dense_regions(mesh, top_k=n_regions, **(locate_kwargs or {}))

    chosen: DenseRegion | None = None
    out_mesh = mesh

    if seed_point is not None:
        out_mesh, _ = crop_around_seed(mesh, seed_point, crop_radius)
    elif auto:
        if not regions:
            raise RuntimeError("자동 구역을 하나도 못 찾았습니다 -- seed_point를 직접 지정하세요.")
        if region_index >= len(regions):
            raise IndexError(f"region_index={region_index}, 구역은 {len(regions)}개뿐입니다.")
        chosen = regions[region_index]
        radius = chosen.extent_radius * crop_expansion_mult
        print(
            f"[pipeline] 자동 선택 구역 #{region_index+1}: score={chosen.score:.3f} "
            f"n_points={chosen.n_points} centroid={chosen.centroid.round(4).tolist()} "
            f"extent_radius={chosen.extent_radius:.5g} crop_radius={radius:.5g} "
            "-- 확정 아닌 휴리스틱 선택이니 결과를 확인할 것"
        )
        out_mesh, _ = crop_around_seed(mesh, chosen.centroid, radius)

    stats = None
    if postprocess:
        out_mesh, stats = postprocess_mesh(out_mesh, **(postprocess_kwargs or {}))

    return ExtractResult(mesh=out_mesh, regions=regions, chosen=chosen, postprocess_stats=stats)
