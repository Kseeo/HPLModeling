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

    세 모드:
    1. seed_point/crop_radius 직접 지정(picker.py로 확인한 좌표) -- 가장 신뢰 높음.
    2. auto=True: find_dense_regions() 형상 점수 최상위 구역을 자동 크롭
       (region_index로 다른 구역 선택) -- 휴리스틱이니 결과 확인 필수.
    3. 둘 다 없으면 크롭 없이 구역 목록만 반환(picker.py로 사람이 고름).

    한계: 발과 배경이 밀도 차이 없이 진짜로 fuse된 경우(실측: project_223)는
    auto=True로도 안 갈라짐(원리적 한계) -- picker.py로 좌표 확인 후
    seed_point/crop_radius(+crop.remove_near_point() 국소 보정) 사용.

    Args:
        crop_expansion_mult: auto=True일 때 구역 extent_radius에 곱해 크롭
            반경을 정함 -- 넉넉히 키워야 발 전체가 살아남음(실측 확인).
        postprocess(True): 크롭 후 finishing.postprocess_mesh() 적용 여부.
        locate_kwargs: find_dense_regions()에 그대로 전달할 추가 인자.

    Returns:
        ExtractResult -- 크롭 안 했으면 mesh는 입력 그대로, chosen은 None.
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
