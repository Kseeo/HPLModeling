"""외부에서 받은 완성된 STL(스캔 품질은 좋지만 배경/다른 물체가 같이 찍힌 경우)에서
발 부위를 찾아 잘라내고 다듬는 독립 패키지.

`foot_engine.sfm`(사진 -> 메쉬 생성)과 완전히 분리돼 있다 -- 입력은 오직 메쉬
파일(STL 등) 하나뿐이고, 사진/카메라/`scripts/` 쪽 공용 유틸에 의존하지 않는다.

하위 모듈:
    locate.py   -- 발 후보 위치 자동 탐색(밀도 + 형상 + 말단 다지 구조 점수)
    crop.py     -- 씨앗점/평면 기반 수동 크롭 도구(사람이 좌표를 짚어 자름)
    finishing.py -- 배경 파편 정리 + 스무딩(mesh_postprocess.py와 같은 계열)
    picker.py   -- 로컬 브라우저 씨앗점 피커(HTML, 자동 후보 오버레이)
    pipeline.py -- 위 전부를 엮는 `extract_foot()`
    cli.py      -- `python -m foot_engine.stl_foot_extract.cli` CLI

사용 예::

    from foot_engine.stl_foot_extract import extract_foot
    import trimesh

    mesh = trimesh.load("project_223.stl", process=True)
    result = extract_foot(mesh)
    result.mesh.export("project_223_clean.stl")
"""

from __future__ import annotations

from .crop import crop_around_seed, crop_to_region, remove_near_point, trim_by_plane
from .finishing import (
    finish_smooth_mesh,
    keep_largest_component,
    postprocess_mesh,
)
from .locate import DenseRegion, FootCandidate, find_dense_regions, suggest_foot_regions
from .pipeline import ExtractResult, extract_foot

__all__ = [
    "extract_foot",
    "ExtractResult",
    "find_dense_regions",
    "DenseRegion",
    "suggest_foot_regions",
    "FootCandidate",
    "crop_around_seed",
    "crop_to_region",
    "remove_near_point",
    "trim_by_plane",
    "keep_largest_component",
    "finish_smooth_mesh",
    "postprocess_mesh",
]
