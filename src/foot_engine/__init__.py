"""foot_engine — 2D 사진/영상 -> dense MVS 3D 발 메쉬 파이프라인.

Quick start::

    from foot_engine.sfm.pipeline import run_pipeline

    result = run_pipeline(video_path="data/samples/test00.mp4", workdir="data/output/run00")

랜드마크 기반 템플릿 워프(`FootMeshDeformer`)/SSM 경로는 2026-08-11 다음
결론에 따라 `archive/deformer_ssm_pipeline/`로 옮겼다: SSM은 대응점 노이즈로
메쉬 생성기로 못 쓴다는 게 이미 확인돼 있었고, 템플릿 워프까지 유지할 이유가
약해져 dense MVS(`foot_engine.sfm`) 결과물 자체를 다듬고 경량화하는 쪽이
낫다고 판단함. 코드는 지우지 않고 archive/에 그대로 남아 있다(참고/복구용).
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
