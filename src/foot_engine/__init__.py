"""foot_engine — 2D 사진/영상 -> dense MVS 3D 발 메쉬 파이프라인.

Quick start::

    from foot_engine.sfm.pipeline import run_pipeline

    result = run_pipeline(video_path="data/samples/test00.mp4", workdir="data/output/run00")

템플릿 워프(`FootMeshDeformer`)/SSM 경로는 `archive/deformer_ssm_pipeline/`로
옮겼다 — 참고/복구용으로 코드는 남아 있다.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
