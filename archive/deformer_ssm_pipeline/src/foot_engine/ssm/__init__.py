"""통계적 형상 모델(SSM) 구축 파이프라인.

    preprocessing.clean_and_normalize()  — 스캔 1개: 노이즈 제거 + 자기 길이로 정규화
        └─ registration.register_template_to_target()  — 공통 템플릿을 그 스캔에 맞춤
              └─ model.fit_ssm()  — 정합된 여러 스캔의 정점 변동을 PCA로 압축

`scripts/build_ssm.py` 가 이 세 모듈을 매니페스트(`scan_dataset.py`) 기반으로 엮는
오케스트레이션 CLI다. 아직 실 데이터(200여 개 STL)가 없어도, 지금 이 골격은
`template_factory.py`의 절차적 템플릿으로 합성 검증까지 마친 상태다.
"""

from __future__ import annotations

from .model import StatisticalShapeModel, fit_ssm
from .preprocessing import (
    clean_and_normalize,
    denoise_mesh,
    measured_length_mm,
    mirror_side,
    pca_axes,
    self_normalize,
)
from .registration import (
    RegistrationResult,
    generalized_procrustes_align,
    register_template_to_target,
    rigid_prealign,
)

__all__ = [
    "clean_and_normalize",
    "denoise_mesh",
    "measured_length_mm",
    "mirror_side",
    "pca_axes",
    "self_normalize",
    "RegistrationResult",
    "generalized_procrustes_align",
    "register_template_to_target",
    "rigid_prealign",
    "StatisticalShapeModel",
    "fit_ssm",
]
