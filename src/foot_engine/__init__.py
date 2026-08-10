"""foot_engine — 2D 랜드마크 기반 3D 발 메쉬 파라메트릭 변형 엔진.

Quick start::

    from foot_engine import FootMeshDeformer

    deformer = FootMeshDeformer("data/templates/base_foot_template.stl")
    mesh = deformer.deform_mesh(landmarks_dict)
    deformer.export_mesh("data/output/output_deformed_foot.stl")
    print(deformer.last_report.summary_lines())
"""

from __future__ import annotations

from .config import DeformConfig
from .deformer import FootMeshDeformer
from .exceptions import (
    DeformationError,
    ExportError,
    FootEngineError,
    LandmarkValidationError,
    MeshQualityError,
    ScanDatasetError,
    TemplateLoadError,
)
from .landmarks import extract_measurements
from .scan_dataset import ScanRecord, category_counts, load_manifest, side_counts
from .schemas import (
    DeformationReport,
    FootMeasurements,
    LandmarkPayload,
    QualityReport,
    parse_payload,
)
from .service import DeformationService, run_deformation
from .template_factory import build_reference_foot, save_reference_template

__version__ = "0.1.0"

__all__ = [
    "FootMeshDeformer",
    "DeformConfig",
    "DeformationService",
    "run_deformation",
    "FootMeasurements",
    "LandmarkPayload",
    "DeformationReport",
    "QualityReport",
    "parse_payload",
    "extract_measurements",
    "build_reference_foot",
    "save_reference_template",
    "FootEngineError",
    "TemplateLoadError",
    "LandmarkValidationError",
    "DeformationError",
    "MeshQualityError",
    "ExportError",
    "ScanDatasetError",
    "ScanRecord",
    "load_manifest",
    "category_counts",
    "side_counts",
    "__version__",
]
