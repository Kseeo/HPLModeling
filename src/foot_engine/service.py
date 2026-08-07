"""애플리케이션 서비스 계층 — CLI 와 FastAPI 가 공유하는 유스케이스.

웹 프레임워크에 의존하지 않는 순수 함수/클래스로 유지해서
`app/api.py`(FastAPI) 는 얇은 어댑터로만 남도록 한다.
"""

from __future__ import annotations

import threading
from pathlib import Path

import trimesh

from .config import DeformConfig
from .deformer import FootMeshDeformer


class DeformationService:
    """템플릿별 `FootMeshDeformer` 를 캐시하는 서비스.

    템플릿 로딩·계측·제어점 생성은 요청마다 반복할 필요가 없으므로 캐시한다.
    변형 자체는 인스턴스 상태(`deformed_mesh`)를 갱신하므로, 동시 요청에서는
    락으로 직렬화한다. (처리량이 필요하면 워커 프로세스를 늘리는 편이 낫다.)
    """

    def __init__(self, conf: DeformConfig | None = None) -> None:
        self.conf = conf or DeformConfig()
        self._cache: dict[str, FootMeshDeformer] = {}
        self._lock = threading.Lock()

    def get_deformer(self, template_path: str | Path) -> FootMeshDeformer:
        """템플릿 경로에 대응하는 변형기를 (필요하면 생성해서) 반환한다."""
        key = str(Path(template_path).resolve())
        with self._lock:
            deformer = self._cache.get(key)
            if deformer is None:
                deformer = FootMeshDeformer(template_path, conf=self.conf)
                self._cache[key] = deformer
            return deformer

    def deform(
        self,
        landmarks_data: dict,
        template_path: str | Path,
        output_path: str | Path | None = None,
    ) -> tuple[trimesh.Trimesh, dict]:
        """변형을 수행하고 (메쉬, 리포트 dict) 를 반환한다.

        Args:
            landmarks_data: 2D 랜드마크 payload.
            template_path: 기본 템플릿 `.stl` 경로.
            output_path: 지정하면 결과를 해당 경로로 저장한다.

        Returns:
            (변형된 메쉬, 리포트 dict). 리포트에는 `output_path` 키가 포함된다.
        """
        deformer = self.get_deformer(template_path)
        with self._lock:  # deformer 내부 상태 보호
            mesh = deformer.deform_mesh(landmarks_data)
            report = deformer.last_report.to_dict() if deformer.last_report else {}
            saved: Path | None = None
            if output_path is not None:
                saved = deformer.export_mesh(output_path, mesh)
        report["output_path"] = str(saved) if saved else None
        report["template_path"] = str(template_path)
        return mesh, report


#: 프로세스 전역 기본 서비스 (FastAPI 의존성 주입에서 그대로 사용 가능)
default_service = DeformationService()


def run_deformation(
    landmarks_data: dict,
    template_path: str | Path,
    output_path: str | Path | None = None,
    conf: DeformConfig | None = None,
) -> dict:
    """한 줄로 쓰는 편의 함수 — 변형 후 리포트 dict 만 돌려준다.

    Example:
        >>> report = run_deformation(landmarks, "base_foot_template.stl",
        ...                          "output_deformed_foot.stl")
        >>> report["quality"]["is_watertight"]
        True
    """
    service = DeformationService(conf) if conf is not None else default_service
    _, report = service.deform(landmarks_data, template_path, output_path)
    return report
