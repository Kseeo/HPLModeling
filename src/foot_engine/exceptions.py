"""엔진 전용 예외 계층.

FastAPI 로 감쌀 때 각 예외를 HTTP status code 로 1:1 매핑할 수 있도록
`http_status` 속성을 함께 정의해 둔다. (예: 400 / 422 / 500)
"""

from __future__ import annotations


class FootEngineError(Exception):
    """엔진의 모든 예외가 상속하는 최상위 예외."""

    http_status: int = 500

    def __init__(self, message: str, *, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        """API 응답 body 로 바로 직렬화하기 위한 헬퍼."""
        return {
            "error": type(self).__name__,
            "message": self.message,
            "detail": self.detail,
        }


class TemplateLoadError(FootEngineError):
    """STL 템플릿 로딩 / 좌표계 정규화 실패."""

    http_status = 500


class LandmarkValidationError(FootEngineError):
    """입력 랜드마크 JSON 의 스키마·단위·캘리브레이션 오류."""

    http_status = 422


class DeformationError(FootEngineError):
    """RBF/TPS 연산 자체가 실패했거나 결과가 발산한 경우."""

    http_status = 500


class MeshQualityError(FootEngineError):
    """변형 결과 메쉬가 품질 기준(watertight, 뒤집힌 면 비율 등)을 통과하지 못한 경우."""

    http_status = 500


class ExportError(FootEngineError):
    """지원하지 않는 확장자이거나 파일 쓰기에 실패한 경우."""

    http_status = 400


class ScanDatasetError(FootEngineError):
    """SSM 학습용 스캔 매니페스트(CSV)의 스키마·값 오류."""

    http_status = 422


class CaptureQualityError(FootEngineError):
    """입력 사진/영상 품질 게이트를 통과한 프레임이 SfM에 필요한 최소치 미만인 경우.

    사용자 입력(촬영 품질)이 원인이므로 4xx로 분류한다 — 서버/엔진 결함이 아니다.
    """

    http_status = 422
