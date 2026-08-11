"""SSM 학습 전처리 — 노이즈 제거 + 자기 자신의 실측 길이로 스케일 정규화.

원본 스캔은 (1) 스캐너 노이즈(뿔처럼 튀는 점)가 섞여 있고, (2) 절대 축척이
자기신고 사이즈에 의존해 신뢰할 수 없다(자세한 배경은 프로젝트 메모리
`scan-dataset-characteristics` 참고). 두 문제 모두 "각 스캔 자신의 형상에서 얻은
값만 쓴다"는 원칙으로 해결한다:
    - 노이즈  → 알고리즘적 스무딩(Taubin) — 사람마다 다르게 손대는 수작업보다
      일관되고, 강도를 코드로 통제할 수 있다.
    - 축척   → 자기신고 라벨이 아니라 메쉬 자신의 뒤꿈치~발끝 길이로 정규화.
      우리가 만드는 SSM은 순수 형상(비율)만 배우고, 자기신고 사이즈는 나중에
      최종 절대 크기를 정할 때 딱 한 번, 곱셈 계수로만 다시 등장한다.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .. import mesh_utils as mu


def pca_axes(points: np.ndarray) -> np.ndarray:
    """점들의 주성분 축을 분산 내림차순으로 반환 — (3,3), 각 열이 축 하나.

    발처럼 길쭉하고 납작한 형태에서는 분산이 큰 순서가 대체로
    [길이, 너비, 높이] 축과 일치한다. `registration.py`의 강체 사전 정렬,
    아래 `mirror_side()`가 공통으로 쓴다.
    """
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)  # 오름차순
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]


def mirror_side(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """발의 좌우를 반전한다(왼발 ↔ 오른발).

    실제 스캔은 좌표축 관례가 파일마다 달라(예: Y가 항상 너비축이라는 보장이
    없음) 특정 축을 고정해 반전할 수 없다. 대신 PCA로 찾은 "너비 축"(분산
    두 번째로 큰 축)을 반전하는 방식이라 좌표축 관례에 의존하지 않는다.
    """
    centroid = mesh.vertices.mean(axis=0)
    axes = pca_axes(mesh.vertices)
    local = (mesh.vertices - centroid) @ axes
    local[:, 1] *= -1.0  # 너비 축 반전
    mirrored_verts = local @ axes.T + centroid
    # 반사 변환은 삼각형의 winding(앞뒤 방향)을 뒤집으므로 정점 순서도 뒤집어 복구한다.
    mirrored_faces = mesh.faces[:, ::-1]
    return trimesh.Trimesh(vertices=mirrored_verts, faces=mirrored_faces, process=False)

#: 정규화 기준 길이(mm) — template_factory.py 의 _REF_LENGTH_MM 과 맞춰서,
#: SSM 산출물을 그 절차적 템플릿과 같은 스케일 감각으로 다룰 수 있게 한다.
DEFAULT_REFERENCE_LENGTH_MM = 250.0


def denoise_mesh(
    mesh: trimesh.Trimesh, *, iterations: int = 5, lamb: float = 0.5, nu: float = -0.53
) -> trimesh.Trimesh:
    """Taubin smoothing으로 스캐너 스파이크 노이즈를 제거한다.

    일반 Laplacian smoothing과 달리 lamb(팽창)/nu(수축) 두 단계를 번갈아 적용해
    저주파 형상(발 전체 윤곽)은 거의 그대로 두고 고주파 노이즈만 감쇠시킨다
    (Taubin, 1995). 기본값은 원 논문의 권장 비율이다.
    """
    mesh = mesh.copy()
    trimesh.smoothing.filter_taubin(mesh, lamb=lamb, nu=nu, iterations=iterations)
    return mesh


def measured_length_mm(mesh: trimesh.Trimesh) -> float:
    """메쉬 자신의 뒤꿈치~발끝 길이. 자기신고 라벨이 아니라 형상 자체에서 측정한다.

    `FootFrame`(X축 extent를 길이로 가정)을 쓰지 않고 PCA로 가장 긴 축을 직접
    찾는다 — 실제 스캔은 촬영/재구성 파이프라인마다 좌표축 관례가 달라 X축이
    길이축이라는 보장이 없다(실측 결과 일부 스캔은 X축 extent가 진짜 길이의
    1/3에도 못 미쳤다). 발은 항상 길이 방향의 분산이 가장 크므로, 축 관례에
    의존하지 않는 이 방법이 훨씬 안전하다.
    """
    axes = pca_axes(mesh.vertices)
    projected = (mesh.vertices - mesh.vertices.mean(axis=0)) @ axes[:, 0]
    return float(projected.max() - projected.min())


def self_normalize(
    mesh: trimesh.Trimesh, *, reference_length_mm: float = DEFAULT_REFERENCE_LENGTH_MM
) -> tuple[trimesh.Trimesh, float]:
    """메쉬를 '자기 자신의 길이'가 `reference_length_mm` 이 되도록 균일 스케일한다.

    Returns:
        (정규화된 메쉬, 정규화 전 원래 측정 길이mm). 뒷값은 최종 절대 크기를
        복원할 때 필요하므로 버리지 말 것.

    Raises:
        ValueError: 메쉬가 퇴화해 길이를 측정할 수 없는 경우.
    """
    own_length = measured_length_mm(mesh)
    if own_length <= 1e-6:
        raise ValueError("메쉬 길이를 측정할 수 없습니다(퇴화된 형상이거나 빈 메쉬).")

    scale = reference_length_mm / own_length
    normalized = mesh.copy()
    normalized.apply_scale(scale)
    return normalized, own_length


def clean_and_normalize(
    mesh: trimesh.Trimesh, *, reference_length_mm: float = DEFAULT_REFERENCE_LENGTH_MM
) -> tuple[trimesh.Trimesh, float]:
    """canonicalize → denoise → self_normalize 를 한 번에 적용하는 전처리 파이프라인.

    Returns:
        (전처리된 메쉬, 정규화 전 원래 측정 길이mm).
    """
    canonical, _ = mu.canonicalize(mesh)
    denoised = denoise_mesh(canonical)
    return self_normalize(denoised, reference_length_mm=reference_length_mm)
