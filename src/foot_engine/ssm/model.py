"""PCA 기반 통계적 형상 모델(SSM).

`registration.py`로 모든 스캔을 같은 위상(정점 개수·순서)으로 맞춘 뒤, 그
정점 좌표들의 변동을 PCA로 압축한다. 최종 목적은 "형태 공간의 계수 몇 개로,
소수의 랜드마크만 보고도 그 사람에게 가장 그럴듯한 형태를 골라내는 것"이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class StatisticalShapeModel:
    """평균 형상 + 주성분. 정점 위상(순서)은 학습에 쓰인 모든 스캔이 공유한다."""

    mean_shape: np.ndarray  # (V, 3)
    components: np.ndarray  # (K, V*3) — 각 행이 단위 주성분 벡터
    explained_variance: np.ndarray  # (K,)
    template_faces: np.ndarray  # (F, 3) — 뷰어 표시·재구성용, 위상은 전 스캔 공유

    @property
    def n_vertices(self) -> int:
        return self.mean_shape.shape[0]

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    def project(self, vertices: np.ndarray) -> np.ndarray:
        """등록된 정점 배열 (V,3) → PCA 계수 (K,). `vertices`는 반드시 같은 위상이어야 한다."""
        flat = (vertices - self.mean_shape).reshape(-1)
        return self.components @ flat

    def generate(self, coefficients: np.ndarray) -> np.ndarray:
        """PCA 계수 (K,) → 정점 배열 (V,3). `coefficients`가 K보다 짧으면 나머지는 0으로 채운다."""
        coeff = np.zeros(self.n_components, dtype=float)
        coeff[: len(coefficients)] = coefficients
        flat = self.mean_shape.reshape(-1) + coeff @ self.components
        return flat.reshape(self.mean_shape.shape)

    def fit_from_landmarks(
        self, landmark_indices: np.ndarray, landmark_positions: np.ndarray, *, ridge: float = 1.0
    ) -> np.ndarray:
        """일부 정점(랜드마크)의 실측 위치만으로 PCA 계수를 역산한다.

        실제 서비스에서는 전체 정점을 알 수 없고, 사진에서 SfM으로 삼각측량한
        랜드마크 몇 개(뒤꿈치, 아치 정점, 볼너비 등)만 있다 — `project()`가
        "전체를 다 안다면"의 상한선을 재는 것과 달리, 이게 실제로 쓰이는 경로다.
        랜드마크 개수가 보통 주성분 개수보다 적어 문제가 미결정(underdetermined)
        이므로 능형회귀(ridge)로 정칙화한다 — 정보가 부족한 방향은 평균 형상
        쪽으로 보수적으로 수렴한다.

        Args:
            landmark_indices: 랜드마크에 해당하는 정점 인덱스, (L,).
            landmark_positions: 그 정점들의 실측 위치(mm), (L,3).
            ridge: 정칙화 강도. 클수록 평균 형상 쪽으로 보수적으로 수렴한다.

        Returns:
            PCA 계수 (K,) — `generate()`에 그대로 넣으면 전체 메쉬가 나온다.
        """
        landmark_indices = np.asarray(landmark_indices)
        mean_at_landmarks = self.mean_shape[landmark_indices]  # (L,3)
        target = (np.asarray(landmark_positions) - mean_at_landmarks).reshape(-1)  # (3L,)

        col_idx = np.stack(
            [landmark_indices * 3, landmark_indices * 3 + 1, landmark_indices * 3 + 2], axis=1
        ).reshape(-1)
        a = self.components[:, col_idx].T  # (3L, K)

        k = a.shape[1]
        return np.linalg.solve(a.T @ a + ridge * np.eye(k), a.T @ target)

    def variance_ratio(self) -> np.ndarray:
        """각 주성분이 설명하는 분산 비율 — 몇 개까지 쓸지 정할 때 참고."""
        total = self.explained_variance.sum()
        return self.explained_variance / total if total > 0 else self.explained_variance

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            mean_shape=self.mean_shape,
            components=self.components,
            explained_variance=self.explained_variance,
            template_faces=self.template_faces,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StatisticalShapeModel":
        data = np.load(path)
        return cls(
            mean_shape=data["mean_shape"],
            components=data["components"],
            explained_variance=data["explained_variance"],
            template_faces=data["template_faces"],
        )


def fit_ssm(
    registered_vertices: list[np.ndarray], faces: np.ndarray, *, n_components: int = 20
) -> StatisticalShapeModel:
    """정합된 정점 배열 목록으로 SSM을 학습한다.

    Args:
        registered_vertices: 각 원소가 (V,3) — 전부 같은 V(정점 수)·같은 위상이어야
            한다(`registration.register_template_to_target()` 결과들).
        faces: 공통 위상의 면 인덱스.
        n_components: 남길 주성분 개수. 스캔 수보다 많이 요청해도 자동으로
            (스캔 수 - 1)로 잘린다(그 이상은 수학적으로 의미가 없으므로).

    Raises:
        ValueError: 스캔이 3개 미만이거나, 정점 개수가 서로 다른 경우.
    """
    if len(registered_vertices) < 3:
        raise ValueError("SSM을 학습하려면 정합된 스캔이 최소 3개 이상 필요합니다.")

    shapes = {v.shape for v in registered_vertices}
    if len(shapes) != 1:
        raise ValueError(f"모든 스캔의 정점 배열 shape가 같아야 합니다: {shapes}")

    stacked = np.stack(registered_vertices)  # (N, V, 3)
    n_samples = stacked.shape[0]
    mean_shape = stacked.mean(axis=0)

    centered = (stacked - mean_shape).reshape(n_samples, -1)  # (N, V*3)

    # SVD 기반 PCA — 표본 수(N)가 차원(V*3)보다 훨씬 적은 전형적인 상황에서
    # 공분산 행렬을 직접 만드는 것보다 훨씬 효율적이고 수치적으로 안정적이다.
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(n_components, n_samples - 1)
    components = vt[:k]
    explained_variance = (singular_values[:k] ** 2) / (n_samples - 1)

    return StatisticalShapeModel(
        mean_shape=mean_shape,
        components=components,
        explained_variance=explained_variance,
        template_faces=np.asarray(faces),
    )
