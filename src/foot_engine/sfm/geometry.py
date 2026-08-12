"""점군/메쉬 정점에 쓰는 순수 기하 유틸 — PCA 축, 자기 자신의 실측 길이.
"""

from __future__ import annotations

import numpy as np


def pca_axes(points: np.ndarray) -> np.ndarray:
    """점들의 주성분 축을 분산 내림차순으로 반환 — (3,3), 각 열이 축 하나.

    발처럼 길쭉하고 납작한 형태에서는 분산이 큰 순서가 대체로
    [길이, 너비, 높이] 축과 일치한다.
    """
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)  # 오름차순
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]


def measured_length(points: np.ndarray) -> float:
    """점군 자신의 PCA 최장축(추정 길이) 크기 — 절대 축척 아님, 스케일 비교용."""
    axes = pca_axes(points)
    projected = (points - points.mean(axis=0)) @ axes[:, 0]
    return float(projected.max() - projected.min())
