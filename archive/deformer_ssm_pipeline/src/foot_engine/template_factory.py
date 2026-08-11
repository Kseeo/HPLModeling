"""테스트용 기준 발 템플릿(`base_foot_template.stl`) 절차적 생성기.

실제 서비스에서는 3D 스캔으로 만든 표준 발 모델을 쓰지만, 개발·CI 단계에서
바로 돌려볼 수 있는 **watertight 한 대체 템플릿**이 필요하다. 이 모듈은
길이방향(X) 단면을 로프트(loft)해 발+발목 형상을 만든다.

    - 각 단면은 '아래는 평평하고 위는 둥근' superellipse
    - 뒤쪽 단면을 높게 만들어 발목/종아리 스텁을 표현 (X-Z 평면의 L 자 형상)
    - 바닥은 내측 가중 프로파일로 들어올려 아치를 만든다
      (변형기 `deformer._arch_weights()` 와 동일한 `config.medial_arch_weight` 사용)

주의: 해부학적 정밀 모델이 아니다. 좌표계 규약·계측 파이프라인·품질 검사를
검증하기 위한 스탠드인(stand-in)이며, 실제 스캔 STL 로 교체하면 동일 코드가
그대로 동작한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator

from . import config as cfg

# ---------------------------------------------------------------------------
# 형상 프로파일 — 길이 250mm 발 기준값을 길이 비율로 정규화한 테이블
# ---------------------------------------------------------------------------

_REF_LENGTH_MM = 250.0

#: 길이방향 정규화 위치 (0 = 뒤꿈치 끝, 1 = 발가락 끝)
_PROFILE_U = np.array(
    [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
)

#: 각 단면의 반너비(mm @ L=250). 볼(u≈0.70)에서 최대, 중족부에서 잘록해진다.
_HALF_WIDTH_MM = np.array(
    [20.0, 26.0, 30.0, 32.0, 32.5, 32.0, 31.0, 30.5, 31.0, 32.5, 35.0,
     39.0, 43.0, 46.0, 47.0, 45.5, 41.0, 35.0, 27.0, 17.0, 6.0]
)

#: 각 단면의 윗면 높이(mm @ L=250). u=0 부근이 발목/종아리 절단면이다.
_TOP_HEIGHT_MM = np.array(
    [110.0, 110.0, 108.0, 103.0, 95.0, 86.0, 78.0, 72.0, 68.0, 65.0, 62.0,
     58.0, 52.0, 47.0, 42.0, 37.0, 32.0, 27.0, 24.0, 20.0, 12.0]
)

#: superellipse 지수 — 클수록 사각형에 가깝다.
_EXPONENT_TOP = 2.3   # 발등: 둥글게
_EXPONENT_BOTTOM = 5.0  # 발바닥: 평평하게

#: 아치가 시작·끝나는 구간 (뒤꿈치 접지 ~ 볼 접지)
_ARCH_SPAN_U = (0.20, 0.66)
#: 발가락 들림(toe spring)이 시작되는 위치와 최대 높이(mm @ L=250)
_TOE_SPRING_START_U = 0.90
_TOE_SPRING_MM = 4.0
#: 발목 위쪽으로 갈수록 단면 폭을 줄이는 비율
_ANKLE_TAPER = 0.18


def _profile(values_mm: np.ndarray) -> PchipInterpolator:
    """단조 보간기(overshoot 없는 PCHIP)로 프로파일 테이블을 매끄럽게 만든다."""
    return PchipInterpolator(_PROFILE_U, values_mm / _REF_LENGTH_MM)


_HALF_WIDTH_F = _profile(_HALF_WIDTH_MM)
_TOP_HEIGHT_F = _profile(_TOP_HEIGHT_MM)


def _arch_bump(u: np.ndarray, arch_shape_mm: float) -> np.ndarray:
    """길이방향 아치 융기 프로파일(mm). 구간 밖에서는 0."""
    lo, hi = _ARCH_SPAN_U
    t = (u - lo) / (hi - lo)
    inside = (t > 0.0) & (t < 1.0)
    bump = np.zeros_like(u, dtype=float)
    bump[inside] = arch_shape_mm * np.sin(np.pi * t[inside]) ** 1.6
    return bump


def _toe_spring(u: np.ndarray, length_mm: float) -> np.ndarray:
    """발가락 끝의 바닥 들림(mm)."""
    t = np.clip((u - _TOE_SPRING_START_U) / (1.0 - _TOE_SPRING_START_U), 0.0, 1.0)
    return (_TOE_SPRING_MM * length_mm / _REF_LENGTH_MM) * t**2


def _cross_section(
    u: float, length_mm: float, arch_shape_mm: float, theta: np.ndarray
) -> np.ndarray:
    """길이방향 위치 u 에서의 닫힌 단면 링을 (M, 3) 좌표로 만든다."""
    half_w = float(_HALF_WIDTH_F(u)) * length_mm
    z_top = float(_TOP_HEIGHT_F(u)) * length_mm
    z_bot = float(_toe_spring(np.array([u]), length_mm)[0])

    z_center = 0.5 * (z_top + z_bot)
    z_half = 0.5 * (z_top - z_bot)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    upper = sin_t >= 0.0
    exponent = np.where(upper, 2.0 / _EXPONENT_TOP, 2.0 / _EXPONENT_BOTTOM)

    y = half_w * np.sign(cos_t) * np.abs(cos_t) ** exponent
    z = z_center + z_half * np.sign(sin_t) * np.abs(sin_t) ** exponent

    # 발목 위쪽 테이퍼 — 종아리가 복사뼈보다 살짝 좁아진다.
    z_frac = (z - z_bot) / max(z_top - z_bot, 1e-9)
    y = y * (1.0 - _ANKLE_TAPER * np.clip((z_frac - 0.55) / 0.45, 0.0, 1.0))

    # 내측 가중 아치 리프트 — 아랫면에만 적용하고 옆면에서 0 으로 수렴시킨다.
    bump = float(_arch_bump(np.array([u]), arch_shape_mm)[0])
    if bump > 0.0:
        v = (y + half_w) / (2.0 * half_w)  # 0 = 내측(y_min), 1 = 외측
        downward = np.clip(-sin_t, 0.0, 1.0)
        z = z + bump * cfg.medial_arch_weight(v) * downward

    x = np.full_like(theta, u * length_mm)
    return np.column_stack([x, y, z])


def build_reference_foot(
    *,
    length_mm: float = 250.0,
    side: str = "right",
    arch_shape_mm: float = 24.0,
    n_stations: int = 120,
    n_ring: int = 64,
) -> trimesh.Trimesh:
    """절차적으로 발+발목 템플릿 메쉬를 생성한다.

    Args:
        length_mm: 뒤꿈치~발가락 끝 길이.
        side: 'right' | 'left'. 'left' 는 Y 를 미러링한다.
        arch_shape_mm: 아치 융기의 **형상 파라미터**. 실제로 계측되는 내측 아치
            높이는 내측 가중 프로파일 때문에 이 값보다 조금 작다
            (기본 24mm → L=250 기준 계측값 약 20mm).
        n_stations: 길이방향 단면 개수. 클수록 매끄럽다.
        n_ring: 단면당 정점 수(짝수 권장).

    Returns:
        watertight 한 `trimesh.Trimesh` (정규 좌표계, 바닥 z=0).
    """
    if length_mm <= 0:
        raise ValueError("length_mm 은 0보다 커야 합니다.")
    if n_stations < 8 or n_ring < 8:
        raise ValueError("n_stations, n_ring 은 각각 8 이상이어야 합니다.")

    theta = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    stations = np.linspace(0.0, 1.0, n_stations)

    rings = np.stack(
        [_cross_section(u, length_mm, arch_shape_mm, theta) for u in stations]
    )  # (K, M, 3)
    vertices = rings.reshape(-1, 3)

    # --- 측면(loft) 사각형을 삼각형 2개로 --------------------------------
    faces: list[tuple[int, int, int]] = []
    for i in range(n_stations - 1):
        base, nxt = i * n_ring, (i + 1) * n_ring
        for j in range(n_ring):
            k = (j + 1) % n_ring
            a, b = base + j, base + k
            c, d = nxt + k, nxt + j
            faces.append((a, b, c))
            faces.append((a, c, d))

    # --- 앞/뒤 캡(cap) — 링 중심으로 부채꼴 삼각형 -----------------------
    heel_center_idx = len(vertices)
    toe_center_idx = heel_center_idx + 1
    vertices = np.vstack(
        [vertices, rings[0].mean(axis=0), rings[-1].mean(axis=0)]
    )
    last_base = (n_stations - 1) * n_ring
    for j in range(n_ring):
        k = (j + 1) % n_ring
        faces.append((heel_center_idx, k, j))                        # 뒤꿈치 캡
        faces.append((toe_center_idx, last_base + j, last_base + k))  # 발가락 캡

    if side.lower() == "left":
        vertices[:, 1] *= -1.0  # 좌우 미러링 (winding 은 아래 fix_normals 가 정리)

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)  # 법선이 모두 바깥을 향하도록

    # 바닥을 z=0 으로 정렬
    mesh.apply_translation((0.0, 0.0, -float(mesh.vertices[:, 2].min())))
    return mesh


def save_reference_template(
    output_path: str | Path, **kwargs
) -> tuple[Path, trimesh.Trimesh]:
    """기준 템플릿을 생성해 파일로 저장한다.

    Args:
        output_path: 저장할 `.stl` 경로 (상위 폴더 자동 생성).
        **kwargs: `build_reference_foot()` 로 전달.

    Returns:
        (저장 경로, 생성된 메쉬)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = build_reference_foot(**kwargs)
    mesh.export(path)
    return path, mesh
