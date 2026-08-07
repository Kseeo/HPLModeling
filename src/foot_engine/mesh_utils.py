"""메쉬 I/O · 좌표계 정규화 · 계측 · 품질 보장 유틸리티.

`deformer.py` 가 얇게 유지되도록 순수 기하 연산은 모두 여기에 모았다.
모든 함수는 부작용을 최소화하며, 메쉬를 수정하는 함수는 이름에 명시한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from . import config as cfg
from .exceptions import MeshQualityError, TemplateLoadError
from .schemas import FootMeasurements, QualityReport

# ---------------------------------------------------------------------------
# 로딩 / 좌표계
# ---------------------------------------------------------------------------


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """STL/OBJ/PLY/GLB 를 단일 `Trimesh` 로 로드한다.

    Scene 으로 로드되는 포맷(glb 등)은 하나의 메쉬로 합친다.

    Raises:
        TemplateLoadError: 파일이 없거나 정점/면이 비어 있는 경우.
    """
    path = Path(path)
    if not path.is_file():
        raise TemplateLoadError(
            f"템플릿 파일을 찾을 수 없습니다: {path}",
            detail={"resolved": str(path.resolve())},
        )

    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:  # trimesh 는 다양한 예외를 던진다
        raise TemplateLoadError(f"메쉬 로딩 실패: {path}", detail=str(exc)) from exc

    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_mesh() if hasattr(loaded, "to_mesh") else trimesh.util.concatenate(
            list(loaded.geometry.values())
        )

    if not isinstance(loaded, trimesh.Trimesh):
        raise TemplateLoadError(
            f"삼각 메쉬로 변환할 수 없습니다: {path}", detail=type(loaded).__name__
        )
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise TemplateLoadError(f"메쉬가 비어 있습니다: {path}")

    # STL 은 면마다 정점을 중복 저장하므로 병합해야 위상(topology)이 살아난다.
    # (병합 전에는 watertight 판정이 항상 False 다)
    loaded.merge_vertices()
    return loaded


def remap_axes(
    mesh: trimesh.Trimesh, order: str = "XYZ", flip: tuple[int, int, int] = (1, 1, 1)
) -> trimesh.Trimesh:
    """축 순서를 바꾸거나 뒤집어 정규 좌표계로 맞춘다.

    외부에서 받은 템플릿이 Y-up 이거나 발가락이 -X 를 향하는 경우 사용한다.

    Args:
        order: 'XZY' 처럼 (새 X, 새 Y, 새 Z) 가 원본의 어느 축인지.
        flip:  각 축의 부호. 예) (1, -1, 1) 은 Y 반전(좌우발 미러링).

    Returns:
        축이 재배열된 **새** 메쉬.
    """
    order = order.upper()
    if sorted(order) != ["X", "Y", "Z"]:
        raise ValueError(f"order 는 X,Y,Z 의 순열이어야 합니다: {order!r}")

    index = [{"X": 0, "Y": 1, "Z": 2}[c] for c in order]
    vertices = mesh.vertices[:, index] * np.asarray(flip, dtype=float)

    out = trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy(), process=False)
    # 홀수 번의 반전은 면의 방향(winding)을 뒤집으므로 되돌린다.
    if np.prod(flip) < 0:
        out.faces = out.faces[:, ::-1]
    return out


def canonicalize(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    """정규 좌표계 가정을 점검하고, 명백히 뒤집힌 경우 자동 교정한다.

    - 발목/뒤꿈치 쪽이 발가락 쪽보다 훨씬 높다는 성질을 이용해 X 방향을 판별한다.
    - 바닥(z_min)이 지면에 오도록 Z 를 0 으로 내린다.

    Returns:
        (교정된 메쉬, 수행한 작업 설명 리스트)
    """
    notes: list[str] = []
    mesh = mesh.copy()
    verts = mesh.vertices
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    extents = hi - lo

    if np.any(extents <= 1e-6):
        raise TemplateLoadError(
            "메쉬가 평면이거나 퇴화(degenerate)했습니다.",
            detail={"extents": extents.tolist()},
        )

    # X 방향 판별: 앞/뒤 20% 구간의 높이를 비교한다.
    u = (verts[:, 0] - lo[0]) / extents[0]
    rear_h = verts[u < 0.20, 2].max() - lo[2]
    front_h = verts[u > 0.80, 2].max() - lo[2]
    if front_h > rear_h * 1.2:
        mesh = remap_axes(mesh, "XYZ", (-1, 1, 1))
        notes.append("발가락이 -X 를 향하고 있어 X 축을 반전했습니다.")
        verts = mesh.vertices
        lo = verts.min(axis=0)

    # 바닥을 z = 0 에 맞춘다 (아치 높이를 절대값으로 다루기 위함).
    if abs(lo[2]) > 1e-9:
        mesh.apply_translation((0.0, 0.0, -lo[2]))
        notes.append(f"바닥이 z=0 이 되도록 {-lo[2]:.3f}mm 이동했습니다.")

    if not mesh.is_watertight:
        notes.append("경고: 템플릿이 watertight 가 아닙니다. 변형 전 복구를 시도합니다.")

    return mesh, notes


def detect_medial_side(mesh: trimesh.Trimesh) -> str:
    """아치 공간이 있는 쪽을 내측(medial)으로 판정한다.

    좌/우발을 별도 플래그 없이도 처리하기 위한 기하학적 판별.

    Returns:
        'ymin' 또는 'ymax'.
    """
    verts = mesh.vertices
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    extents = np.maximum(hi - lo, 1e-9)
    u = (verts[:, 0] - lo[0]) / extents[0]
    v = (verts[:, 1] - lo[1]) / extents[1]
    w = (verts[:, 2] - lo[2]) / extents[2]

    lo_u, hi_u = cfg.ARCH_WINDOW_U
    in_arch = (u >= lo_u) & (u <= hi_u) & (w < 0.30)
    if not in_arch.any():
        return "ymin"

    # 아치가 있는 쪽은 바닥면이 더 높이 떠 있다.
    left = in_arch & (v < 0.35)
    right = in_arch & (v > 0.65)
    if not left.any() or not right.any():
        return "ymin"

    return "ymin" if verts[left, 2].max() >= verts[right, 2].max() else "ymax"


@dataclass(slots=True)
class FootFrame:
    """메쉬의 Bounding Box 기반 정규화 프레임.

    `v` 좌표는 항상 0 = 내측(medial), 1 = 외측(lateral) 이 되도록 보정된다.
    좌발/우발을 동일한 코드로 다루기 위한 장치다.
    """

    origin: np.ndarray  # (3,) bbox 최소점
    extents: np.ndarray  # (3,) bbox 크기
    medial_side: str = "ymin"  # 'ymin' | 'ymax'

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, medial_side: str | None = None) -> "FootFrame":
        lo = mesh.vertices.min(axis=0)
        hi = mesh.vertices.max(axis=0)
        return cls(
            origin=lo,
            extents=np.maximum(hi - lo, 1e-9),
            medial_side=medial_side or detect_medial_side(mesh),
        )

    # --- 좌표 변환 -----------------------------------------------------
    def to_uvw(self, points: np.ndarray) -> np.ndarray:
        """월드 좌표(mm) → 정규화 좌표 (u, v, w)."""
        uvw = (np.asarray(points, dtype=float) - self.origin) / self.extents
        if self.medial_side == "ymax":
            uvw[:, 1] = 1.0 - uvw[:, 1]
        return uvw

    def to_world(self, uvw: np.ndarray) -> np.ndarray:
        """정규화 좌표 (u, v, w) → 월드 좌표(mm)."""
        uvw = np.array(uvw, dtype=float, copy=True)
        if uvw.ndim == 1:
            uvw = uvw[None, :]
        if self.medial_side == "ymax":
            uvw[:, 1] = 1.0 - uvw[:, 1]
        return uvw * self.extents + self.origin

    @property
    def length(self) -> float:
        return float(self.extents[0])

    @property
    def y_center(self) -> float:
        return float(self.origin[1] + self.extents[1] * 0.5)

    @property
    def x_heel(self) -> float:
        return float(self.origin[0])

    @property
    def z_floor(self) -> float:
        return float(self.origin[2])


# ---------------------------------------------------------------------------
# 계측
# ---------------------------------------------------------------------------


def sole_vertex_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    """바닥면(plantar surface)에 속하는 정점 마스크.

    정점 법선의 z 성분이 충분히 아래를 향하는 정점을 바닥면으로 본다.
    법선 계산이 실패하면 하단 25% 높이로 대체한다.
    """
    try:
        normals = mesh.vertex_normals
        mask = normals[:, 2] < cfg.SOLE_NORMAL_Z_MAX
        if mask.any():
            return mask
    except Exception:  # 법선 계산 불가(퇴화 메쉬 등)
        pass

    z = mesh.vertices[:, 2]
    height = max(z.max() - z.min(), 1e-9)
    return (z - z.min()) / height < 0.25


def arch_apex(
    mesh: trimesh.Trimesh, frame: FootFrame | None = None
) -> tuple[float, int | None]:
    """내측 아치의 정점(apex) 높이와 해당 정점 인덱스를 찾는다.

    정의: '아치 구간(u) × 내측 밴드(v) 의 바닥면(plantar) 중 가장 높은 점'.
    측면 사진에서 읽는 아치 실루엣의 최고점과 같은 정의이므로,
    2D 계측치와 3D 결과를 같은 잣대로 비교할 수 있다.

    Returns:
        (바닥(z_min) 기준 높이 mm, 정점 인덱스). 해당 영역이 없으면 (0.0, None).
    """
    frame = frame or FootFrame.from_mesh(mesh)
    uvw = frame.to_uvw(mesh.vertices)
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]

    band_lo, band_hi = cfg.ARCH_MEDIAL_BAND_V
    win_lo, win_hi = cfg.ARCH_WINDOW_U
    region = (u >= win_lo) & (u <= win_hi) & (v >= band_lo) & (v <= band_hi)

    mask = region & sole_vertex_mask(mesh)
    if not mask.any():  # 법선 기준이 실패하면 높이 기준으로 대체
        mask = region & (w < 0.25)
    if not mask.any():
        return 0.0, None

    indices = np.flatnonzero(mask)
    apex = int(indices[np.argmax(mesh.vertices[indices, 2])])
    return float(mesh.vertices[apex, 2] - frame.z_floor), apex


def measure_foot(mesh: trimesh.Trimesh, frame: FootFrame | None = None) -> FootMeasurements:
    """메쉬에서 변형 구동용 계측치를 추출한다.

    템플릿과 변형 결과에 **같은 함수**를 적용하므로 계측 정의가 자기일관적이다.
    (예: 아치 높이는 '내측 밴드에서 바닥면의 최고점'으로 항상 동일하게 정의)

    Args:
        mesh: 정규 좌표계로 맞춰진 메쉬.
        frame: 재사용할 정규화 프레임. None 이면 새로 계산.

    Returns:
        모든 필드가 채워진 `FootMeasurements`.
    """
    frame = frame or FootFrame.from_mesh(mesh)
    verts = mesh.vertices
    uvw = frame.to_uvw(verts)
    u, w = uvw[:, 0], uvw[:, 2]
    y, z = verts[:, 1], verts[:, 2]
    z_floor = frame.z_floor

    def y_span(mask: np.ndarray, fallback: float) -> float:
        return float(y[mask].max() - y[mask].min()) if mask.any() else fallback

    def in_u(window: tuple[float, float]) -> np.ndarray:
        return (u >= window[0]) & (u <= window[1])

    m = FootMeasurements()
    m.foot_length_mm = frame.length

    # 너비 계열
    m.heel_width_mm = y_span(in_u(cfg.HEEL_WINDOW_U), float(frame.extents[1]))
    m.ball_width_mm = y_span(in_u(cfg.BALL_WINDOW_U), float(frame.extents[1]))

    ankle_band = in_u(cfg.REAR_WINDOW_U) & (
        np.abs(w - cfg.ANKLE_HEIGHT_W_FRACTION) < cfg.ANKLE_BAND_W
    )
    m.ankle_width_mm = y_span(ankle_band, m.heel_width_mm)

    # 높이 계열
    instep_mask = in_u(cfg.INSTEP_WINDOW_U)
    m.instep_height_mm = (
        float(z[instep_mask].max() - z_floor) if instep_mask.any()
        else float(frame.extents[2])
    )
    # 복사뼈 높이는 제어점 규약(config.ANKLE_HEIGHT_W_FRACTION)으로 정의한다.
    m.ankle_height_mm = float(frame.extents[2] * cfg.ANKLE_HEIGHT_W_FRACTION)

    # 아치 높이 = 내측 밴드에서 바닥면(plantar)의 최고점 (측면 사진 실루엣과 동일 정의)
    m.arch_height_mm, _ = arch_apex(mesh, frame)

    for name in m.field_names():
        m.sources[name] = ["mesh-geometry"]
    return m


# ---------------------------------------------------------------------------
# 품질 보장
# ---------------------------------------------------------------------------


def _open_edge_count(mesh: trimesh.Trimesh) -> int:
    """경계(한 번만 등장하는) 에지 개수 — 구멍의 크기를 가늠하는 지표."""
    try:
        groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
        return int(len(groups))
    except Exception:
        return 0


def count_flipped_faces(
    reference: trimesh.Trimesh, deformed: trimesh.Trimesh
) -> tuple[int, int]:
    """변형 전후로 방향이 뒤집힌 면의 개수를 센다.

    두 메쉬는 **동일한 face 인덱스**를 가져야 한다(변형은 정점만 옮기므로 성립).
    면 법선의 내적이 음수면 그 삼각형은 뒤집힌 것으로 본다.

    Returns:
        (뒤집힌 면 수, 비교한 면 수)
    """
    if reference.faces.shape != deformed.faces.shape:
        return 0, 0

    def face_normals(mesh: trimesh.Trimesh) -> np.ndarray:
        tri = mesh.vertices[mesh.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(norm, 1e-12)

    dots = np.einsum("ij,ij->i", face_normals(reference), face_normals(deformed))
    return int((dots < 0.0).sum()), int(len(dots))


def ensure_quality(
    mesh: trimesh.Trimesh,
    *,
    reference: trimesh.Trimesh | None = None,
    conf: cfg.DeformConfig | None = None,
) -> tuple[trimesh.Trimesh, QualityReport]:
    """메쉬를 복구하고 품질 리포트를 만든다.

    수행 순서
        1. (reference 가 있으면) 뒤집힌 면 비율 측정 — 복구로 토폴로지가 바뀌기 전에
        2. 비정상 값(NaN/Inf) 제거
        3. 퇴화 삼각형 / 미참조 정점 제거
        4. 구멍 메우기 (`trimesh.repair.fill_holes`)
        5. winding·법선 일관성 복구 (`fix_winding`, `fix_inversion`, `fix_normals`)
        6. watertight / 체적 부호 / Euler number 검사

    Args:
        mesh: 검사·복구할 메쉬 (복사본을 만들어 원본은 건드리지 않는다).
        reference: 변형 전 메쉬. 면 뒤집힘 비교용.
        conf: 임계값 설정.

    Returns:
        (복구된 메쉬, QualityReport)

    Raises:
        MeshQualityError: `conf.strict_quality=True` 이고 기준 미달인 경우.
    """
    conf = conf or cfg.DeformConfig()
    report = QualityReport()
    mesh = mesh.copy()

    # 1) 뒤집힌 면 비율 (토폴로지 변경 전에 측정해야 인덱스가 일치한다)
    if reference is not None:
        flipped, total = count_flipped_faces(reference, mesh)
        report.flipped_face_ratio = flipped / total if total else 0.0
        if report.flipped_face_ratio > conf.max_flipped_face_ratio:
            report.warnings.append(
                f"변형으로 면 {flipped}/{total} 개({report.flipped_face_ratio:.2%})가 "
                f"뒤집혔습니다. 제어점 목표치가 과도한지 확인하세요."
            )

    if conf.auto_repair:
        report.repaired = True
        face_count_before = len(mesh.faces)

        # 2) NaN/Inf 정점 제거
        if not np.isfinite(mesh.vertices).all():
            report.warnings.append("비정상(NaN/Inf) 정점이 발견되어 제거했습니다.")
            mesh.remove_infinite_values()

        # 3) 퇴화 삼각형 · 중복 정점 · 미참조 정점 정리
        try:
            mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
            mesh.update_faces(mesh.unique_faces())
        except Exception:  # trimesh 버전에 따라 helper 명이 다를 수 있다
            pass
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        report.degenerate_faces_removed = max(0, face_count_before - len(mesh.faces))

        # 4) 구멍 메우기
        if not mesh.is_watertight:
            open_before = _open_edge_count(mesh)
            trimesh.repair.fill_holes(mesh)
            open_after = _open_edge_count(mesh)
            report.holes_filled = max(0, open_before - open_after)
            if not mesh.is_watertight:
                report.warnings.append(
                    f"구멍을 완전히 메우지 못했습니다. 남은 경계 에지: {open_after}개"
                )

        # 5) 면 방향 일관성 복구
        try:
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_inversion(mesh)
            trimesh.repair.fix_normals(mesh)
        except Exception as exc:
            report.warnings.append(f"법선 복구 중 경고: {exc}")

    # 6) 최종 검사
    report.is_watertight = bool(mesh.is_watertight)
    report.is_winding_consistent = bool(mesh.is_winding_consistent)
    report.vertex_count = int(len(mesh.vertices))
    report.face_count = int(len(mesh.faces))
    report.euler_number = int(mesh.euler_number)
    try:
        report.volume_mm3 = float(mesh.volume)
    except Exception:
        report.volume_mm3 = 0.0
    report.is_volume_positive = report.volume_mm3 > 0.0

    if not report.is_watertight:
        report.warnings.append("결과 메쉬가 watertight 가 아닙니다(3D 프린팅 불가).")
    if not report.is_winding_consistent:
        report.warnings.append("면의 winding 이 일관되지 않습니다.")
    if not report.is_volume_positive:
        report.warnings.append(f"체적이 양수가 아닙니다: {report.volume_mm3:.1f}mm³")

    if conf.strict_quality and not report.is_ok:
        raise MeshQualityError(
            "메쉬 품질 기준을 통과하지 못했습니다.", detail=report.to_dict()
        )

    return mesh, report
