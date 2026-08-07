"""파라메트릭 발 메쉬 변형기 (Parametric Foot Mesh Deformation).

파이프라인
----------
    2D 랜드마크 dict
        └─ landmarks.extract_measurements()  →  목표 계측치(mm)
              └─ 템플릿 계측치와 비교 → 부위별 스케일 계수
                    └─ 제어점(control point)의 목표 위치 산출
                          └─ RBF / TPS 로 전체 정점에 변위장(displacement field) 보간
                                └─ 아치 국소 Z 곡률 보정 (수렴 루프)
                                      └─ 품질 검사·복구 → Trimesh 반환

설계 원칙
    * 계측 정의는 템플릿과 결과에 **동일 함수**를 적용해 자기일관성을 유지한다.
    * 측정되지 않은 항목은 템플릿 값을 유지(scale=1.0)하므로 이미지 장수와 무관하게 동작한다.
    * 상태(state)는 인스턴스에 캐시하되, `deform_mesh()` 는 원본 템플릿을 변경하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import RBFInterpolator

from . import config as cfg
from . import mesh_utils as mu
from .exceptions import DeformationError, ExportError
from .landmarks import extract_measurements
from .mesh_utils import FootFrame
from .schemas import (
    DeformationReport,
    FootMeasurements,
    LandmarkPayload,
    parse_payload,
)


class FootMeshDeformer:
    """발/발목 3D 템플릿을 2D 랜드마크 계측치에 맞춰 변형하는 엔진.

    Example:
        >>> deformer = FootMeshDeformer("data/templates/base_foot_template.stl")
        >>> mesh = deformer.deform_mesh(landmarks_dict)
        >>> deformer.export_mesh("data/output/output_deformed_foot.stl")

    한 인스턴스를 재사용하면 템플릿 로딩·계측이 1회만 수행되므로,
    FastAPI 에서는 앱 startup 시 싱글턴으로 생성해 두는 것을 권장한다.
    """

    def __init__(
        self,
        template_stl_path: str | Path,
        conf: cfg.DeformConfig | None = None,
        *,
        medial_side: str | None = None,
    ) -> None:
        """템플릿을 로드하고 기준 정보(BBox, 계측치, 제어점)를 초기화한다.

        Args:
            template_stl_path: 기본 발 템플릿 `.stl` 경로.
            conf: 변형 파라미터. None 이면 기본값.
            medial_side: 'ymin' | 'ymax'. None 이면 아치 위치로 자동 판별.

        Raises:
            TemplateLoadError: 파일이 없거나 메쉬가 비어 있는 경우.
        """
        self.conf = conf or cfg.DeformConfig()
        self.conf.validate()

        self.template_path = Path(template_stl_path)
        self.setup_notes: list[str] = []

        # --- 1) 로드 & 정규 좌표계 정렬 ---------------------------------------
        raw = mu.load_mesh(self.template_path)
        template, notes = mu.canonicalize(raw)
        self.setup_notes.extend(notes)

        # --- 2) 템플릿 자체 품질 확보 (구멍 뚫린 템플릿이 들어오는 경우 대비) ---
        template, self.template_quality = mu.ensure_quality(template, conf=self.conf)
        if self.template_quality.warnings:
            self.setup_notes.extend(
                f"템플릿 품질: {w}" for w in self.template_quality.warnings
            )

        self.template_mesh: trimesh.Trimesh = template
        self.frame = FootFrame.from_mesh(template, medial_side=medial_side)

        # --- 3) 기본 정보 캐시 ------------------------------------------------
        self.vertices: np.ndarray = np.asarray(template.vertices, dtype=float)
        self.faces: np.ndarray = np.asarray(template.faces, dtype=np.int64)
        self.bounds: np.ndarray = np.array(
            [self.frame.origin, self.frame.origin + self.frame.extents]
        )
        self.template_measurements: FootMeasurements = mu.measure_foot(template, self.frame)

        # --- 4) 고정 제어점 세팅 ---------------------------------------------
        self.control_point_names: list[str]
        self.control_points: np.ndarray  # (N, 3) 월드 좌표(mm)
        self.control_point_names, self.control_points = self._build_control_points()

        # --- 5) 마지막 실행 결과 ----------------------------------------------
        self.deformed_mesh: trimesh.Trimesh | None = None
        self.last_report: DeformationReport | None = None

    # ==================================================================
    # 제어점
    # ==================================================================

    def _build_control_points(self) -> tuple[list[str], np.ndarray]:
        """해부학적 제어점 + 원거리장 고정용 격자 앵커를 생성한다.

        격자 앵커는 BBox 를 `lattice_padding` 만큼 확장한 위치에 두어,
        RBF 의 외삽(extrapolation)이 발산하지 않도록 잡아주는 역할을 한다.

        Returns:
            (제어점 이름 리스트, (N,3) 월드 좌표 배열)
        """
        names = list(cfg.ANATOMICAL_CONTROL_POINTS)
        uvw = np.array([cfg.ANATOMICAL_CONTROL_POINTS[n] for n in names], dtype=float)

        pad = self.conf.lattice_padding
        axis = np.linspace(-pad, 1.0 + pad, self.conf.lattice_resolution)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)

        lattice_names = [f"lattice_{i:02d}" for i in range(len(grid))]
        all_uvw = np.vstack([uvw, grid])
        all_names = names + lattice_names

        world = self.frame.to_world(all_uvw)

        # 중복 제어점은 RBF 행렬을 특이(singular)하게 만들므로 제거한다.
        _, keep = np.unique(np.round(world, 6), axis=0, return_index=True)
        keep = np.sort(keep)
        return [all_names[i] for i in keep], world[keep]

    # ==================================================================
    # 스케일 계수 & 목표 위치
    # ==================================================================

    def _scale_factors(self, target: FootMeasurements) -> dict[str, float]:
        """목표 계측치 ÷ 템플릿 계측치. 1.0 이면 해당 부위는 그대로 둔다."""
        base = self.template_measurements
        factors: dict[str, float] = {}
        for name in FootMeasurements.field_names():
            t = getattr(base, name)
            g = getattr(target, name)
            factors[name] = float(g / t) if t and g else 1.0
        return factors

    def _apply_scale_field(
        self, points: np.ndarray, factors: dict[str, float]
    ) -> np.ndarray:
        """부위별 스케일 필드를 적용해 임의 점들의 '목표 위치'를 계산한다.

        축별 앵커
            X : 뒤꿈치 끝(x_min) 고정 → 길이 스케일
            Y : 발 중심선 고정 → 뒤꿈치/볼 너비를 u 를 따라 보간, 발목 영역은 발목 너비로 블렌딩
            Z : 바닥(z=0) 고정 → 발목/발등 높이를 u 를 따라 보간

        Args:
            points: (N,3) 월드 좌표.
            factors: `_scale_factors()` 결과.

        Returns:
            (N,3) 변형 목표 좌표.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        uvw = self.frame.to_uvw(points)
        u, w = uvw[:, 0], uvw[:, 2]

        s_len = factors["foot_length_mm"]
        s_heel_w = factors["heel_width_mm"]
        s_ball_w = factors["ball_width_mm"]
        s_ankle_w = factors["ankle_width_mm"]
        s_ankle_h = factors["ankle_height_mm"]
        s_instep_h = factors["instep_height_mm"]

        # --- X: 길이 ---------------------------------------------------
        x = self.frame.x_heel + (points[:, 0] - self.frame.x_heel) * s_len

        # --- Y: 너비 (길이방향 보간 + 발목 영역 블렌딩) -------------------
        width_nodes = [
            s_heel_w,
            s_heel_w,
            0.5 * (s_heel_w + s_ball_w),  # 중족부는 뒤꿈치/볼의 중간
            s_ball_w,
            s_ball_w,
        ]
        s_w = np.interp(u, cfg.WIDTH_PROFILE_U, width_nodes)

        # 뒤쪽 & 높은 곳일수록 '발목 너비'가 지배하도록 섞는다.
        # 복사뼈 계측 밴드에서 가중치가 정확히 1.0 이 되도록 구간을 잡았다.
        u_lo, u_hi = cfg.ANKLE_BLEND_U_RANGE
        w_lo, w_hi = cfg.ANKLE_BLEND_W_RANGE
        ankle_blend = np.clip((u_hi - u) / (u_hi - u_lo), 0.0, 1.0) * np.clip(
            (w - w_lo) / (w_hi - w_lo), 0.0, 1.0
        )
        s_w = s_w * (1.0 - ankle_blend) + s_ankle_w * ankle_blend
        y_center = self.frame.y_center
        y = y_center + (points[:, 1] - y_center) * s_w

        # --- Z: 높이 ---------------------------------------------------
        height_nodes = [s_ankle_h, s_ankle_h, s_instep_h, s_instep_h, s_instep_h]
        s_h = np.interp(u, cfg.HEIGHT_PROFILE_U, height_nodes)
        z_floor = self.frame.z_floor
        z = z_floor + (points[:, 2] - z_floor) * s_h

        return np.column_stack([x, y, z])

    # ==================================================================
    # RBF / TPS 변위장
    # ==================================================================

    def _solve_displacement_field(
        self, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """제어점 변위를 RBF 로 보간해 전체 정점에 적용한 새 정점 배열을 만든다.

        위치가 아니라 **변위(displacement)** 를 보간한다. 변위는 위치보다
        훨씬 완만한 장이라 원거리 외삽이 안정적이다.

        Raises:
            DeformationError: RBF 해가 발산하거나 특이 행렬인 경우.
        """
        try:
            rbf = RBFInterpolator(
                source,
                target - source,
                kernel=self.conf.rbf_kernel,
                smoothing=self.conf.rbf_smoothing,
                degree=self.conf.rbf_degree,
            )
        except Exception as exc:
            raise DeformationError(
                "RBF/TPS 보간기 생성에 실패했습니다. 제어점이 중복되었거나 "
                "동일 평면에 놓여 있는지 확인하세요.",
                detail=str(exc),
            ) from exc

        verts = self.vertices
        displacement = np.empty_like(verts)
        chunk = max(1, self.conf.rbf_chunk_size)
        for start in range(0, len(verts), chunk):  # 대용량 메쉬 메모리 보호
            end = start + chunk
            displacement[start:end] = rbf(verts[start:end])

        if not np.isfinite(displacement).all():
            raise DeformationError("RBF 결과에 NaN/Inf 가 포함되어 있습니다.")

        max_disp = float(np.abs(displacement).max())
        limit = self.conf.max_displacement_ratio * self.frame.length
        if max_disp > limit:
            raise DeformationError(
                f"변위가 비정상적으로 큽니다 (최대 {max_disp:.1f}mm > 허용 {limit:.1f}mm). "
                f"입력 계측치 단위(mm)를 확인하세요.",
                detail={"max_displacement_mm": max_disp},
            )

        self._last_max_displacement = max_disp
        return verts + displacement

    # ==================================================================
    # 아치 국소 변형
    # ==================================================================

    def _arch_weights(self, uvw: np.ndarray) -> np.ndarray:
        """아치 변형의 정점별 가중치 (0~1).

        길이방향 가우시안 × 내측 가중 × 바닥으로부터의 감쇠 의 곱.
        발등·발목은 가중치가 0 이므로 아치만 국소적으로 움직인다.
        """
        u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]

        w_len = np.exp(-0.5 * ((u - self.conf.arch_center_u) / self.conf.arch_sigma_u) ** 2)
        w_med = cfg.medial_arch_weight(v, floor=self.conf.arch_lateral_floor)
        w_up = cfg.smoothstep(1.0 - w / self.conf.arch_z_falloff_w)
        return w_len * w_med * w_up

    def _apply_arch_height(
        self, mesh: trimesh.Trimesh, target_arch_mm: float
    ) -> tuple[trimesh.Trimesh, int]:
        """아치 높이를 목표값에 맞도록 바닥면 Z 를 국소적으로 들어올린다.

        가중치 때문에 아치 정점의 실제 상승량은 `delta * ω*` 이므로
        `delta / ω*` 를 가해 1~2회 만에 수렴시킨다.

        **발산 방지(2026-08-07 실측 확인)**: `arch_apex()`가 찾는 "높이 최고점"과
        `_arch_weights()`가 주는 가중치 최고점이 항상 같은 정점이라는 보장이
        없다 — 특히 SfM 점군처럼 계측치가 잡음 섞인 입력에서, 최고점 정점의
        가중치(ω)가 우연히 작으면 `delta/ω`가 과도하게 커지고, 그 결과 다음
        반복에서 "최고점"이 완전히 다른(방금 과도하게 밀린) 정점으로 옮겨가며
        진동·발산한다(실측: 700~3300% 오차로 폭주하는 스파이크 확인). 두 가지
        안전장치로 막는다: (1) 한 반복에서 가하는 최대 변위를
        `_solve_displacement_field()`와 같은 기준(`max_displacement_ratio *
        frame.length`)으로 상한을 두고, (2) 반복 후 실제로 목표에 더 가까워지지
        않았으면(개선 실패) 그 반복을 버리고 즉시 멈춘다 — "밑져야 본전"이
        아니라 "밑지면 그 시도는 무효"로 만들어 최악의 경우에도 시작 상태보다
        나빠지지 않게 한다.

        Args:
            mesh: RBF 변형이 끝난 메쉬 (내부에서 복사본을 사용).
            target_arch_mm: 목표 아치 높이(mm).

        Returns:
            (보정된 메쉬, 사용한 반복 횟수)
        """
        mesh = mesh.copy()
        used = 0
        step_limit = self.conf.max_displacement_ratio * self.frame.length

        frame = FootFrame.from_mesh(mesh, medial_side=self.frame.medial_side)
        current, apex_index = mu.arch_apex(mesh, frame)
        best_error = abs(target_arch_mm - current)

        for _ in range(max(0, self.conf.arch_iterations)):
            delta = target_arch_mm - current
            if abs(delta) <= self.conf.arch_tolerance_mm:
                break

            uvw = frame.to_uvw(mesh.vertices)
            weights = self._arch_weights(uvw)
            omega = float(weights[apex_index]) if apex_index is not None else 1.0
            omega = max(omega, 0.25)  # 0 근처에서 폭주하지 않도록 하한

            step = delta / omega
            max_step = step_limit / max(float(weights.max()), 1e-6)
            step = float(np.clip(step, -max_step, max_step))

            verts = np.array(mesh.vertices, dtype=float, copy=True)
            verts[:, 2] += step * weights
            candidate = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)

            new_frame = FootFrame.from_mesh(candidate, medial_side=self.frame.medial_side)
            new_current, new_apex_index = mu.arch_apex(candidate, new_frame)
            new_error = abs(target_arch_mm - new_current)
            if new_error >= best_error:
                # 이번 반복이 오히려 목표에서 멀어졌다 — 적용하지 않고 멈춘다.
                break

            mesh, frame, current, apex_index, best_error = (
                candidate, new_frame, new_current, new_apex_index, new_error,
            )
            used += 1

        # 접지면 유지: 최저점을 z=0 으로
        verts = np.array(mesh.vertices, dtype=float, copy=True)
        verts[:, 2] -= verts[:, 2].min()
        mesh.vertices = verts
        return mesh, used

    # ==================================================================
    # 메인 API
    # ==================================================================

    def deform_mesh(self, landmarks_data: dict) -> trimesh.Trimesh:
        """2D 랜드마크 데이터를 반영해 템플릿을 변형한다.

        Args:
            landmarks_data: `schemas.parse_payload()` 가 이해하는 dict.
                이미지 개수는 자유이며, 부족한 계측 항목은 템플릿 값을 유지한다.

        Returns:
            변형된 `trimesh.Trimesh`. 인스턴스의 `deformed_mesh` 에도 저장된다.
            상세 리포트는 `self.last_report` 로 확인한다.

        Raises:
            LandmarkValidationError: 입력 스키마/단위 오류.
            DeformationError: RBF 연산 실패 또는 변위 발산.
            MeshQualityError: strict_quality=True 이고 품질 기준 미달.
        """
        payload: LandmarkPayload = parse_payload(landmarks_data)
        measured, warnings = extract_measurements(payload)
        return self.deform_from_measurements(
            measured, side=payload.side, image_count=len(payload.images),
            control_point_offsets_mm=payload.control_point_offsets_mm, warnings=warnings,
        )

    def deform_from_measurements(
        self,
        measured: FootMeasurements,
        *,
        side: str = "right",
        image_count: int = 0,
        control_point_offsets_mm: dict[str, list[float]] | None = None,
        warnings: list[str] | None = None,
    ) -> trimesh.Trimesh:
        """이미 계측된 `FootMeasurements`로 템플릿을 변형한다(`deform_mesh()`의 공통 핵심).

        `deform_mesh()`는 2D 사진 랜드마크에서 계측치를 뽑는 경로 하나만 지원했는데,
        SfM 점군에서 직접 계측하거나(예: `fit_deformer_to_pointcloud.py`) 실제 스캔의
        측정치를 재사용하는 등 계측치를 다른 경로로 이미 확보한 경우에도 변형 로직을
        그대로 쓸 수 있도록 분리했다.

        Args:
            measured: 계측치(비어 있는 항목은 템플릿 값으로 채워짐).
            side: 리포트에 남길 좌우 표기.
            image_count: 리포트에 남길 입력 이미지 수(사진 경로가 아니면 0).
            control_point_offsets_mm: 제어점 이름별 미세 조정(mm).
            warnings: 상위 호출자가 이미 만든 경고 목록에 이어 붙인다.

        Returns:
            변형된 `trimesh.Trimesh`. 인스턴스의 `deformed_mesh`/`last_report`에도 저장된다.
        """
        warnings = list(warnings) if warnings else []
        control_point_offsets_mm = control_point_offsets_mm or {}

        target = measured.filled_with(self.template_measurements)
        factors = self._scale_factors(target)

        # --- 제어점 목표 위치 --------------------------------------------
        source_cp = self.control_points
        target_cp = self._apply_scale_field(source_cp, factors)

        unknown_offsets = set(control_point_offsets_mm) - set(self.control_point_names)
        if unknown_offsets:
            warnings.append(
                f"알 수 없는 제어점 오프셋은 무시했습니다: {sorted(unknown_offsets)}"
            )
        index_of = {name: i for i, name in enumerate(self.control_point_names)}
        for name, offset in control_point_offsets_mm.items():
            if name in index_of:
                target_cp[index_of[name]] += np.asarray(offset, dtype=float)

        # --- RBF/TPS 전역 변형 --------------------------------------------
        self._last_max_displacement = 0.0
        new_vertices = self._solve_displacement_field(source_cp, target_cp)
        deformed = trimesh.Trimesh(
            vertices=new_vertices, faces=self.faces.copy(), process=False
        )

        # --- 아치 국소 Z 곡률 보정 ------------------------------------------
        arch_target = target.arch_height_mm or self.template_measurements.arch_height_mm
        deformed, arch_iters = self._apply_arch_height(deformed, float(arch_target))

        # --- 품질 검사·복구 --------------------------------------------------
        pre_quality_reference = trimesh.Trimesh(
            vertices=self.vertices, faces=self.faces, process=False
        )
        deformed, quality = mu.ensure_quality(
            deformed, reference=pre_quality_reference, conf=self.conf
        )

        # --- 결과 계측 & 리포트 -----------------------------------------------
        achieved = mu.measure_foot(
            deformed, FootFrame.from_mesh(deformed, medial_side=self.frame.medial_side)
        )
        errors = {
            name: float(getattr(achieved, name) - getattr(target, name))
            for name in FootMeasurements.field_names()
            if getattr(target, name) is not None
        }
        for name, err in errors.items():
            goal = getattr(target, name)
            if goal and abs(err) / goal * 100.0 > self.conf.measurement_tolerance_pct:
                warnings.append(
                    f"'{name}' 오차 {err:+.2f}mm "
                    f"({abs(err) / goal * 100:.1f}%) 가 허용치를 초과했습니다."
                )

        self.last_report = DeformationReport(
            side=side,
            image_count=image_count,
            template_measurements=self.template_measurements.to_dict(),
            target_measurements=target.to_dict(),
            achieved_measurements=achieved.to_dict(),
            scale_factors=factors,
            measurement_error_mm=errors,
            control_point_count=len(source_cp),
            max_displacement_mm=self._last_max_displacement,
            arch_iterations_used=arch_iters,
            quality=quality,
            warnings=warnings + quality.warnings,
        )
        self.deformed_mesh = deformed
        return deformed

    def export_mesh(
        self, output_path: str | Path, mesh: trimesh.Trimesh | None = None
    ) -> Path:
        """변형 결과를 파일로 저장한다.

        Args:
            output_path: `.stl` / `.glb` / `.ply` / `.obj` 경로. 상위 폴더는 자동 생성.
            mesh: 저장할 메쉬. None 이면 마지막 `deform_mesh()` 결과.

        Returns:
            저장된 파일의 `Path`.

        Raises:
            ExportError: 지원하지 않는 확장자이거나 저장할 메쉬가 없는 경우.
        """
        mesh = mesh if mesh is not None else self.deformed_mesh
        if mesh is None:
            raise ExportError(
                "저장할 메쉬가 없습니다. deform_mesh() 를 먼저 호출하세요."
            )

        path = Path(output_path)
        suffix = path.suffix.lower()
        if suffix not in cfg.SUPPORTED_EXPORT_SUFFIXES:
            raise ExportError(
                f"지원하지 않는 확장자: '{suffix}'. "
                f"가능: {sorted(cfg.SUPPORTED_EXPORT_SUFFIXES)}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            mesh.export(path)
        except Exception as exc:
            raise ExportError(f"파일 저장 실패: {path}", detail=str(exc)) from exc

        if not path.is_file() or path.stat().st_size == 0:
            raise ExportError(f"파일이 생성되지 않았거나 비어 있습니다: {path}")
        return path

    # ==================================================================
    # 부가 정보
    # ==================================================================

    def describe_template(self) -> dict:
        """템플릿 요약 정보 — API 의 `GET /template` 응답 등에 사용."""
        return {
            "path": str(self.template_path),
            "vertices": int(len(self.vertices)),
            "faces": int(len(self.faces)),
            "bounds_mm": self.bounds.tolist(),
            "medial_side": self.frame.medial_side,
            "measurements_mm": self.template_measurements.to_dict(),
            "control_points": len(self.control_points),
            "quality": self.template_quality.to_dict(),
            "notes": self.setup_notes,
        }


# ---------------------------------------------------------------------------
# 실행 예시 (스모크 테스트)
#   python -m foot_engine.deformer
# 더 다양한 옵션은 scripts/run_deform_demo.py 참고.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .template_factory import save_reference_template

    ROOT = Path(__file__).resolve().parents[2]
    TEMPLATE = ROOT / "data" / "templates" / "base_foot_template.stl"
    OUTPUT = ROOT / "data" / "output" / "output_deformed_foot.stl"

    # 1) 템플릿이 없으면 절차적 기준 템플릿을 만들어 둔다.
    if not TEMPLATE.is_file():
        save_reference_template(TEMPLATE, length_mm=250.0, side="right")
        print(f"[setup] 기준 템플릿 생성: {TEMPLATE}")

    # 2) 더미 2D 랜드마크 (이미지 2장 — 개수는 자유롭게 늘릴 수 있다)
    DUMMY_LANDMARKS = {
        "meta": {"subject_id": "SMOKE-TEST"},
        "side": "right",
        "images": [
            {
                "view": "top",
                "image_size_px": [1200, 1600],
                "scale_mm_per_px": 0.2,          # px → mm 캘리브레이션
                "landmarks": {
                    "heel_center": [600, 1500],
                    "toe_tip": [600, 160],       # 길이 1340px × 0.2 = 268mm
                    "mtp1_medial": [345, 668],
                    "mtp5_lateral": [855, 640],  # 볼 너비 ≈ 102mm
                    "heel_medial": [435, 1420],
                    "heel_lateral": [765, 1420], # 뒤꿈치 너비 = 66mm
                },
            },
            {
                "view": "medial",
                "image_size_px": [1600, 1200],
                # scale 이 없으면 위 뷰에서 확정된 발 길이를 기준자로 자동 캘리브레이션
                "landmarks": {
                    "heel_back": [145, 872],
                    "toe_tip": [1485, 935],
                    "ground_ref": [800, 950],
                    "arch_apex": [700, 830],     # 아치 높이 = 120px × 0.2 = 24mm
                    "instep_top": [620, 620],    # 발등 높이 = 66mm
                    "medial_malleolus": [300, 590],
                },
            },
        ],
    }

    # 3) 변형 → 저장
    engine = FootMeshDeformer(TEMPLATE)
    result = engine.deform_mesh(DUMMY_LANDMARKS)
    saved = engine.export_mesh(OUTPUT)

    # 4) 리포트 출력
    report = engine.last_report
    assert report is not None
    print("\n".join(report.summary_lines()))
    print(f"\nwatertight : {report.quality.is_watertight}")
    print(f"volume     : {report.quality.volume_mm3:,.0f} mm³")
    print(f"saved      : {saved}")
    for warning in report.warnings:
        print(f"[warn] {warning}")
