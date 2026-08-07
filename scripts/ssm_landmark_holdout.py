"""SfM 랜드마크 기반 SSM 피팅 — 실제 흐름 그대로 홀드아웃 검증 + 시각화용 메쉬 저장.

`ssm_cross_validate.py`는 정합된 전체 정점(7,682개)을 다 안다고 가정하고 SSM에
투영했다 — 이건 SSM 표현력의 상한선을 재는 것이지, 실제 서비스 흐름과는 다르다.
실제로는 사진에서 SfM으로 삼각측량한 **랜드마크 몇 개**만 갖고 전체 형태를
역산해야 한다. 이 스크립트는 그 흐름을 그대로 흉내낸다:

    1. 홀드아웃 스캔에서 미리 정한 랜드마크 정점(뒤꿈치·아치·볼너비 등) 몇 개의
       "실측" 위치만 남기고 나머지 정점은 안 본 것처럼 가린다.
    2. `StatisticalShapeModel.fit_from_landmarks()`로 그 몇 개만으로 계수를 역산.
    3. 복원된 전체 메쉬와 실제 홀드아웃 스캔을 `mesh_utils.measure_foot()`로 재서
       계측치별(아치 높이, 볼너비 등) 오차를 비교한다 — 메쉬 전체 평균이 아니라
       실제 인솔 설계에 쓰이는 항목별로 본다.
    4. 인솔 설계에 실제로 중요한 건 계측치 스칼라 몇 개가 아니라 **족저면(발바닥)
       + 그로부터 몇 cm 위까지의 곡률**이므로, 그 구간에 속한 정점만 따로 골라
       정점 단위 오차(RMS/평균/중앙값/최대)를 추가로 낸다(`--plantar-band-mm`).
       발등·발목은 인솔 설계에 중요하지 않으므로 이 지표에서 제외된다.
    5. 예시 몇 개는 실제/복원 메쉬를 STL로 내보내 눈으로 직접 비교할 수 있게 한다.

사용 예::

    python scripts/ssm_landmark_holdout.py data/output/registered_cache.npz --n-components 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from foot_engine import config as cfg  # noqa: E402
from foot_engine import mesh_utils as mu  # noqa: E402
from foot_engine.schemas import FootMeasurements  # noqa: E402
from foot_engine.ssm import clean_and_normalize, fit_ssm  # noqa: E402
from foot_engine.template_factory import build_reference_foot  # noqa: E402

#: 실제 촬영 사진에서 뽑을 수 있다고 가정하는 랜드마크
#: (7개 계측 항목의 근거가 되는 지점들 — `config.ANATOMICAL_CONTROL_POINTS`의 부분집합)
LANDMARK_NAMES = [
    "heel_back_center",
    "heel_medial",
    "heel_lateral",
    "ankle_medial",
    "ankle_lateral",
    "arch_apex",
    "ball_medial_mtp1",
    "ball_lateral_mtp5",
    "toe_tip_center",
]


def resolve_landmark_indices(template: trimesh.Trimesh, names: list[str]) -> dict[str, int]:
    """이름 붙은 해부학적 지점(uvw)을 템플릿에서 가장 가까운 정점 인덱스로 변환."""
    frame = mu.FootFrame.from_mesh(template)
    result = {}
    for name in names:
        uvw = np.array(cfg.ANATOMICAL_CONTROL_POINTS[name])
        world = frame.to_world(uvw[None, :])[0]
        idx = int(np.argmin(np.linalg.norm(template.vertices - world, axis=1)))
        result[name] = idx
    return result


def _measure(vertices: np.ndarray, faces: np.ndarray) -> FootMeasurements:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mu.measure_foot(mesh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="랜드마크 기반 SSM 피팅 홀드아웃 검증")
    parser.add_argument("cache", type=Path, help="build_ssm.py --cache-registered 로 저장한 .npz")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0, help="랜드마크 피팅 정칙화 강도")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--export-examples", type=int, default=3,
        help="실제/복원 메쉬 쌍을 STL로 내보낼 예시 개수(기본 3)",
    )
    parser.add_argument(
        "--plantar-band-mm", type=float, default=40.0,
        help="바닥(각 스캔 자신의 z 최솟값)에서 이 높이(mm)까지의 정점만 골라 "
             "'족저면+아치 곡률' 오차를 따로 낸다(기본 40mm — 인솔이 실제로 접촉/"
             "지지하는 범위). 발등·발목처럼 이 범위 밖은 인솔 설계에 중요하지 않다.",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "output" / "ssm_landmark_holdout")
    args = parser.parse_args(argv)

    data = np.load(args.cache, allow_pickle=True)
    vertices = data["vertices"]  # (N, V, 3)
    categories = data["categories"]
    scan_ids = data["scan_ids"]
    faces = data["faces"]
    n = len(vertices)
    print(f"[cache] {n}건 로드 (정점 {vertices.shape[1]:,}개)")

    template = build_reference_foot()
    template_norm, _ = clean_and_normalize(template)
    landmark_idx_by_name = resolve_landmark_indices(template_norm, LANDMARK_NAMES)
    landmark_indices = np.array(list(landmark_idx_by_name.values()))
    print(f"[랜드마크] {len(landmark_indices)}개 지점 사용: {list(landmark_idx_by_name.keys())}")

    rng = np.random.default_rng(args.rng_seed)
    order = rng.permutation(n)
    folds = np.array_split(order, args.k_folds)

    measurement_errors: dict[str, list[float]] = {name: [] for name in FootMeasurements.field_names()}
    overall_rms: list[float] = []
    plantar_errors: list[float] = []  # 스캔당 족저 밴드 정점 RMS
    plantar_errors_by_cat: dict[str, list[float]] = {}
    examples: list[tuple[str, np.ndarray, np.ndarray]] = []  # (scan_id, true_verts, recon_verts)

    for i in range(args.k_folds):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(args.k_folds) if j != i])
        if len(train_idx) - 1 < args.n_components:
            continue
        ssm = fit_ssm([vertices[j] for j in train_idx], faces, n_components=args.n_components)

        for j in test_idx:
            true_verts = vertices[j]
            landmark_positions = true_verts[landmark_indices]
            coeffs = ssm.fit_from_landmarks(landmark_indices, landmark_positions, ridge=args.ridge)
            recon_verts = ssm.generate(coeffs)

            overall_rms.append(float(np.sqrt(np.mean(np.sum((recon_verts - true_verts) ** 2, axis=1)))))

            # 족저면+아치 밴드: 실제 스캔 자신의 바닥(z 최솟값) 기준으로 정의해야
            # GPA로 정점이 어디로 평행이동됐든(중심 정렬이라 z=0=바닥이 아님)
            # 항상 "그 발의 바닥"을 가리킨다.
            floor_z = true_verts[:, 2].min()
            band_mask = (true_verts[:, 2] - floor_z) <= args.plantar_band_mm
            if band_mask.any():
                band_err = np.linalg.norm(recon_verts[band_mask] - true_verts[band_mask], axis=1)
                scan_rms = float(np.sqrt(np.mean(band_err**2)))
                plantar_errors.append(scan_rms)
                plantar_errors_by_cat.setdefault(str(categories[j]), []).append(scan_rms)

            true_m = _measure(true_verts, faces)
            recon_m = _measure(recon_verts, faces)
            for name in FootMeasurements.field_names():
                t, r = getattr(true_m, name), getattr(recon_m, name)
                if t is not None and r is not None:
                    measurement_errors[name].append(abs(t - r))

            if len(examples) < args.export_examples:
                examples.append((str(scan_ids[j]), true_verts, recon_verts))

    print(f"\n[전체 메쉬 RMS] 평균 {np.mean(overall_rms):.3f}mm (참고용 — 발등/발목까지 포함한 상한선)")

    print(
        f"\n[족저면+아치 밴드 오차(mm)] 바닥에서 {args.plantar_band_mm:.0f}mm 이내 정점만"
        "(인솔이 실제로 접촉/지지하는 범위 — 가장 중요한 지표)"
    )
    print(
        f"  전체            평균 {np.mean(plantar_errors):6.2f}mm   중앙값 {np.median(plantar_errors):6.2f}mm   "
        f"최대 {np.max(plantar_errors):6.2f}mm  (n={len(plantar_errors)})"
    )
    for cat, errs in sorted(plantar_errors_by_cat.items()):
        print(f"  {cat:<16}평균 {np.mean(errs):6.2f}mm   중앙값 {np.median(errs):6.2f}mm  (n={len(errs)})")

    print("\n[계측치별 절대 오차(mm)] — 랜드마크 몇 개만으로 역산한 결과 vs 실제 (참고용)")
    for name, errs in measurement_errors.items():
        if not errs:
            continue
        print(
            f"  {name:<20}평균 {np.mean(errs):6.2f}mm   중앙값 {np.median(errs):6.2f}mm   "
            f"최대 {np.max(errs):6.2f}mm  (n={len(errs)})"
        )

    if args.export_examples:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[예시 메쉬 저장] {args.out_dir}")
        for scan_id, true_v, recon_v in examples:
            true_path = args.out_dir / f"{scan_id}_true.stl"
            recon_path = args.out_dir / f"{scan_id}_reconstructed.stl"
            trimesh.Trimesh(vertices=true_v, faces=faces, process=False).export(true_path)
            trimesh.Trimesh(vertices=recon_v, faces=faces, process=False).export(recon_path)
            print(f"  {scan_id}: {true_path.name} (실제) / {recon_path.name} (랜드마크 {len(landmark_indices)}개로 복원)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
