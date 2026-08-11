"""SfM sparse 포인트클라우드에 SSM을 직접 피팅해서 메쉬 하나를 만드는 엔드투엔드 프로토타입.

`ssm_landmark_holdout.py`는 사람이 이름 붙인 9개 랜드마크의 "정확한" 위치를 안다고
가정했다 — 실제 SfM 파이프라인에는 아직 그 9개를 자동으로 짚어주는 단계
(`triangulate_landmarks.py`, 미구현)가 없다. 대신 지금 당장 손에 있는 건
`sparse_sfm_prototype.py`가 만든 **대응관계 없는 점 수천 개**(배경 잔여 노이즈 포함)
뿐이다. 이 스크립트는 그 점들에 SSM을 직접 들이대는 방식으로 첫 엔드투엔드 결과물을
만든다:

    1. 점군은 자체 스케일이 없다(monocular SfM 특성 — 실측 기준선/마커가 없으면
       절대 축척을 알 수 없다). 일단 점군 자신의 PCA 최장축 길이를 250mm 기준으로
       스케일링해 "형태만" 비교 가능하게 만든다(최종 절대 크기는 나중에 자기신고
       사이즈로 다시 보정해야 한다 — `scan-dataset-characteristics` 메모 참고).
    2. `registration.py`의 강체 사전정렬과 같은 방식(PCA축 맞추기 + 4가지 부호
       조합 중 ICP 비용 최저)으로 점군을 SSM의 캐노니컬 좌표계로 가져온다.
    3. SSM 평균 형상에서 시작해 반복한다: 현재 복원 메쉬의 각 정점에서 점군의
       최근접점을 찾고(대응 없는 점은 outlier로 제외), 그 대응을
       `fit_from_landmarks()`와 같은 능형회귀에 넣어 계수를 갱신, 메쉬를 다시
       생성 — Amberg 류 비강체 ICP를 SSM의 선형 부분공간으로 제한한 버전이다.
    4. 계수가 학습 분포(표준편차 몇 배) 밖으로 벗어나면 경고한다 — 노이즈가 SSM을
       비현실적인 방향으로 잡아당기고 있다는 신호다.

사용 예::

    python scripts/fit_ssm_to_pointcloud.py data/output/ssm_normal_spectrum.npz `
        data/output/sfm_prototype/test02_run/sparse_points.ply --out data/output/test02_fit.stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from foot_engine.sfm.fitting import measured_length as _measured_length  # noqa: E402
from foot_engine.sfm.fitting import rigid_prealign_points as _rigid_prealign_points  # noqa: E402
from foot_engine.ssm import StatisticalShapeModel  # noqa: E402
from foot_engine.ssm.preprocessing import DEFAULT_REFERENCE_LENGTH_MM  # noqa: E402

# `_measured_length`/`_rigid_prealign_points`는 이전엔 이 파일에 독립적으로 구현돼
# 있었다 — `foot_engine.sfm.fitting`(2026-08-07 패키징)에 같은 알고리즘(raw 점군
# 대상 PCA+ICP 강체 정렬)이 이미 있어 중복이었으므로 그쪽을 재사용하도록 정리했다.
# 동작은 동일(단순 이동, 로직 변경 없음).


def fit_ssm_to_points(
    ssm: StatisticalShapeModel,
    target_points: np.ndarray,
    *,
    n_iterations: int = 15,
    outlier_distance_mm: float = 20.0,
    ridge: float = 5.0,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """SSM 평균 형상을 점군에 반복 피팅해 계수를 구한다.

    Returns:
        (coefficients, aligned_target_points, 진단정보 dict)
    """
    aligned_cloud = _rigid_prealign_points(target_points, ssm.mean_shape, rng_seed=rng_seed)
    tree = cKDTree(aligned_cloud)
    coeffs = np.zeros(ssm.n_components)

    inlier_history = []
    rms_history = []
    for _ in range(n_iterations):
        # 방향 주의: "메쉬 정점마다 가장 가까운 점군 점"을 찾는다(그 반대가 아님) —
        # 점군(2,825개)이 메쉬 정점(7,682개)보다 훨씬 성기므로, 여러 정점이 같은
        # 점군 점 하나에 몰릴 수 있다(다대일). len(current) 기준으로 세야 한다.
        current = ssm.generate(coeffs)
        dist, idx = tree.query(current)
        inlier = dist < outlier_distance_mm
        inlier_history.append((int(inlier.sum()), len(current)))
        rms_history.append(float(np.sqrt(np.mean(dist[inlier] ** 2))) if inlier.any() else float("nan"))
        if inlier.sum() < ssm.n_components * 2:
            break

        landmark_indices = np.flatnonzero(inlier)
        landmark_positions = aligned_cloud[idx[inlier]]
        coeffs = ssm.fit_from_landmarks(landmark_indices, landmark_positions, ridge=ridge)

    diagnostics = {"inlier_history": inlier_history, "rms_history": rms_history}
    return coeffs, aligned_cloud, diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SfM 점군에 SSM 피팅 (엔드투엔드 프로토타입)")
    parser.add_argument("ssm", type=Path, help="build_ssm.py가 만든 SSM .npz")
    parser.add_argument("points", type=Path, help="sparse_sfm_prototype.py가 만든 sparse_points.ply")
    parser.add_argument("--out", type=Path, required=True, help="복원된 메쉬 저장 경로(.stl)")
    parser.add_argument("--out-cloud", type=Path, default=None, help="정렬된 점군도 같이 저장(디버깅용 .ply)")
    parser.add_argument("--n-iterations", type=int, default=15)
    parser.add_argument("--outlier-distance-mm", type=float, default=20.0)
    parser.add_argument("--ridge", type=float, default=5.0)
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args(argv)

    ssm = StatisticalShapeModel.load(args.ssm)
    cloud = trimesh.load(args.points)
    raw_points = np.asarray(cloud.vertices)
    print(f"[입력] 점군 {len(raw_points):,}개, SSM 주성분 {ssm.n_components}개")

    own_length = _measured_length(raw_points)
    scale = DEFAULT_REFERENCE_LENGTH_MM / own_length
    scaled_points = raw_points * scale
    print(
        f"[스케일] 점군 자체 PCA 길이 {own_length:.3f} (SfM 임의 단위) -> "
        f"{DEFAULT_REFERENCE_LENGTH_MM:.0f}mm 기준으로 스케일(x{scale:.4f}) "
        "— 절대 축척 아님, 형태 비교용 임시값(실제 크기는 자기신고 사이즈로 추후 보정 필요)"
    )

    coeffs, aligned_cloud, diag = fit_ssm_to_points(
        ssm,
        scaled_points,
        n_iterations=args.n_iterations,
        outlier_distance_mm=args.outlier_distance_mm,
        ridge=args.ridge,
        rng_seed=args.rng_seed,
    )

    print("\n[반복별 진행] inlier 정점 수(메쉬 정점 기준) / 대응점 RMS(mm)")
    for i, ((n_in, n_total), rms) in enumerate(zip(diag["inlier_history"], diag["rms_history"])):
        print(f"  iter {i:2d}: inlier {n_in:5d} / {n_total}   RMS {rms:.2f}mm")

    sigma = np.sqrt(ssm.explained_variance)
    z_scores = coeffs / sigma
    print("\n[계수 진단] 학습 분포 대비 표준편차 배수(|z|>3이면 비현실적 외삽 의심)")
    for i, z in enumerate(z_scores):
        flag = "  <-- 주의" if abs(z) > 3 else ""
        print(f"  PC{i:02d}: coeff={coeffs[i]:8.3f}  z={z:6.2f}{flag}")

    recon = ssm.generate(coeffs)
    mesh = trimesh.Trimesh(vertices=recon, faces=ssm.template_faces, process=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.out)
    print(f"\n[결과] 복원 메쉬 저장: {args.out}")

    if args.out_cloud:
        trimesh.PointCloud(aligned_cloud).export(args.out_cloud)
        print(f"[결과] 정렬된 점군 저장(디버깅용): {args.out_cloud}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
