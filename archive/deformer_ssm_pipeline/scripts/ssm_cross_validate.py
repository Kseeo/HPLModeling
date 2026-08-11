"""캐시된 정합 결과로 SSM 주성분 개수를 K-fold 홀드아웃 검증하는 CLI.

정합(rigid_prealign + 비강체 정합)은 스캔당 20초 이상 걸리는 무거운 단계라,
`build_ssm.py --cache-registered`로 한 번 저장해둔 결과를 재사용한다. 그 덕에
이 스크립트는 몇 초 안에 끝나고, 주성분 개수를 자유롭게 실험할 수 있다.

각 fold에서 일부 스캔을 빼고(홀드아웃) 나머지로만 SSM을 학습한 뒤, 뺀 스캔들에
대해 재구성 오차를 잰다. 학습에 쓰인 데이터로 재는 오차(홀드인)는 항상 낙관적이라
실제 새 발에 얼마나 잘 맞을지의 근거가 될 수 없다 — 홀드아웃 오차만이 진짜 지표다.

사용 예::

    python scripts/ssm_cross_validate.py data/output/registered_cache.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

import numpy as np  # noqa: E402

from foot_engine.ssm import fit_ssm  # noqa: E402


def _k_fold_indices(
    n: int, k: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, test_idx) 쌍을 k개 만든다."""
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    return [
        (np.concatenate([folds[j] for j in range(k) if j != i]), folds[i])
        for i in range(k)
    ]


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SSM 주성분 개수 K-fold 홀드아웃 검증")
    parser.add_argument("cache", type=Path, help="build_ssm.py --cache-registered 로 저장한 .npz")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument(
        "--component-range", default="2,4,6,8,10,15,20,30,40",
        help="시험할 주성분 개수 목록(쉼표 구분)",
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args(argv)

    data = np.load(args.cache, allow_pickle=True)
    vertices = data["vertices"]  # (N, V, 3)
    categories = data["categories"]
    faces = data["faces"]
    n = len(vertices)
    print(f"[cache] {n}건 로드 (정점 {vertices.shape[1]:,}개, {args.k_folds}-fold)")

    rng = np.random.default_rng(args.rng_seed)
    folds = _k_fold_indices(n, args.k_folds, rng)
    component_counts = [int(c) for c in args.component_range.split(",")]

    print(f"\n{'주성분':>6}{'홀드아웃 RMS(mm)':>20}{'홀드인 RMS(mm)':>20}   과적합 격차")
    best_k, best_err = None, float("inf")
    category_report: dict[int, dict[str, list[float]]] = {}

    for k_components in component_counts:
        holdout_errors: list[float] = []
        holdin_errors: list[float] = []
        cat_errors: dict[str, list[float]] = {}

        for train_idx, test_idx in folds:
            if len(train_idx) - 1 < k_components:
                continue  # 이 fold 학습 표본으로는 이 주성분 개수를 못 만듦
            ssm = fit_ssm([vertices[i] for i in train_idx], faces, n_components=k_components)

            for i in train_idx:
                recon = ssm.generate(ssm.project(vertices[i]))
                holdin_errors.append(_rms(recon, vertices[i]))

            for i in test_idx:
                recon = ssm.generate(ssm.project(vertices[i]))
                err = _rms(recon, vertices[i])
                holdout_errors.append(err)
                cat_errors.setdefault(str(categories[i]), []).append(err)

        if not holdout_errors:
            print(f"{k_components:>6}   (표본 부족으로 건너뜀)")
            continue

        mean_holdout = float(np.mean(holdout_errors))
        mean_holdin = float(np.mean(holdin_errors))
        gap = mean_holdout - mean_holdin
        print(f"{k_components:>6}{mean_holdout:>20.3f}{mean_holdin:>20.3f}   {gap:+.3f}mm")
        category_report[k_components] = cat_errors

        if mean_holdout < best_err:
            best_err, best_k = mean_holdout, k_components

    if best_k is not None:
        print(f"\n[추천] 홀드아웃 오차가 가장 낮은 주성분 개수: {best_k} (RMS {best_err:.3f}mm)")
        print(f"\n[카테고리별 홀드아웃 오차] (주성분 {best_k}개 기준)")
        for cat, errs in sorted(category_report[best_k].items()):
            print(f"  {cat:<16}평균 {np.mean(errs):.3f}mm  (n={len(errs)})")

    print(
        "\n[해석] 홀드아웃 오차가 더 안 줄어들거나 다시 늘어나기 시작하는 지점이 "
        "적정 주성분 개수입니다. 홀드인 대비 홀드아웃 오차가 크게 벌어지면(과적합 "
        "격차가 큼) 그 주성분 개수는 표본 수 대비 과합니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
