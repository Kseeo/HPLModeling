"""SSM 파이프라인 합성 검증 데모.

실제 스캔 데이터가 아직 없어도, `template_factory.py`의 절차적 템플릿으로 다양한
"가짜 발"(길이·아치 형태를 다르게 준 것)을 만들어 전처리 → 정합 → PCA 전체
파이프라인이 실제로 동작하는지 확인한다. 이 스크립트가 통과하면, 진짜 매니페스트가
준비됐을 때 `scripts/build_ssm.py`로 그대로 이어붙일 수 있다.

사용 예::

    python scripts/ssm_synthetic_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

import numpy as np  # noqa: E402

from foot_engine.ssm import (  # noqa: E402
    clean_and_normalize,
    fit_ssm,
    register_template_to_target,
)
from foot_engine.template_factory import build_reference_foot  # noqa: E402

#: 합성 "발" 몇 개를 길이·아치 형태를 다르게 해서 만든다(실제 개인차의 대용).
_SYNTHETIC_FEET = [
    dict(length_mm=245.0, arch_shape_mm=18.0),
    dict(length_mm=255.0, arch_shape_mm=22.0),
    dict(length_mm=265.0, arch_shape_mm=26.0),
    dict(length_mm=250.0, arch_shape_mm=30.0),
    dict(length_mm=260.0, arch_shape_mm=14.0),
    dict(length_mm=270.0, arch_shape_mm=20.0),
]


def main() -> int:
    print("[1/4] 템플릿 + 합성 스캔 생성 중...")
    template = build_reference_foot(length_mm=250.0, arch_shape_mm=24.0)
    scans = [build_reference_foot(**params) for params in _SYNTHETIC_FEET]
    print(f"      템플릿 정점 {len(template.vertices):,}개, 합성 스캔 {len(scans)}개")

    print("\n[2/4] 전처리(노이즈 제거 + 자기 길이 정규화) 중...")
    template_norm, template_len = clean_and_normalize(template)
    normalized_scans = []
    for i, scan in enumerate(scans):
        norm, own_len = clean_and_normalize(scan)
        normalized_scans.append(norm)
        print(f"      스캔 {i}: 원래 길이 {own_len:.1f}mm -> 정규화 후 250mm 기준으로 통일")

    print("\n[3/4] 비강체 정합 중 (템플릿 -> 각 스캔)...")
    registered_vertices = []
    for i, target in enumerate(normalized_scans):
        result = register_template_to_target(template_norm, target)
        registered_vertices.append(result.vertices)
        print(
            f"      스캔 {i}: {result.iterations_used}회 반복, "
            f"RMS 오차 {result.rms_error_mm:.3f}mm, 대응점 비율 {result.inlier_ratio:.1%}"
        )

    print("\n[4/4] PCA 로 SSM 학습 중...")
    ssm = fit_ssm(registered_vertices, template_norm.faces, n_components=5)
    ratios = ssm.variance_ratio()
    print(f"      정점 {ssm.n_vertices:,}개, 주성분 {ssm.n_components}개")
    print(f"      설명 분산 비율(누적): {np.cumsum(ratios).round(3).tolist()}")

    # --- 검증: 학습에 쓰인 형태를 SSM 계수로 되돌렸을 때 원래 정합 결과와 얼마나 가까운가 ---
    print("\n[검증] 각 스캔을 SSM에 투영 후 복원 -> 정합 결과와의 오차:")
    recon_errors = []
    for i, verts in enumerate(registered_vertices):
        coeffs = ssm.project(verts)
        recon = ssm.generate(coeffs)
        err = float(np.sqrt(np.mean(np.sum((recon - verts) ** 2, axis=1))))
        recon_errors.append(err)
        print(f"      스캔 {i}: RMS 재구성 오차 {err:.4f}mm")

    max_err = max(recon_errors)
    if max_err < 1.0:
        print(f"\n[결과] 통과 — 최대 재구성 오차 {max_err:.4f}mm (< 1mm)")
        print("       파이프라인(전처리/정합/PCA) 세 단계 모두 정상 동작 확인.")
        return 0
    else:
        print(f"\n[결과] 주의 — 최대 재구성 오차 {max_err:.4f}mm 로 다소 큽니다. "
              "정합 파라미터(outlier_distance_mm, 반복 횟수)를 조정해보세요.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
