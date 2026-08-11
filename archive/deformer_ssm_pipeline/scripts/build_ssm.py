"""매니페스트 기반 SSM 구축 오케스트레이션 CLI.

    load_manifest()                         (scan_dataset.py)
        └─ 스캔별: load_mesh → clean_and_normalize   (ssm/preprocessing.py)
              └─ register_template_to_target          (ssm/registration.py)
                    └─ fit_ssm                          (ssm/model.py)
                          └─ 저장 (.npz)

카테고리를 여러 개 묶어 하나의 SSM으로 학습한다(`--categories`, 기본값은 정상 +
연속 변이 계열: normal/flat_foot/cavus/hallux_valgus/other). 평발·요족은 정상과
다른 종류의 발이 아니라 "같은 아치 높이 축의 양 극단"이라, 따로 쪼개기보다 같이
학습시켜야 PCA가 그 변이를 하나의 축으로 제대로 배운다 — 나눠서 학습하면 표본만
쪼개질 뿐이다.

절단(amputee_forefoot/amputee_toe)은 기본값에서 **제외**된다 — 조직이 없는 부위는
정상 발의 연속 변형으로 표현할 수 없어 원천적으로 다른 위상(별도 템플릿)이
필요하기 때문이다. injury도 손상 양상에 따라 다를 수 있어 기본 제외했다.
필요하면 `--categories`로 직접 조합을 지정할 것.

정합이 심하게 실패한 스캔(RMS 오차가 `--max-rms-mm` 초과)은 자동으로 제외하고
경고로 보고한다 — 노이즈가 심하거나 위상이 크게 다른 이상 스캔이 SSM 전체를
망치는 것을 막기 위함이다.

사용 예::

    python scripts/build_ssm.py data/scans/manifest.csv --out data/output/ssm_normal_spectrum.npz
    python scripts/build_ssm.py data/scans/manifest.csv --categories normal,flat_foot --out data/output/ssm.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

import numpy as np  # noqa: E402

from foot_engine import ScanDatasetError, load_manifest  # noqa: E402
from foot_engine import mesh_utils as mu  # noqa: E402
from foot_engine.ssm import (  # noqa: E402
    clean_and_normalize,
    fit_ssm,
    generalized_procrustes_align,
    mirror_side,
    register_template_to_target,
)
from foot_engine.template_factory import build_reference_foot  # noqa: E402

#: 템플릿(`build_reference_foot()`)이 기본으로 만드는 발 방향 — 매니페스트의
#: side가 이와 다르면 정합 전에 `mirror_side()`로 뒤집어 좌우를 맞춘다.
_TEMPLATE_SIDE = "right"

#: 연속적인 형태 변이로 보고 함께 학습시키는 기본 카테고리 조합.
#: 절단(amputee_*)은 위상 자체가 달라 기본에서 제외 — 별도 템플릿이 필요하다.
_DEFAULT_CATEGORIES = "normal,flat_foot,cavus,hallux_valgus,other"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="매니페스트로 SSM 구축")
    parser.add_argument("manifest", type=Path, help="매니페스트 CSV 경로")
    parser.add_argument(
        "--categories", default=_DEFAULT_CATEGORIES,
        help=f"쉼표로 구분된 카테고리 목록, 전부 하나의 SSM으로 묶어 학습 "
             f"(기본: {_DEFAULT_CATEGORIES})",
    )
    parser.add_argument("--stl-root", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="매니페스트가 아직 미완성이어도 채워진 행만 사용",
    )
    parser.add_argument(
        "--template", type=Path, default=None,
        help="정합 기준 템플릿 STL. 생략하면 절차적 기준 템플릿을 자동 생성해서 사용",
    )
    parser.add_argument("--n-components", type=int, default=20, help="SSM 주성분 개수")
    parser.add_argument(
        "--max-rms-mm", type=float, default=5.0,
        help="정합 RMS 오차가 이보다 크면 그 스캔은 제외(기본 5mm)",
    )
    parser.add_argument("--out", type=Path, required=True, help="SSM 저장 경로 (.npz)")
    parser.add_argument(
        "--cache-registered", type=Path, default=None,
        help="정합된 정점 배열(PCA 이전)을 별도로 저장할 경로(.npz). 정합은 스캔당 "
             "20초 이상 걸리는 무거운 단계라, 저장해두면 이후 홀드아웃 검증이나 "
             "주성분 개수 실험을 정합 없이 즉시 반복할 수 있다.",
    )
    args = parser.parse_args(argv)

    try:
        records, skip_warnings = load_manifest(
            args.manifest, stl_root=args.stl_root, skip_incomplete=args.allow_incomplete
        )
    except ScanDatasetError as exc:
        print(f"[error] {exc.message}", file=sys.stderr)
        return 1

    if skip_warnings:
        print(f"[manifest] {len(skip_warnings)}건 건너뜀(미완성)")

    categories = {c.strip().lower() for c in args.categories.split(",") if c.strip()}
    subset = [r for r in records if r.category in categories]

    print(f"[manifest] 카테고리 {sorted(categories)} 묶어서 총 {len(subset)}건")
    for cat in sorted(categories):
        n = sum(1 for r in subset if r.category == cat)
        print(f"  {cat:<16}{n:>5}건")
    if len(subset) < 3:
        print(
            f"[error] SSM을 만들기엔 표본이 부족합니다({len(subset)}건, 최소 3건 필요). "
            "--categories 조합을 넓히거나 데이터를 더 채워주세요.",
            file=sys.stderr,
        )
        return 1

    print("[template] 정합 기준 템플릿 준비 중...")
    if args.template:
        template = mu.load_mesh(args.template)
    else:
        template = build_reference_foot()
        print("         템플릿 STL이 지정되지 않아 절차적 기준 템플릿을 사용합니다.")
    template_norm, _ = clean_and_normalize(template)

    print(f"\n[전처리+정합] {len(subset)}건 처리 중...")
    registered: list = []
    registered_categories: list[str] = []
    registered_scan_ids: list[str] = []
    excluded: list[str] = []
    for record in subset:
        try:
            mesh = mu.load_mesh(record.stl_path)
            normalized, _ = clean_and_normalize(mesh)
            if record.side != _TEMPLATE_SIDE:
                normalized = mirror_side(normalized)
            result = register_template_to_target(template_norm, normalized)
        except Exception as exc:  # noqa: BLE001 — 스캔 하나의 실패로 전체가 멈추면 안 됨
            excluded.append(f"{record.scan_id}: 처리 실패 ({exc})")
            continue

        if result.rms_error_mm > args.max_rms_mm:
            excluded.append(
                f"{record.scan_id}: 정합 RMS {result.rms_error_mm:.2f}mm "
                f"> 허용 {args.max_rms_mm}mm — 제외"
            )
            continue

        registered.append(result.vertices)
        registered_categories.append(record.category)
        registered_scan_ids.append(record.scan_id)
        print(
            f"  {record.scan_id} [{record.category}]: RMS {result.rms_error_mm:.3f}mm, "
            f"대응점 {result.inlier_ratio:.1%}"
        )

    if excluded:
        print(f"\n[제외됨] {len(excluded)}건:")
        for msg in excluded:
            print(f"  - {msg}")

    if len(registered) < 3:
        print(
            f"[error] 정합에 성공한 스캔이 {len(registered)}건뿐이라 SSM을 만들 수 없습니다.",
            file=sys.stderr,
        )
        return 1

    # `register_template_to_target()`의 출력은 각 타겟 스캔 자신의 원본 좌표계에
    # 놓여 있다 — 실제 스캔 파일마다 절대 위치/회전 관례가 제각각이라, 이 상태로
    # 바로 PCA를 돌리면 주성분이 발 모양이 아니라 스캔별 자세(회전+이동) 차이를
    # 먼저 흡수해버린다. GPA로 공통 좌표계에 정렬한 뒤에야 PCA 의미가 있다.
    print(f"\n[정렬] {len(registered)}건 GPA(Generalized Procrustes)로 공통 좌표계 정렬 중...")
    # reference=template_norm.vertices: 데이터 자신의 평균으로 수렴시키면(고전적
    # GPA) 좌표계가 임의 방향이 되어 measure_foot()이 가정하는 X=길이·Z=높이
    # 관례가 깨진다 — 템플릿(이미 그 관례를 따름)에 고정 정렬해 관례를 물려받는다.
    aligned = generalized_procrustes_align(
        np.stack(registered), reference=template_norm.vertices
    )
    registered = list(aligned)

    if args.cache_registered:
        args.cache_registered.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.cache_registered,
            vertices=aligned,
            categories=np.array(registered_categories),
            scan_ids=np.array(registered_scan_ids),
            faces=template_norm.faces,
        )
        print(f"[캐시] 정합+정렬된 정점 {len(registered)}건 저장: {args.cache_registered}")

    print(f"\n[PCA] {len(registered)}건으로 SSM 학습 중...")
    ssm = fit_ssm(registered, template_norm.faces, n_components=args.n_components)
    ssm.save(args.out)

    ratios = ssm.variance_ratio().cumsum()
    print(f"\n[결과] {args.out} 저장 완료")
    print(f"  정점 {ssm.n_vertices:,}개, 주성분 {ssm.n_components}개")
    print(f"  누적 설명 분산: {[round(float(r), 3) for r in ratios]}")

    # 카테고리별 재구성 오차 — 소수 카테고리(요족/무지외반 등)를 정상군과 묶어 학습했을 때
    # SSM이 그 변이도 잘 담아냈는지 확인하는 지표. 이 홀드인(hold-in) 오차가 크다고
    # "묶어서 학습한 게 잘못"인 건 아니고, "그 방향 데이터가 더 필요하다"는 신호로 본다.
    print("\n[카테고리별 재구성 오차] (학습에 쓰인 데이터 기준 — 진짜 검증은 홀드아웃 필요)")
    per_category_errors: dict[str, list[float]] = {}
    for verts, cat in zip(registered, registered_categories):
        coeffs = ssm.project(verts)
        recon = ssm.generate(coeffs)
        err = float(np.sqrt(np.mean(np.sum((recon - verts) ** 2, axis=1))))
        per_category_errors.setdefault(cat, []).append(err)
    for cat, errs in sorted(per_category_errors.items()):
        print(f"  {cat:<16}평균 {sum(errs) / len(errs):.3f}mm  (n={len(errs)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
