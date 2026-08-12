"""SfM sparse 포인트클라우드에서 배경을 걷어내고 피사체(발)만 남기는 CLI.

실제 로직은 `foot_engine.sfm.cleaning`에 있다 — 이 파일은 인자 파싱과 출력
포맷만 담당하는 얇은 CLI 래퍼다. 방식 설명(마스크 기반 vs 기하학적 폴백,
`--intersect`/`--cluster`의 실측된 위험/이점)은 그 모듈의 docstring 참고.

**최종 권장 조합: `--masks-dir ... --cluster` (교집합 없이).**

사용 예(권장 조합)::

    python scripts/generate_foot_masks.py data/output/sfm_prototype/test02_run/images `
        --out data/output/sfm_prototype/test02_run/masks
    python scripts/clean_point_cloud.py data/output/sfm_prototype/test02_run/sparse/0 `
        --masks-dir data/output/sfm_prototype/test02_run/masks `
        --cluster `
        --out data/output/sfm_prototype/test02_run/cleaned_points.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _cli_common  # noqa: F401  -- sys.path 설정 + 콘솔 UTF-8 고정(부작용 import)

import pycolmap  # noqa: E402
import trimesh  # noqa: E402

from foot_engine.sfm import cleaning  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SfM 포인트클라우드에서 배경 제거")
    parser.add_argument("recon_dir", type=Path, help="pycolmap sparse 복원 폴더")
    parser.add_argument("--out", type=Path, required=True, help="정리된 포인트클라우드 저장 경로(.ply)")
    parser.add_argument(
        "--masks-dir", type=Path, default=None,
        help="`generate_foot_masks.py`로 만든 마스크 폴더. 지정하면 마스크 기반 분류를 "
             "쓴다(권장 — 평면 가정이 없어 더 안전하다). 생략하면 아래 기하학적 방식으로 폴백.",
    )
    parser.add_argument(
        "--intersect", action="store_true",
        help="--masks-dir와 함께 쓰면 마스크 분류와 기하학적 방식을 둘 다 돌려서 "
             "교집합(둘 다 배경이라고 동의한 점)만 제거한다. 실험적 — 발 형상을 깎아낼 "
             "위험이 실측으로 확인됐다(`foot_engine.sfm.cleaning` docstring 참고).",
    )
    parser.add_argument(
        "--mask-min-ratio", type=float, default=0.5,
        help="점의 관측 중 마스크 안쪽 비율이 이 이상이면 발로 분류(기본 0.5).",
    )
    parser.add_argument("--n-planes", type=int, default=2, help="[폴백] 제거할 지배적 평면 개수(기본 2)")
    parser.add_argument(
        "--plane-threshold", type=float, default=None,
        help="[폴백] 평면으로 볼 거리 임계값. 생략하면 점군 크기에서 자동 추정(MAD 기반).",
    )
    parser.add_argument(
        "--max-plane-ratio", type=float, default=0.4,
        help="[폴백] 후보 평면이 전체 점의 이 비율을 넘게 먹으면 버린다(기본 0.4) — "
             "임계값이 잘못 커져 피사체까지 잘라내는 사고를 막는 안전장치.",
    )
    parser.add_argument(
        "--focus-percentile", type=float, default=80.0,
        help="[폴백] 카메라들의 공통 초점에서 가까운 몇 %%를 남길지(기본 80). 너무 낮추면 "
             "피사체 일부까지 잘려나갈 수 있으니 보수적으로 잡는다.",
    )
    parser.add_argument("--cluster-eps", type=float, default=None, help="DBSCAN 반경. 생략하면 자동 추정")
    parser.add_argument(
        "--cluster", action="store_true",
        help="정리 뒤에 군집화(DBSCAN)까지 추가로 적용한다. 기본은 꺼져 있다 — sparse한 "
             "점군에서 군집화가 피사체를 조각내는 부작용이 실측으로 확인됐다(모듈 docstring 참고).",
    )
    args = parser.parse_args(argv)

    if args.intersect and args.masks_dir is None:
        print("[error] --intersect는 --masks-dir와 같이 써야 합니다.", file=sys.stderr)
        return 2

    recon = pycolmap.Reconstruction(args.recon_dir)
    result = cleaning.clean_point_cloud(
        recon,
        masks_dir=args.masks_dir,
        intersect=args.intersect,
        mask_min_ratio=args.mask_min_ratio,
        n_planes=args.n_planes,
        plane_threshold=args.plane_threshold,
        max_plane_ratio=args.max_plane_ratio,
        focus_percentile=args.focus_percentile,
        cluster=args.cluster,
        cluster_eps=args.cluster_eps,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(result).export(args.out)
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
