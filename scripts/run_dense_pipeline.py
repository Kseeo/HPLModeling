"""Sparse SfM 결과 -> OpenMVS dense 포인트클라우드/메쉬. `foot_engine.sfm.dense` CLI.

실제 로직은 `foot_engine.sfm.dense`에 있다 — 이 파일은 인자 파싱/출력만
담당하는 얇은 CLI 래퍼다. 선택적 단계 — 실제 3D 메쉬가 필요할 때만 쓴다.

**OpenMVS 별도 설치 필요**: `OPENMVS_BIN_DIR` 환경변수를 실행파일 폴더로
설정하거나 `--openmvs-bin`으로 직접 넘길 것. 설치 방법은 README 참고.

dense 파라미터만 바꿔 재실행하려면(sparse SfM 재계산 없이) 1번을
`--keep-intermediates`로 돌려 images/sparse를 남겨둬야 한다. 재튜닝이
끝났으면 `--cleanup-sfm-scratch`로 남은 sparse SfM 찌꺼기를 지운다.

사용 예::

    # 1) 먼저 sparse SfM까지 (dense 파라미터를 나중에 따로 튜닝할 거라 산출물 보존)
    python scripts/run_sfm_pipeline.py --video data/samples/test02.mp4 `
        --workdir data/output/sfm_pipeline/test02_run --out data/output/test02_fit.stl `
        --keep-intermediates

    이후 #2, 3 중 택 1

    # 2) RefineMesh 제외, dense mesh 생성
    python scripts/run_dense_pipeline.py data/output/sfm_pipeline/test02_run

    # 3) 최종 품질까지 원하면 RefineMesh 포함 (느림, 전체 시간의 70%+ 차지)
    python scripts/run_dense_pipeline.py data/output/sfm_pipeline/test02_run --refine
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import trimesh

import _cli_common  # noqa: F401  -- sys.path 설정 + 콘솔 UTF-8 고정(부작용 import)
from _dense_cli_args import add_dense_args  # noqa: E402

from foot_engine.sfm.dense import finalize_mesh, largest_sparse_dir, run_dense_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="sparse SfM 결과 -> OpenMVS dense 포인트클라우드/메쉬 (선택적 단계)"
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="run_sfm_pipeline.py의 --workdir (images/, sparse/ 를 포함하는 폴더 — "
             "그 실행에서 --keep-intermediates를 켰어야 함)",
    )
    parser.add_argument(
        "--sparse-dir", type=Path, default=None,
        help="sparse 재구성 폴더를 직접 지정(생략 시 <run_dir>/sparse 아래에서 등록 이미지가 "
             "가장 많은 폴더를 자동으로 찾는다 — 번호가 항상 크기순은 아니다).",
    )
    parser.add_argument(
        "--masks-dir", type=Path, default=None,
        help="마스크 폴더(생략 시 <run_dir>/masks_dense 재사용, 없으면 dilate=0으로 새로 생성).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="산출물 폴더(기본 <run_dir>/dense_mvs)")
    parser.add_argument(
        "--reference-length-mm", type=float, default=None,
        help="자기신고 발길이(mm). 있으면 최종 메쉬를 이 길이에 맞춰 스케일링한다"
             "(SfM은 절대 축척이 없음). 생략하면 250mm placeholder로 스케일링하고 경고 출력.",
    )
    parser.add_argument(
        "--cleanup-sfm-scratch", action="store_true",
        help="완료 후 run_dir 안에서 dense 단계가 이미 다 써서 더는 필요 없는 sparse SfM "
             "찌꺼기(masks/, database.db, sparse_points.ply, cleaned_points.ply)를 지운다. "
             "dense 파라미터 재튜닝에 또 쓰이는 images/sparse/masks_dense는 건드리지 않는다.",
    )
    add_dense_args(parser)  # --openmvs-bin, --refine, --max-threads 등 (dense.py와 공유)
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    sparse_dir = args.sparse_dir or largest_sparse_dir(run_dir / "sparse")
    print(f"[dense] sparse 폴더: {sparse_dir}")

    masks_dir = args.masks_dir
    if masks_dir is None:
        masks_dir = run_dir / "masks_dense"
        if masks_dir.is_dir() and any(masks_dir.iterdir()):
            # 있으면 재사용(재생성 안 함)
            print(f"[dense] 기존 마스크 재사용: {masks_dir}")
        else:
            from foot_engine.sfm.masking import generate_masks
            stats = generate_masks(run_dir / "images", masks_dir, dilate=0)
            print(f"[dense] 마스크(dilate=0) 새로 생성: {stats}")

    out_dir = args.out_dir or (run_dir / "dense_mvs")
    mesh_path = run_dense_pipeline(
        sparse_dir=sparse_dir,
        images_dir=run_dir / "images",
        masks_dir=masks_dir,
        workdir=out_dir,
        openmvs_bin=args.openmvs_bin,
        refine=args.refine,
        postprocess_dmaps=args.postprocess_dmaps,
        max_threads=args.max_threads,
        visibility_filter_threshold=args.visibility_filter_threshold,
        grazing_filter_min_score=args.grazing_filter_min_score,
        reprojection_consistency_min_vote=args.reprojection_consistency_min_vote,
        free_space_support=args.free_space_support,
        thickness_factor=args.thickness_factor,
        quality_factor=args.quality_factor,
        refine_decimate=args.refine_decimate,
        refine_regularity_weight=args.refine_regularity_weight,
        smooth_high_curvature=args.smooth_high_curvature,
        curvature_percentile=args.curvature_percentile,
        curvature_rings=args.curvature_rings,
        curvature_iterations=args.curvature_iterations,
        curvature_alpha=args.curvature_alpha,
        fill_holes=args.fill_holes,
        sand_surface_enabled=args.sand_surface_enabled,
        prune_protrusions=args.prune_protrusions,
        keep_intermediates=args.keep_intermediates,
    )

    # 축 정렬 + 스케일링 + 바닥 정착 후처리, 같은 자리에 덮어쓴다.
    mesh = trimesh.load(mesh_path, process=False)
    mesh, scale_factor = finalize_mesh(mesh, reference_length_mm=args.reference_length_mm)
    mesh.export(mesh_path)

    print(f"\n최종 메쉬: {mesh_path} (정점 {len(mesh.vertices):,}개, 스케일 x{scale_factor:.4f})")

    if args.cleanup_sfm_scratch:
        removed = []
        for name in ("masks", "database.db", "sparse_points.ply", "cleaned_points.ply"):
            target = run_dir / name
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                removed.append(name)
            elif target.is_file():
                target.unlink()
                removed.append(name)
        if removed:
            print(f"[정리] sparse SfM 찌꺼기 삭제: {', '.join(removed)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
