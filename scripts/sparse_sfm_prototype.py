"""Sparse SfM — 촬영 사진에서 카메라 포즈 + sparse 포인트클라우드 복원 (CLI).

실제 로직은 `foot_engine.sfm.reconstruction`에 있다 — 이 파일은 인자 파싱과
출력 포맷만 담당하는 얇은 CLI 래퍼다. 다른 코드에서 재사용하려면
`foot_engine.sfm`을 직접 import할 것.

사용 예::

    python scripts/sparse_sfm_prototype.py --video data/samples/foot_test.mp4
    python scripts/sparse_sfm_prototype.py --images-dir data/samples/foot_frames --interval 0.5
    python scripts/sparse_sfm_prototype.py --video data/samples/foot_test.mp4 --blur-keep-ratio 0.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _cli_common import ROOT  # sys.path 설정 + 콘솔 UTF-8 고정(부작용 import) + ROOT 경로

from foot_engine.sfm import reconstruction as sfm  # noqa: E402

DEFAULT_WORKDIR = ROOT / "data" / "output" / "sfm_prototype"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sparse SfM (COLMAP/pycolmap, exhaustive matching)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="순서대로 촬영된 영상 파일")
    src.add_argument("--images-dir", type=Path, help="이미 추출된, 순서가 보존된 이미지 폴더")
    parser.add_argument("--interval", type=float, default=0.5, help="영상에서 프레임을 뽑을 간격(초)")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="중간 산출물 저장 위치")
    parser.add_argument(
        "--blur-keep-ratio", type=float, default=1.0,
        help="선명도 상위 몇 %%만 쓸지(0~1, 기본 1.0=전체 사용). 0.7이면 흐린 하위 30%% 제외.",
    )
    parser.add_argument(
        "--start-time", type=float, default=0.0,
        help="영상에서 사용할 구간의 시작 시각(초). 관계없는 장면을 잘라내고 싶을 때.",
    )
    parser.add_argument(
        "--end-time", type=float, default=None,
        help="영상에서 사용할 구간의 끝 시각(초). 생략하면 끝까지.",
    )
    parser.add_argument(
        "--masks-dir", type=Path, default=None,
        help="배경 제외용 마스크 폴더(`generate_foot_masks.py`로 미리 생성). "
             "지정하면 마스크 밖(배경) 픽셀은 특징점 추출에서 제외한다. 배경이 "
             "복잡/혼재할 때만 쓸 것 — 텍스처 있는 바닥 등 피사체와 함께 고정된 "
             "배경에서는 오히려 등록률을 깎을 수 있다(실측 확인, run_sparse_sfm docstring 참고).",
    )
    parser.add_argument(
        "--max-features", type=int, default=sfm.DEFAULT_MAX_FEATURES,
        help=f"이미지당 최대 SIFT 특징점 수(기본 {sfm.DEFAULT_MAX_FEATURES} — 저텍스처 "
             "피부 대상 실측 튜닝값, COLMAP 기본은 8192).",
    )
    parser.add_argument(
        "--peak-threshold", type=float, default=sfm.DEFAULT_PEAK_THRESHOLD,
        help=f"SIFT 특징점 채택 임계값, 낮을수록 약한 특징점까지 채택(기본 "
             f"{sfm.DEFAULT_PEAK_THRESHOLD} — 실측 튜닝값, COLMAP 기본은 0.00667).",
    )
    parser.add_argument(
        "--ransac-threshold", type=float, default=sfm.DEFAULT_RANSAC_MAX_ERROR,
        help=f"매칭 기하 검증(RANSAC) 인라이어 허용 오차, px 단위(기본 "
             f"{sfm.DEFAULT_RANSAC_MAX_ERROR} — 실측 튜닝값, COLMAP 기본은 4.0). 낮출수록 "
             "엄격해져 재투영 오차가 줄어든다.",
    )
    args = parser.parse_args(argv)

    args.workdir.mkdir(parents=True, exist_ok=True)

    if args.video:
        if not args.video.is_file():
            print(f"[error] 영상 파일이 없습니다: {args.video}", file=sys.stderr)
            return 2
        images_dir = sfm.extract_frames(
            args.video, args.workdir / "images", args.interval,
            start_time=args.start_time, end_time=args.end_time,
        )
    else:
        images_dir = args.images_dir
        if not images_dir.is_dir():
            print(f"[error] 이미지 폴더가 없습니다: {images_dir}", file=sys.stderr)
            return 2

    all_names = sorted(
        p.name for p in images_dir.iterdir() if p.suffix.lower() in sfm.IMAGE_EXTS
    )

    image_names = None
    if args.blur_keep_ratio < 1.0:
        image_names = sfm.filter_blurry_frames(images_dir, args.blur_keep_ratio)
    # 등록 실패 진단은 실제로 SfM에 넘긴 후보(블러 필터링된 경우 그 부분집합) 기준.
    candidate_names = image_names if image_names is not None else all_names
    total_images = len(candidate_names)

    try:
        recon = sfm.run_sparse_sfm(
            images_dir, args.workdir, image_names=image_names, masks_dir=args.masks_dir,
            max_features=args.max_features, peak_threshold=args.peak_threshold,
            ransac_max_error=args.ransac_threshold,
        )
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    ply_path = args.workdir / "sparse_points.ply"
    kept, removed = sfm.export_point_cloud(recon, ply_path)

    print("\n[결과]")
    print(f"  등록된 이미지: {recon.num_reg_images()} / {total_images}")
    print(f"  3D 포인트 수: {recon.num_points3D()} (이상치 {removed}개 제거 -> {len(kept)}개 저장)")
    print(f"  평균 재투영 오차: {recon.compute_mean_reprojection_error():.3f} px")
    print(f"  평균 트랙 길이: {recon.compute_mean_track_length():.2f}")
    print(f"  포인트클라우드 저장: {ply_path}")
    sfm.report_unregistered_frames(images_dir, recon, candidate_names)
    print(
        "\n[다음 확인할 것] 위 .ply 파일을 MeshLab/CloudCompare 등으로 열어 "
        "발 모양이 그럴듯하게 점으로 맺히는지 육안으로 확인하세요. 등록된 이미지 수가 "
        "전체보다 많이 적다면, 촬영 중 발이 움직였거나 사진이 흐릿했을 가능성이 높습니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
