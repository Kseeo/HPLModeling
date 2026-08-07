"""영상/사진 한 벌 -> 변형된 발 메쉬. `foot_engine.sfm` 전체를 한 번에 돌리는 CLI.

내부적으로 아래 네 단계를 순서대로 실행한다(각 단계를 따로 돌리고 싶으면
`sparse_sfm_prototype.py` / `generate_foot_masks.py` / `clean_point_cloud.py` /
`fit_deformer_to_pointcloud.py`를 개별적으로 쓸 것 — 중간 결과를 실제 뷰어로
확인하고 싶을 때 유용하다):

    1. (영상이면) 프레임 추출
    2. 발/피부 마스크 생성 (rembg + MediaPipe 피부 정제)
    3. Sparse SfM 복원 (pycolmap)
    4. 배경 제거 + 템플릿 워프

사용 예::

    python scripts/run_sfm_pipeline.py --video data/samples/test00.mp4 `
        --template data/templates/S0001_real_template.stl `
        --side right --template-side left `
        --out data/output/test00_pipeline_fit.stl `
        --workdir data/output/sfm_pipeline/test00_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

# cp949 등 비-UTF8 콘솔에서 한글 출력이 깨지거나 죽는 문제 방지(실측 확인, 2026-08-07).
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

from foot_engine.sfm.fitting import default_template_path  # noqa: E402
from foot_engine.sfm.pipeline import run_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="영상/사진 -> 발 메쉬, 파이프라인 전체 실행")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="순서대로 촬영된 영상 파일")
    src.add_argument("--images-dir", type=Path, help="이미 추출된, 순서가 보존된 이미지 폴더")
    parser.add_argument("--workdir", type=Path, required=True, help="중간 산출물 저장 폴더")
    parser.add_argument("--out", type=Path, required=True, help="최종 변형된 메쉬 저장 경로(.stl)")
    parser.add_argument("--template", type=Path, default=None, help="기준 템플릿 STL(생략 시 절차적 생성)")
    parser.add_argument(
        "--side", choices=["left", "right"], default=None,
        help="입력 영상이 실제로 어느 쪽 발인지. --template-side와 다르면 자동 미러링한다. "
             "생략하면 좌우 확인을 건너뛴다(주의: 다르면 결과가 뒤틀림).",
    )
    parser.add_argument(
        "--template-side", choices=["left", "right"], default="right",
        help="템플릿의 실제 발 좌우(기본 right).",
    )
    parser.add_argument("--interval", type=float, default=0.5, help="영상에서 프레임을 뽑을 간격(초)")
    parser.add_argument("--start-time", type=float, default=0.0, help="사용할 구간의 시작 시각(초)")
    parser.add_argument("--end-time", type=float, default=None, help="사용할 구간의 끝 시각(초)")
    parser.add_argument(
        "--no-quality-gate", dest="quality_gate", action="store_false", default=True,
        help="SfM 전 프레임별 절대기준 QC(파일 손상/해상도/노출)를 끈다(기본 on).",
    )
    parser.add_argument(
        "--min-sharpness", type=float, default=None,
        help="QC 블러 절대 임계값(라플라시안 분산). 기본 None=검사 안 함 — "
             "scripts/inspect_frame_quality.py로 실제 영상의 선명도 분포를 보고 정할 것.",
    )
    parser.add_argument(
        "--min-frames", type=int, default=8,
        help="QC/마스킹 게이트 통과 후 최소로 남아야 하는 프레임 수(기본 8).",
    )
    parser.add_argument(
        "--blur-keep-ratio", type=float, default=1.0,
        help="선명도 상위 몇 %%만 쓸지(0~1, 기본 1.0=전체 사용).",
    )
    parser.add_argument(
        "--mask-during-extraction", action="store_true",
        help="특징점 추출 단계에서도 마스크를 적용한다. 배경이 복잡/혼재할 때만 — "
             "텍스처 있는 배경이 피사체와 함께 고정된 촬영에서는 오히려 손해였다(실측 확인).",
    )
    parser.add_argument(
        "--no-skin-refine", dest="skin_refine", action="store_false", default=True,
        help="MediaPipe 피부 정제(옷/장신구 제외)를 끈다.",
    )
    parser.add_argument(
        "--no-cluster", dest="cluster", action="store_false", default=True,
        help="정리된 점군에 DBSCAN 군집화를 적용하지 않는다(기본은 적용).",
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args(argv)

    template_path = args.template or default_template_path(ROOT)

    result = run_pipeline(
        workdir=args.workdir,
        template_path=template_path,
        out_mesh=args.out,
        video=args.video,
        images_dir=args.images_dir,
        side=args.side,
        template_side=args.template_side,
        interval=args.interval,
        start_time=args.start_time,
        end_time=args.end_time,
        quality_gate=args.quality_gate,
        min_sharpness=args.min_sharpness,
        min_frames=args.min_frames,
        blur_keep_ratio=args.blur_keep_ratio,
        mask_during_extraction=args.mask_during_extraction,
        skin_refine=args.skin_refine,
        cluster=args.cluster,
        rng_seed=args.rng_seed,
    )

    print("\n[파이프라인 요약]")
    if result.quality_stats:
        print(f"  QC: {result.quality_stats}")
    print(f"  등록된 이미지: {result.n_points_registered_images} / {result.n_points_total_images}")
    print(f"  sparse 점: {result.n_points_raw:,}개 -> 정리 후: {result.n_points_cleaned:,}개")
    print(f"  마스크: {result.mask_stats}")
    print(f"  최종 메쉬: {result.output_mesh_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
