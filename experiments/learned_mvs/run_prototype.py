"""학습 기반 MVS(PatchmatchNet) 프로토타입 -- 발끝/발날 관측-부족 문제 완화 시도.

`run_sfm_pipeline.py --keep-intermediates`로 만든 run 폴더(images/+sparse/ 포함)를
받아 벤더 PatchmatchNet으로 depth 추정 + 융합까지 실행하고 점군(fused.ply)을 낸다.
메쉬 생성/후처리는 하지 않는다 -- 먼저 점군 밀도가 실제로 나아지는지만 본다.

사전 조건: `setup_vendor.py` 1회 실행 + requirements.txt 설치 + GPU(CUDA).
자세한 사용법은 이 폴더의 README.md 참고.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).parent
VENDOR_DIR = EXPERIMENT_ROOT / "_vendor" / "PatchmatchNet"
DEFAULT_CHECKPOINT = VENDOR_DIR / "checkpoints" / "params_000007.ckpt"

# 메인 파이프라인의 undistort_for_dense()를 재사용 -- OpenMVS로 넘기기 직전과
# 동일한 COLMAP undistort 워크스페이스(images/+sparse/)를 만들어준다.
sys.path.insert(0, str(EXPERIMENT_ROOT.parent.parent / "src"))
from foot_engine.sfm import dense  # noqa: E402


def _check_vendor() -> None:
    if not VENDOR_DIR.is_dir():
        raise FileNotFoundError(
            f"벤더 코드가 없습니다: {VENDOR_DIR} -- 먼저 `python "
            f"{EXPERIMENT_ROOT / 'setup_vendor.py'}`를 실행하세요."
        )
    if not DEFAULT_CHECKPOINT.exists():
        raise FileNotFoundError(f"사전학습 체크포인트가 없습니다: {DEFAULT_CHECKPOINT}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="학습 기반 MVS(PatchmatchNet) 프로토타입")
    parser.add_argument(
        "run_dir", type=Path,
        help="run_sfm_pipeline.py --keep-intermediates 로 만든 run 폴더(images/, sparse/ 포함)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="산출물 폴더(기본 <run_dir>/pmnet_mvs)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="PatchmatchNet 체크포인트 경로")
    parser.add_argument(
        "--num-views", type=int, default=10,
        help="depth 추정 시 참조할 소스뷰 개수(기본 10, PatchmatchNet 기본값). "
             "촬영 프레임 수가 적으면 그보다 낮출 것.",
    )
    parser.add_argument("--image-max-dim", type=int, default=2048, help="추정 전 이미지 최대 변 길이(기본 2048)")
    parser.add_argument(
        "--geo-mask-thres", type=int, default=3,
        help="기하 일관성 필터 -- 이 개수 이상의 다른 뷰와 일치해야 살아남는다(기본 3, "
             "PatchmatchNet 논문 기본은 5지만 발 촬영은 뷰 수가 적어 낮춤).",
    )
    parser.add_argument("--photo-thres", type=float, default=0.5, help="광도(포토메트릭) 신뢰도 임계값(기본 0.5)")
    parser.add_argument(
        "--python", type=str, default=sys.executable,
        help="벤더 스크립트를 실행할 파이썬(기본: 지금 이 스크립트를 실행 중인 인터프리터 -- "
             "torch가 설치된 venv와 같아야 함)",
    )
    args = parser.parse_args(argv)

    _check_vendor()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir가 없습니다: {run_dir}")
    # 벤더 스크립트는 자기 폴더를 cwd로 실행하므로(아래 cwd=VENDOR_DIR), 상대경로를
    # 그대로 넘기면 엉뚱한 위치를 가리킨다 -- 전부 절대경로로 해석해서 넘긴다.
    out_dir = (args.out_dir or (run_dir / "pmnet_mvs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()

    sparse_dir = dense.largest_sparse_dir(run_dir / "sparse")
    images_dir = run_dir / "images"

    print("[1/3] COLMAP undistort 워크스페이스 생성 (OpenMVS 직전 산출물과 동일)...")
    colmap_workspace = out_dir / "colmap_undistorted"
    dense.undistort_for_dense(sparse_dir, images_dir, colmap_workspace)

    print("[2/3] PatchmatchNet 입력 포맷으로 변환 (cams/, pair.txt)...")
    mvsnet_dir = out_dir / "mvsnet_input"
    mvsnet_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            args.python, str(VENDOR_DIR / "colmap_input.py"),
            "--input_folder", str(colmap_workspace),
            "--output_folder", str(mvsnet_dir),
        ],
        check=True, cwd=str(VENDOR_DIR),
    )

    print("[3/3] depth 추정 + 융합 (GPU 필요)...")
    subprocess.run(
        [
            args.python, str(VENDOR_DIR / "eval.py"),
            "--input_folder", str(mvsnet_dir),
            "--output_folder", str(out_dir),
            "--checkpoint_path", str(checkpoint),
            "--num_views", str(args.num_views),
            "--image_max_dim", str(args.image_max_dim),
            "--geo_mask_thres", str(args.geo_mask_thres),
            "--photo_thres", str(args.photo_thres),
        ],
        check=True, cwd=str(VENDOR_DIR),
    )

    fused_ply = out_dir / "fused.ply"
    print(f"\n완료: {fused_ply if fused_ply.exists() else '(fused.ply를 찾을 수 없음 -- 로그 확인)'}")
    print("기존 dense_mvs 점군과 발끝/발날 부위 밀도를 육안 비교해볼 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
