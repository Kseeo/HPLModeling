"""사진 1장(top view) → 실루엣 랜드마크 자동 추출 → deformer 3D 모델. 엔드투엔드 데모.

`silhouette_landmarks.extract_top_view_landmarks()`로 뽑은 픽셀 좌표를 그대로
`landmarks.py`/`deformer.py`가 기대하는 payload에 넣어 실제로 변형까지 돌린다 —
"표준 템플릿 + 이름 붙은 landmark + 사진 대응" 경로의 첫 실사진 종단 테스트.

절대 축척은 아직 모르므로(단안 사진 1장 + 기준자 없음), 추출된 픽셀 길이를
250mm 기준으로 임시 정규화한다 — 지금까지와 같은 한계다.

사용 예::

    python scripts/photo_to_deformer_demo.py `
        data/output/sfm_prototype/test02_run/images/frame_00000.jpg `
        data/output/sfm_prototype/test02_run/masks/frame_00000.jpg.png `
        --out data/output/photo_demo_fit.stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 패키지 설치 없이도 실행되도록

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from foot_engine.deformer import FootMeshDeformer  # noqa: E402
from foot_engine.schemas import parse_payload  # noqa: E402
from foot_engine.silhouette_landmarks import extract_top_view_landmarks  # noqa: E402
from foot_engine.template_factory import save_reference_template  # noqa: E402

_REFERENCE_LENGTH_MM = 250.0  # 절대 축척 미확정 — 형태 비교용 임시 기준


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="사진 1장 → 실루엣 랜드마크 → deformer 3D 모델")
    parser.add_argument("image", type=Path, help="원본 사진(top view)")
    parser.add_argument("mask", type=Path, help="같은 사진의 배경 제거 마스크(rembg 출력)")
    parser.add_argument("--out", type=Path, required=True, help="변형된 메쉬 저장 경로(.stl)")
    parser.add_argument("--side", default="right", choices=["left", "right"])
    parser.add_argument("--template", type=Path, default=None)
    args = parser.parse_args(argv)

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[error] 마스크를 읽을 수 없습니다: {args.mask}", file=sys.stderr)
        return 1
    img = cv2.imread(str(args.image))
    if img is None:
        print(f"[error] 사진을 읽을 수 없습니다: {args.image}", file=sys.stderr)
        return 1

    landmarks_px = extract_top_view_landmarks(mask)
    print("[랜드마크 추출]")
    for name, (x, y) in landmarks_px.items():
        print(f"  {name:<15} ({x:.0f}, {y:.0f}) px")

    # 절대 축척 없음 — 뽑힌 발 길이(px)를 250mm 기준으로 임시 스케일링.
    heel = np.array(landmarks_px["heel_center"])
    toe = np.array(landmarks_px["toe_tip"])
    length_px = float(np.linalg.norm(toe - heel))
    scale_mm_per_px = _REFERENCE_LENGTH_MM / length_px
    print(f"\n[스케일] 픽셀 길이 {length_px:.1f}px -> {_REFERENCE_LENGTH_MM:.0f}mm 기준(임시, 절대 축척 아님)")

    landmarks_data = {
        "side": args.side,
        "images": [
            {
                "view": "top",
                "image_size_px": [img.shape[1], img.shape[0]],
                "scale_mm_per_px": scale_mm_per_px,
                "landmarks": {name: list(pt) for name, pt in landmarks_px.items()},
            }
        ],
    }

    template_path = args.template
    if template_path is None:
        template_path = ROOT / "data" / "templates" / "base_foot_template.stl"
        if not template_path.is_file():
            save_reference_template(template_path, length_mm=250.0, side="right")

    engine = FootMeshDeformer(template_path)
    deformed = engine.deform_mesh(landmarks_data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    deformed.export(args.out)

    report = engine.last_report
    assert report is not None
    print("\n" + "\n".join(report.summary_lines()))
    print(f"\n저장: {args.out}")
    for warning in report.warnings:
        print(f"[warn] {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
