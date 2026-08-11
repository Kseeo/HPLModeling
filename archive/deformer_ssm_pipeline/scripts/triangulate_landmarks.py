"""2D 랜드마크 픽셀 좌표 -> 3D 삼각측량, 특징점/3D 모델 저장.

Sparse SfM 복원(카메라 포즈, `sparse_sfm_prototype.py`의 결과물)이 있는 상태에서,
여러 사진에 사람이 찍어둔 2D 랜드마크(픽셀 좌표)를 그 사진들의 카메라 포즈를
이용해 3D 위치로 삼각측량한다. `_scale_reference`가 주어지면 두 랜드마크 사이의
실제 길이(mm)를 기준자로 삼아 COLMAP의 임의 단위를 실측 mm로 환산한다
(`landmarks.py`의 `resolve_scale()`과 같은 원리 — 여기서는 픽셀 거리 대신
삼각측량된 3D 거리를 기준으로 삼는다).

출력은 두 가지:
    - landmarks_3d.json       : 랜드마크 이름 -> 3D 위치 + 신뢰도(사용 뷰 수/인라이어 수)
    - model_with_landmarks.ply: sparse 포인트클라우드(회색) + 랜드마크(빨강) 결합

랜드마크 입력 형식(JSON)::

    {
      "_scale_reference": {"a": "heel", "b": "toe_tip", "real_length_mm": 260.0},
      "heel":     {"frame_00001.jpg": [512.0, 780.0], "frame_00035.jpg": [498.0, 760.0]},
      "toe_tip":  {"frame_00003.jpg": [900.0, 400.0]}
    }

사용 예::

    python scripts/triangulate_landmarks.py \\
        --recon-dir data/output/sfm_prototype/sparse/0 \\
        --landmarks data/samples/landmarks_pixels_demo.json \\
        --out-dir data/output/sfm_prototype
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
import trimesh

_SCALE_KEY = "_scale_reference"


def load_landmark_specs(path: Path) -> tuple[dict[str, dict[str, list[float]]], dict | None]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scale_ref = raw.pop(_SCALE_KEY, None)
    if not raw:
        raise ValueError(f"{path}: 랜드마크가 하나도 없습니다.")
    return raw, scale_ref


def triangulate_landmarks(
    recon: pycolmap.Reconstruction,
    specs: dict[str, dict[str, list[float]]],
) -> dict[str, dict]:
    """각 랜드마크 이름에 대해, 등록된 뷰들의 카메라 포즈로 3D 위치를 삼각측량한다."""
    images_by_name = {im.name: im for im in recon.images.values()}
    results: dict[str, dict] = {}

    for name, views in specs.items():
        pts2d, cams_from_world, cameras, used_frames = [], [], [], []
        for frame_name, px in views.items():
            image = images_by_name.get(frame_name)
            if image is None:
                print(f"[warn] '{name}': '{frame_name}' 은 등록되지 않은(재구성 실패) 프레임이라 건너뜁니다.")
                continue
            pts2d.append(px)
            cams_from_world.append(image.cam_from_world())
            cameras.append(image.camera)
            used_frames.append(frame_name)

        if len(pts2d) < 2:
            print(f"[warn] '{name}': 유효한 관측이 {len(pts2d)}개뿐이라 건너뜁니다(최소 2개 필요).")
            continue

        estimate = pycolmap.estimate_triangulation(
            np.array(pts2d, dtype=np.float64), cams_from_world, cameras
        )
        if estimate is None:
            print(f"[warn] '{name}': 삼각측량에 실패했습니다(뷰 간 시차 부족 또는 관측 불일치).")
            continue

        inliers = estimate["inliers"]
        results[name] = {
            "xyz": estimate["xyz"],
            "views_used": used_frames,
            "num_views": len(used_frames),
            "num_inliers": int(np.sum(inliers)),
        }
        if not all(inliers):
            outlier_frames = [f for f, ok in zip(used_frames, inliers) if not ok]
            print(f"[warn] '{name}': {outlier_frames} 뷰가 아웃라이어로 제외됐습니다 — 좌표를 다시 확인하세요.")

    return results


def resolve_scale(results: dict[str, dict], scale_ref: dict | None) -> float:
    """scale_ref 가 있으면 실측mm/SfM단위 배율을 계산, 없으면 1.0(임의 단위 유지)."""
    if scale_ref is None:
        print("[scale] 기준 정보가 없어 SfM 임의 단위를 그대로 사용합니다(실측 mm 아님).")
        return 1.0

    a, b, real_mm = scale_ref["a"], scale_ref["b"], float(scale_ref["real_length_mm"])
    if a not in results or b not in results:
        raise ValueError(f"scale_reference 랜드마크 '{a}'/'{b}' 가 삼각측량 결과에 없습니다.")

    sfm_dist = float(np.linalg.norm(results[a]["xyz"] - results[b]["xyz"]))
    if sfm_dist <= 1e-9:
        raise ValueError("scale_reference 두 랜드마크의 SfM 거리가 0입니다.")

    scale = real_mm / sfm_dist
    print(f"[scale] '{a}'~'{b}' SfM거리={sfm_dist:.4f} -> 실측 {real_mm}mm 기준 배율={scale:.4f}")
    return scale


def save_outputs(
    recon: pycolmap.Reconstruction,
    results: dict[str, dict],
    scale: float,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = "mm" if scale != 1.0 else "sfm_units"

    # --- 특징점(랜드마크) JSON ---------------------------------------------
    landmarks_json = {
        name: {
            "xyz": (r["xyz"] * scale).tolist(),
            "unit": unit,
            "num_views": r["num_views"],
            "num_inliers": r["num_inliers"],
            "views_used": r["views_used"],
        }
        for name, r in results.items()
    }
    landmarks_path = out_dir / "landmarks_3d.json"
    landmarks_path.write_text(
        json.dumps(landmarks_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- 3D 모델(sparse 포인트클라우드 + 랜드마크 강조) ----------------------
    cloud_xyz = np.array([p.xyz for p in recon.points3D.values()]) * scale
    cloud_rgb = np.tile(np.array([[160, 160, 160]], dtype=np.uint8), (len(cloud_xyz), 1))

    landmark_xyz = np.array([r["xyz"] for r in results.values()]) * scale
    landmark_rgb = np.tile(np.array([[220, 30, 30]], dtype=np.uint8), (len(landmark_xyz), 1))

    all_xyz = np.vstack([cloud_xyz, landmark_xyz]) if len(landmark_xyz) else cloud_xyz
    all_rgb = np.vstack([cloud_rgb, landmark_rgb]) if len(landmark_xyz) else cloud_rgb

    model_path = out_dir / "model_with_landmarks.ply"
    trimesh.PointCloud(vertices=all_xyz, colors=all_rgb).export(model_path)

    return landmarks_path, model_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SfM 카메라 포즈로 2D 랜드마크를 3D 삼각측량")
    parser.add_argument("--recon-dir", type=Path, required=True, help="pycolmap sparse 복원 폴더")
    parser.add_argument("--landmarks", type=Path, required=True, help="2D 랜드마크 픽셀 좌표 JSON")
    parser.add_argument("--out-dir", type=Path, required=True, help="결과 저장 위치")
    args = parser.parse_args(argv)

    if not args.recon_dir.is_dir():
        print(f"[error] 복원 폴더가 없습니다: {args.recon_dir}", file=sys.stderr)
        return 2
    if not args.landmarks.is_file():
        print(f"[error] 랜드마크 파일이 없습니다: {args.landmarks}", file=sys.stderr)
        return 2

    specs, scale_ref = load_landmark_specs(args.landmarks)
    recon = pycolmap.Reconstruction(args.recon_dir)

    results = triangulate_landmarks(recon, specs)
    if not results:
        print("[error] 삼각측량에 성공한 랜드마크가 하나도 없습니다.", file=sys.stderr)
        return 1

    try:
        scale = resolve_scale(results, scale_ref)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    landmarks_path, model_path = save_outputs(recon, results, scale, args.out_dir)

    print("\n[결과]")
    for name, r in results.items():
        xyz = r["xyz"] * scale
        print(
            f"  {name}: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})  "
            f"[{r['num_inliers']}/{r['num_views']} 뷰 사용]"
        )
    print(f"  랜드마크 저장: {landmarks_path}")
    print(f"  포인트클라우드(+랜드마크) 저장: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
