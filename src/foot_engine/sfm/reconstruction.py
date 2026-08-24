"""사진/영상 -> 카메라 포즈 + sparse 포인트클라우드 (pycolmap 기반).

매칭은 항상 exhaustive(모든 사진 쌍 비교) — 순서/인접성 무관하게 이어붙일
수 있다. 사진 수백 장 이상이면 느려질 수 있다.

촬영 전제 조건(알고리즘으로 못 고침): 발이 촬영 내내 완전히 고정돼야 하고
(의자 등에 자연스럽게 올려둘 것), 사진이 선명해야 한다.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import trimesh
from scipy.spatial import cKDTree

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# COLMAP 기본 max_num_features=8192/peak_threshold=0.00667은 발 피부처럼
# 저텍스처인 피사체에서 특징점이 부족해 등록률을 깎는다 -- 값을 키우고
# threshold를 낮춰 더 약한 특징점까지 채택하도록 튜닝한 기본값.
DEFAULT_MAX_FEATURES = 16384
DEFAULT_PEAK_THRESHOLD = 0.003

# 매칭 단계 RANSAC 임계값. COLMAP 기본(4.0px)보다 낮춰 더 엄격하게 인라이어를
# 거른다 -- 등록률/포인트 수 손해 없이 평균 재투영 오차를 낮춘다.
DEFAULT_RANSAC_MAX_ERROR = 2.0


def extract_frames(
    video_path: Path,
    out_dir: Path,
    interval_sec: float,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> Path:
    """영상의 [start_time, end_time] 구간에서 `interval_sec` 간격으로 프레임을 추출.

    관계없는 장면이 섞인 영상에서 실제 피사체가 나오는 구간만 잘라 쓰고 싶을 때
    `start_time`/`end_time`(초)을 지정한다.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir.mkdir(parents=True, exist_ok=True)
    saved, frame_idx, next_sample_time = 0, 0, start_time
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        frame_idx += 1
        if end_time is not None and t > end_time:
            break
        if t < start_time:
            continue
        if t >= next_sample_time:
            cv2.imwrite(str(out_dir / f"frame_{saved:05d}.jpg"), frame)
            saved += 1
            next_sample_time += interval_sec
    cap.release()

    if saved < 8:
        raise RuntimeError(
            f"추출된 프레임이 {saved}장뿐입니다(최소 8장 권장). "
            "interval을 줄이거나 구간(start_time/end_time)을 넓히세요."
        )
    print(
        f"[frames] {video_path.name} -> {saved}장 추출 "
        f"(구간 {start_time}s~{end_time if end_time is not None else '끝'}s, "
        f"간격 {interval_sec}s, 원본 {fps:.1f}fps)"
    )
    return out_dir


def compute_sharpness(image_path: Path) -> float:
    """라플라시안 분산으로 선명도를 잰다 — 낮을수록 블러가 심하다."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def filter_blurry_frames(
    images_dir: Path, keep_ratio: float, names: list[str] | None = None
) -> list[str]:
    """선명도 상위 `keep_ratio`만 골라 파일명 목록을 반환한다.

    Args:
        names: 지정하면 이 파일들 중에서만 순위를 매긴다(다른 QC 게이트를 이미
            통과한 후보 목록 등). None이면 `images_dir`의 전체 이미지 파일.
    """
    if names is not None:
        paths = [images_dir / name for name in names]
    else:
        paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    scores = [(p.name, compute_sharpness(p)) for p in paths]
    scores.sort(key=lambda item: item[1], reverse=True)

    keep_n = max(8, round(len(scores) * keep_ratio))
    kept_names = sorted(name for name, _ in scores[:keep_n])
    dropped = len(scores) - len(kept_names)
    print(f"[blur] 선명도 하위 {dropped}장 제외 -> {len(kept_names)}/{len(scores)}장 사용")
    return kept_names


def filter_outlier_points(points: np.ndarray, k: int = 8, std_ratio: float = 2.0) -> np.ndarray:
    """통계적 이상치 제거 — 이웃 k개까지의 평균 거리가 유별나게 먼 점을 걸러낸다.

    Returns:
        inlier 여부를 나타내는 (N,) bool 배열.
    """
    if len(points) <= k:
        return np.ones(len(points), dtype=bool)

    tree = cKDTree(points)
    neighbor_dist, _ = tree.query(points, k=k + 1)  # 0번째는 자기 자신(거리 0)
    mean_dist = neighbor_dist[:, 1:].mean(axis=1)

    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    return mean_dist <= threshold


def run_sparse_sfm(
    images_dir: Path,
    workdir: Path,
    image_names: list[str] | None = None,
    masks_dir: Path | None = None,
    max_features: int = DEFAULT_MAX_FEATURES,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    ransac_max_error: float = DEFAULT_RANSAC_MAX_ERROR,
) -> pycolmap.Reconstruction:
    """이미지 폴더에서 exhaustive 매칭 + incremental SfM으로 sparse 복원을 만든다.

    - image_names: 지정하면 이 파일들만 사용(블러 필터링된 목록 등).
    - masks_dir: 지정하면 <파일명>.png 마스크(0=제외)로 배경 특징점 추출을
      막는다(masking.generate_masks() 결과). 배경이 피사체와 함께 완전히
      고정된 촬영에서는 오히려 손해(저텍스처 피부보다 SIFT를 더 잘 잡아줌)
      -- 배경이 복잡/혼재할 때만 쓸 것.
    - max_features/peak_threshold: SIFT 특징점 추출 파라미터(모듈 상단 기본값).
    - ransac_max_error: 매칭 단계 RANSAC 인라이어 허용 오차(px).
    """
    db_path = workdir / "database.db"
    sparse_dir = workdir / "sparse"
    if db_path.exists():
        db_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True)

    print(f"[sfm] 특징점 추출 중 (SIFT, max_features={max_features}, peak_threshold={peak_threshold})...")
    reader_options = pycolmap.ImageReaderOptions(mask_path=str(masks_dir) if masks_dir else "")
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = max_features
    extraction_options.sift.peak_threshold = peak_threshold
    pycolmap.extract_features(
        db_path, images_dir, image_names=image_names or [],
        reader_options=reader_options, extraction_options=extraction_options,
    )

    print(
        f"[sfm] exhaustive 매칭 중 (모든 사진 쌍을 비교 — 순서/인접성에 의존하지 않음, "
        f"RANSAC 오차 허용치={ransac_max_error}px)..."
    )
    verification_options = pycolmap.TwoViewGeometryOptions()
    verification_options.ransac.max_error = ransac_max_error
    pycolmap.match_exhaustive(db_path, verification_options=verification_options)

    print("[sfm] incremental mapping 중 (카메라 포즈 + sparse 포인트클라우드)...")
    reconstructions = pycolmap.incremental_mapping(db_path, images_dir, sparse_dir)
    if not reconstructions:
        raise RuntimeError(
            "재구성에 실패했습니다 — 촬영 중 피사체(발)가 움직였거나, 사진이 흐릿하거나, "
            "사진 간 시차(각도 변화)가 부족했을 가능성이 높습니다."
        )

    # 연결이 끊긴 구간이 있으면 여러 조각으로 나뉘어 재구성된다 — 가장 큰 조각을 사용.
    sizes = sorted((r.num_reg_images() for r in reconstructions.values()), reverse=True)
    if len(sizes) > 1:
        print(
            f"[sfm] 재구성이 {len(sizes)}개 조각으로 나뉨(사용: {sizes[0]}장, "
            f"버려짐: {sizes[1:]}) — 촬영 중 매칭이 끊긴 구간이 있었을 가능성이 높습니다."
        )
    best = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    return best


def report_unregistered_frames(
    images_dir: Path, recon: pycolmap.Reconstruction, candidate_names: list[str]
) -> None:
    """등록 실패한 프레임을 선명도 오름차순으로 보여준다 -- 실패 원인 진단용.

    - 블러가 가장 흔한 원인. 미등록 프레임 선명도가 유독 낮으면 블러 의심,
      정상 범위인데도 미등록이면 자세 흔들림/각도 편차 의심.
    """
    registered = {im.name for im in recon.images.values()}
    missing = [name for name in candidate_names if name not in registered]
    if not missing:
        return

    scored = sorted(
        ((name, compute_sharpness(images_dir / name)) for name in missing),
        key=lambda item: item[1],
    )
    print(f"\n[미등록 프레임] {len(missing)}장 (선명도 낮은 순 — 블러가 원인일 가능성부터 확인):")
    shown = scored[:15]
    for name, sharpness in shown:
        print(f"  {name}  (선명도 {sharpness:.1f})")
    if len(scored) > len(shown):
        print(f"  ... 외 {len(scored) - len(shown)}장")


def export_point_cloud(recon: pycolmap.Reconstruction, out_path: Path) -> tuple[np.ndarray, int]:
    """복원 결과의 3D 점을 이상치 제거 후 .ply로 내보낸다.

    Returns:
        (남은 점 배열, 제거된 이상치 개수).
    """
    points = np.array([p.xyz for p in recon.points3D.values()])
    if len(points) == 0:
        kept, removed = points, 0
    else:
        inliers = filter_outlier_points(points)
        kept, removed = points[inliers], len(points) - int(inliers.sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(kept).export(out_path)
    return kept, removed
