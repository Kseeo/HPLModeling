"""사진별 발/피부 세그멘테이션 마스크 생성 (rembg + MediaPipe 피부 정제).

피사체 분리는 `rembg`(`u2net_human_seg`), 옷/장신구 제외는 MediaPipe
Selfie Multiclass가 맡는다.

마스크는 COLMAP 관례(`<파일명>.png`, 0=제외·그 외=포함)로 저장, 발 가장자리
보존을 위해 살짝 팽창(dilate)한다.

`u2net_human_seg`는 "사람 전체"만 분리하고 옷/장신구는 못 거른다 —
`skin_refine=True`(기본)가 MediaPipe로 `body-skin`/`face-skin` 클래스만
남겨 rembg 마스크와 AND한다.

한계: 피부-옷 경계선 대비가 강해 고리 모양 잔여 오염이 남을 수 있다 —
촬영 시 옷을 안 보이게 찍는 게 근본 해결책.

세그멘테이션/피부 정제가 애매하게 실패한 프레임은 후보에서 제외한다."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# rembg는 generate_foot_masks()(사진 기반 SfM 파이프라인 전용, 텍스처 크롭
# 경로는 안 씀)에서만 필요해 그 함수 안에서 지연 임포트한다 -- 모듈 최상단에
# 두면 PyInstaller로 얼린 exe에서 rembg의 하위 의존성 pymatting이 자기
# 버전을 importlib.metadata로 못 찾아 죽는 문제가 있었다(실측 확인: 텍스처
# 크롭만 쓰는 웹앱 마법사는 이 임포트 자체가 필요 없는데도 죽었음).

from .reconstruction import IMAGE_EXTS

# MediaPipe Selfie Multiclass 클래스 id. body-skin/face-skin만 "피부"로 취급 —
# clothes(옷)/hair(머리카락)/others(장신구)는 제외한다. 주의: 손도 body-skin으로
# 잡히므로 이걸로 "발에 닿은 손" 오염은 못 막는다(별도 문제, 캡처 가이드로 대응).
SKIN_CLASS_IDS = frozenset({2, 3})
SELFIE_MULTICLASS_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
def _default_skin_model_path() -> Path:
    """PyInstaller로 얼린(frozen) 상태에서도 모델 파일을 찾는다.

    평소엔 소스 트리 기준(`parents[3]`)이지만, 얼린 exe에서는 그 경로가
    깨진다 -- PyInstaller가 번들 데이터 파일을 실제로 푸는 위치는
    `sys._MEIPASS`다(onedir는 `_internal/`, onefile은 임시 추출 폴더 --
    `sys.executable`의 부모 폴더가 아님, 실측으로 확인).
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "data" / "models" / "selfie_multiclass_256x256.tflite"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[3] / "data" / "models" / "selfie_multiclass_256x256.tflite"


DEFAULT_SKIN_MODEL_PATH = _default_skin_model_path()


def load_skin_segmenter(model_path: Path = DEFAULT_SKIN_MODEL_PATH):
    """MediaPipe Selfie Multiclass 세그멘터를 로드한다. 모델 파일이 없으면 내려받는다."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[안내] 피부 정제 모델이 없어 내려받습니다: {model_path}")
        urllib.request.urlretrieve(SELFIE_MULTICLASS_URL, model_path)

    options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_category_mask=True,
    )
    return vision.ImageSegmenter.create_from_options(options)


def skin_only_mask(segmenter, bgr_image: np.ndarray, *, erode: int = 8) -> np.ndarray:
    """rembg 마스크를 정제할 "피부만" 바이너리 마스크(0/255)를 만든다.

    옷과 피부의 경계선 자체가 대비가 강해 SIFT가 좋아하는 특징점이라, 클래스
    경계에 걸친 점들이 그 경계(예: 소매 밑단 솔기)를 따라 얇은 고리로
    삼각측량될 수 있다. 경계에서 `erode`px만큼 안쪽으로 깎아 그 경계선
    자체를 피부 마스크에서 미리 제외한다.
    """
    import mediapipe as mp

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = segmenter.segment(mp_image)
    cat_mask = result.category_mask.numpy_view()
    skin = np.isin(cat_mask, list(SKIN_CLASS_IDS)).astype(np.uint8) * 255
    if erode > 0:
        kernel = np.ones((erode, erode), np.uint8)
        skin = cv2.erode(skin, kernel)
    return skin


def remove_thin_appendages(mask: np.ndarray, *, erode_size: int = 21) -> np.ndarray:
    """마스크에서 가는 목으로 겨우 이어진 돌출부(오분류된 배경)를 잘라낸다.

    Opening by reconstruction: 침식으로 얇은 연결부를 끊고, 가장 큰 연결요소만
    씨앗으로 남긴 뒤, 원본 마스크 경계를 넘지 않는 선에서 다시 팽창시켜
    복원한다 — 발 경계 자체는 원래 모양대로 복원되고 끊어진 돌출부만 안 자란다.

    한계: 오분류된 배경이 발과 폭 넓게 이어져 있으면(가는 목이 아니라)
    못 거른다.

    Args:
        erode_size: 끊어낼 "목"의 최대 굵기(px) 기준.

    Returns:
        돌출부가 제거된 마스크(0/255).
    """
    if erode_size <= 0 or not mask.any():
        return mask

    kernel = np.ones((erode_size, erode_size), np.uint8)
    eroded = cv2.erode(mask, kernel)
    n_labels, labels = cv2.connectedComponents(eroded)
    if n_labels <= 1:
        return mask  # 침식으로 씨앗이 하나도 안 남음 -- 안전하게 원본 유지

    sizes = [int((labels == i).sum()) for i in range(1, n_labels)]
    largest_label = 1 + int(np.argmax(sizes))
    seed = np.where(labels == largest_label, np.uint8(255), np.uint8(0))

    # geodesic dilation: 원본 마스크를 넘지 않는 선에서 씨앗을 반복 팽창.
    # 재구성용 커널을 침식 때와 같은 크기로 써서 수렴을 빠르게 한다 --
    # 매 단계 원본과 AND로 잘라내므로 커널이 커도 경계가 흐트러지지 않는다.
    prev = seed
    while True:
        grown = cv2.dilate(prev, kernel)
        grown = cv2.bitwise_and(grown, mask)
        if np.array_equal(grown, prev):
            return grown
        prev = grown


def generate_masks(
    images_dir: Path,
    out_dir: Path,
    *,
    names: list[str] | None = None,
    model: str = "u2net_human_seg",
    dilate: int = 15,
    min_coverage: float = 0.03,
    reject_coverage: float = 0.005,
    skin_refine: bool = True,
    skin_model: Path = DEFAULT_SKIN_MODEL_PATH,
    skin_erode: int = 8,
    extra_dilations: list[tuple[Path, int]] | None = None,
) -> dict:
    """이미지 폴더(또는 그 안 일부)의 발/피부 마스크를 생성해 `out_dir`에 저장한다.

    세그멘테이션/피부 정제가 애매하게 실패하면 원본 전체나 정제 전 마스크로
    대체하지 않고 그 프레임을 후보에서 제외한다 — 배경이 섞여 들어갈 수 있어서.

    Args:
        images_dir: 원본 이미지 폴더.
        out_dir: 마스크(`<파일명>.png`) 저장 폴더.
        names: 지정하면 이 파일들만 처리. None이면 `images_dir`의 전체 이미지.
        model: rembg 모델 이름.
        dilate: 마스크 경계 팽창 폭(px).
        min_coverage: 이 비율 미만이면 세그멘테이션 실패로 보고 제외.
        reject_coverage: 이 비율 미만이면 발 자체가 안 보인다고 보고 제외
            (`min_coverage`보다 낮게).
        skin_refine: MediaPipe로 옷/장신구를 추가 제외할지.
        skin_model: MediaPipe Selfie Multiclass 모델(.tflite) 경로.
        skin_erode: 피부 마스크 경계를 깎는 폭(px) — 옷-피부 경계선 잔여 오염 완화용.
        extra_dilations: `(out_dir, dilate)` 쌍 목록. rembg/피부 정제 추론
            (비싼 부분)은 한 번만 돌리고, 각 쌍에 대해 팽창 폭만 다르게 적용해
            추가로 저장한다 — sparse SfM용(dilate=15)과 dense MVS용(dilate=0)
            마스크를 한 번에 만들 때 세그멘테이션 추론이 중복 실행되는 걸 막는다.
            판정(수락/제외)은 팽창 전 단계에서 정해지므로 모든 출력이 동일한
            판정을 공유한다.

    Returns:
        {"total": 처리한 장 수, "refined": 피부 정제가 적용된 장 수,
         "rejected": 제외된 장 수, "rejected_names": 제외된 파일명 목록,
         "rejected_reasons": 사유별 집계(no_foot/low_coverage/skin_refine_collapsed)}.
    """
    from rembg import new_session, remove

    if not images_dir.is_dir():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {images_dir}")

    outputs = [(out_dir, dilate)] + list(extra_dilations or [])
    kernels = [
        (out, np.ones((d, d), np.uint8) if d > 0 else None) for out, d in outputs
    ]
    for out, _ in outputs:
        out.mkdir(parents=True, exist_ok=True)

    session = new_session(model)

    skin_segmenter = None
    if skin_refine:
        try:
            skin_segmenter = load_skin_segmenter(skin_model)
        except Exception as exc:
            print(f"[warn] 피부 정제 모델 로드 실패, 이 단계를 건너뜁니다: {exc}")

    if names is not None:
        paths = [images_dir / name for name in names]
    else:
        paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    refined_count = 0
    rejected_names: list[str] = []
    rejected_reasons = {"no_foot": 0, "low_coverage": 0, "skin_refine_collapsed": 0}
    try:
        for path in paths:
            img = cv2.imread(str(path))
            rgba = remove(img, session=session)  # rembg는 BGR ndarray도 그대로 받는다
            alpha = rgba[:, :, 3]
            mask = (alpha > 10).astype(np.uint8) * 255
            rejected = False

            coverage = (mask > 0).mean()
            if coverage < reject_coverage:
                # 발이 아예 프레임에 없음.
                rejected_reasons["no_foot"] += 1
                rejected = True
            elif coverage < min_coverage:
                # 세그멘테이션이 애매하게 실패.
                rejected_reasons["low_coverage"] += 1
                rejected = True
            elif skin_segmenter is not None:
                # rembg가 잡은 "사람" 영역 안에서 옷/장신구만 추가로 뺀다.
                skin = skin_only_mask(skin_segmenter, img, erode=skin_erode)
                refined = cv2.bitwise_and(mask, skin)
                if (refined > 0).mean() >= min_coverage:
                    mask = refined
                    refined_count += 1
                else:
                    # 정제 결과가 너무 작아짐 — 정제 전 마스크로 되돌아가면 배경이
                    # 섞일 수 있어 이 프레임을 제외한다.
                    rejected_reasons["skin_refine_collapsed"] += 1
                    rejected = True

            if rejected:
                mask = np.zeros(mask.shape, dtype=np.uint8)
                rejected_names.append(path.name)

            for out, kernel in kernels:
                saved = mask if rejected or kernel is None else cv2.dilate(mask, kernel)
                cv2.imwrite(str(out / f"{path.name}.png"), saved)
    finally:
        # GC가 나중에 아무 때나 __del__로 닫게 두면(예: 한참 뒤 다른 코드가
        # 도는 도중 가비지 컬렉션이 도는 시점), MediaPipe의 내부 디스패처가
        # 그 시점에 걸려 영원히 멈추는 경우가 있다 -- 쓰자마자 바로 닫는다.
        if skin_segmenter is not None:
            skin_segmenter.close()

    return {
        "total": len(paths),
        "refined": refined_count,
        "rejected": len(rejected_names),
        "rejected_names": sorted(rejected_names),
        "rejected_reasons": rejected_reasons,
    }
