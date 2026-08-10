"""사진별 발/피부 세그멘테이션 마스크 생성 (rembg + MediaPipe 피부 정제).

이전에 시도한 "네 모서리를 배경으로 가정" 방식은 피사체가 모서리까지 꽉 찬
클로즈업이나 복잡한 배경에서 실패했다. 실제 피사체 분리 모델(`rembg`,
`u2net_human_seg`)을 1차로 쓰고, MediaPipe Selfie Multiclass로 옷/장신구를
추가로 걸러낸다.

마스크는 COLMAP 관례(`<파일명>.png`, 픽셀값 0=제외·그 외=포함)로 저장하고,
안전 여유를 위해 살짝 팽창(dilate)한다 — 세그멘테이션이 살짝 타이트해도 발
가장자리가 잘려나가지 않게 하기 위함이다(경계에 배경이 조금 남는 것은 이후
`cleaning.py`의 평면 제거/초점 거리 필터로 마저 걸러낼 수 있다).

`u2net_human_seg`는 "사람 전체"를 배경과 분리할 뿐, 옷/장신구를 피부와
구분하지 못한다 — 2026-08-07 test00에서 발목 위 소매 커프가 그대로
마스크에 포함되어 점군의 ~1/3을 오염시킨 사례를 pycolmap으로 점→픽셀
역추적해 확인함(`test00-sleeve-cuff-contamination` 메모리 참고). 이를 막기
위해 `skin_refine=True`(기본)로 MediaPipe의 Selfie Multiclass 세그멘터
(`background/hair/body-skin/face-skin/clothes/others` 6클래스)를 추가로
돌려 `body-skin`|`face-skin` 클래스만 남기고 rembg 마스크와 AND한다 —
rembg가 잡아낸 "사람" 영역 안에서 옷/장신구만 추가로 제거하는 정제
단계이므로, 이미 튜닝된 배경 제거 동작 자체는 건드리지 않는다.

한계(실측 확인, 완전히 해결되지 않음): 피부-옷 경계선 자체가 대비가 강해
고리 모양 잔여 오염이 남을 수 있다.
촬영 시 옷이 보이지 않도록 찍는 것이 근본적인 해결책
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
from rembg import new_session, remove

from .reconstruction import IMAGE_EXTS

# MediaPipe Selfie Multiclass 클래스 id. body-skin/face-skin만 "피부"로 취급 —
# clothes(옷)/hair(머리카락)/others(장신구)는 제외한다. 주의: 손도 body-skin으로
# 잡히므로 이걸로 "발에 닿은 손" 오염은 못 막는다(별도 문제, 캡처 가이드로 대응).
SKIN_CLASS_IDS = frozenset({2, 3})
SELFIE_MULTICLASS_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
DEFAULT_SKIN_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "selfie_multiclass_256x256.tflite"
)


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
    경계에 걸친 점들이 그 경계(예: 소매 밑단 솔기)를 따라 얇은 고리로 삼각측량
    되는 게 실측으로 확인됐다. 
    경계에서 `erode`px만큼 안쪽으로 깎아 그 경계선 자체를 피부 마스크에서
    미리 제외한다.
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
) -> dict:
    """이미지 폴더(또는 그 안 일부)의 발/피부 마스크를 생성해 `out_dir`에 저장한다.

    Args:
        images_dir: 원본 이미지 폴더.
        out_dir: 마스크(`<파일명>.png`) 저장 폴더.
        names: 지정하면 이 파일들만 처리(다른 QC 게이트를 이미 통과한 후보 목록 등).
            None이면 `images_dir`의 전체 이미지 파일.
        model: rembg 모델 이름.
        dilate: 마스크 경계 팽창 폭(px).
        min_coverage: 이 비율 미만이면 세그멘테이션 실패로 보고 원본 전체를 사용
            (`reject_coverage`보다는 커야 함 — 그 사이는 "애매하니 안전하게 전체 사용").
        reject_coverage: 이 비율 미만이면 세그멘테이션 실패가 아니라 애초에 발이
            프레임에 없다고 보고(카메라가 다른 곳을 비췄거나 초점을 벗어난 경우 등)
            해당 프레임을 후보 목록에서 아예 제외한다. `min_coverage`보다 훨씬
            낮게 잡아야 한다 — 발이 실제로 있는데 세그멘테이션만 살짝 실패한
            프레임까지 통째로 버리면 안 되기 때문.
        skin_refine: MediaPipe로 옷/장신구를 추가 제외할지.
        skin_model: MediaPipe Selfie Multiclass 모델(.tflite) 경로.
        skin_erode: 피부 마스크 경계를 깎는 폭(px) — 옷-피부 경계선 잔여 오염 완화용.

    Returns:
        {"total": 처리한 장 수, "fallback": 원본 전체를 쓴 장 수,
         "refined": 피부 정제가 적용된 장 수, "rejected": 제외된 장 수,
         "rejected_names": 제외된 파일명 목록(정렬됨)}.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {images_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    session = new_session(model)
    kernel = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None

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
    fallback_count = 0
    refined_count = 0
    rejected_names: list[str] = []
    for path in paths:
        img = cv2.imread(str(path))
        rgba = remove(img, session=session)  # rembg는 BGR ndarray도 그대로 받는다
        alpha = rgba[:, :, 3]
        mask = (alpha > 10).astype(np.uint8) * 255

        coverage = (mask > 0).mean()
        if coverage < reject_coverage:
            # 세그멘테이션 실패가 아니라 애초에 발이 프레임에 안 보이는 경우로 본다
            # — 원본 전체를 쓰면 배경이 통째로 SfM/점군에 섞여 들어가므로, 이
            # 프레임은 마스크만 빈 채로 남기고 후보 목록에서 제외한다.
            mask = np.zeros(mask.shape, dtype=np.uint8)
            rejected_names.append(path.name)
        elif coverage < min_coverage:
            # 세그멘테이션이 애매하게 실패한 경우 — 안전하게 원본 전체를 사용
            mask = np.full(mask.shape, 255, dtype=np.uint8)
            fallback_count += 1
        else:
            if skin_segmenter is not None:
                # rembg가 잡은 "사람" 영역 안에서 옷/장신구(예: 소매 커프)만 추가로 뺀다.
                # 정제 후 커버리지가 너무 작아지면(옷을 사람으로 오인해 거의 다 날아간
                # 경우 등) 오히려 발이 통째로 사라질 위험이 있어 안전장치로 원래 마스크를 쓴다.
                skin = skin_only_mask(skin_segmenter, img, erode=skin_erode)
                refined = cv2.bitwise_and(mask, skin)
                if (refined > 0).mean() >= min_coverage:
                    mask = refined
                    refined_count += 1
            if kernel is not None:
                mask = cv2.dilate(mask, kernel)

        cv2.imwrite(str(out_dir / f"{path.name}.png"), mask)

    return {
        "total": len(paths),
        "fallback": fallback_count,
        "refined": refined_count,
        "rejected": len(rejected_names),
        "rejected_names": sorted(rejected_names),
    }
