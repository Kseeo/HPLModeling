# HPLModeling (foot_deform_engine)

2D 사진/영상 → 3D 발 메쉬를 만드는 엔진. 최종 목표는 이 3D 모델을
체중부하(weight-bearing) 변형 AI에 넣어 평발/부상/절단 등 다양한 발 형태에
맞는 **맞춤 인솔**을 자동 설계하는 것이다. 이 저장소는 그 파이프라인의
"사진/영상 → 3D 메쉬" 구간을 담당한다.

## 목차

- [파이프라인](#파이프라인)
- [디렉터리 구조](#디렉터리-구조)
- [설치](#설치)
- [`data/` 디렉터리 준비 (필수 — git에 없음)](#data-디렉터리-준비--필수--git에-없음)
- [빠른 시작](#빠른-시작)
- [단계별로 나눠 돌리기](#단계별로-나눠-돌리기)
- [모듈 설명 (`src/foot_engine`)](#모듈-설명-srcfoot_engine)
- [스크립트 목록 (`scripts/`)](#스크립트-목록-scripts)
- [알려진 한계 / 진행 상황](#알려진-한계--진행-상황)
- [`archive/` — 더 이상 쓰지 않는 경로](#archive--더-이상-쓰지-않는-경로)
- [개발 메모](#개발-메모)

## 파이프라인

외부 유료 포토그래메트리(VRIN)를 대체하기 위한 in-house 경로다. `pycolmap`
기반 sparse SfM으로 사진 여러 장에서 카메라 포즈 + sparse 포인트클라우드를
복원하고, 배경/노이즈를 제거한 뒤 OpenMVS 기반 dense MVS로 실제 3D 메쉬를
만든다. 메쉬 생성기는 **dense MVS 하나뿐**이다(2026-08-11 결론 — 한때
템플릿 워프/SSM 경로도 있었으나 [`archive/`](#archive--더-이상-쓰지-않는-경로)
절 참고, dense 메쉬 자체를 다듬고 경량화하는 쪽이 낫다고 판단함).

```
frame_quality.assess_frames()   — SfM 전 프레임별 절대기준 QC
    └─ reconstruction.run_sparse_sfm()
          └─ masking.generate_masks()  — 여기서 발 미검출 프레임을 추가로 제외
                └─ cleaning.clean_point_cloud()  — QA용 sparse 정리(최종 메쉬엔 안 씀)
                └─ masking.generate_masks(dilate=0)  — dense 전용 마스크
                      └─ dense.run_dense_pipeline()  — OpenMVS densify + 메싱 + 파편 제거
                            └─ (선택) 스케일 보정 — geometry.measured_length() 기준
```

`sfm/pipeline.py`의 `run_pipeline()`이 위 전부를 한 번에 엮는다. 아래
[단계별로 나눠 돌리기](#단계별로-나눠-돌리기)에서 각 단계를 개별 실행하는
방법도 다룬다.

## 디렉터리 구조

```
foot_deform_engine/
├── src/foot_engine/          # 패키지 본체 (pip install 안 해도 sys.path로 바로 씀)
│   ├── exceptions.py         #   엔진 전용 예외 계층 (HTTP status 매핑 포함)
│   └── sfm/                  #   사진/영상 → 3D 파이프라인 (활성 경로 전체)
│       ├── frame_quality.py  #     프레임별 절대기준 QC
│       ├── reconstruction.py #     pycolmap 기반 sparse SfM
│       ├── masking.py        #     발/피부 마스크 (rembg + MediaPipe)
│       ├── cleaning.py       #     포인트클라우드 배경/노이즈 제거
│       ├── dense.py          #     OpenMVS 기반 dense MVS 메쉬 생성 (메쉬 생성기 본체)
│       ├── geometry.py       #     PCA 축/자기 길이 추정 (스케일 보정용 소형 유틸)
│       └── pipeline.py       #     위 전부를 엮는 오케스트레이션
├── scripts/                  # 각 기능을 독립적으로 돌려볼 수 있는 CLI들
├── archive/                  # 더 이상 안 쓰는 코드(템플릿 워프/SSM) — 지우지 않고 참고용 보관
│   └── deformer_ssm_pipeline/README.md  #   왜/무엇을 옮겼는지 설명
├── data/                     # git에 없음 — 아래 절 참고
│   ├── samples/               #   촬영 샘플 영상
│   ├── models/                 #   MediaPipe 등 다운로드되는 모델 가중치
│   └── output/                 #   스크립트 산출물 (자동 생성)
└── .venv/                    # 로컬 가상환경 (git에 없음)
```

## 설치

Python 3.11(개발 환경은 3.11.8)에서 확인됨. 이 저장소는 `pyproject.toml`/
`requirements.txt`가 아직 없다 — `scripts/` 아래 모든 CLI가
`sys.path.insert(0, str(ROOT / "src"))`로 `src/`를 직접 바라보므로
**패키지를 pip install 하지 않아도** 곧바로 실행된다.

```powershell
# 1) 가상환경 생성 (프로젝트 루트에서)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) 의존성 설치 — 현재 개발 venv 기준 핵심 패키지 목록
pip install numpy scipy scikit-learn trimesh[easy] pycolmap opencv-python-headless `
    opencv-contrib-python rembg mediapipe onnxruntime pillow networkx pytest
```

> `pip freeze > requirements.txt`로 현재 venv를 그대로 고정하는 게
> 가장 정확하다 — 위 목록은 `import` 기준으로 추린 핵심 패키지만 나열한
> 것이라 부차 의존성(예: `trimesh`가 쓰는 `manifold3d`/`embreex`,
> `rembg`가 쓰는 `PyMatting` 등)은 pip가 알아서 딸려 온다.

CUDA/GPU는 **필요 없다** — sparse SfM(`pycolmap`)과 dense MVS(`dense.py`)
모두 CPU로 돈다(실측 확인 — COLMAP/OpenMVS의 CUDA 빌드는 최신 CUDA 13.2를
요구해 드라이버 업그레이드 부담이 커서 채택하지 않았다).

### OpenMVS 설치 (필수)

dense 메쉬 생성(`dense.py`/`scripts/run_dense_pipeline.py`)에는 OpenMVS
CLI 실행파일이 별도로 필요하다(pip 패키지 아님, 이 저장소 `requirements`에도
없음):

1. [OpenMVS 릴리즈 페이지](https://github.com/cdcseacave/openMVS/releases)에서
   `OpenMVS_Windows_x64.zip`(CPU 빌드로 충분, CUDA 불필요 — 실측 확인)을
   받아 아무 폴더에나 압축을 푼다.
2. 그 안의 `vc17/x64/Release/` (또는 실행파일이 직접 있는 폴더)를
   `OPENMVS_BIN_DIR` 환경변수로 지정하거나, `run_dense_pipeline.py`
   실행 시 `--openmvs-bin`으로 넘긴다.

```powershell
$env:OPENMVS_BIN_DIR = "C:\tools\openmvs\vc17\x64\Release"
python scripts/run_dense_pipeline.py data/output/sfm_pipeline/test02_run
```

한글 콘솔 출력: Windows의 기본 콘솔 코드페이지(cp949)에서 한글이 깨지거나
죽는 문제가 있어, `scripts/*.py`는 실행 시작 시 `sys.stdout`/`stderr`를
UTF-8로 강제 `reconfigure`한다. 새 스크립트를 추가할 때도 이 패턴을
따를 것 (`run_sfm_pipeline.py` 상단 참고).

## `data/` 디렉터리 준비 (필수 — git에 없음)

`data/` 전체가 `.gitignore`되어 있다 — 실제 스캔/촬영 샘플에 **개인 식별
가능한 실측 데이터**가 섞여 있어서다. 즉 이 저장소를 새로 클론하면 `data/`
폴더가 아예 없다 — 아래 구조대로 로컬에 직접 만들거나, 별도 저장소/스토리지에서
받아와야 한다.

```
data/
├── samples/
│   └── test00.mp4 ~ test10.mp4    # 촬영 샘플 영상 (SfM 파이프라인 입력)
├── models/
│   └── selfie_multiclass_256x256.tflite  # MediaPipe 피부 세그멘테이션 모델
│                                          # — masking.py가 없으면 최초 1회 자동 다운로드함
└── output/                        # 각 스크립트의 산출물이 여기 쌓인다 (직접 만들 필요 없음, 자동 생성)
```

시험해보려면 발이 나온 영상 1개(`.mp4` 등, 최소 8프레임 이상 뽑힐 분량)를
`data/samples/`에 아무 이름으로 넣고 그 경로를 `--video`로 넘기면 된다.
`test00.mp4` 같은 기존 샘플 이름 자체는 중요하지 않다 — 스크립트들은
`--video`/`--images-dir` 인자로 경로를 받으므로 원하는 위치에 아무 영상이나
두고 그 경로를 지정하면 된다.

경로를 스크립트 기본값과 다르게 두고 싶으면 항상 `--video`/`--images-dir`/
`--out`/`--workdir` 같은 명시적 인자로 넘기면 된다 — 이 저장소의 CLI들은
`data/` 하위 경로를 하드코딩하지 않고 전부 인자로 받는다.

## 빠른 시작

```powershell
# 영상 1개로 SfM + dense MVS 파이프라인 전체 실행 (사진 → 3D)
python scripts/run_sfm_pipeline.py --video data/samples/test00.mp4 `
    --workdir data/output/sfm_pipeline/test00_run `
    --out data/output/test00_pipeline_fit.stl
```

## 단계별로 나눠 돌리기

중간 산출물을 뷰어로 직접 확인하고 싶을 때:

```powershell
# 1) sparse SfM 복원
python scripts/sparse_sfm_prototype.py --video data/samples/test00.mp4

# 2) 발/피부 마스크 생성
python scripts/generate_foot_masks.py data/output/sfm_prototype/test00_run/images `
    --out data/output/sfm_prototype/test00_run/masks

# 3) 배경/노이즈 제거 (QA용 — 최종 dense 메쉬에는 안 쓰인다, 확인용)
python scripts/clean_point_cloud.py data/output/sfm_prototype/test00_run/sparse/0 `
    --masks-dir data/output/sfm_prototype/test00_run/masks --cluster `
    --out data/output/sfm_prototype/test00_run/cleaned_points.ply

# 4) dense MVS 메쉬 생성 (RefineMesh 제외, 빠름 — 몇 분 내)
python scripts/run_dense_pipeline.py data/output/sfm_prototype/test00_run

# 5) 최종 품질까지 원하면 RefineMesh 포함 (느림 — 전체 소요시간의 70%+ 차지, 실측)
python scripts/run_dense_pipeline.py data/output/sfm_prototype/test00_run --refine
```

촬영 시 지켜야 하는 두 가지 전제(코드로 못 고침)는
[`reconstruction.py`](src/foot_engine/sfm/reconstruction.py) docstring에
정리돼 있다 — 요약하면 **촬영 내내 발이 완전히 고정**, **사진이 선명**해야
한다. 그 외 실측으로 확인된 촬영 가이드(소매/바지단이 보이면 안 됨,
손으로 발을 잡으면 안 됨)는 [`sfm/__init__.py`](src/foot_engine/sfm/__init__.py)
docstring 참고.

프레임별 QC 임계값(선명도 등)을 직접 정하고 싶다면:

```powershell
python scripts/inspect_frame_quality.py data/output/sfm_prototype/test00_run/images
```

튜닝 근거(마스크는 densify 이전에 dilate=0으로 적용할 것, DBSCAN 최대군집
방식은 발바닥 같은 성긴 진짜 부위를 삭제하는 버그가 있어 금지, `--smooth 0`,
`--postprocess-dmaps`로 저텍스처 평면 공백 완화 등)는 전부
[`dense.py`](src/foot_engine/sfm/dense.py) 모듈 docstring에 실측 수치와
함께 정리돼 있다 — 파라미터를 바꾸기 전에 먼저 읽을 것.

## 모듈 설명 (`src/foot_engine`)

| 모듈 | 역할 |
|---|---|
| `exceptions.py` | 엔진 전용 예외 계층, 각 예외에 HTTP status 매핑 포함 |
| `sfm/frame_quality.py` | SfM 전 프레임별 절대기준 QC(파일 손상/해상도/노출) |
| `sfm/reconstruction.py` | `pycolmap` 기반 sparse SfM (exhaustive 매칭 + incremental mapping) |
| `sfm/masking.py` | 발/피부 세그멘테이션 마스크 (rembg + MediaPipe Selfie Multiclass 피부 정제) |
| `sfm/cleaning.py` | 포인트클라우드 배경 제거(마스크 기반/기하학적) + 이상치 제거 + DBSCAN 군집화 |
| `sfm/dense.py` | OpenMVS 기반 dense MVS — sparse 결과를 실제 3D 메쉬로. 메쉬 생성기 본체 |
| `sfm/geometry.py` | PCA 축(`pca_axes`)/점군 자기 길이(`measured_length`) — dense 메쉬 스케일 보정용 소형 유틸 |
| `sfm/pipeline.py` | 위 전부를 엮는 오케스트레이션 (`run_pipeline()`) |

## 스크립트 목록 (`scripts/`)

| 스크립트 | 용도 |
|---|---|
| `run_sfm_pipeline.py` | 영상/사진 → 마스크 → SfM → dense MVS, 전체 자동 실행 |
| `sparse_sfm_prototype.py` | sparse SfM만 단독 실행 |
| `inspect_frame_quality.py` | 프레임별 선명도/밝기/클리핑 지표 표로 출력 (QC 임계값 튜닝용) |
| `generate_foot_masks.py` | 발/피부 마스크만 단독 생성 |
| `clean_point_cloud.py` | sparse 점군 배경/노이즈 제거만 단독 실행 (QA용) |
| `run_dense_pipeline.py` | sparse SfM 결과 → OpenMVS dense 메쉬. 별도 설치 필요(OpenMVS) |
| `compare_stl_pair.py` | STL 두 개를 정점/면/부피/watertight 기준으로 비교하는 진단 도구 |

각 스크립트의 정확한 인자와 예시는 파일 상단 docstring에 있다 — 위 표는
요약이다. 모두 `--help`로 인자 목록을 볼 수 있다.

## 알려진 한계 / 진행 상황

- **Dense MVS(`sfm/dense.py`)는 알려진 한계가 있다.**
  OpenMVS(CPU 빌드, CUDA 불필요 — `pycolmap` 내장 dense 스테레오/COLMAP
  공식 CUDA 바이너리는 최신 CUDA 13.2를 요구해 드라이버 업그레이드 부담이
  커서 채택 안 함)로 densify한다. 남은 한계:
    - **발 실루엣 경계의 MVS 깊이 노이즈는 마스크로 못 거른다.** 각 점을
      관측한 모든 카메라에 재투영해 마스크와 대조해도(만장일치 기준에서도)
      93.8%가 마스크 안쪽으로 판정된다 — 배경이 아니라 경계 자체의
      깊이 추정 노이즈라 마스크 정확도를 아무리 올려도 원리적으로
      못 거른다. `RefineMesh`(사진 광도일관성 보정)가 어느 정도 줄여주는
      게 실측 확인됐지만 완전히 없애지는 못한다.
    - **저텍스처 평면(발등/발바닥)에서 깊이 추정 자체가 비어 채워지지
      않는 경우가 있다.** `DensifyPointCloud --postprocess-dmaps 3`
      (remove-speckles+fill-gaps)로 일부 완화되나(실측: 정점 13.8%↑)
      "꽉 찬 느낌"까지는 아니라는 육안 평가 있음 — 추가로 시도해볼 것:
      `--sub-resolution-levels` 상향, `--number-views-fuse` 조정.
    - **DBSCAN 최대 군집 유지 방식은 절대 쓰지 말 것.** 발바닥처럼 촬영
      각도상 원래 점이 성긴 진짜 부위를 배경 노이즈로 오판해 통째로
      삭제하는 버그가 실측으로 확인됐다(발바닥 노멀 방향 점 60,937개 →
      0개). `dense.py`의 `clean_dense_point_cloud()`는 통계적 이상치
      제거만 쓴다.
    - **`DensifyPointCloud`가 간헐적으로 크래시한다**(ACCESS_VIOLATION/
      힙손상, 원인 불명 — 멀티스레드 경쟁 상태로 추정). `--max-threads 8`로
      낮추면 이 저장소 검증 중 재현 안 됨(기본값에 반영돼 있음).
    - **`RefineMesh`가 압도적 병목**(전체 소요시간의 70~72%, 실측
      1.5~9분)이라 `run_dense_pipeline.py` 기본값은 꺼져 있다
      (`--refine`로 켤 것).
    - **발바닥 미촬영 프로토콜 특성상 접지면에 큰 구멍이 남는다.**
      `align_sole_down()`(평탄도 기반 바닥면 정렬)으로 대응 중.
    - sparse 재구성 폴더(`sparse/0`, `sparse/1`, ...)는 **번호가 크기순이
      아니다** — `dense.py.largest_sparse_dir()`로 항상 실제 등록 이미지
      수를 비교해서 골라야 한다(실측: 어떤 촬영에서 `sparse/1`이 108장,
      `sparse/0`은 2장짜리 파편이었다).
  실험 산출물: `data/output/dense_mvs_results/`(각 결과의 상태 요약 포함).
- **스캔 데이터에 신뢰할 수 있는 ground-truth 축척이 없다** — 자기신고
  사이즈만 있고 실측 기준자가 없어, 절대 크기 정확도의 가장 큰 리스크로
  남아 있다.
- **다음으로 볼 것: 배경 제거/발 분리(마스킹) 품질.** 메쉬 정밀도나
  정렬보다 이쪽이 지금 가장 유력한 오차 원인으로 보인다(2026-08-11 기준).

## `archive/` — 더 이상 쓰지 않는 경로

`archive/deformer_ssm_pipeline/`에 랜드마크 기반 템플릿 워프
(`FootMeshDeformer`)와 SSM(통계적 형상 모델) 코드가 있다. 둘 다 한때 메쉬
생성기 후보였지만:

- **SSM**은 nearest-point ICP 기반 대응점 산출이 해부학적 인식이 없어,
  정확한 랜드마크를 줘도 PCA 기저 자체를 오염시키는 구조적 결함이 확인돼
  일찌감치 폐기됐다.
- **템플릿 워프**는 SSM 대신 한동안 실제 경로였지만, dense MVS가 궤도에
  오르면서 "템플릿을 워프하느니 dense MVS 결과물 자체를 다듬고 경량화하는
  게 낫다"는 결론이 나 2026-08-11에 폐기됐다.

지우지 않고 참고/회귀검증용으로 남겨뒀다 — 자세한 목록과 주의사항(그대로는
실행 안 됨)은 [`archive/deformer_ssm_pipeline/README.md`](archive/deformer_ssm_pipeline/README.md)
참고.

## 개발 메모

- 모든 코드 주석/docstring/CLI 출력은 한국어로 작성한다(기존 컨벤션 유지).
- Windows cp949 콘솔에서 한글 출력이 깨지는 문제가 있어, 새 CLI
  스크립트를 만들 때는 `run_sfm_pipeline.py` 상단의 `stdout`/`stderr`
  UTF-8 `reconfigure` 패턴을 그대로 따를 것.
- `data/`는 git 이력에 절대 올리지 않는다(`.gitignore` 참고, 과거 실수로
  올라갔던 이력은 `git-filter-repo`로 제거하고 강제 push된 상태). 새
  샘플/스캔을 커밋하려는 실수를 하지 않도록 주의.
- `.claude/settings.local.json`도 로컬 전용(절대경로 포함)이라 커밋하지
  않는다 — 팀과 공유할 설정은 `.claude/settings.json`에.
