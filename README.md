# HPLModeling (foot_deform_engine)

2D 사진(또는 사람이 찍은 랜드마크) → 3D 발 메쉬를 만드는 엔진. 최종 목표는 이
3D 모델을 체중부하(weight-bearing) 변형 AI에 넣어 평발/부상/절단 등 다양한
발 형태에 맞는 **맞춤 인솔**을 자동 설계하는 것이다. 이 저장소는 그 파이프라인의
"사진/랜드마크 → 3D 메쉬" 구간을 담당한다.

## 목차

- [저장소 두 축](#저장소-두-축)
- [디렉터리 구조](#디렉터리-구조)
- [설치](#설치)
- [`data/` 디렉터리 준비 (필수 — git에 없음)](#data-디렉터리-준비--필수--git에-없음)
- [빠른 시작](#빠른-시작)
- [시나리오별 사용법](#시나리오별-사용법)
- [모듈 설명 (`src/foot_engine`)](#모듈-설명-srcfoot_engine)
- [스크립트 목록 (`scripts/`)](#스크립트-목록-scripts)
- [알려진 한계 / 진행 상황](#알려진-한계--진행-상황)
- [개발 메모](#개발-메모)

## 저장소 두 축

이 프로젝트는 서로 독립적으로도 쓸 수 있는 두 개의 축으로 이뤄져 있다.

1. **랜드마크 기반 변형 엔진** (`foot_engine` 최상위 패키지) — 표준 발
   템플릿(STL) 하나를 놓고, 사진에서 뽑은(또는 사람이 찍은) 이름 붙은
   2D 랜드마크 몇 개로 그 템플릿을 RBF/TPS 변형한다. **검증 완료, 가장
   신뢰할 수 있는 경로**다. 이미지 장수가 1장이든 4장이든 동일한 코드
   경로로 동작하며, 측정 안 된 항목은 템플릿 값을 그대로 쓴다.
2. **SfM 기반 사진→3D 파이프라인** (`foot_engine.sfm`) — 외부 유료
   포토그래메트리(VRIN)를 대체하기 위한 in-house 경로. `pycolmap`으로
   사진 여러 장에서 카메라 포즈 + sparse 포인트클라우드를 복원하고,
   배경/노이즈를 제거한 뒤 위 1번 변형 엔진에 태워 최종 메쉬를 만든다.
   `fitting.py` 경로는 스칼라 계측치 몇 개만 필요해서 **sparse 복원까지만**
   있으면 되지만, 실제 3D 메쉬(시각화/QA용)가 필요하면
   [`dense.py`](src/foot_engine/sfm/dense.py)로 OpenMVS 기반 dense MVS를
   선택적으로 이어붙일 수 있다 — 아래 [시나리오 E](#e-dense-mvs-메쉬-생성-선택-별도-설치-필요)와
   [알려진 한계](#알려진-한계--진행-상황) 절 참고.

세 번째로 `foot_engine.ssm`(통계적 형상 모델, PCA 기반)이 있는데, 한때 메쉬
생성기로 쓰려 했으나 대응점(correspondence) 노이즈가 PCA 기저 자체를
오염시키는 근본적 결함이 확인되어 **현재 사용하지 않는다**. 코드는
참고/회귀검증용으로 남아 있다(지우지 말라는 명시적 지침).

## 디렉터리 구조

```
foot_deform_engine/
├── src/foot_engine/          # 패키지 본체 (pip install 안 해도 sys.path로 바로 씀)
│   ├── config.py             #   좌표계 규약 + DeformConfig
│   ├── deformer.py           #   FootMeshDeformer — 핵심 변형 엔진
│   ├── landmarks.py          #   2D 랜드마크 → mm 계측치
│   ├── mesh_utils.py         #   메쉬 I/O·계측·품질검사 유틸
│   ├── schemas.py            #   FootMeasurements/LandmarkPayload 등 데이터 구조
│   ├── service.py            #   FastAPI 등에서 재사용할 서비스 계층
│   ├── template_factory.py   #   절차적 기준 템플릿 생성기
│   ├── scan_dataset.py       #   SSM 학습용 스캔 매니페스트(CSV) 스키마
│   ├── silhouette_landmarks.py #  세그멘테이션 마스크 → 랜드마크 자동 추출
│   ├── exceptions.py         #   엔진 전용 예외 계층 (HTTP status 매핑 포함)
│   ├── sfm/                  #   사진/영상 → 3D 파이프라인
│   │   ├── frame_quality.py  #     프레임별 절대기준 QC
│   │   ├── reconstruction.py #     pycolmap 기반 sparse SfM
│   │   ├── masking.py        #     발/피부 마스크 (rembg + MediaPipe)
│   │   ├── cleaning.py       #     포인트클라우드 배경/노이즈 제거
│   │   ├── dense.py          #     (선택) OpenMVS 기반 dense MVS 메쉬 생성
│   │   ├── fitting.py        #     점군 → 계측치 → 변형 엔진 연결
│   │   └── pipeline.py       #     위 전부를 엮는 오케스트레이션
│   └── ssm/                  #   통계적 형상 모델 (현재 보류, 참고용)
│       ├── preprocessing.py
│       ├── registration.py
│       └── model.py
├── scripts/                  # 각 기능을 독립적으로 돌려볼 수 있는 CLI들
├── data/                     # git에 없음 — 아래 절 참고
│   ├── templates/            #   기준 템플릿 STL
│   ├── samples/               #   촬영 샘플 영상/랜드마크 JSON
│   ├── scans/                 #   SSM 학습용 원본 스캔 STL
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
    opencv-contrib-python rembg mediapipe onnxruntime pillow matplotlib `
    open3d pydantic tqdm pytest
```

> `pip freeze > requirements.txt`로 현재 venv를 그대로 고정하는 게
> 가장 정확하다 — 위 목록은 `import` 기준으로 추린 핵심 패키지만 나열한
> 것이라 부차 의존성(예: `trimesh`가 쓰는 `manifold3d`/`embreex`,
> `rembg`가 쓰는 `PyMatting` 등)은 pip가 알아서 딸려 온다.

CUDA/GPU는 **필요 없다** — sparse SfM(`pycolmap`)과 변형 엔진 모두 CPU로
돈다. dense MVS(`dense.py`, 선택 기능)도 CPU로 돈다(실측 확인 — COLMAP/
OpenMVS의 CUDA 빌드는 최신 CUDA 13.2를 요구해 드라이버 업그레이드 부담이
커서 채택하지 않았다).

### OpenMVS 설치 (dense MVS 쓸 때만 필요, 선택)

실제 3D 메쉬가 필요해 `dense.py`/`scripts/run_dense_pipeline.py`를 쓰려면
OpenMVS CLI 실행파일이 별도로 필요하다(pip 패키지 아님, 이 저장소
`requirements`에도 없음):

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

`data/` 전체가 `.gitignore`되어 있다 — 실제 스캔(`data/scans`), 촬영
샘플(`data/samples`), 거기서 파생된 템플릿에 **개인 식별 가능한 실측
데이터**가 섞여 있어서다. 즉 이 저장소를 새로 클론하면 `data/` 폴더가
아예 없다 — 아래 구조대로 로컬에 직접 만들거나, 별도 저장소/스토리지에서
받아와야 한다.

```
data/
├── templates/
│   ├── base_foot_template.stl     # 없어도 됨 — 처음 실행 시 자동 생성(절차적, 250mm 오른발)
│   └── S0001_real_template.stl    # 선택: 실제 스캔 기반 템플릿 (워터타이트 보수 필요, 아래 한계 참고)
├── samples/
│   ├── test00.mp4 ~ test10.mp4    # 촬영 샘플 영상 (SfM 파이프라인 입력)
│   ├── landmarks_2views.json      # 2D 랜드마크 payload 샘플 (랜드마크 경로 입력)
│   ├── landmarks_4views.json
│   └── scan_manifest_template.csv # SSM 매니페스트 CSV 샘플 (scan_id,stl_path,side,...)
├── scans/
│   └── S0001.stl ~ S0102.stl      # SSM 학습용 원본 스캔 (현재 SSM 자체는 보류 상태)
├── models/
│   └── selfie_multiclass_256x256.tflite  # MediaPipe 피부 세그멘테이션 모델
│                                          # — masking.py가 없으면 최초 1회 자동 다운로드함
└── output/                        # 각 스크립트의 산출물이 여기 쌓인다 (직접 만들 필요 없음, 자동 생성)
```

**최소로 시작하려면 아무것도 필요 없다** — `data/templates/base_foot_template.stl`이
없으면 `foot_engine.sfm.fitting.default_template_path()`나
`scripts/generate_template.py`가 절차적으로 만들어준다. 실제 예제를
그대로 돌려보고 싶다면:

- 랜드마크 경로만 시험 → `data/samples/landmarks_2views.json` 같은 JSON
  1개만 있으면 됨 (형식은 아래 [시나리오 A](#a-2d-랜드마크-json--3d-메쉬-사진-처리-불필요) 참고, 직접 만들어도 됨).
- 사진/영상 경로를 시험 → 발이 나온 영상 1개(`.mp4` 등, 최소 8프레임
  이상 뽑힐 분량)를 `data/samples/`에 아무 이름으로 넣고 그 경로를
  `--video`로 넘기면 된다. `test00.mp4` 같은 기존 샘플 이름 자체는
  중요하지 않다 — 스크립트들은 `--video`/`--images-dir` 인자로 경로를
  받으므로 원하는 위치에 아무 영상이나 두고 그 경로를 지정하면 된다.

경로를 스크립트 기본값과 다르게 두고 싶으면 항상 `--video`/`--images-dir`/
`--template`/`--out`/`--workdir` 같은 명시적 인자로 넘기면 된다 — 이
저장소의 CLI들은 `data/` 하위 경로를 하드코딩하지 않고 전부 인자로 받는다
(예외: 템플릿 인자를 아예 생략하면 `data/templates/base_foot_template.stl`을
기본값으로 씀).

## 빠른 시작

```powershell
# 1) 기준 템플릿이 없다면 생성 (있으면 건너뛰어도 됨 — 다른 스크립트가 자동 생성)
python scripts/generate_template.py

# 2) 랜드마크 JSON 샘플로 변형 데모 (사진 처리 없이 엔진 자체만 확인)
python scripts/run_deform_demo.py --landmarks data/samples/landmarks_2views.json `
    --out data/output/foot.stl

# 3) 영상 1개로 SfM 파이프라인 전체 실행 (사진 → 3D, 실험적)
python scripts/run_sfm_pipeline.py --video data/samples/test00.mp4 `
    --out data/output/test00_pipeline_fit.stl `
    --workdir data/output/sfm_pipeline/test00_run
```

## 시나리오별 사용법

### A. 2D 랜드마크 JSON → 3D 메쉬 (사진 처리 불필요)

가장 단순하고 가장 신뢰할 수 있는 경로. 사람(또는 다른 파이프라인)이 이미
이름 붙은 랜드마크의 픽셀 좌표를 알고 있을 때 쓴다.

```powershell
python scripts/run_deform_demo.py                                    # 기본 4뷰 샘플
python scripts/run_deform_demo.py --landmarks data/samples/landmarks_2views.json
python scripts/run_deform_demo.py --out data/output/foot.glb --json-report
```

입력 JSON 형식(`data/samples/landmarks_2views.json` 참고)::

```json
{
  "side": "right",
  "measurements": { "foot_length_mm": 242.0 },
  "images": [
    {
      "view": "top",
      "image_size_px": [1000, 1400],
      "landmarks": { "heel_center": [500, 1300], "toe_tip": [500, 90], "...": "..." }
    }
  ]
}
```

측정되지 않은 항목은 템플릿 기본값을 유지하므로 이미지/랜드마크가 몇 개든
동작한다. 픽셀→mm 변환 규칙, 뷰별로 뽑을 수 있는 계측 항목 목록은
[`landmarks.py`](src/foot_engine/landmarks.py) 상단 docstring 참고.

### B. 사진/영상 → 3D 메쉬 (SfM 파이프라인, 실험적)

**한 번에 전체 실행**:

```powershell
python scripts/run_sfm_pipeline.py --video data/samples/test00.mp4 `
    --template data/templates/S0001_real_template.stl `
    --side right --template-side left `
    --out data/output/test00_pipeline_fit.stl `
    --workdir data/output/sfm_pipeline/test00_run
```

**단계별로 나눠 돌리기**(중간 산출물을 뷰어로 직접 확인하고 싶을 때):

```powershell
# 1) sparse SfM 복원
python scripts/sparse_sfm_prototype.py --video data/samples/test00.mp4

# 2) 발/피부 마스크 생성
python scripts/generate_foot_masks.py data/output/sfm_prototype/test00_run/images `
    --out data/output/sfm_prototype/test00_run/masks

# 3) 배경/노이즈 제거 (권장 조합: 마스크 기반 + 군집화)
python scripts/clean_point_cloud.py data/output/sfm_prototype/test00_run/sparse/0 `
    --masks-dir data/output/sfm_prototype/test00_run/masks --cluster `
    --out data/output/sfm_prototype/test00_run/cleaned_points.ply

# 4) 계측치 추출 + 템플릿 변형
python scripts/fit_deformer_to_pointcloud.py `
    data/output/sfm_prototype/test00_run/cleaned_points.ply `
    --out data/output/test00_deformer_fit.stl
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

### C. 세그멘테이션 마스크 → 랜드마크 자동 추출

`silhouette_landmarks.py`가 기존 마스크(rembg 결과)의 윤곽선에서 이름 붙은
랜드마크 픽셀 좌표를 기하학적으로 뽑아낸다.

```powershell
python scripts/photo_to_deformer_demo.py `
    data/output/sfm_prototype/test00_run/images/frame_00000.jpg `
    data/output/sfm_prototype/test00_run/masks/frame_00000.jpg.png `
    --out data/output/photo_demo_fit.stl
```

**주의**: top-view(위에서 내려다본 사진) 추출만 검증됐고 신뢰할 수 있다.
side-view(옆모습) 추출은 발끝/뒤꿈치 방향과 발등/발바닥(sole/dorsum) 판별
로직에 알려진 버그가 있어 arch_height/instep_height/ankle_height 등 높이
계측치를 신뢰하면 안 된다(아래 [알려진 한계](#알려진-한계--진행-상황) 참고).

### D. SSM (통계적 형상 모델) — 현재 보류, 참고용

메쉬 생성기로는 쓰지 않지만 코드는 유지되며, 합성 데이터로 파이프라인
자체는 계속 돌려볼 수 있다.

```powershell
python scripts/ssm_synthetic_demo.py                    # 합성 발 데이터로 전체 파이프라인 검증
python scripts/audit_scan_manifest.py data/scans/manifest.csv        # 매니페스트 검증
python scripts/audit_scan_meshes.py data/samples/scan_manifest_template.csv `
    --stl-root data --allow-incomplete                    # STL 지오메트리 품질 감사
python scripts/build_ssm.py data/scans/manifest.csv --out data/output/ssm.npz
```

### E. Dense MVS 메쉬 생성 (선택, 별도 설치 필요)

`fitting.py` 경로(스칼라 계측치만 필요)는 이거 없이도 그대로 동작한다.
실제 3D 메쉬(시각화/QA/향후 고정밀 활용)가 필요할 때만 쓴다. OpenMVS
설치는 위 [설치](#openmvs-설치-dense-mvs-쓸-때만-필요-선택) 절 참고.

```powershell
# 1) 먼저 sparse SfM + 마스크를 만들어 둔다 (시나리오 B의 1~2단계와 동일)
python scripts/run_sfm_pipeline.py --video data/samples/test02.mp4 `
    --workdir data/output/sfm_pipeline/test02_run --out data/output/test02_fit.stl

# 2) dense 메쉬 생성 (RefineMesh 제외, 빠름 — 몇 분 내)
python scripts/run_dense_pipeline.py data/output/sfm_pipeline/test02_run

# 3) 최종 품질까지 원하면 RefineMesh 포함 (느림 — 전체 소요시간의 70%+ 차지, 실측)
python scripts/run_dense_pipeline.py data/output/sfm_pipeline/test02_run --refine
```

튜닝 근거(마스크는 densify 이전에 dilate=0으로 적용할 것, DBSCAN 최대군집
방식은 발바닥 같은 성긴 진짜 부위를 삭제하는 버그가 있어 금지, `--smooth 0`,
`--postprocess-dmaps`로 저텍스처 평면 공백 완화 등)는 전부
[`dense.py`](src/foot_engine/sfm/dense.py) 모듈 docstring에 실측 수치와
함께 정리돼 있다 — 파라미터를 바꾸기 전에 먼저 읽을 것.

## 모듈 설명 (`src/foot_engine`)

| 모듈 | 역할 |
|---|---|
| `config.py` | 좌표계 규약(X=길이, Y=너비, Z=높이), 해부학적 제어점 좌표, `DeformConfig` 튜닝 파라미터 |
| `deformer.py` | `FootMeshDeformer` — 계측치 → 제어점 목표 위치 → RBF/TPS 변형 → 품질검사, 엔진 핵심 |
| `landmarks.py` | 2D 랜드마크 dict → mm 계측치(`FootMeasurements`) 변환, 뷰별 추출 규칙 테이블 |
| `mesh_utils.py` | 메쉬 로딩/좌표계 정규화/계측/품질검사 — 순수 기하 연산 모음 |
| `schemas.py` | `FootMeasurements`/`LandmarkPayload`/`DeformationReport`/`QualityReport` 데이터 구조 |
| `service.py` | 템플릿별 `FootMeshDeformer`를 캐시하는 서비스 계층 (FastAPI 등에서 재사용) |
| `template_factory.py` | 절차적 기준 템플릿(`base_foot_template.stl`) 생성기 — 실 스캔 없이도 개발 가능 |
| `scan_dataset.py` | SSM 학습용 스캔 매니페스트(CSV) 스키마 정의 + 검증 |
| `silhouette_landmarks.py` | 세그멘테이션 마스크 윤곽선 → 랜드마크 픽셀 좌표 자동 추출 |
| `exceptions.py` | 엔진 전용 예외 계층, 각 예외에 HTTP status 매핑 포함 |
| `sfm/frame_quality.py` | SfM 전 프레임별 절대기준 QC(파일 손상/해상도/노출) |
| `sfm/reconstruction.py` | `pycolmap` 기반 sparse SfM (exhaustive 매칭 + incremental mapping) |
| `sfm/masking.py` | 발/피부 세그멘테이션 마스크 (rembg + MediaPipe Selfie Multiclass 피부 정제) |
| `sfm/cleaning.py` | 포인트클라우드 배경 제거(마스크 기반/기하학적) + 이상치 제거 + DBSCAN 군집화 |
| `sfm/dense.py` | (선택) OpenMVS 기반 dense MVS — sparse 결과를 실제 3D 메쉬로 |
| `sfm/fitting.py` | SfM 점군 → 강체 정렬 → 계측 → `FootMeshDeformer` 연결 |
| `sfm/pipeline.py` | 위 다섯 단계를 엮는 오케스트레이션 (`run_pipeline()`) |
| `ssm/preprocessing.py` | 스캔 노이즈 제거 + 자기 길이 기준 정규화 |
| `ssm/registration.py` | 공통 템플릿을 개별 스캔에 비강체 정합 |
| `ssm/model.py` | 정합된 스캔들의 정점 변동을 PCA로 압축(SSM 학습) |

## 스크립트 목록 (`scripts/`)

| 스크립트 | 용도 |
|---|---|
| `run_deform_demo.py` | 랜드마크 JSON → 변형 → STL/GLB, 전체 파이프라인 데모 |
| `generate_template.py` | 절차적 기준 템플릿 생성 |
| `run_sfm_pipeline.py` | 영상/사진 → 마스크 → SfM → 정리 → 변형, 전체 자동 실행 |
| `sparse_sfm_prototype.py` | sparse SfM만 단독 실행 |
| `inspect_frame_quality.py` | 프레임별 선명도/밝기/클리핑 지표 표로 출력 (QC 임계값 튜닝용) |
| `generate_foot_masks.py` | 발/피부 마스크만 단독 생성 |
| `clean_point_cloud.py` | sparse 점군 배경/노이즈 제거만 단독 실행 |
| `run_dense_pipeline.py` | (선택) sparse SfM 결과 → OpenMVS dense 메쉬. 별도 설치 필요 |
| `fit_deformer_to_pointcloud.py` | 점군 → 계측치 → 템플릿 변형만 단독 실행 |
| `photo_to_deformer_demo.py` | 사진 1장(top view) → 실루엣 랜드마크 자동 추출 → 변형, 종단 데모 |
| `triangulate_landmarks.py` | 여러 사진의 2D 랜드마크 픽셀 좌표 → 3D 삼각측량 |
| `compare_stl_pair.py` | STL 두 개를 정점/면/부피/watertight 기준으로 비교하는 진단 도구 |
| `audit_scan_manifest.py` | 스캔 매니페스트 CSV 스키마 검증 |
| `audit_scan_meshes.py` | 매니페스트의 STL들 지오메트리 품질 감사 |
| `build_ssm.py` | 매니페스트 기반 SSM 구축 오케스트레이션 |
| `ssm_synthetic_demo.py` | 합성 발 데이터로 SSM 파이프라인 자체 검증 (실 스캔 없이 가능) |
| `ssm_cross_validate.py` | 캐시된 정합 결과로 SSM 주성분 개수 K-fold 검증 |
| `ssm_landmark_holdout.py` | 실제 서비스 흐름을 흉내낸 랜드마크 기반 SSM 홀드아웃 검증 |
| `fit_ssm_to_pointcloud.py` | SfM 점군에 SSM을 직접 피팅하는 엔드투엔드 프로토타입 |

각 스크립트의 정확한 인자와 예시는 파일 상단 docstring에 있다 — 위 표는
요약이다. 모두 `--help`로 인자 목록을 볼 수 있다.

## 알려진 한계 / 진행 상황

- **SSM은 메쉬 생성기로 쓰지 않는다.** nearest-point ICP 기반 대응점 산출이
  해부학적 인식이 없어 정확한 랜드마크를 줘도 PCA 기저 자체를 오염시키는
  구조적 결함이 확인됨. 현재 활성 경로는 `FootMeshDeformer`(템플릿 워프)다.
- **`silhouette_landmarks.extract_side_view_landmarks()`는 미완성.**
  발끝/뒤꿈치 방향 판별과 발등/발바닥(sole/dorsum) 판별 휴리스틱이
  합성 데이터 검증에서 틀린 답을 냈다 — arch_height/instep_height/
  ankle_height 등 옆모습 기반 높이 계측치는 현재 신뢰하면 안 된다.
  top-view 추출(`extract_top_view_landmarks()`)만 검증되어 안전하다.
- **Dense MVS(`sfm/dense.py`)는 선택 기능이고 알려진 한계가 있다.**
  `fitting.py` 경로는 이거 없이도 그대로 동작한다(스칼라 계측치만 필요).
  실제 3D 메쉬가 필요할 때 OpenMVS(CPU 빌드, CUDA 불필요 — `pycolmap`
  내장 dense 스테레오/COLMAP 공식 CUDA 바이너리는 최신 CUDA 13.2를
  요구해 드라이버 업그레이드 부담이 커서 채택 안 함)로 densify한다.
  남은 한계:
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
    - sparse 재구성 폴더(`sparse/0`, `sparse/1`, ...)는 **번호가 크기순이
      아니다** — `dense.py.largest_sparse_dir()`로 항상 실제 등록 이미지
      수를 비교해서 골라야 한다(실측: 어떤 촬영에서 `sparse/1`이 108장,
      `sparse/0`은 2장짜리 파편이었다).
  실험 산출물: `data/output/dense_mvs_results/`(README에 각 결과의 상태
  요약 포함).
- **`FootMeshDeformer._apply_arch_height()`**는 한때 폭주 피드백 루프가
  있었으나 수정됨(스텝 클램프 + 미개선 시 중단). ball_width_mm는 아직
  ~40% 오차가 남아 있어 별도로 봐야 한다.
- **실 스캔을 템플릿으로 쓸 때** `ssm.clean_and_normalize()`로 먼저
  워터타이트 보수를 해야 한다(원본 스캔은 mm 단위가 아니고 발목 절단면이
  열려 있음). `S0001_real_template.stl`은 이미 보수된 버전이다.
- **좌우(카이랄리티) 처리**: 발은 카이랄 형상이라 회전만으로 정렬 안 됨.
  `fit_point_cloud_to_template()`/`run_sfm_pipeline.py` 사용 시 실제 촬영
  발과 템플릿의 좌우를 안다면 반드시 `--side`/`--template-side`를 넘길 것
  — 안 넘기면 결과가 뒤틀릴 수 있다.
- **스캔 데이터에 신뢰할 수 있는 ground-truth 축척이 없다** — 자기신고
  사이즈만 있고 실측 기준자가 없어, 절대 크기 정확도의 가장 큰 리스크로
  남아 있다.

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
