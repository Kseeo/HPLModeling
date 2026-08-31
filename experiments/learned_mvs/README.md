# 프로토타입: 학습 기반 MVS (PatchmatchNet)로 발끝/발날 관측-부족 문제 완화 시도

## 왜

`dense_mvs_replaces_deformer_pipeline.md` 메모 2026-08-25 항목 참고: 발끝/발날이
납작해지는 문제를 순수 기하학적 MVS(OpenMVS) 후처리 튜닝으로는 해결 못 한다는 게
3단계 비교(스무딩 켬/끔/원본)로 확인됨 -- 재구성 자체(관측 부족 지역을 점이 없으면
채울 수 없음)의 한계. 학습 기반 depth 추정 모델은 사진 유사도만이 아니라 학습된
prior로 그 지역을 보완할 가능성이 있어 시도해본다.

**아직 검증 안 됨** -- 이 폴더는 실제로 개선되는지 확인하기 위한 프로토타입이지,
파이프라인에 편입 결정된 게 아니다.

## 왜 이 구조(격리)

메인 파이프라인(`src/foot_engine/sfm/`)을 전혀 건드리지 않는다. 효과가 없으면
`experiments/learned_mvs/` 폴더 통째로 지우고, 이것 때문에 설치한
`torch`/`torchvision`도 지우면(`pip uninstall torch torchvision`) 원상복구 끝.
`_vendor/`(외부 코드 clone)는 `.gitignore`로 커밋 안 되므로 삭제 시 흔적도 안 남는다.

## 구성

- `setup_vendor.py` -- PatchmatchNet 원본 저장소를 `_vendor/PatchmatchNet`에
  고정 커밋(`8dc6cb4`)으로 clone한다(MIT 라이선스, 사전학습 체크포인트 내장이라
  별도 다운로드 불필요). 1회 실행.
- `run_prototype.py` -- 실제 실행 스크립트. `run_sfm_pipeline.py --keep-intermediates`로
  만든 run 폴더 하나를 받아:
  1. 우리 코드의 `dense.undistort_for_dense()`를 재사용해 COLMAP undistort
     워크스페이스(`images/`+`sparse/`)를 만든다(OpenMVS로 넘기기 직전과 동일한 산출물).
  2. 벤더 `colmap_input.py`로 PatchmatchNet 입력 포맷(`cams/`, `pair.txt`)으로 변환.
  3. 벤더 `eval.py`로 depth 추정 + 융합(fusion)까지 실행 -- 결과는
     `<out-dir>/fused.ply` (점군, 메쉬 아님).
- `requirements.txt` -- 이 프로토타입 전용 추가 의존성(`torch`/`torchvision` 등).
  메인 프로젝트 의존성에는 안 넣는다.

## 사용법

```powershell
# 1회: 벤더 코드 준비
python experiments/learned_mvs/setup_vendor.py

# 2회 이상 필요할 때: 추가 의존성 설치 (CUDA 12.1 wheel 예시 -- 드라이버에 맞게 조정)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python plyfile

# 먼저 기존 파이프라인을 --keep-intermediates로 한 번 돌려 sparse SfM 산출물을 남긴다
python scripts/run_sfm_pipeline.py --video data/samples/test01.mp4 `
    --workdir data/output/test01_pmnet_run --out data/output/test01_pmnet_run/dummy.stl `
    --keep-intermediates

# 학습 기반 MVS 프로토타입 실행
python experiments/learned_mvs/run_prototype.py data/output/test01_pmnet_run
```

결과: `data/output/test01_pmnet_run/pmnet_mvs/fused.ply` -- 기존
`sparse_points.ply`/`dense_mvs`의 원본 densify 점군과 발끝/발날 부위 밀도를
육안 비교(같은 `render_mesh_views.py` 계열 스크립트 또는 아무 뷰어)로 확인할 것.

## GPU 필요

PatchmatchNet의 `eval.py`는 CPU 폴백이 없다(`model.cuda()` 하드코딩) -- NVIDIA GPU
필수. 이 머신은 RTX 4070 확인됨(OpenMVS CUDA 빌드는 드라이버/툴킷 버전 불일치로
실패했었지만, PyTorch wheel은 자체 CUDA 런타임을 번들하므로 별개 문제).

## 다음 확인할 것 (아직 안 함)

- `fused.ply`가 실제로 발끝/발날에 점이 더 많이/정확히 잡히는지 육안 비교
- 개선되면: 이 점군을 OpenMVS `ReconstructMesh`에 먹여서 메쉬까지 뽑아 최종 비교
  (지금은 점군 단계까지만 -- 메싱까지 가치 있는지 먼저 판단 후 진행)
- 처리 시간 실측 (지금 파이프라인 5~10분 예산 안에 들어오는지)
