# webapp

GLB 하나를 업로드하면 4단계를 순서대로 눌러 실행하고, 각 단계 결과를 미리보기+다운로드할
수 있는 로컬 웹앱.

| 단계 | 내용 | 실행 위치 |
|---|---|---|
| 1 | 발 검출(크롭) + 정렬 + 발목 절단 + 해상도 맞춤 | 이 서버(같은 프로세스) |
| 2 | 스무딩(파편/구멍 정리 + 사포질 + 고곡률 스무딩 + 마감) | 이 서버(같은 프로세스) |
| 3 | hplAI GNN 추론(체크포인트로 하중 변형 예측) | `mesh` conda env (subprocess) |
| 4 | GNN 결과 스무딩 + 바닥 재접지 | 이 서버(같은 프로세스) |

## 실행

```
C:/Users/cani0/foot_deform_engine/.venv/Scripts/python.exe webapp/app.py
```

브라우저에서 http://127.0.0.1:5050 접속.

## 3단계(GNN) 전에 확인할 것

- `checkpoint`: 드롭다운에 `hplAI/checkpoints_local/*.pt` 목록이 자동으로 뜬다.
- `train_dataset_path`: **C3_bio.pt가 들어있는 폴더 경로를 직접 입력**해야 한다
  (정규화 통계 + 입력 차원을 재구성하는 데만 쓰이고, 이 파일 자체를 예측하지 않음).
  이 파일 위치를 못 찾으면 3단계는 에러 로그를 그대로 화면에 보여준다.
- `mesh` conda env(`C:/Users/cani0/miniconda3/envs/mesh`)에 torch/PyG가 설치돼 있어야
  한다 -- `glb_preprocess/README.md` 참고.

## 코드 경로

- `app.py`: Flask 서버. 1/2/4단계는 `foot_engine.stl_foot_extract.*` /
  `foot_engine.sfm.dense`를 직접 import(`src/`를 sys.path에 추가), 3단계는
  `hplAI/glb_preprocess/{build_dataset,predict,export_glb}.py`를 그대로
  subprocess로 호출한다(새 로직을 만들지 않고 기존 스크립트 재사용).
- 업로드/중간 산출물은 `webapp/jobs/<job_id>/`에 쌓인다(0_input -> 1_crop_ankle ->
  2_smoothed -> 3_gnn -> 4_final).
