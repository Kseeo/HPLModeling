# archive/deformer_ssm_pipeline

2026-08-11에 여기로 옮긴, **더 이상 메쉬 생성기로 안 쓰는** 두 경로의 코드다.

- **SSM(통계적 형상 모델, PCA 기반)** — nearest-point ICP 기반 대응점 산출이
  해부학적 인식이 없어, 정확한 랜드마크를 줘도 PCA 기저 자체를 오염시키는
  구조적 결함이 진작에 확인돼 있었다(`ssm/`).
- **랜드마크 기반 템플릿 워프(`FootMeshDeformer`)** — SSM 대신 한동안 실제
  메쉬 생성 경로였지만, dense MVS(`src/foot_engine/sfm/dense.py`)가 궤도에
  오르면서 "템플릿을 워프하느니 dense MVS 결과물 자체를 다듬고 경량화하는 게
  낫다"는 결론이 났다(`deformer.py`, `template_factory.py`, `service.py`,
  `sfm/fitting.py`, 관련 스크립트들).

현재 활성 경로는 `src/foot_engine/sfm/pipeline.py`(dense MVS)뿐이다. 이 폴더는
지우지 않고 **참고/회귀검증용으로만** 남겨둔다 — 코드 리뷰나 "그때 왜 이렇게
했더라" 확인 용도다.

## 그대로는 실행되지 않는다

여기 있는 파일들은 옮겨질 당시의 상대 임포트(`from ..config`,
`from .. import mesh_utils` 등)를 그대로 갖고 있다. 그런데 `exceptions.py`처럼
지금도 활성 파이프라인이 쓰는 파일은 `src/foot_engine/`에 그대로 남았고,
`sfm/` 패키지도 `fitting.py` 하나만 여기 있고 나머지(`__init__.py`,
`dense.py` 등)는 없다 — 즉 이 폴더 자체를 `sys.path`에 얹어 바로
`import foot_engine`을 하면 안 깨진다는 보장이 없다.

다시 돌려보고 싶으면:
1. 이 폴더를 별도 위치로 복사
2. `src/foot_engine/exceptions.py`(그리고 필요하면 `sfm/__init__.py` 등
   활성 코드가 참조하는 나머지 파일)를 그 복사본에도 채워 넣고
3. `scripts/`의 각 파일 상단 `sys.path.insert(...)`가 그 복사본의 `src`를
   가리키도록 조정

정확히 무엇이 왜 폐기됐는지는 프로젝트 메모리
(`ssm-pipeline-build-notes`, `deformer-photo-pipeline-status`,
`dense-mvs-replaces-deformer-pipeline`)에 더 자세히 적혀 있다.

## 옮긴 목록

```
src/foot_engine/
├── deformer.py
├── template_factory.py
├── service.py
├── landmarks.py
├── silhouette_landmarks.py
├── config.py
├── schemas.py
├── mesh_utils.py
├── scan_dataset.py
├── ssm/
│   ├── __init__.py
│   ├── model.py
│   ├── preprocessing.py
│   └── registration.py
└── sfm/
    └── fitting.py

scripts/
├── build_ssm.py
├── fit_deformer_to_pointcloud.py
├── fit_ssm_to_pointcloud.py
├── generate_template.py
├── photo_to_deformer_demo.py
├── run_deform_demo.py
├── ssm_cross_validate.py
├── ssm_landmark_holdout.py
├── ssm_synthetic_demo.py
├── triangulate_landmarks.py
├── audit_scan_manifest.py
└── audit_scan_meshes.py
```

`fitting.py`가 갖고 있던 순수 기하 유틸(`pca_axes`/`measured_length`)만은
dense MVS 파이프라인(`dense.py`, `pipeline.py`)이 스케일 추정에 계속 써서,
`src/foot_engine/sfm/geometry.py`로 복사해 살려뒀다 — 이 폴더의 `fitting.py`
사본은 그 원본일 뿐 더 이상 활성 코드에서 참조되지 않는다.
