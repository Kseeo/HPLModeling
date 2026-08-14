"""영상/사진 업로드 -> (영상이면) 프레임 추출 -> 이미지 선택/삭제 -> SfM-dense 파이프라인 실행 웹 UI.

`streamlit run scripts/ui_app.py`로 실행한다(일반 `python`으로 실행하지 않음).
실제 로직은 `foot_engine.sfm`을 그대로 호출한다 — 이 파일은 4단계 위저드
화면만 담당하는 얇은 UI 래퍼다. OpenMVS 설치/`OPENMVS_BIN_DIR` 설정은
README 참고(dense 단계에 필요).
"""

from __future__ import annotations

import contextlib
import io
import shutil
import time
from pathlib import Path

import streamlit as st

import _cli_common  # noqa: F401  -- sys.path 설정 + 콘솔 UTF-8 고정(부작용 import)

from foot_engine.exceptions import CaptureQualityError  # noqa: E402
from foot_engine.sfm import reconstruction  # noqa: E402
from foot_engine.sfm.pipeline import run_pipeline  # noqa: E402
from foot_engine.sfm.reconstruction import compute_sharpness  # noqa: E402

RUNS_ROOT = _cli_common.ROOT / "data" / "output" / "ui_runs"

st.set_page_config(page_title="발 스캔 파이프라인", layout="wide")


def _new_run_dir() -> Path:
    run_dir = RUNS_ROOT / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _reset() -> None:
    st.session_state.clear()
    st.session_state.stage = "upload"


if "stage" not in st.session_state:
    st.session_state.stage = "upload"

st.title("발 스캔 파이프라인")
st.caption("1. 영상/사진 업로드 → 2. (영상이면) 프레임 추출 → 3. 이미지 선택/삭제 → 4. SfM-dense 실행")

if st.session_state.stage != "upload":
    st.button("처음부터 다시", on_click=_reset)

# ---------------------------------------------------------------- 1. 업로드
if st.session_state.stage == "upload":
    upload_mode = st.radio("입력 방식", ["영상", "사진 여러 장"], horizontal=True)

    if upload_mode == "영상":
        uploaded = st.file_uploader("영상 파일 업로드", type=["mp4", "mov", "avi", "mkv"])
        if uploaded is not None:
            run_dir = _new_run_dir()
            video_path = run_dir / f"input{Path(uploaded.name).suffix}"
            video_path.write_bytes(uploaded.getvalue())
            st.session_state.run_dir = run_dir
            st.session_state.video_path = video_path
            st.session_state.stage = "extract"
            st.rerun()
    else:
        uploaded_images = st.file_uploader(
            "사진 여러 장 업로드 (촬영 순서대로 선택하면 이후 검토 화면도 그 순서로 정렬됩니다)",
            type=["jpg", "jpeg", "png"], accept_multiple_files=True,
        )
        if uploaded_images:
            st.write(f"{len(uploaded_images)}장 선택됨")
            if st.button("업로드", type="primary", disabled=len(uploaded_images) < 8):
                run_dir = _new_run_dir()
                images_dir = run_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                # 업로드 순서를 프레임 번호로 그대로 매긴다 -- 원본 파일명은 기기마다
                # 제각각이라 정렬 기준으로 못 쓴다(select 단계는 이름순 정렬에 의존).
                for i, f in enumerate(uploaded_images):
                    ext = Path(f.name).suffix.lower() or ".jpg"
                    (images_dir / f"frame_{i:05d}{ext}").write_bytes(f.getvalue())
                st.session_state.run_dir = run_dir
                st.session_state.images_dir = images_dir
                st.session_state.stage = "select"
                st.rerun()
            if len(uploaded_images) < 8:
                st.warning("SfM에는 최소 8장이 필요합니다.")

# ---------------------------------------------------------------- 2. 프레임 추출
elif st.session_state.stage == "extract":
    st.video(str(st.session_state.video_path))
    interval = st.slider("프레임 추출 간격(초)", 0.1, 2.0, 0.5, 0.1)
    col1, col2 = st.columns(2)
    start_time = col1.number_input("시작 시각(초)", min_value=0.0, value=0.0)
    end_time = col2.number_input("끝 시각(초, 0=끝까지)", min_value=0.0, value=0.0)

    if st.button("프레임 추출", type="primary"):
        images_dir = st.session_state.run_dir / "images"
        with st.spinner("프레임 추출 중..."):
            try:
                reconstruction.extract_frames(
                    st.session_state.video_path, images_dir, interval,
                    start_time=start_time, end_time=end_time or None,
                )
            except Exception as e:  # noqa: BLE001 -- 사용자에게 원인 그대로 보여줌
                st.error(f"프레임 추출 실패: {e}")
                st.stop()
        st.session_state.images_dir = images_dir
        st.session_state.stage = "select"
        st.rerun()

# ---------------------------------------------------------------- 3. 선택/삭제
elif st.session_state.stage == "select":
    images_dir: Path = st.session_state.images_dir
    names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    # 선명도는 계산 비용이 있어(라플라시안 분산), 파일 목록이 바뀔 때만 다시 계산한다.
    if st.session_state.get("sharpness_names") != names:
        st.session_state.sharpness = {n: compute_sharpness(images_dir / n) for n in names}
        st.session_state.sharpness_names = names
    sharpness = st.session_state.sharpness

    st.write(f"추출된 이미지 {len(names)}장 — 촬영 순서대로 정렬했습니다. 지울 사진을 체크한 뒤 삭제하세요.")

    with st.expander("흐린 사진 자동 선택"):
        pct = st.slider("선명도 하위 몇 %를 삭제 후보로 미리 체크할지", 0, 50, 0, 5)
        if st.button("적용") and pct > 0:
            cutoff_idx = max(1, int(len(names) * pct / 100))
            worst = sorted(names, key=lambda n: sharpness[n])[:cutoff_idx]
            for n in worst:
                st.session_state[f"del_{n}"] = True
            st.rerun()

    to_delete = []
    n_cols = 5
    cols = st.columns(n_cols)
    for i, name in enumerate(names):
        with cols[i % n_cols]:
            st.image(str(images_dir / name), caption=f"{name} (선명도 {sharpness[name]:.1f})", use_container_width=True)
            if st.checkbox("삭제", key=f"del_{name}"):
                to_delete.append(name)

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button(f"선택한 {len(to_delete)}장 삭제", disabled=not to_delete):
        for name in to_delete:
            (images_dir / name).unlink(missing_ok=True)
            sharpness.pop(name, None)
            st.session_state.pop(f"del_{name}", None)
        st.rerun()

    remaining = len(names) - len(to_delete)
    if col2.button(f"남은 {remaining}장으로 파이프라인 실행", type="primary", disabled=remaining < 8):
        st.session_state.stage = "run"
        st.rerun()
    if remaining < 8:
        st.warning("SfM에는 최소 8장이 필요합니다.")

# ---------------------------------------------------------------- 4. 파이프라인 실행
elif st.session_state.stage == "run":
    reference_length_mm = st.number_input(
        "자기신고 발길이(mm, 없으면 비워두면 250mm placeholder 사용)",
        min_value=0.0, value=0.0,
    )
    refine = st.checkbox(
        "RefineMesh(정밀 보정) 사용 — 표면 노이즈를 확실히 줄이지만 전체 소요 시간이 크게 늘어남",
        value=False,
    )
    trim_leg = st.checkbox(
        "발목 위 다리 자동 트림 — 발목까지만이 아니라 다리(정강이)까지 찍혀서 축 정렬이 "
        "다리 쪽으로 쏠리는 경우를 겨냥. 패턴이 뚜렷할 때만 자르고, 애매하면 안 자름. "
        "소수 사례로만 검증됨",
        value=False,
    )
    # resolution_level/scales를 낮추면(성긴 해상도/적은 반복) 빨라지지만, 같은 영상도
    # 실행마다 폭 치수가 달라지는 문제가 있어 프리셋으로 노출하지 않고 파이프라인
    # 기본값(최고 정밀도)을 그대로 쓴다.

    # (sand_min_neighbors, sand_max_neighbors, sand_iterations,
    #  curvature_percentile, curvature_max_radius_mult, curvature_iterations, curvature_alpha)
    SMOOTH_PRESETS = {
        "기본": (16, 32, 3, 60.0, 25.0, 150, 0.7),
        "강하게(디테일 희생)": (40, 80, 8, 20.0, 40.0, 250, 0.8),
    }
    smooth_strength = st.select_slider("표면 매끄러움", options=list(SMOOTH_PRESETS), value="기본")
    (
        sand_min_neighbors, sand_max_neighbors, sand_iterations,
        curvature_percentile, curvature_max_radius_mult, curvature_iterations, curvature_alpha,
    ) = SMOOTH_PRESETS[smooth_strength]

    if st.button("실행", type="primary"):
        run_dir = st.session_state.run_dir
        out_mesh = run_dir / "result.stl"
        log_buffer = io.StringIO()
        spinner_msg = "SfM + dense MVS 파이프라인 실행 중... (수 분"
        spinner_msg += "~수십 분" if refine else ""
        spinner_msg += " 소요될 수 있습니다)"
        with st.spinner(spinner_msg):
            try:
                with contextlib.redirect_stdout(log_buffer):
                    result = run_pipeline(
                        workdir=run_dir / "pipeline",
                        out_mesh=out_mesh,
                        images_dir=st.session_state.images_dir,
                        reference_length_mm=reference_length_mm or None,
                        refine=refine,
                        sand_min_neighbors=sand_min_neighbors,
                        sand_max_neighbors=sand_max_neighbors,
                        sand_iterations=sand_iterations,
                        curvature_percentile=curvature_percentile,
                        curvature_max_radius_mult=curvature_max_radius_mult,
                        curvature_iterations=curvature_iterations,
                        curvature_alpha=curvature_alpha,
                        trim_leg=trim_leg,
                        keep_intermediates=True,
                    )
            except CaptureQualityError as e:
                st.error(f"촬영 품질 문제로 중단됨: {e.message}")
                st.text(log_buffer.getvalue())
                st.stop()
            except Exception as e:  # noqa: BLE001
                st.error(f"파이프라인 실패: {e}")
                st.text(log_buffer.getvalue())
                st.stop()

        st.success("완료!")
        st.write(
            f"등록된 이미지: {result.n_points_registered_images}/{result.n_points_total_images}, "
            f"메쉬 정점 {result.n_mesh_vertices:,}개, 면 {result.n_mesh_faces:,}개, "
            f"스케일 x{result.scale_factor:.4f}"
        )
        st.download_button(
            "결과 메쉬 다운로드(.stl)", data=out_mesh.read_bytes(),
            file_name="result.stl", mime="model/stl",
        )
        with st.expander("실행 로그"):
            st.text(log_buffer.getvalue())
