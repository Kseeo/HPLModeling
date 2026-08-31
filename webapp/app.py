"""로컬 웹 뷰어: GLB 업로드 -> 4단계 파이프라인 -> 각 단계 결과 GLB 저장/미리보기.

  1단계: 발 검출(다중뷰 피부투표 크롭) + 정렬 + 발목 절단 + 해상도(target_vertices) 맞춤
         (스무딩은 아직 안 함) -- process_glb_to_foot(postprocess=False)
  2단계: 1단계 결과에 스무딩(배경 파편/구멍 정리 + 사포질 + 고곡률 스무딩 + 마감 라플라시안)
         -- finishing.postprocess_mesh()
  3단계: hplAI GNN 추론(체크포인트로 하중 변형 예측) -- 별도 conda env(`mesh`)를
         subprocess로 호출(build_dataset.py -> predict.py -> export_glb.py 그대로 재사용)
  4단계: GNN 결과 후처리 스무딩 + 바닥 재접지 -- finish_smooth_mesh() + rest_on_floor()

1, 2, 4단계는 이 서버(foot_deform_engine .venv)와 같은 프로세스에서 바로 실행한다.
3단계만 별도 conda env가 필요해 subprocess로 뺀다(torch/PyG가 이 venv엔 없음).

실행:
    C:/Users/cani0/foot_deform_engine/.venv/Scripts/python.exe webapp/app.py
    -> http://127.0.0.1:5050 접속
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np
import trimesh
from flask import Flask, jsonify, render_template, request, send_from_directory

# ---------------------------------------------------------------------------
# 경로 설정 -- 필요하면 여기만 고치면 됨.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FOOT_ENGINE_SRC = REPO_ROOT / "src"
if str(FOOT_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(FOOT_ENGINE_SRC))

# cp949 등 비-UTF8 콘솔에서 foot_engine 쪽 한글/em-dash 출력이 깨지거나
# UnicodeEncodeError로 죽는 문제 방지 (cli.py와 같은 조치).
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

HPLAI_DIR = Path(r"C:/Users/cani0/OneDrive/바탕 화면/김서현/hplAI")
GLB_PREPROCESS_DIR = HPLAI_DIR / "glb_preprocess"
MESH_PYTHON = Path(r"C:/Users/cani0/miniconda3/envs/mesh/python.exe")
CHECKPOINTS_DIR = HPLAI_DIR / "checkpoints_local"

JOBS_DIR = Path(__file__).resolve().parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

from foot_engine.sfm.dense import find_floor_contact_mask, rest_on_floor  # noqa: E402
from foot_engine.stl_foot_extract.finishing import finish_smooth_mesh, postprocess_mesh  # noqa: E402

STAGE1_WORKER = Path(__file__).resolve().parent / "stage1_worker.py"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512MB


def job_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id
    if not d.is_dir():
        raise FileNotFoundError(f"알 수 없는 job_id: {job_id}")
    return d


def floor_contact_path(glb_path: Path) -> Path:
    return glb_path.with_name(f"{glb_path.stem}_floor_contact.npy")


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    checkpoints = sorted(p.name for p in CHECKPOINTS_DIR.glob("*.pt")) if CHECKPOINTS_DIR.is_dir() else []
    return render_template("index.html", checkpoints=checkpoints)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="파일이 없습니다"), 400

    job_id = uuid.uuid4().hex[:12]
    d = JOBS_DIR / job_id
    d.mkdir(parents=True)
    suffix = Path(f.filename).suffix or ".glb"
    input_path = d / f"0_input{suffix}"
    f.save(input_path)
    return jsonify(job_id=job_id, filename=input_path.name)


@app.route("/api/stage1/<job_id>", methods=["POST"])
def api_stage1(job_id):
    """크롭 + 정렬 + 발목 절단 + 해상도 맞춤 (스무딩 전).

    GLB 다중뷰 렌더링(pyglet)이 서버 프로세스 안에서 반복 호출되면 크래시해서
    서버 전체가 죽는 걸 실측으로 확인했다(재현: 여러 파일을 연달아 stage1
    돌리면 어느 시점에 프로세스가 트레이스백도 없이 죽음) -- 그래서 이 단계만
    `stage1_worker.py`를 매 요청마다 새 subprocess로 띄운다. 거기서 죽어도
    이 서버 프로세스는 살아있다.
    """
    try:
        d = job_dir(job_id)
        inputs = list(d.glob("0_input.*"))
        if not inputs:
            return jsonify(error="0단계(업로드) 결과가 없습니다"), 400

        args = request.get_json(silent=True) or {}
        target_vertices = int(args.get("target_vertices", 18119))
        trim_leg = bool(args.get("trim_leg", True))
        reference_length_mm = args.get("reference_length_mm")
        two_pass = bool(args.get("two_pass", False))

        out_path = d / "1_crop_ankle.glb"
        result_json = d / "1_crop_ankle_result.json"
        cmd = [
            sys.executable, str(STAGE1_WORKER),
            "--input", str(inputs[0]), "--output", str(out_path),
            "--target_vertices", str(target_vertices),
            "--trim_leg", "1" if trim_leg else "0",
            "--two_pass", "1" if two_pass else "0",
            "--result_json", str(result_json),
        ]
        if reference_length_mm:
            cmd += ["--reference_length_mm", str(float(reference_length_mm))]

        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        log = f"$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
        if proc.returncode != 0 or not result_json.exists():
            return jsonify(error="1단계 워커가 실패했습니다(렌더링 크래시 가능성)", log=log), 500

        info = json.loads(result_json.read_text(encoding="utf-8"))
        return jsonify(**info)
    except subprocess.TimeoutExpired:
        return jsonify(error="1단계 시간 초과(10분)"), 500
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/stage2/<job_id>", methods=["POST"])
def api_stage2(job_id):
    """1단계 결과에 스무딩(구멍/파편 정리 + 사포질 + 고곡률 스무딩 + 마감)."""
    try:
        d = job_dir(job_id)
        src_path = d / "1_crop_ankle.glb"
        if not src_path.exists():
            return jsonify(error="1단계 결과가 없습니다"), 400

        mesh = trimesh.load(src_path, force="mesh", process=False)
        n_before = len(mesh.vertices)

        args = request.get_json(silent=True) or {}
        fill_round_holes_enabled = bool(args.get("fill_round_holes", True))
        fill_round_holes_min_circularity = float(args.get("fill_round_holes_min_circularity", 0.5))

        smoothed, stats = postprocess_mesh(
            mesh, keep_largest=False,
            fill_round_holes_enabled=fill_round_holes_enabled,
            fill_round_holes_min_circularity=fill_round_holes_min_circularity,
        )

        out_path = d / "2_smoothed.glb"
        smoothed.export(out_path)

        # floor_contact 마스크는 정점 순서 그대로 유지되므로(구멍 메움만 정점을 뒤에
        # 덧붙일 수 있음) 같은 길이로 맞춰 이어 붙여서 넘겨준다(3단계 node_type용).
        src_mask_path = floor_contact_path(src_path)
        if src_mask_path.exists():
            mask = np.load(src_mask_path)
            n_after = len(smoothed.vertices)
            if n_after > len(mask):
                mask = np.concatenate([mask, np.zeros(n_after - len(mask), dtype=bool)])
            else:
                mask = mask[:n_after]
            np.save(floor_contact_path(out_path), mask)

        return jsonify(
            file=out_path.name,
            n_vertices=len(smoothed.vertices),
            n_faces=len(smoothed.faces),
            steps_applied=stats.steps_applied,
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/api/stage3/<job_id>", methods=["POST"])
def api_stage3(job_id):
    """hplAI GNN 추론(`mesh` conda env에서 build_dataset.py -> predict.py -> export_glb.py)."""
    try:
        d = job_dir(job_id)
        src_path = d / "2_smoothed.glb"
        if not src_path.exists():
            return jsonify(error="2단계 결과가 없습니다"), 400
        if not MESH_PYTHON.exists():
            return jsonify(error=f"mesh conda env python이 없습니다: {MESH_PYTHON}"), 500

        args = request.get_json(silent=True) or {}
        checkpoint = args.get("checkpoint")
        train_dataset_path = args.get("train_dataset_path")
        train_dataset_file = args.get("train_dataset_file") or "C3_bio.pt"
        target_faces = args.get("target_faces") or None
        use_cpu = bool(args.get("cpu", False))
        if not checkpoint:
            return jsonify(error="checkpoint를 선택하세요"), 400
        if not train_dataset_path:
            return jsonify(error="train_dataset_path(학습셋 폴더, C3_bio.pt 위치)를 입력하세요"), 400

        ckpt_path = CHECKPOINTS_DIR / checkpoint
        if not ckpt_path.exists():
            return jsonify(error=f"체크포인트가 없습니다: {ckpt_path}"), 400

        env = dict(**os.environ)
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        logs = []

        scan_pt = d / "scan.pt"
        build_cmd = [
            str(MESH_PYTHON), "glb_preprocess/build_dataset.py",
            "--input_dir", str(d), "--pattern", src_path.name,
            "--output", str(scan_pt), "--node_type_mode", "floor_contact",
        ]
        if target_faces:
            build_cmd += ["--target_faces", str(int(target_faces))]
        logs.append(_run_env(build_cmd, HPLAI_DIR, env))

        pred_pt = d / "predictions.pt"
        predict_cmd = [
            str(MESH_PYTHON), "glb_preprocess/predict.py",
            "--checkpoint", str(ckpt_path),
            "--train_dataset_path", str(train_dataset_path),
            "--train_dataset_file", str(train_dataset_file),
            "--scan_dataset", str(scan_pt),
            "--output", str(pred_pt),
        ]
        if use_cpu:
            predict_cmd.append("--cpu")
        logs.append(_run_env(predict_cmd, HPLAI_DIR, env))

        export_cmd = [
            str(MESH_PYTHON), "glb_preprocess/export_glb.py",
            "--predictions", str(pred_pt), "--output_dir", str(d), "--no-original_too",
        ]
        logs.append(_run_env(export_cmd, HPLAI_DIR, env))

        predicted = d / f"{src_path.stem}_predicted.glb"
        if not predicted.exists():
            return jsonify(error="predicted.glb가 생성되지 않았습니다", log="\n\n".join(logs)), 500

        out_path = d / "3_gnn.glb"
        shutil.move(str(predicted), out_path)
        mesh = trimesh.load(out_path, force="mesh", process=False)

        return jsonify(
            file=out_path.name,
            n_vertices=len(mesh.vertices),
            n_faces=len(mesh.faces),
            log="\n\n".join(logs),
        )
    except RuntimeError as e:
        return jsonify(error="GNN 단계 실패", log=str(e)), 500
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


def _run_env(cmd: list[str], cwd: Path, env: dict) -> str:
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    log = f"$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        raise RuntimeError(log)
    return log


@app.route("/api/stage4/<job_id>", methods=["POST"])
def api_stage4(job_id):
    """GNN 결과 스무딩 + 바닥 재접지 (foot_pipeline_postsmooth.py와 동일 로직)."""
    try:
        d = job_dir(job_id)
        src_path = d / "3_gnn.glb"
        if not src_path.exists():
            return jsonify(error="3단계 결과가 없습니다"), 400

        args = request.get_json(silent=True) or {}
        iterations = int(args.get("iterations", 10))
        lamb = float(args.get("lamb", 0.5))
        rest = bool(args.get("rest_on_floor", True))

        mesh = trimesh.load(src_path, force="mesh", process=False)
        smoothed = finish_smooth_mesh(mesh, lamb=lamb, iterations=iterations)
        if rest:
            smoothed = rest_on_floor(smoothed, floor_percentile=0.5)

        out_path = d / "4_final.glb"
        smoothed.export(out_path)

        return jsonify(
            file=out_path.name,
            n_vertices=len(smoothed.vertices),
            n_faces=len(smoothed.faces),
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/jobs/<job_id>/<path:filename>")
def serve_job_file(job_id, filename):
    return send_from_directory(job_dir(job_id), filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
