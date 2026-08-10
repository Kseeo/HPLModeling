"""Sparse SfM 결과 -> OpenMVS 기반 dense 포인트클라우드/메쉬 복원.

`reconstruction.py`의 sparse 복원은 `fitting.py`가 스칼라 계측치 몇 개만
필요로 해서 원래 여기까지 다룰 필요가 없었다(모듈 상단 주석 참고). 이 모듈은
실제 3D 메쉬(시각화/QA/향후 고정밀 활용)가 필요할 때만 쓰는 **선택적** 다음
단계다 — `fitting.py` 경로는 이 모듈 없이도 그대로 동작한다.

파이프라인(2026-08-10 실측 검증)::

    undistort_for_dense()          -- pycolmap, sparse 카메라를 PINHOLE로 왜곡보정
        └─ convert_masks_for_openmvs()  -- masking.py 마스크를 OpenMVS 명명 규칙으로
              └─ run_interface_colmap()     -- COLMAP 형식 -> OpenMVS 씬(.mvs)
                    └─ run_densify_point_cloud()  -- 마스크 기반 dense 포인트클라우드
                          └─ clean_dense_point_cloud()  -- 통계적 이상치 제거만
                                └─ run_reconstruct_mesh()   -- Delaunay+그래프컷 메싱
                                      └─ run_refine_mesh()      -- (선택) 사진 광도일관성 보정

OpenMVS는 pip 패키지가 아니라 **별도 설치한 CLI 실행파일**이 필요하다
(`InterfaceCOLMAP`/`DensifyPointCloud`/`ReconstructMesh`/`RefineMesh`,
CPU 빌드로 충분 -- 실측 확인, CUDA 불필요). 설치 경로는 `OPENMVS_BIN_DIR`
환경변수로 지정하거나 `openmvs_bin` 인자로 직접 넘긴다. 설치 방법은
README 참고.

실측으로 확인된, 코드에 그대로 반영된 결론들
--------------------------------------------
1. **마스크는 densify 이전에 적용해야 한다** (`DensifyPointCloud
   --mask-path`) -- 사후에 sparse 점 정리하듯 numpy 배열만 필터링하면 dense
   경로와 완전히 단절된다(실측: 배경 섞인 원본 그대로 densify됨).
2. **마스크 팽창(dilate)은 0으로** -- `masking.generate_masks()`의 기본
   dilate=15는 sparse 등록/랜드마크 추출용으로 안전 여유를 준 것이지,
   dense masking에는 그 여유가 배경 누출로 직결된다(실측: dilate=15 대비
   dilate=0이 인접 배경 잔여물을 뚜렷이 줄임).
3. **DBSCAN 최대 군집 유지 방식은 쓰지 말 것** -- 발바닥처럼 촬영 각도상
   원래 점이 성긴 진짜 부위를, 배경처럼 동떨어진 노이즈로 오판해 통째로
   삭제하는 버그가 실측으로 확인됐다(발바닥 노멀 방향 점 60,937개 -> 0개).
   통계적 이상치 제거(`filter_outlier_points`)만 쓴다 -- 배경 경계의
   잔여 노이즈는 남지만, 발 형상 자체를 깎아내는 것보다 안전한 실패 방향이다.
4. **마스크 기반 재투영 분류도 경계 노이즈엔 무력** -- 남은 노이즈를 각
   점이 관측된 모든 카메라에 재투영해 마스크와 대조해봐도(만장일치
   기준에서도) 93.8%가 마스크 안쪽으로 판정된다 -- 이건 배경이 아니라
   발 실루엣 경계 자체의 MVS 깊이 노이즈라 마스크로 원리적으로 못 거른다.
5. **`ReconstructMesh --smooth 0`** -- 기본 스무딩(2회)이 형상 정교함을
   깎아낸다는 사용자 육안 확인(raw 메쉬가 스무딩된 버전보다 발 형태에
   더 정확히 맞았음).
6. **`--postprocess-dmaps 3`**(remove-speckles + fill-gaps)로 저텍스처
   평면(발등/발바닥 등)의 깊이 추정 공백을 일부 메울 수 있다(실측:
   정점 13.8% 증가) -- 다만 완전한 해결책은 아니다.
7. **`RefineMesh`가 압도적 병목이다**(전체 소요시간의 70~72%, 실측
   1.5~9분) -- 사진 광도일관성 기반 보정이라 노이즈/뿔 감소에 실제
   효과가 있지만(실측 확인), 빠른 반복 실험 시엔 건너뛸 만하다.
   `--cuda-device` 기본값이 -2(CPU)라 CUDA 없이도 동작한다.
8. **OpenMVS 크래시**(ACCESS_VIOLATION/힙손상, 원인 불명 -- 아마 멀티스레드
   경쟁 상태)가 이 저장소 검증 중 두 차례 발생했다 -- `--max-threads`를
   낮추면(예: 8) 재현 안 됨. `run_densify_point_cloud()` 기본값에 반영.
9. **sparse 재구성 폴더 번호는 크기순이 아니다** -- `run_sparse_sfm()`이
   반환하는 `pycolmap.Reconstruction`을 직접 쓰면 되지만, 이미 디스크에
   저장된 `sparse/0`, `sparse/1`, ... 중 어느 게 가장 큰(등록 이미지 많은)
   것인지는 매번 직접 비교해야 한다(`largest_sparse_dir()` 참고) --
   `sparse/0`을 무조건 가정했다가 2장짜리 파편을 densify한 실패 사례 있음.
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pycolmap

from .reconstruction import filter_outlier_points

#: OpenMVS CLI 실행파일이 있는 폴더. 환경변수로 덮어쓸 것 -- 설치 방법은 README 참고.
DEFAULT_OPENMVS_BIN_DIR = os.environ.get("OPENMVS_BIN_DIR", "")

#: `DensifyPointCloud`의 간헐적 네이티브 크래시(원인 불명, 멀티스레드 경쟁
#: 상태로 추정) 재현을 막기 위한 기본 스레드 상한. 실측: 8로 낮추면 두 차례
#: 재현된 크래시(ACCESS_VIOLATION, 힙 손상)가 재발하지 않았다.
DEFAULT_MAX_THREADS = 8

#: `--postprocess-dmaps` 비트마스크: 1=remove-speckles, 2=fill-gaps.
#: 저텍스처 평면(발등/발바닥)의 깊이 추정 공백을 일부 메운다(실측: 점 13.8%↑).
DEFAULT_POSTPROCESS_DMAPS = 3


def _resolve_openmvs_bin(openmvs_bin: str | Path | None) -> Path:
    resolved = Path(openmvs_bin) if openmvs_bin else Path(DEFAULT_OPENMVS_BIN_DIR)
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"OpenMVS 실행파일 폴더를 찾을 수 없습니다: {resolved} -- "
            "OPENMVS_BIN_DIR 환경변수를 설정하거나 openmvs_bin 인자로 직접 넘기세요. "
            "설치 방법은 README의 'Dense MVS(선택)' 절 참고."
        )
    return resolved


def largest_sparse_dir(sparse_root: Path) -> Path:
    """`sparse/0`, `sparse/1`, ... 중 등록 이미지가 가장 많은 폴더를 찾는다.

    번호가 항상 크기순은 아니다(실측 확인: 어떤 촬영에서는 `sparse/1`이
    108장, `sparse/0`은 2장짜리 파편이었다) -- `sparse/0`을 무조건 가정하면
    안 된다.
    """
    best_dir, best_n = None, -1
    for d in sorted(sparse_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            n = pycolmap.Reconstruction(d).num_reg_images()
        except Exception:
            continue
        if n > best_n:
            best_dir, best_n = d, n
    if best_dir is None:
        raise FileNotFoundError(f"유효한 sparse 재구성을 찾을 수 없습니다: {sparse_root}")
    return best_dir


def undistort_for_dense(sparse_dir: Path, images_dir: Path, output_dir: Path) -> Path:
    """sparse 카메라를 PINHOLE(왜곡보정)로 바꿔 OpenMVS가 기대하는 COLMAP dense
    워크스페이스(`images/`, `sparse/`, `stereo/`)를 만든다.

    `pycolmap.undistort_images()`로 처리한다 -- 별도 `colmap.exe` 실행파일 없이도
    된다(실측 확인).
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # pycolmap이 상위 폴더까지는 안 만들어줌
    pycolmap.undistort_images(
        output_path=str(output_dir),
        input_path=str(sparse_dir),
        image_path=str(images_dir),
        output_type="COLMAP",
    )
    return output_dir


def convert_masks_for_openmvs(masks_dir: Path, out_dir: Path) -> Path:
    """`masking.py` 마스크(`<원본파일명>.png`)를 OpenMVS 명명 규칙
    (`<확장자 뺀 파일명>.mask.png`)으로 복사한다.

    OpenMVS의 `Util::getFileName()`이 확장자를 뗀 이름 뒤에 `.mask.png`를
    붙여 마스크를 찾으므로(소스 확인), 원본 마스크 파일명 그대로는 못 찾는다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in masks_dir.glob("*.png"):
        # "frame_00000.jpg.png" -> "frame_00000" (원본 확장자까지 통째로 뗀다)
        stem = f.name
        for ext in (".jpg.png", ".jpeg.png", ".png.png", ".bmp.png"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        else:
            stem = f.stem
        (out_dir / f"{stem}.mask.png").write_bytes(f.read_bytes())
        n += 1
    print(f"[dense] 마스크 {n}개를 OpenMVS 명명 규칙으로 변환: {out_dir}")
    return out_dir


def _run_openmvs(
    exe_name: str,
    args: list[str],
    workdir: Path,
    openmvs_bin: Path,
    log_name: str,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(openmvs_bin / exe_name)] + args + ["-w", str(workdir), "-v", "3"]
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    log_path = workdir / log_name
    log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"{exe_name} 실패 (exit {result.returncode}) -- 상세 로그: {log_path} "
            f"(OpenMVS 자체 로그는 workdir의 {exe_name}-*.log 참고)"
        )


def run_interface_colmap(dense_dir: Path, workdir: Path, *, openmvs_bin: str | Path | None = None) -> Path:
    """COLMAP dense 워크스페이스를 OpenMVS 씬 파일(`scene.mvs`)로 변환한다."""
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    _run_openmvs(
        "InterfaceCOLMAP.exe",
        ["-i", str(Path(dense_dir).resolve()), "-o", "scene.mvs"],
        workdir, bin_dir, "log_interface_colmap.txt",
    )
    return workdir / "scene.mvs"


def run_densify_point_cloud(
    scene_mvs: Path,
    workdir: Path,
    *,
    masks_dir: Path | None = None,
    openmvs_bin: str | Path | None = None,
    max_threads: int = DEFAULT_MAX_THREADS,
    postprocess_dmaps: int = DEFAULT_POSTPROCESS_DMAPS,
    output_name: str = "scene_dense.mvs",
) -> Path:
    """마스크 기반 dense 포인트클라우드를 만든다 (`scene_dense.ply` + `.mvs`).

    Args:
        masks_dir: `convert_masks_for_openmvs()`로 변환된(=`.mask.png`
            규칙) 마스크 폴더. 지정하면 배경 픽셀에서는 깊이 계산 자체를
            생략한다 -- 사후 필터링보다 근본적인 배경 제거(실측 확인:
            raw depth 계산량이 정확히 절반으로 줆). `masking.generate_masks()`를
            `dilate=0`으로 호출해 만들 것(모듈 docstring 2번 참고).
        postprocess_dmaps: 기본 3(remove-speckles+fill-gaps). 저텍스처 평면
            깊이 공백을 메운다(모듈 docstring 6번 참고). 0이면 비활성.
    """
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    args = [scene_mvs.name, "-o", output_name, "--max-threads", str(max_threads)]
    if postprocess_dmaps:
        args += ["--postprocess-dmaps", str(postprocess_dmaps)]
    if masks_dir is not None:
        args += ["--mask-path", str(Path(masks_dir).resolve()) + os.sep, "--ignore-mask-label", "0"]
    _run_openmvs("DensifyPointCloud.exe", args, workdir, bin_dir, "log_densify.txt")
    return workdir / output_name.replace(".mvs", ".ply")


def clean_dense_point_cloud(
    dense_ply_path: Path,
    out_path: Path,
    *,
    k: int = 8,
    std_ratio: float = 2.0,
) -> tuple[int, int]:
    """dense 포인트클라우드에서 통계적 이상치만 제거한다 (DBSCAN 사용 안 함).

    OpenMVS의 dense PLY는 표준 필드(xyz/rgb/normal) 외에 커스텀 리스트
    속성(`view_indices`/`view_weights` -- 각 점을 관측한 카메라 가시성
    정보로, 이후 `ReconstructMesh`의 그래프컷 가중치 계산에 쓰인다)을 갖고
    있어 `trimesh`/`open3d`로 읽고 다시 쓰면 그 정보가 소실된다(둘 다
    조용히 무시함) -- 그 상태로 `ReconstructMesh -p`에 넘기면 크래시한다
    (실측 확인: ACCESS_VIOLATION). 그래서 원본 바이트를 그대로 보존한 채
    살아남는 점의 레코드만 골라 이어붙이는 방식으로 직접 파싱/저장한다.

    DBSCAN(최대 군집 유지)은 의도적으로 안 쓴다 -- 모듈 docstring 3번 참고.

    Returns:
        (원본 점 개수, 정리 후 점 개수).
    """
    data = dense_ply_path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    body = data[header_end:]
    n = int([line for line in header.splitlines() if line.startswith("element vertex")][0].split()[-1])

    # 고정 필드 27바이트: x,y,z(f4)*3 + red,green,blue(u1)*3 + nx,ny,nz(f4)*3.
    # 그 뒤로 가변 길이 리스트 두 개(view_indices: u1 count + u4*count,
    # view_weights: u1 count + f4*count)가 이어진다 -- OpenMVS dense PLY 고정 스키마.
    fixed_size = 27
    offsets = np.empty(n + 1, dtype=np.int64)
    xyz = np.empty((n, 3), dtype=np.float32)
    pos = 0
    for i in range(n):
        offsets[i] = pos
        xyz[i] = struct.unpack_from("<3f", body, pos)
        pos += fixed_size
        n_idx = body[pos]
        pos += 1 + 4 * n_idx
        n_w = body[pos]
        pos += 1 + 4 * n_w
    offsets[n] = pos
    if pos != len(body):
        raise ValueError(
            f"{dense_ply_path} 파싱 실패 -- 예상 스키마(xyz+rgb+normal+view_indices"
            "+view_weights)와 다른 형식일 수 있습니다."
        )

    inliers = filter_outlier_points(xyz, k=k, std_ratio=std_ratio)
    kept_n = int(inliers.sum())
    print(f"[dense] 점단위 정리(통계적 이상치 제거만): {n:,} -> {kept_n:,} ({kept_n/n:.1%} 유지)")

    chunks = [body[offsets[i]:offsets[i + 1]] for i in np.where(inliers)[0]]
    new_body = b"".join(chunks)
    new_header = header.replace(f"element vertex {n}\n", f"element vertex {kept_n}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(new_header.encode("ascii") + new_body)
    return n, kept_n


def run_reconstruct_mesh(
    scene_mvs: Path,
    point_cloud_ply: Path,
    workdir: Path,
    *,
    openmvs_bin: str | Path | None = None,
    smooth: int = 0,
    output_name: str = "scene_mesh.mvs",
) -> Path:
    """Delaunay 사면체화 + 그래프컷으로 점군을 메쉬로 만든다.

    Args:
        smooth: 기본 0(끔). OpenMVS 기본값(2)이 형상 정교함을 깎아낸다는
            육안 확인 있음(모듈 docstring 5번 참고) -- raw 메쉬가 더 정확했다.
    """
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    _run_openmvs(
        "ReconstructMesh.exe",
        [scene_mvs.name, "-p", point_cloud_ply.name, "-o", output_name, "--smooth", str(smooth)],
        workdir, bin_dir, "log_reconstruct_mesh.txt",
    )
    return workdir / output_name.replace(".mvs", ".ply")


def run_refine_mesh(
    scene_mvs: Path,
    mesh_ply: Path,
    workdir: Path,
    *,
    openmvs_bin: str | Path | None = None,
    output_name: str = "scene_mesh_refined.mvs",
) -> Path:
    """사진 광도일관성 기반으로 메쉬 정점 위치를 보정한다 (선택적, 가장 느린 단계).

    CPU로도 동작한다(`--cuda-device` 기본값 -2). 노이즈/뿔 형태 결함을
    실제로 줄이는 효과가 육안 확인됐지만, 전체 파이프라인 소요시간의
    70~72%를 차지하는 압도적 병목이다(실측: 1.5~9분) -- 빠른 반복
    실험에서는 건너뛸 것.
    """
    bin_dir = _resolve_openmvs_bin(openmvs_bin)
    _run_openmvs(
        "RefineMesh.exe",
        [scene_mvs.name, "-m", mesh_ply.name, "-o", output_name],
        workdir, bin_dir, "log_refine_mesh.txt",
    )
    return workdir / output_name.replace(".mvs", ".ply")


def run_dense_pipeline(
    *,
    sparse_dir: Path,
    images_dir: Path,
    masks_dir: Path,
    workdir: Path,
    openmvs_bin: str | Path | None = None,
    refine: bool = False,
    postprocess_dmaps: int = DEFAULT_POSTPROCESS_DMAPS,
    max_threads: int = DEFAULT_MAX_THREADS,
) -> Path:
    """위 단계 전부를 엮는 오케스트레이션. 최종 메쉬 경로를 반환한다.

    Args:
        sparse_dir: `reconstruction.run_sparse_sfm()`이 만든 sparse 재구성
            폴더(예: `<workdir>/sparse/0`). 폴더 번호가 가장 큰 것이라는
            보장이 없으면 `largest_sparse_dir()`로 먼저 확인할 것.
        masks_dir: `masking.generate_masks(..., dilate=0)`로 만든 마스크
            폴더 (dilate 값이 커지면 배경 경계 노이즈가 늘어난다 -- 모듈
            docstring 2번 참고).
        refine: RefineMesh(사진 광도일관성 보정)까지 돌릴지. 기본 False --
            전체 시간의 70% 이상을 차지하는 병목이라(모듈 docstring 7번),
            빠른 확인이 필요하면 끄고 최종 산출물만 켤 것.
    """
    # OpenMVS 서브프로세스는 -w(workdir)를 cwd로 실행되므로, 그 외 입력 경로는
    # 전부 절대경로로 넘겨야 한다 -- 상대경로를 그대로 두면 cwd가 바뀐 뒤
    # 엉뚱한 곳을 가리키게 된다(실측으로 확인된 버그, 조용히 실패하고 OpenMVS
    # 자체 로그도 안 남아 원인 파악이 어려웠다).
    sparse_dir = Path(sparse_dir).resolve()
    images_dir = Path(images_dir).resolve()
    masks_dir = Path(masks_dir).resolve()
    workdir = Path(workdir).resolve()

    dense_dir = undistort_for_dense(sparse_dir, images_dir, workdir / "dense")
    openmvs_masks_dir = convert_masks_for_openmvs(masks_dir, workdir / "openmvs_masks")
    openmvs_dir = workdir / "openmvs"

    scene_mvs = run_interface_colmap(dense_dir, openmvs_dir, openmvs_bin=openmvs_bin)
    dense_ply = run_densify_point_cloud(
        scene_mvs, openmvs_dir, masks_dir=openmvs_masks_dir, openmvs_bin=openmvs_bin,
        max_threads=max_threads, postprocess_dmaps=postprocess_dmaps,
    )
    cleaned_ply = openmvs_dir / "scene_dense_cleaned.ply"
    clean_dense_point_cloud(dense_ply, cleaned_ply)
    mesh_ply = run_reconstruct_mesh(scene_mvs, cleaned_ply, openmvs_dir, openmvs_bin=openmvs_bin)

    if refine:
        mesh_ply = run_refine_mesh(scene_mvs, mesh_ply, openmvs_dir, openmvs_bin=openmvs_bin)

    print(f"[dense] 완료: {mesh_ply}")
    return mesh_ply
