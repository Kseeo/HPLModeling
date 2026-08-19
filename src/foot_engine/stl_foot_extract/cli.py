"""`foot_engine.stl_foot_extract` CLI -- `scripts/`의 공용 유틸(`_cli_common` 등)에
의존하지 않는 독립 진입점.

사용 예::

    # 이미 분리된 조각을 색으로 보여주는 피커(가장 신뢰도 높은 경로 -- 우선 시도할 것)
    python -m foot_engine.stl_foot_extract.cli component-pick project_229.stl

    # 피커에서 고른 조각 번호로 추출 + 정리
    python -m foot_engine.stl_foot_extract.cli extract project_229.stl \\
        --component-index 0 --out project_229_clean.stl

    # 씨앗점 피커 열기(자동 후보 힌트 포함) -- 조각이 발과 fuse돼 안 나뉠 때
    python -m foot_engine.stl_foot_extract.cli pick project_229.stl

    # 사람이 피커에서 확인한 좌표로 크롭 + 정리
    python -m foot_engine.stl_foot_extract.cli extract project_229.stl \\
        --seed-point 0.01,0.08,0.09 --crop-radius 0.03 --out project_229_clean.stl

    # 자동 후보 1위로 바로 크롭(휴리스틱, 결과 확인 필수) + 정리
    python -m foot_engine.stl_foot_extract.cli extract project_229.stl --auto --out project_229_clean.stl

    # 후보만 출력(크롭 없이)
    python -m foot_engine.stl_foot_extract.cli suggest project_229.stl

    # 색/텍스처 있는 입력(GLB 등): 발 검출(다중뷰 피부투표) -> 정리/스무딩 -> 정렬/스케일/
    # 바닥 접지까지 기본 적용(--no-align으로 끌 수 있음). 실제 구현은 postprocess_pipeline.py
    # 의 process_glb_to_foot() -- CLI 아닌 코드에서 재사용할 땐 그 함수를 직접 호출할 것.
    python -m foot_engine.stl_foot_extract.cli texture-extract project_232.glb --out project_232_clean.glb

    # 위 전체 체인 + 발목 절단 + 다른 데이터셋 해상도(정점 수) 맞춤 + 접지 노드 마스크까지
    python -m foot_engine.stl_foot_extract.cli texture-extract project_232.glb \\
        --out project_232_gnn.glb --trim-leg --target-vertices 18119 --floor-contact-tolerance-mm 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

from foot_engine.sfm.dense import find_floor_contact_mask, finalize_mesh

from .branch_cut import suggest_bend_components
from .finishing import postprocess_mesh
from .locate import find_dense_regions, list_components
from .picker import open_component_picker, open_picker
from .pipeline import extract_foot
from .postprocess_pipeline import export_result, process_glb_to_foot

# cp949 등 비-UTF8 콘솔에서 한글 출력이 깨지거나 죽는 문제 방지 -- scripts/_cli_common.py와
# 같은 부작용이지만, 이 패키지는 scripts/에 의존하지 않아야 해서 여기 자체적으로 둔다.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def _parse_point(text: str) -> tuple[float, float, float]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"'x,y,z' 형식이어야 합니다: {text!r}")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"숫자 파싱 실패: {text!r} ({e})") from e


def _add_common_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-postprocess", dest="postprocess", action="store_false",
                         help="배경 파편 정리 + 스무딩(finishing.postprocess_mesh)을 끈다. 기본 켜짐.")
    parser.add_argument("--sand-iterations", type=int, default=3, help="사포질 반복 횟수(기본 3).")
    parser.add_argument("--curvature-iterations", type=int, default=150, help="고곡률 스무딩 반복 횟수(기본 150).")
    parser.add_argument("--finish-smooth-iterations", type=int, default=40, help="마감 라플라시안 반복 횟수(기본 40).")
    parser.add_argument("--fill-holes-max-diameter-ratio", type=float, default=0.05,
                         help="이 비율(메쉬 전체 대각선 대비) 이하인 구멍만 메움(기본 0.05). "
                              "발목 절단면처럼 원래 열려있어야 하는 큰 구멍까지 메우지 않도록 낮게 유지할 것.")
    parser.add_argument("--fill-round-holes", action="store_true",
                         help="원형 구멍(단순 미관측 결손 추정)은 크더라도 자동으로 메움. 기본 꺼짐 -- "
                              "결과를 확인하며 쓸 것(절단면과 헷갈릴 소지 있음).")
    parser.add_argument("--fill-round-holes-min-circularity", type=float, default=0.7,
                         help="--fill-round-holes일 때 이 원형도 이상만 메움(기본 0.7, 1.0=완전한 원)")


def _add_common_align_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-align", dest="align", action="store_false",
                         help="정렬(축 정렬 + 스케일 + 바닥 접지, sfm.dense.finalize_mesh)을 끈다. 기본 켜짐.")
    parser.add_argument("--reference-length-mm", type=float, default=None,
                         help="자기신고 발길이(mm) -- 없으면 placeholder 250mm 기준(절대 축척 아님, 경고 출력).")
    parser.add_argument("--trim-leg", action="store_true",
                         help="정렬 전 다리(정강이)까지 찍힌 패턴이 뚜렷하면 잘라낸다. 기본 꺼짐 -- "
                              "소수 사례로만 검증됨, 애매한 케이스를 과잉절단할 수 있음.")
    parser.add_argument("--target-vertices", type=int, default=None,
                         help="지정하면 이 정점 수 근방까지 단순화(쿼드릭 축약)+마감 스무딩한다 "
                              "-- 다른 데이터셋/모델의 메쉬 해상도에 맞출 때 사용.")
    parser.add_argument("--floor-contact-tolerance-mm", type=float, default=None,
                         help="지정하면 바닥(접지면)에서 이 거리(mm) 이내인 정점을 접지 노드로 "
                              "표시해 '<out파일명>_floor_contact.npy'(정점 순서와 1:1 대응하는 "
                              "불리언 배열)로 같이 저장한다. 기본 꺼짐.")
    parser.set_defaults(align=True)


def _apply_align(mesh: trimesh.Trimesh, args: argparse.Namespace, out_path: Path) -> trimesh.Trimesh:
    # glTF(.glb/.gltf) 스펙은 Y-up이 규약이라, finalize_mesh() 기본값인 Z-up(STL/슬라이서 관례)으로
    # 내보내면 표준을 지키는 뷰어에서 다시 옆으로 누운 것처럼 보인다 -- 확장자로 분기.
    z_up = out_path.suffix.lower() not in (".glb", ".gltf")
    aligned, scale_factor = finalize_mesh(
        mesh, reference_length_mm=args.reference_length_mm, trim_leg=args.trim_leg, z_up=z_up,
        target_vertices=args.target_vertices,
    )
    up_axis = "Z" if z_up else "Y"
    print(f"[align] 정렬 완료(x{scale_factor:.4f} 스케일, {up_axis}-up, 발바닥={up_axis}0)")

    if args.floor_contact_tolerance_mm is not None:
        mask = find_floor_contact_mask(
            aligned, tolerance_mm=args.floor_contact_tolerance_mm, up_axis=2 if z_up else 1,
        )
        mask_path = out_path.with_name(f"{out_path.stem}_floor_contact.npy")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, mask)
        print(f"[floor-contact] 저장: {mask_path}")

    return aligned


def cmd_suggest(args: argparse.Namespace) -> int:
    mesh = trimesh.load(args.mesh, process=True)
    regions = find_dense_regions(mesh, top_k=args.top_k)
    if not regions:
        print("구역을 하나도 못 찾았습니다.")
        return 1
    for i, r in enumerate(regions):
        print(
            f"#{i+1} score={r.score:.3f} (구형성={r.sphericity_score:.2f} 다지구조={r.toe_score:.2f}) "
            f"점개수={r.n_points} 중심={r.centroid.round(6).tolist()} "
            f"권장여유={r.density_radius * 2.0:.5g}(판정반경*2.0)"
        )
    return 0


def cmd_bend_split(args: argparse.Namespace) -> int:
    mesh = trimesh.load(args.mesh, process=True)
    components = suggest_bend_components(
        mesh, corner_angle_deg=args.corner_angle, tip_top_k=args.top_k,
    )
    if len(components) <= 1:
        print("코너를 못 찾아 조각이 안 갈라졌습니다(직선으로 곧게 이어진 형태이거나 임계각이 너무 큼).")
        return 1

    out_dir = args.out_dir or args.mesh.with_name(f"{args.mesh.stem}_bend_parts")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(components):
        out_path = out_dir / f"part{i+1}.stl"
        c.mesh.export(out_path)
        dx, dy, dz = c.bbox_size
        print(
            f"#{i+1} {out_path.name}: 정점 {c.n_vertices:,}개, "
            f"bbox=({dx:.4f}, {dy:.4f}, {dz:.4f}) "
            f"구형성={c.sphericity_score:.2f} 다지구조={c.toe_score:.2f}"
        )
    print(f"[bend-split] {len(components)}개 조각을 {out_dir}에 저장했습니다 -- 열어보고 발인 조각을 고르세요.")
    return 0


def cmd_texture_extract(args: argparse.Namespace) -> int:
    out_path = args.out or args.mesh.with_name(f"{args.mesh.stem}_extracted{args.mesh.suffix}")
    # glTF(.glb/.gltf) 스펙은 Y-up이 규약이라, 그 외 확장자(STL 등)는 대부분 뷰어/슬라이서
    # 관례인 Z-up으로 내보낸다 -- 안 그러면 표준을 지키는 뷰어에서 옆으로 누운 것처럼 보임.
    z_up = out_path.suffix.lower() not in (".glb", ".gltf")

    result = process_glb_to_foot(
        args.mesh,
        n_views=args.n_views, resolution=args.resolution,
        vote_threshold=args.vote_threshold, close_gap_radius_mult=args.close_gap_radius,
        postprocess=args.postprocess,
        sand_iterations=args.sand_iterations,
        curvature_iterations=args.curvature_iterations,
        finish_smooth_iterations=args.finish_smooth_iterations,
        fill_holes_max_diameter_ratio=args.fill_holes_max_diameter_ratio,
        fill_round_holes_enabled=args.fill_round_holes,
        fill_round_holes_min_circularity=args.fill_round_holes_min_circularity,
        align=args.align,
        reference_length_mm=args.reference_length_mm,
        trim_leg=args.trim_leg,
        z_up=z_up,
        target_vertices=args.target_vertices,
        floor_contact_tolerance_mm=args.floor_contact_tolerance_mm,
    )
    export_result(result, out_path)
    return 0


def cmd_pick(args: argparse.Namespace) -> int:
    open_picker(args.mesh, n_regions=args.top_k, auto_open=not args.no_open)
    return 0


def cmd_component_pick(args: argparse.Namespace) -> int:
    open_component_picker(
        args.mesh, min_component_vertices=args.min_vertices, auto_open=not args.no_open,
    )
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    mesh = trimesh.load(args.mesh, process=True)
    print(f"[extract] 입력: {args.mesh} (정점 {len(mesh.vertices):,}개, 면 {len(mesh.faces):,}개)")

    if args.component_index is not None:
        comps = list_components(mesh, min_vertices=args.min_vertices)
        if args.component_index >= len(comps):
            print(f"[extract] component-index={args.component_index}, 조각은 {len(comps)}개뿐입니다.")
            return 1
        chosen = comps[args.component_index]
        print(
            f"[extract] 조각 #{args.component_index+1} 선택: 정점 {chosen.n_vertices:,}개, "
            f"bbox={chosen.bbox_size.round(4).tolist()}"
        )
        out_mesh = chosen.mesh
        if args.postprocess:
            out_mesh, _ = postprocess_mesh(
                out_mesh, keep_largest=False,  # 이미 단일 연결요소이니 재분리 불필요
                sand_iterations=args.sand_iterations,
                curvature_iterations=args.curvature_iterations,
                finish_smooth_iterations=args.finish_smooth_iterations,
                fill_holes_max_diameter_ratio=args.fill_holes_max_diameter_ratio,
                fill_round_holes_enabled=args.fill_round_holes,
                fill_round_holes_min_circularity=args.fill_round_holes_min_circularity,
            )
    else:
        seed_point = _parse_point(args.seed_point) if args.seed_point else None
        result = extract_foot(
            mesh,
            seed_point=seed_point,
            crop_radius=args.crop_radius,
            auto=args.auto,
            region_index=args.region_index,
            postprocess=args.postprocess,
            postprocess_kwargs=dict(
                sand_iterations=args.sand_iterations,
                curvature_iterations=args.curvature_iterations,
                finish_smooth_iterations=args.finish_smooth_iterations,
                fill_holes_max_diameter_ratio=args.fill_holes_max_diameter_ratio,
                fill_round_holes_enabled=args.fill_round_holes,
                fill_round_holes_min_circularity=args.fill_round_holes_min_circularity,
            ),
        )
        out_mesh = result.mesh

    out_path = args.out or args.mesh.with_name(f"{args.mesh.stem}_extracted{args.mesh.suffix}")
    if args.align:
        out_mesh = _apply_align(out_mesh, args, out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(out_path)
    print(f"[extract] 저장: {out_path} (정점 {len(out_mesh.vertices):,}개, 면 {len(out_mesh.faces):,}개)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m foot_engine.stl_foot_extract.cli",
        description="외부 STL(스캔 품질은 좋으나 배경/다른 물체가 같이 찍힌 경우)에서 발 부위 추출",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_suggest = sub.add_parser("suggest", help="발 구역 후보를 점수와 함께 출력(크롭 없음)")
    p_suggest.add_argument("mesh", type=Path)
    p_suggest.add_argument("--top-k", type=int, default=5)
    p_suggest.set_defaults(func=cmd_suggest)

    p_bend = sub.add_parser("bend-split", help="말단에서 꺾이는 지점(코너)을 찾아 조각으로 나눠 각각 저장")
    p_bend.add_argument("mesh", type=Path)
    p_bend.add_argument("--out-dir", type=Path, default=None)
    p_bend.add_argument("--corner-angle", type=float, default=55.0, help="코너 판정 임계각(도, 기본 55)")
    p_bend.add_argument("--top-k", type=int, default=6, help="탐색할 말단 개수(기본 6)")
    p_bend.set_defaults(func=cmd_bend_split)

    p_tex = sub.add_parser(
        "texture-extract",
        help="색/텍스처 있는 입력(GLB 등)에서 다중뷰 피부분할 투표로 발을 자동 크롭(가장 신뢰도 높음)",
    )
    p_tex.add_argument("mesh", type=Path)
    p_tex.add_argument("--out", type=Path, default=None)
    p_tex.add_argument("--n-views", type=int, default=16, help="가상 카메라 개수(기본 16)")
    p_tex.add_argument("--resolution", type=int, default=640, help="렌더 가로 해상도(기본 640, 세로는 0.75배)")
    p_tex.add_argument("--vote-threshold", type=float, default=0.5,
                        help="관측된 뷰 중 이 비율 이상 피부로 보여야 채택(기본 0.5)")
    p_tex.add_argument("--close-gap-radius", type=float, default=12.0,
                        help="빈틈 닫힘(팽창-침식) 반경, 전형적 정점 간격의 배수(기본 12, 0=끔)")
    _add_common_postprocess_args(p_tex)
    _add_common_align_args(p_tex)
    p_tex.set_defaults(func=cmd_texture_extract)

    p_pick = sub.add_parser("pick", help="씨앗점 피커(로컬 HTML, 자동 구역 힌트 포함)를 연다")
    p_pick.add_argument("mesh", type=Path)
    p_pick.add_argument("--top-k", type=int, default=5, help="같이 표시할 자동 구역 개수(기본 5, 0=끔)")
    p_pick.add_argument("--no-open", action="store_true", help="생성 후 브라우저로 자동으로 열지 않는다")
    p_pick.set_defaults(func=cmd_pick)

    p_comp_pick = sub.add_parser(
        "component-pick",
        help="이미 분리된 연결 요소를 색으로 구분해 보여주고 사람이 고르게 하는 피커를 연다",
    )
    p_comp_pick.add_argument("mesh", type=Path)
    p_comp_pick.add_argument("--min-vertices", type=int, default=30, help="이보다 작은 조각은 표시 안 함(기본 30)")
    p_comp_pick.add_argument("--no-open", action="store_true", help="생성 후 브라우저로 자동으로 열지 않는다")
    p_comp_pick.set_defaults(func=cmd_component_pick)

    p_extract = sub.add_parser("extract", help="발 부위를 크롭 + 정리해 저장")
    p_extract.add_argument("mesh", type=Path)
    p_extract.add_argument("--out", type=Path, default=None)
    p_extract.add_argument("--seed-point", type=str, default=None, help="'x,y,z' -- pick으로 확인한 좌표")
    p_extract.add_argument("--crop-radius", type=float, default=None, help="--seed-point 크롭 반지름")
    p_extract.add_argument("--auto", action="store_true",
                            help="자동 구역으로 바로 크롭(휴리스틱, 결과 확인 필수) -- --seed-point 대안")
    p_extract.add_argument("--region-index", type=int, default=0, help="--auto일 때 몇 번째 구역을 쓸지(기본 0=1위)")
    p_extract.add_argument("--component-index", type=int, default=None,
                            help="component-pick으로 고른 조각 번호(0=1위) -- --seed-point/--auto 대안, "
                                 "이미 분리된 조각을 그대로 쓸 때 가장 신뢰도 높음")
    p_extract.add_argument("--min-vertices", type=int, default=30,
                            help="--component-index일 때 이보다 작은 조각은 무시(기본 30, component-pick과 맞출 것)")
    _add_common_postprocess_args(p_extract)
    _add_common_align_args(p_extract)
    p_extract.set_defaults(func=cmd_extract)

    args = parser.parse_args(argv)
    if args.command == "extract":
        modes = [args.seed_point is not None, args.auto, args.component_index is not None]
        if sum(modes) != 1:
            parser.error("extract는 --seed-point(+--crop-radius) / --auto / --component-index 중 정확히 하나가 필요합니다")
        if (args.seed_point is None) != (args.crop_radius is None):
            parser.error("--seed-point와 --crop-radius는 같이 지정해야 합니다")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
