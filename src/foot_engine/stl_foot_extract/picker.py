"""오염이 심한 메쉬에서 사람이 뷰어에서 직접 발 위치를 클릭해 좌표를 얻기 위한
로컬 HTML 뷰어를 만든다. `locate.suggest_foot_regions()` 자동 후보를 빨간
점으로 같이 보여준다(참고용, 확정 아님).

**Artifact가 아니라 로컬 파일이다**: three.js를 CDN에서 불러온다 -- 사용자
브라우저에서 `file://`로 직접 열 것(퍼블리시/공유 대상 아님).
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import numpy as np
import trimesh

from .locate import find_dense_regions, list_components

#: 조각 색상 팔레트(순환) -- 구분만 되면 되므로 색맹 접근성까지는 신경 안 씀.
_COMPONENT_COLORS = [
    "#66ccff", "#ff6666", "#66ff99", "#ffcc66", "#cc66ff", "#ff99cc",
    "#99ff66", "#66ffff", "#ff9966", "#9966ff", "#ccff66", "#ff66cc",
]

#: 브라우저 렌더링 성능을 위한 표시 포인트 수 상한.
DEFAULT_MAX_DISPLAY_POINTS = 150_000

_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>foot seed picker -- {title}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #111; overflow: hidden; font-family: sans-serif; }}
  #panel {{
    position: fixed; top: 8px; left: 8px; z-index: 10; background: rgba(20,20,20,0.85);
    color: #eee; padding: 10px 14px; border-radius: 8px; max-width: 420px; font-size: 13px;
    max-height: 90vh; overflow-y: auto;
  }}
  #panel h3 {{ margin: 0 0 6px; font-size: 14px; }}
  #panel code {{ background: #333; padding: 1px 4px; border-radius: 3px; }}
  #history {{ max-height: 180px; overflow-y: auto; margin-top: 6px; }}
  .pt {{ padding: 3px 0; border-bottom: 1px solid #333; }}
  button {{ cursor: pointer; margin-top: 4px; }}
</style>
</head>
<body>
<div id="panel">
  <h3>{title}</h3>
  <div>정점 {n_points:,}개 표시 중 (원본 {n_total:,}개) -- 바운딩 대각선 약 <code>{diag:.5f}</code></div>
  <div>클릭: 점 선택 (마지막 클릭 = 씨앗점 후보). 최근 2개 사이 거리 = 반지름 가늠용.</div>
  <div id="last">아직 클릭 없음</div>
  <button id="copyBtn">마지막 좌표 복사 (--seed-point 형식)</button>
  <div id="history"></div>
  <div style="margin-top:8px; border-top:1px solid #444; padding-top:6px;">
    <b>발 후보 힌트</b>(빨간 점, 자동 추정 -- 확정 아님, 참고만 할 것)
    <div id="candidates"></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const positions = {positions_json};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.0001, 1000);
const renderer = new THREE.WebGLRenderer({{antialias: true}});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
geometry.computeBoundingSphere();
const material = new THREE.PointsMaterial({{color: 0x66ccff, size: {point_size}}});
const points = new THREE.Points(geometry, material);
scene.add(points);

const center = geometry.boundingSphere.center;
const radius = geometry.boundingSphere.radius || 1;
camera.position.set(center.x + radius*1.5, center.y + radius*1.5, center.z + radius*1.5);
camera.lookAt(center);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.update();

// 발 후보 힌트(휴리스틱, 확정 아님) -- 빨간 점으로 표시 + 카메라 이동 버튼.
const candidates = {candidates_json};
if (candidates.length > 0) {{
  const candPositions = candidates.flatMap(c => c.point);
  const candGeom = new THREE.BufferGeometry();
  candGeom.setAttribute('position', new THREE.Float32BufferAttribute(candPositions, 3));
  const candMat = new THREE.PointsMaterial({{color: 0xff3333, size: {point_size} * 6}});
  scene.add(new THREE.Points(candGeom, candMat));

  document.getElementById('candidates').innerHTML = candidates.map((c, i) => {{
    const [x, y, z] = c.point;
    return `<div class="pt">#${{i+1}} score=${{c.score.toFixed(2)}} `
      + `(점${{c.n_points}}/구형성${{c.sphericity.toFixed(2)}}/다지${{c.toe.toFixed(2)}}) `
      + `${{x.toFixed(5)}}, ${{y.toFixed(5)}}, ${{z.toFixed(5)}} `
      + `<button data-idx="${{i}}" class="flyto">이동</button></div>`;
  }}).join('');
  document.querySelectorAll('.flyto').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const [x, y, z] = candidates[parseInt(btn.dataset.idx)].point;
      const target = new THREE.Vector3(x, y, z);
      camera.position.set(x + radius*0.3, y + radius*0.3, z + radius*0.3);
      controls.target.copy(target);
      controls.update();
    }});
  }});
}}

// 클릭 위치 근방에서 반지름 내 가장 가까운 점 하나만 골라 좌표를 읽는다.
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = radius * 0.01;
const mouse = new THREE.Vector2();
const history = [];

function onClick(event) {{
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(points);
  if (hits.length === 0) return;
  const p = hits[0].point;
  history.unshift(p);
  if (history.length > 8) history.pop();
  render_panel();
}}
renderer.domElement.addEventListener('click', onClick);

function render_panel() {{
  const last = history[0];
  let lastText = `마지막: x=${{last.x.toFixed(5)}}, y=${{last.y.toFixed(5)}}, z=${{last.z.toFixed(5)}}`;
  if (history.length >= 2) {{
    const d = last.distanceTo(history[1]);
    lastText += `<br>최근 2점 사이 거리: <code>${{d.toFixed(5)}}</code> (크롭 반지름 가늠용)`;
  }}
  document.getElementById('last').innerHTML = lastText;
  document.getElementById('history').innerHTML = history.map(
    (p,i) => `<div class="pt">#${{history.length-i}} ${{p.x.toFixed(5)}}, ${{p.y.toFixed(5)}}, ${{p.z.toFixed(5)}}</div>`
  ).join('');
}}

document.getElementById('copyBtn').addEventListener('click', () => {{
  if (history.length === 0) return;
  const p = history[0];
  const text = `${{p.x.toFixed(6)}},${{p.y.toFixed(6)}},${{p.z.toFixed(6)}}`;
  navigator.clipboard.writeText(text).catch(() => {{}});
  alert('복사됨: ' + text + '\\n(클립보드 접근이 막히면 위 텍스트를 직접 옮겨 적으세요)');
}});

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""


def build_picker_html(
    mesh_path: Path,
    *,
    max_display_points: int = DEFAULT_MAX_DISPLAY_POINTS,
    n_regions: int = 5,
    rng: np.random.Generator | None = None,
) -> str:
    """메쉬 정점을 서브샘플링해 클릭-피킹 가능한 로컬 HTML 문자열을 만든다.

    `n_regions` > 0이면 `find_dense_regions()` 힌트(빨간 점 = 구역 중심,
    확정 아님)도 같이 표시한다 -- `0`으로 끌 수 있다.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    mesh = trimesh.load(mesh_path, process=True)
    v = mesh.vertices
    n_total = len(v)
    if n_total > max_display_points:
        idx = rng.choice(n_total, size=max_display_points, replace=False)
        v = v[idx]

    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    point_size = max(diag * 0.0015, 1e-6)

    candidates_payload = []
    if n_regions > 0:
        for r in find_dense_regions(mesh, top_k=n_regions, rng=rng):
            candidates_payload.append({
                "point": r.centroid.round(6).tolist(), "score": r.score,
                "n_points": r.n_points, "sphericity": r.sphericity_score, "toe": r.toe_score,
            })

    return _HTML_TEMPLATE.format(
        title=mesh_path.name,
        n_points=len(v),
        n_total=n_total,
        diag=diag,
        positions_json=json.dumps(v.astype(np.float64).round(6).flatten().tolist()),
        candidates_json=json.dumps(candidates_payload),
        point_size=point_size,
    )


_COMPONENT_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>foot component picker -- {title}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #111; overflow: hidden; font-family: sans-serif; }}
  #panel {{
    position: fixed; top: 8px; left: 8px; z-index: 10; background: rgba(20,20,20,0.9);
    color: #eee; padding: 10px 14px; border-radius: 8px; max-width: 380px; font-size: 13px;
    max-height: 94vh; overflow-y: auto;
  }}
  #panel h3 {{ margin: 0 0 6px; font-size: 14px; }}
  #panel code {{ background: #333; padding: 1px 4px; border-radius: 3px; }}
  .comp {{ padding: 6px 8px; border-radius: 6px; margin: 4px 0; border: 1px solid #333; cursor: pointer; }}
  .comp:hover {{ border-color: #888; }}
  .comp.active {{ border-color: #fff; background: rgba(255,255,255,0.08); }}
  .swatch {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
  .comp-stats {{ color: #aaa; font-size: 11.5px; margin-top: 2px; }}
  button {{ cursor: pointer; margin-top: 4px; }}
  label {{ display: block; margin-top: 8px; }}
  #selected-box {{ margin-top: 10px; padding-top: 8px; border-top: 1px solid #444; }}
</style>
</head>
<body>
<div id="panel">
  <h3>{title}</h3>
  <div>연결 요소 {n_components}개 (정점 {n_points:,}개 표시 중, 실제 {n_total:,}개) -- 색 = 조각.</div>
  <div>클릭: 3D에서 점 클릭 또는 아래 목록에서 선택.</div>
  <label><input type="checkbox" id="isolateToggle"> 선택한 조각만 보기</label>
  <div id="selected-box">아직 선택 안 됨</div>
  <div id="legend"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const positions = {positions_json};
const colors = {colors_json};
const compIds = {comp_ids_json};
const components = {components_json};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.0001, 1000);
const renderer = new THREE.WebGLRenderer({{antialias: true}});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
const baseColors = new Float32Array(colors);
geometry.setAttribute('color', new THREE.Float32BufferAttribute(baseColors.slice(), 3));
geometry.computeBoundingSphere();
const material = new THREE.PointsMaterial({{size: {point_size}, vertexColors: true}});
const points = new THREE.Points(geometry, material);
scene.add(points);

const center = geometry.boundingSphere.center;
const radius = geometry.boundingSphere.radius || 1;
camera.position.set(center.x + radius*1.5, center.y + radius*1.5, center.z + radius*1.5);
camera.lookAt(center);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.update();

let selected = -1;
let isolate = false;

function applyColors() {{
  const colorAttr = geometry.getAttribute('color');
  for (let i = 0; i < compIds.length; i++) {{
    const isSel = selected < 0 || compIds[i] === selected;
    const vis = (!isolate || isSel);
    const dim = (selected >= 0 && !isSel) ? 0.12 : 1.0;
    const r0 = baseColors[i*3], g0 = baseColors[i*3+1], b0 = baseColors[i*3+2];
    if (vis) {{
      colorAttr.setXYZ(i, r0*dim, g0*dim, b0*dim);
    }} else {{
      colorAttr.setXYZ(i, 0, 0, 0);
    }}
  }}
  colorAttr.needsUpdate = true;
}}

function selectComponent(idx) {{
  selected = idx;
  document.querySelectorAll('.comp').forEach(el => el.classList.toggle('active', parseInt(el.dataset.idx) === idx));
  const c = components[idx];
  document.getElementById('selected-box').innerHTML = idx < 0 ? '아직 선택 안 됨' : (
    `선택: 조각 #${{idx+1}} (정점 ${{c.n_vertices.toLocaleString()}}개)<br>` +
    `bbox=(${{c.bbox[0].toFixed(4)}}, ${{c.bbox[1].toFixed(4)}}, ${{c.bbox[2].toFixed(4)}})<br>` +
    `구형성=${{c.sphericity.toFixed(2)}} 다지구조=${{c.toe.toFixed(2)}}<br>` +
    `<button id="copyBtn">CLI 인자 복사 (--component-index ${{idx}})</button>`
  );
  if (idx >= 0) {{
    document.getElementById('copyBtn').addEventListener('click', () => {{
      const text = `--component-index ${{idx}}`;
      navigator.clipboard.writeText(text).catch(() => {{}});
      alert('복사됨: ' + text);
    }});
  }}
  applyColors();
}}

function renderLegend() {{
  document.getElementById('legend').innerHTML = components.map((c, i) => {{
    return `<div class="comp" data-idx="${{i}}">` +
      `<span class="swatch" style="background:${{c.color}}"></span>` +
      `<b>#${{i+1}}</b> 정점 ${{c.n_vertices.toLocaleString()}}` +
      `<div class="comp-stats">bbox=(${{c.bbox[0].toFixed(3)}}, ${{c.bbox[1].toFixed(3)}}, ${{c.bbox[2].toFixed(3)}}) ` +
      `구형성=${{c.sphericity.toFixed(2)}} 다지=${{c.toe.toFixed(2)}}</div></div>`;
  }}).join('');
  document.querySelectorAll('.comp').forEach(el => {{
    el.addEventListener('click', () => selectComponent(parseInt(el.dataset.idx)));
  }});
}}
renderLegend();

document.getElementById('isolateToggle').addEventListener('change', (e) => {{
  isolate = e.target.checked;
  applyColors();
}});

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = radius * 0.01;
const mouse = new THREE.Vector2();

function onClick(event) {{
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(points);
  if (hits.length === 0) return;
  selectComponent(compIds[hits[0].index]);
}}
renderer.domElement.addEventListener('click', onClick);

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""


def build_component_picker_html(
    mesh_path: Path,
    *,
    max_display_points: int = DEFAULT_MAX_DISPLAY_POINTS,
    min_component_vertices: int = 30,
    rng: np.random.Generator | None = None,
) -> str:
    """이미 공간적으로 분리된 연결 요소(`locate.list_components()`)를 색으로
    구분해 보여주고, 클릭 또는 목록에서 사람이 하나를 고르게 하는 로컬 HTML.

    발과 배경이 물리적으로 붙어있지 않은(가장 흔한) 케이스를 위한 도구 --
    자동 판별 없이 사람이 직접 눈으로 확인해서 고른다.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    mesh = trimesh.load(mesh_path, process=True)
    n_total = len(mesh.vertices)
    comps = list_components(mesh, min_vertices=min_component_vertices)
    if not comps:
        raise RuntimeError("연결 요소를 하나도 못 찾았습니다(메쉬가 비어있을 수 있음).")

    # 조각별로 정점 수에 비례해 표시 예산을 배분하되, 작은 조각도 최소한은 보이게 한다.
    total_vertices = sum(c.n_vertices for c in comps)
    min_per_comp = 200
    budgets = []
    remaining = max_display_points
    for c in comps:
        share = int(max_display_points * c.n_vertices / total_vertices)
        budget = min(c.n_vertices, max(share, min_per_comp))
        budgets.append(budget)

    all_positions: list[float] = []
    all_colors: list[float] = []
    all_comp_ids: list[int] = []
    components_meta = []
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    for i, (c, budget) in enumerate(zip(comps, budgets)):
        v = c.mesh.vertices
        if len(v) > budget:
            idx = rng.choice(len(v), size=budget, replace=False)
            v = v[idx]
        color_hex = _COMPONENT_COLORS[i % len(_COMPONENT_COLORS)]
        r = int(color_hex[1:3], 16) / 255.0
        g = int(color_hex[3:5], 16) / 255.0
        b = int(color_hex[5:7], 16) / 255.0
        for p in v:
            all_positions.extend([float(p[0]), float(p[1]), float(p[2])])
            all_colors.extend([r, g, b])
            all_comp_ids.append(i)
        components_meta.append({
            "n_vertices": c.n_vertices,
            "bbox": c.bbox_size.round(6).tolist(),
            "sphericity": c.sphericity_score,
            "toe": c.toe_score,
            "color": color_hex,
        })

    point_size = max(diag * 0.0018, 1e-6)

    return _COMPONENT_HTML_TEMPLATE.format(
        title=mesh_path.name,
        n_components=len(comps),
        n_points=len(all_comp_ids),
        n_total=n_total,
        positions_json=json.dumps([round(x, 6) for x in all_positions]),
        colors_json=json.dumps([round(x, 6) for x in all_colors]),
        comp_ids_json=json.dumps(all_comp_ids),
        components_json=json.dumps(components_meta),
        point_size=point_size,
    )


def open_component_picker(
    mesh_path: str | Path,
    *,
    out_path: str | Path | None = None,
    max_display_points: int = DEFAULT_MAX_DISPLAY_POINTS,
    min_component_vertices: int = 30,
    auto_open: bool = True,
) -> Path:
    """`build_component_picker_html()`을 파일로 저장하고(기본) 브라우저로 연다."""
    mesh_path = Path(mesh_path)
    html = build_component_picker_html(
        mesh_path, max_display_points=max_display_points, min_component_vertices=min_component_vertices,
    )
    resolved_out = Path(out_path) if out_path else mesh_path.with_name(f"{mesh_path.stem}_components.html")
    resolved_out.write_text(html, encoding="utf-8")
    print(f"[picker] 저장: {resolved_out}")
    if auto_open:
        webbrowser.open(resolved_out.resolve().as_uri())
    return resolved_out


def open_picker(
    mesh_path: str | Path,
    *,
    out_path: str | Path | None = None,
    max_display_points: int = DEFAULT_MAX_DISPLAY_POINTS,
    n_regions: int = 5,
    auto_open: bool = True,
) -> Path:
    """`build_picker_html()`을 파일로 저장하고(기본) 브라우저로 연다."""
    mesh_path = Path(mesh_path)
    html = build_picker_html(mesh_path, max_display_points=max_display_points, n_regions=n_regions)
    resolved_out = Path(out_path) if out_path else mesh_path.with_name(f"{mesh_path.stem}_picker.html")
    resolved_out.write_text(html, encoding="utf-8")
    print(f"[picker] 저장: {resolved_out}")
    if auto_open:
        webbrowser.open(resolved_out.resolve().as_uri())
    return resolved_out
