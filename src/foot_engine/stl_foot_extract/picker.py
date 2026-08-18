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

from .locate import find_dense_regions

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
