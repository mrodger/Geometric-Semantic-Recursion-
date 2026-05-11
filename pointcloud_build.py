#!/usr/bin/env python3
"""
pointcloud_build.py — Semantic 3D point cloud from conversation data.

Extracts user/assistant message pairs from a SQLite database, embeds them
using OpenAI text-embedding-3-small, projects to 3D via UMAP, and writes a
self-contained Three.js HTML viewer where points are coloured by context label
(persona, category, or any string tag on each message).

Usage:
    python pointcloud_build.py
    python pointcloud_build.py --db path/to/your.db --limit 2000 --out viewer.html

Environment:
    OPENAI_API_KEY  — required (or place in ~/.secrets.env as KEY=VALUE)

Database schema expected:
    conversations(id, persona, title)
    messages(id, conversation_id, role, content, created_at)

    role values: 'user' | 'assistant'
    persona: any string label — used for colouring in the viewer
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# ── Load secrets ──────────────────────────────────────────────────────────────
secrets_path = Path.home() / ".secrets.env"
if secrets_path.exists():
    for line in secrets_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

BATCH_SIZE = 100
EMBED_MODEL = "text-embedding-3-small"


# ── Extract pairs ─────────────────────────────────────────────────────────────
def extract_pairs(db_path: Path, limit: int) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT
            m1.id        AS user_msg_id,
            m1.content   AS user_msg,
            m2.content   AS assistant_msg,
            c.persona,
            c.title,
            m1.created_at
        FROM messages m1
        JOIN conversations c ON c.id = m1.conversation_id
        JOIN messages m2 ON (
            m2.conversation_id = m1.conversation_id
            AND m2.role = 'assistant'
            AND m2.created_at = (
                SELECT MIN(created_at) FROM messages
                WHERE conversation_id = m1.conversation_id
                  AND role = 'assistant'
                  AND created_at > m1.created_at
            )
        )
        WHERE m1.role = 'user'
          AND length(m1.content) > 10
        ORDER BY m1.created_at ASC
        LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ── Embed ─────────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> "np.ndarray":
    import numpy as np
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"  Embedding batch {i // BATCH_SIZE + 1}/{-(-len(texts) // BATCH_SIZE)} ({len(batch)} items)...")
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([e.embedding for e in resp.data])
    return np.array(vectors, dtype=np.float32)


# ── Build HTML viewer ─────────────────────────────────────────────────────────
def build_html(points: list[dict], out_path: Path):
    data_json = json.dumps(points)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Semantic Point Cloud</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f1923; overflow: hidden; font-family: 'JetBrains Mono', monospace; }}
  #tooltip {{
    position: fixed;
    background: rgba(13,22,31,0.95);
    border: 1px solid #2b567e;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    max-width: 340px;
    pointer-events: none;
    display: none;
    line-height: 1.6;
    color: #c8d8e8;
    z-index: 100;
  }}
  #tooltip .label  {{ color: #c89632; font-weight: 600; margin-bottom: 4px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
  #tooltip .title  {{ color: #5a8fb0; font-size: 10px; margin-bottom: 4px; }}
  #tooltip .msg    {{ color: #8e9fad; font-size: 11px; }}
  #info {{
    position: fixed;
    bottom: 16px;
    left: 16px;
    font-size: 11px;
    color: #3a5268;
    font-family: monospace;
  }}
</style>
</head>
<body>
<div id="tooltip">
  <div class="label"></div>
  <div class="title"></div>
  <div class="msg"></div>
</div>
<div id="info"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const POINTS = {data_json};

// ── Scene ──────────────────────────────────────────────────────────────────
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 1000);
camera.position.set(0, 0, 30);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x0f1923);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.4;

// ── Normalize coords to [-10, 10] ──────────────────────────────────────────
const xs = POINTS.map(p => p.x), ys = POINTS.map(p => p.y), zs = POINTS.map(p => p.z || 0);
const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
const cz = (Math.min(...zs) + Math.max(...zs)) / 2;
const spread = Math.max(
  Math.max(...xs) - Math.min(...xs),
  Math.max(...ys) - Math.min(...ys),
  Math.max(...zs) - Math.min(...zs)
) || 1;
const scale = 20 / spread;

// ── Label → colour palette ──────────────────────────────────────────────────
// Default palette for known Datum personas. Unknown labels get auto-assigned.
const PRESET_COLOURS = {{
  developer:    0x4a9eff,
  researcher:   0x4ecda4,
  designer:     0xe87fac,
  manager:      0xc89632,
  security:     0xff6b4a,
  'gis-analyst': 0x7ec8e3,
  chat:         0xa78bfa,
}};
const AUTO_PALETTE = [
  0x64a0dc, 0xf0a050, 0x6bcb77, 0xf05050, 0xc084fc,
  0xfbbf24, 0x34d399, 0xf472b6, 0x60a5fa, 0xa3e635,
];
const labelColour = {{}};
let autoPaletteIdx = 0;
function getColour(label) {{
  if (labelColour[label]) return labelColour[label];
  const hex = PRESET_COLOURS[label] ||
    AUTO_PALETTE[autoPaletteIdx++ % AUTO_PALETTE.length];
  labelColour[label] = hex;
  return hex;
}}

// ── Build instanced mesh ───────────────────────────────────────────────────
const geo = new THREE.SphereGeometry(0.13, 6, 6);
const mat = new THREE.MeshBasicMaterial({{ vertexColors: true }});
const mesh = new THREE.InstancedMesh(geo, mat, POINTS.length);
const dummy = new THREE.Object3D();
const colorBuf = new Float32Array(POINTS.length * 3);
const col = new THREE.Color();

POINTS.forEach((p, i) => {{
  dummy.position.set(
    (p.x - cx) * scale,
    (p.y - cy) * scale,
    ((p.z || 0) - cz) * scale
  );
  dummy.updateMatrix();
  mesh.setMatrixAt(i, dummy.matrix);
  col.setHex(getColour(p.label || p.persona || 'unknown'));
  colorBuf[i * 3]     = col.r;
  colorBuf[i * 3 + 1] = col.g;
  colorBuf[i * 3 + 2] = col.b;
}});
mesh.instanceColor = new THREE.InstancedBufferAttribute(colorBuf, 3);
mesh.instanceColor.needsUpdate = true;
scene.add(mesh);

// ── Hover ──────────────────────────────────────────────────────────────────
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip  = document.getElementById('tooltip');
let hoveredIdx = -1;
let dragging   = false;

renderer.domElement.addEventListener('mousemove', e => {{
  mouse.x =  (e.clientX / innerWidth)  * 2 - 1;
  mouse.y = -(e.clientY / innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(mesh);
  if (hits.length > 0) {{
    const idx = hits[0].instanceId;
    if (idx !== hoveredIdx) {{
      hoveredIdx = idx;
      const p = POINTS[idx];
      tooltip.querySelector('.label').textContent = p.label || p.persona || '';
      tooltip.querySelector('.title').textContent = p.title || '';
      tooltip.querySelector('.msg').textContent   = p.user_msg || p.text || '';
    }}
    const tx = Math.min(e.clientX + 16, innerWidth  - 360);
    const ty = Math.min(e.clientY - 8,  innerHeight - 130);
    tooltip.style.left    = tx + 'px';
    tooltip.style.top     = ty + 'px';
    tooltip.style.display = 'block';
    if (!dragging) controls.autoRotate = false;
  }} else {{
    hoveredIdx = -1;
    tooltip.style.display = 'none';
    if (!dragging) controls.autoRotate = true;
  }}
}});

renderer.domElement.addEventListener('mousedown', () => {{ dragging = true;  controls.autoRotate = false; }});
renderer.domElement.addEventListener('mouseup',   () => {{ dragging = false; }});
renderer.domElement.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

// ── Legend ─────────────────────────────────────────────────────────────────
const legendEl = document.createElement('div');
legendEl.style.cssText = 'position:fixed;top:16px;left:16px;font-size:11px;font-family:monospace;line-height:2;';
Object.entries(labelColour).forEach(([name, hex]) => {{
  const swatch = '#' + hex.toString(16).padStart(6, '0');
  legendEl.innerHTML += `<span style="color:${{swatch}}">&#9632;</span> <span style="color:#8e9fad">${{name}}</span>&nbsp;&nbsp;`;
}});
document.body.appendChild(legendEl);

document.getElementById('info').textContent =
  `${{POINTS.length}} points · drag to rotate · scroll to zoom · hover for detail`;

// ── Resize + render ────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});

(function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}})();
</script>
</body>
</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import numpy as np
    import umap as umap_lib

    parser = argparse.ArgumentParser(description="Build a semantic 3D point cloud from conversation data.")
    parser.add_argument("--db",    type=Path, default=Path("data/datum-ui.db"), help="Path to SQLite database")
    parser.add_argument("--limit", type=int,  default=1000, help="Max message pairs to embed")
    parser.add_argument("--out",   type=Path, default=Path("pointcloud.html"),  help="Output HTML path")
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings_cache.npy"), help="Embedding cache path")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing cache")
    args = parser.parse_args()

    if not args.db.exists():
        sys.exit(f"Database not found: {args.db}")

    print(f"Extracting up to {args.limit} pairs from {args.db}...")
    pairs = extract_pairs(args.db, args.limit)
    print(f"  Got {len(pairs)} pairs")

    if args.cache.exists() and not args.no_cache:
        print(f"Loading cached embeddings from {args.cache}...")
        vectors = np.load(args.cache)
        if len(vectors) != len(pairs):
            print("  Cache size mismatch — re-embedding...")
            vectors = embed_texts([f"Q: {p['user_msg'][:400]}\nA: {p['assistant_msg'][:400]}" for p in pairs])
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.cache, vectors)
    else:
        print("Embedding pairs...")
        texts = [f"Q: {p['user_msg'][:400]}\nA: {p['assistant_msg'][:400]}" for p in pairs]
        vectors = embed_texts(texts)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.cache, vectors)
        print(f"  Saved to {args.cache}")

    print("Running UMAP projection (3D)...")
    reducer = umap_lib.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(vectors)
    print("  Done")

    points = []
    for i, p in enumerate(pairs):
        points.append({
            "x":        float(coords[i, 0]),
            "y":        float(coords[i, 1]),
            "z":        float(coords[i, 2]),
            "label":    p["persona"],       # colour key — change to any string field
            "persona":  p["persona"],
            "title":    p.get("title", ""),
            "user_msg": p["user_msg"][:200],
            "created_at": p.get("created_at", ""),
        })

    build_html(points, args.out)
    print(f"Viewer written to {args.out}")
    print(f"Open in browser: file://{args.out.resolve()}")


if __name__ == "__main__":
    main()
