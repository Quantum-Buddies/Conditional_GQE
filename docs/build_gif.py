#!/usr/bin/env python3
"""Render the GIC 2026 journey loop to an animated GIF.

Approach: since headless-browser/ffmpeg tooling isn't available on this
machine, we synthesize N static SVG frames (one per animation step),
rasterize each with `rsvg-convert`, then assemble into a looping GIF with
Pillow. No SMIL/JS execution required at render time.
"""
import subprocess
import shutil
import os
from pathlib import Path
from PIL import Image

OUT_DIR = Path("/scratch/kcwp264/Conditional-GQE_materials/docs")
FRAME_DIR = OUT_DIR / "_gif_frames"
FRAME_DIR.mkdir(exist_ok=True)

W, H = 800, 450
FRAMES_PER_SCENE = 7
FRAME_MS = 260  # duration per frame in the final GIF

STYLE = """
    .canvas { fill: #f0eee6; }
    .ink { stroke: #191919; fill: none; stroke-width: 1.4; stroke-linecap: round; stroke-linejoin: round; }
    .ink-fill { fill: #191919; }
    .card { fill: #ffffff; stroke: #191919; stroke-width: 1; }
    .muted { fill: #737373; }
    .clay { stroke: #cc785c; fill: none; stroke-width: 1.6; }
    .clay-fill { fill: #cc785c; }
    .label { font-family: Georgia, 'Tiempos', serif; font-size: 20px; fill: #191919; }
    .cap { font-family: 'DejaVu Sans', Inter, system-ui, sans-serif; font-size: 13px; fill: #737373; }
    .tok { font-family: 'DejaVu Sans Mono', 'JetBrains Mono', monospace; font-size: 14px; fill: #191919; }
"""


def svg_header():
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n<style>{STYLE}</style>\n<rect class="canvas" width="{W}" height="{H}"/>\n'


HEX_R = 50


def hex_points(cx, cy, r=HEX_R):
    import math
    pts = []
    for k in range(6):
        a = math.pi / 2 + k * math.pi / 3
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    return pts


def scene1(frac):
    """Molecule -> graph: benzene ring with atom nodes + a GNN message
    pulse traveling around the ring (edge-aware message passing)."""
    op = min(1.0, 0.4 + frac * 3)
    cx, cy = 320, 190
    pts = hex_points(cx, cy)
    edges = ""
    for i in range(6):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 6]
        edges += f'<line class="ink" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>\n'
    nodes = ""
    for (x, y) in pts:
        nodes += f'<circle class="card" cx="{x:.1f}" cy="{y:.1f}" r="6"/>\n'
    pulse_idx = frac * 6
    i0 = int(pulse_idx) % 6
    i1 = (i0 + 1) % 6
    t = pulse_idx - int(pulse_idx)
    px = pts[i0][0] + (pts[i1][0] - pts[i0][0]) * t
    py = pts[i0][1] + (pts[i1][1] - pts[i0][1]) * t
    return f'''
<g opacity="{op:.2f}">
  <text x="60" y="60" class="label">A molecule becomes a graph.</text>
  <g>
    {edges}
    {nodes}
    <circle class="clay-fill" cx="{px:.1f}" cy="{py:.1f}" r="5"/>
  </g>
  <text x="60" y="405" class="cap">Chemistry GNN — edge-aware message passing over atoms/bonds</text>
  <text x="60" y="425" class="cap">Nodes: Z, hybridization, charge  ·  Edges: bond order, R_ij</text>
</g>'''


def scene2(frac):
    """Encoder-decoder cross-attention: Hamiltonian encoder feeds a
    decoder that autoregressively writes Pauli-word tokens."""
    tokens = ["Y", "Z", "X", "I", "Y"]
    active = min(4, int(round(frac * 4)))
    boxes = ""
    for i, t in enumerate(tokens):
        cls = "clay" if i == active else "card"
        x = 430 + i * 58
        boxes += f'<rect class="{cls}" x="{x}" y="150" width="46" height="46"/>\n'
        boxes += f'<text x="{x+23}" y="180" text-anchor="middle" class="tok">{t}</text>\n'
    n_bars = 5
    bars = ""
    for i in range(n_bars):
        h = 14 + (i % 3) * 10
        bars += f'<rect class="card" x="{60+i*20}" y="{200-h}" width="12" height="{h}"/>\n'
    return f'''
<g>
  <text x="60" y="60" class="label">Encoder + decoder write the sequence.</text>
  <rect class="card" x="40" y="110" width="180" height="130"/>
  <text x="130" y="135" text-anchor="middle" class="cap">Hamiltonian encoder</text>
  {bars}
  <path class="ink" d="M 220 175 C 320 175 340 175 420 173" stroke-dasharray="3,3"/>
  <text x="320" y="165" text-anchor="middle" class="cap">cross-attention</text>
  {boxes}
  <text x="60" y="405" class="cap">Decoder autoregressively emits Pauli words from the UCCSD pool</text>
</g>'''


def scene3(frac):
    """Diagonal-collapse mitigation: Z-only sequence (struck out) vs
    an entangled UCCSD sequence that the policy is pushed toward."""
    n_show = min(4, int(round(frac * 4)))
    row2 = ""
    for i, t in enumerate(["X", "Y", "Y", "X"]):
        visible = i <= n_show
        cls = "clay" if (visible and i == n_show) else "card"
        op = 1.0 if visible else 0.15
        x = 260 + i * 58
        row2 += f'<g opacity="{op}"><rect class="{cls}" x="{x}" y="0" width="46" height="46"/>' \
                f'<text x="{x+23}" y="30" text-anchor="middle" class="tok">{t}</text></g>\n'
    check_op = 1.0 if n_show >= 3 else 0.0
    return f'''
<g>
  <text x="60" y="60" class="label">Collapse, avoided.</text>

  <text x="60" y="120" class="cap">Z-only — commutes, zero gradient, traps at Hartree-Fock</text>
  <g transform="translate(60,140)">
    <rect class="card" x="0"   y="0" width="46" height="46"/>
    <rect class="card" x="58"  y="0" width="46" height="46"/>
    <rect class="card" x="116" y="0" width="46" height="46"/>
    <rect class="card" x="174" y="0" width="46" height="46"/>
    <text x="23"  y="30" text-anchor="middle" class="tok muted">Z</text>
    <text x="81"  y="30" text-anchor="middle" class="tok muted">Z</text>
    <text x="139" y="30" text-anchor="middle" class="tok muted">Z</text>
    <text x="197" y="30" text-anchor="middle" class="tok muted">Z</text>
    <line class="clay" x1="-6" y1="-6" x2="226" y2="52"/>
  </g>

  <text x="60" y="270" class="cap">UCCSD pool — every operator carries X/Y, entangles by construction</text>
  <g transform="translate(60,290)">
    {row2}
    <text x="440" y="30" class="tok clay-fill" opacity="{check_op:.2f}">&#10003;</text>
  </g>
  <text x="60" y="405" class="cap">Reward penalty on commutator fraction reinforces the entangled path</text>
</g>'''


def scene4(frac):
    """MAP-Elites quality-diversity archive filling with elite circuits,
    axes = entanglement density x circuit depth."""
    grid_n = 6
    fill_order = [0, 7, 14, 21, 28, 35, 3, 10, 17, 24, 31, 5, 12, 19, 26, 33,
                  1, 8, 15, 22, 29, 4, 11, 18, 25, 32, 2, 9, 16, 23, 30, 6, 13, 20, 27, 34]
    n_filled = int(round(frac * len(fill_order)))
    filled = set(fill_order[:n_filled])
    cell = 34
    cells = ""
    for r in range(grid_n):
        for c in range(grid_n):
            idx = r * grid_n + c
            cls = "clay-fill" if idx in filled else "card"
            x = c * cell
            y = r * cell
            if cls == "clay-fill":
                cells += f'<rect x="{x}" y="{y}" width="{cell-4}" height="{cell-4}" fill="#cc785c"/>\n'
            else:
                cells += f'<rect class="card" x="{x}" y="{y}" width="{cell-4}" height="{cell-4}"/>\n'
    coverage = int(100 * len(filled) / (grid_n * grid_n))
    return f'''
<g>
  <text x="60" y="60" class="label">Quality-diversity, not mode collapse.</text>
  <g transform="translate(400,110)">
    {cells}
  </g>
  <text x="400" y="335" class="cap">entanglement density &#8594;</text>
  <text x="380" y="325" class="cap" transform="rotate(-90 380 325)">depth &#8594;</text>
  <text x="60" y="150" class="cap">MAP-Elites archive, 6&#215;6 (paper uses 10&#215;10)</text>
  <text x="60" y="175" class="cap">Coverage: {coverage}%</text>
  <text x="60" y="405" class="cap">DAPO + novelty bonus fills unvisited topological niches</text>
</g>'''


def scene5(frac):
    """Multi-vendor QPU deployment: circuit dispatched to Rigetti / IonQ
    / IQM via qBraid."""
    vendors = [("Rigetti", 120), ("IonQ", 320), ("IQM", 520)]
    active = min(2, int(frac * 3))
    chips = ""
    for i, (name, x) in enumerate(vendors):
        cls = "clay" if i == active else "card"
        chips += f'<rect class="{cls}" x="{x}" y="230" width="120" height="80"/>\n'
        for gx in range(3):
            for gy in range(2):
                chips += f'<circle class="ink" cx="{x+25+gx*35}" cy="{250+gy*35}" r="3"/>\n'
        chips += f'<text x="{x+60}" y="325" text-anchor="middle" class="cap">{name}</text>\n'
    px = 60 + frac * (vendors[active][1] - 60)
    return f'''
<g>
  <text x="60" y="60" class="label">One workspace. Any hardware.</text>
  <rect class="card" x="40" y="110" width="120" height="70"/>
  <text x="100" y="150" text-anchor="middle" class="cap">Circuit</text>
  <path class="ink" d="M 100 180 C 100 220 {px:.0f} 220 {px:.0f} 228" stroke-dasharray="3,3"/>
  <circle class="clay-fill" cx="{px:.0f}" cy="225" r="4"/>
  {chips}
  <text x="60" y="405" class="cap">qBraid dispatch — QWC-grouped OpenQASM, single API key</text>
</g>'''


def scene6(frac):
    dash = 1000 * frac
    gap = 1000 - dash
    dot_op = 1.0 if frac >= 0.95 else 0.0
    return f'''
<g>
  <text x="60" y="60" class="label">Energy finds its floor.</text>
  <g transform="translate(100,320)">
    <line class="ink" x1="0" y1="0" x2="600" y2="0"/>
    <line class="ink" x1="0" y1="0" x2="0" y2="-220"/>
    <text x="-10" y="-225" class="cap" text-anchor="end">&#x27E8;H&#x27E9;</text>
    <text x="600" y="18" class="cap" text-anchor="end">RL steps</text>
    <path class="ink" pathLength="1000" stroke-dasharray="{dash:.1f},{gap:.1f}"
          d="M0,-30 C 80,-60 160,-150 260,-185 C 360,-205 460,-210 600,-210"/>
    <circle class="clay-fill" cx="600" cy="-210" r="5" opacity="{dot_op}"/>
  </g>
  <text x="60" y="405" class="cap">Ground-state energy converges &#8804; 1.6 mHa — chemical accuracy</text>
</g>'''


SCENES = [scene1, scene2, scene3, scene4, scene5, scene6]


def build_frames():
    idx = 0
    paths = []
    for scene_fn in SCENES:
        for step in range(FRAMES_PER_SCENE):
            frac = step / (FRAMES_PER_SCENE - 1)
            svg = svg_header() + scene_fn(frac) + "\n</svg>"
            svg_path = FRAME_DIR / f"frame_{idx:03d}.svg"
            svg_path.write_text(svg)
            png_path = FRAME_DIR / f"frame_{idx:03d}.png"
            subprocess.run(
                ["rsvg-convert", "-w", str(W), "-h", str(H), "-o", str(png_path), str(svg_path)],
                check=True,
            )
            paths.append(png_path)
            idx += 1
    return paths


def assemble_gif(png_paths, out_path):
    frames = [Image.open(p).convert("RGB") for p in png_paths]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=False,
    )


if __name__ == "__main__":
    pngs = build_frames()
    out_gif = OUT_DIR / "gic2026_journey_loop.gif"
    assemble_gif(pngs, out_gif)
    shutil.rmtree(FRAME_DIR)
    size_kb = out_gif.stat().st_size / 1024
    print(f"Wrote {out_gif} ({size_kb:.1f} KB, {len(pngs)} frames)")
