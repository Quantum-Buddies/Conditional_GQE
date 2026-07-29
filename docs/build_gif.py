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
PIPELINE_STAGES = ["Chemistry", "AI Synthesis", "Energy Eval", "RL Update", "Deploy"]


def pipeline_bar(active):
    """5-dot progress bar at top showing current pipeline stage."""
    dots = ""
    for i, label in enumerate(PIPELINE_STAGES):
        cx = 80 + i * 145
        cy = 28
        if i == active:
            dots += f'<circle cx="{cx}" cy="{cy}" r="5" class="clay-fill"/>\n'
            dots += f'<text x="{cx}" y="{cy+18}" text-anchor="middle" class="cap" style="font-weight:bold">{label}</text>\n'
        elif i < active:
            dots += f'<circle cx="{cx}" cy="{cy}" r="4" fill="#cc785c" opacity="0.4"/>\n'
            dots += f'<text x="{cx}" y="{cy+18}" text-anchor="middle" class="cap" opacity="0.4">{label}</text>\n'
        else:
            dots += f'<circle cx="{cx}" cy="{cy}" r="4" fill="#ffffff" stroke="#737373" stroke-width="1"/>\n'
            dots += f'<text x="{cx}" y="{cy+18}" text-anchor="middle" class="cap" opacity="0.3">{label}</text>\n'
        if i < len(PIPELINE_STAGES) - 1:
            nx = 80 + (i + 1) * 145
            col = "#cc785c" if i < active else "#737373"
            op = "0.6" if i < active else "0.2"
            dots += f'<line x1="{cx+8}" y1="{cy}" x2="{nx-8}" y2="{cy}" stroke="{col}" stroke-width="1" opacity="{op}"/>\n'
    return f'<g>{dots}</g>'


def hex_points(cx, cy, r=HEX_R):
    import math
    pts = []
    for k in range(6):
        a = math.pi / 2 + k * math.pi / 3
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    return pts


def scene1(frac):
    """Chemistry input: molecule → atom graph + Hamiltonian (two parallel outputs)."""
    op = min(1.0, 0.4 + frac * 3)
    cx, cy = 200, 200
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
    # Hamiltonian panel on the right
    ham_op = min(1.0, max(0.0, (frac - 0.3) * 2))
    return f'''
{pipeline_bar(0)}
<g opacity="{op:.2f}">
  <text x="60" y="80" class="label">Every molecule becomes a graph.</text>
  <g>
    {edges}
    {nodes}
    <circle class="clay-fill" cx="{px:.1f}" cy="{py:.1f}" r="5"/>
  </g>
  <path class="ink" d="M 270 200 L 400 200" stroke-dasharray="3,3" opacity="{ham_op:.2f}"/>
  <g opacity="{ham_op:.2f}">
    <rect class="card" x="400" y="120" width="180" height="160"/>
    <text x="490" y="145" text-anchor="middle" class="cap" style="font-weight:bold">Hamiltonian</text>
    <text x="490" y="165" text-anchor="middle" class="cap">H = &#931; h&#8202;&#8217; P&#8202;&#8217;</text>
    <text x="490" y="190" text-anchor="middle" class="tok" style="font-size:11px">YZXI  +0.12</text>
    <text x="490" y="208" text-anchor="middle" class="tok" style="font-size:11px">XZYI  -0.08</text>
    <text x="490" y="226" text-anchor="middle" class="tok" style="font-size:11px">ZZII  +0.34</text>
    <text x="490" y="244" text-anchor="middle" class="tok" style="font-size:11px">IYZX  +0.05</text>
    <text x="490" y="262" text-anchor="middle" class="cap">Jordan-Wigner mapped</text>
  </g>
  <text x="60" y="425" class="cap">Chemistry GNN encodes atoms/bonds  ·  Hamiltonian encoder reads Pauli terms</text>
</g>'''


def scene2(frac):
    """AI synthesis: GNN prefix + Hamiltonian cross-attention → decoder → operator tokens."""
    tokens = ["Y", "Z", "X", "I", "Y"]
    active = min(4, int(round(frac * 4)))
    boxes = ""
    for i, t in enumerate(tokens):
        cls = "clay" if i == active else "card"
        x = 430 + i * 58
        boxes += f'<rect class="{cls}" x="{x}" y="170" width="46" height="46"/>\n'
        boxes += f'<text x="{x+23}" y="200" text-anchor="middle" class="tok">{t}</text>\n'
    n_bars = 5
    bars = ""
    for i in range(n_bars):
        h = 14 + (i % 3) * 10
        bars += f'<rect class="card" x="{60+i*20}" y="{220-h}" width="12" height="{h}"/>\n'
    # GNN box
    gnn_op = min(1.0, frac * 4)
    return f'''
{pipeline_bar(1)}
<g>
  <text x="60" y="80" class="label">Conditioned generation.</text>
  <g opacity="{gnn_op:.2f}">
    <rect class="card" x="40" y="100" width="120" height="60"/>
    <text x="100" y="125" text-anchor="middle" class="cap" style="font-weight:bold">GNN</text>
    <text x="100" y="145" text-anchor="middle" class="cap">prefix tokens</text>
  </g>
  <rect class="card" x="200" y="100" width="160" height="140"/>
  <text x="280" y="125" text-anchor="middle" class="cap" style="font-weight:bold">Hamiltonian enc.</text>
  {bars}
  <text x="280" y="250" text-anchor="middle" class="cap">Pauli terms</text>
  <path class="ink" d="M 160 130 L 200 130" stroke-dasharray="3,3"/>
  <text x="180" y="122" text-anchor="middle" class="cap" style="font-size:10px">prefix</text>
  <path class="ink" d="M 360 170 C 390 170 400 193 420 193" stroke-dasharray="3,3"/>
  <text x="390" y="162" text-anchor="middle" class="cap" style="font-size:10px">cross-attn</text>
  <rect class="card" x="420" y="130" width="100" height="30"/>
  <text x="470" y="150" text-anchor="middle" class="cap" style="font-weight:bold">Decoder</text>
  {boxes}
  <text x="60" y="425" class="cap">GNN soft-prefix + Hamiltonian cross-attention  →  autoregressive Pauli tokens</text>
</g>'''


def scene3(frac):
    """Two-stage optimization: discrete topology → L-BFGS-B continuous angles → CUDA-Q energy."""
    stage = int(frac * 3)
    # Stage 0: operator sequence
    seq_op = 1.0
    # Stage 1: L-BFGS-B angles
    angle_op = min(1.0, max(0.0, (frac - 0.15) * 3))
    # Stage 2: CUDA-Q energy
    energy_op = min(1.0, max(0.0, (frac - 0.5) * 3))
    toks = ["Y", "Z", "X", "I"]
    seq_svg = ""
    for i, t in enumerate(toks):
        x = 60 + i * 52
        cls = "clay" if stage == 0 and i == min(stage, 3) else "card"
        seq_svg += f'<rect class="{cls}" x="{x}" y="120" width="44" height="44"/>\n'
        seq_svg += f'<text x="{x+22}" y="148" text-anchor="middle" class="tok">{t}</text>\n'
    # Angle dials
    angles_svg = ""
    for i in range(4):
        x = 60 + i * 52
        ay = 220 + int(20 * (i % 3 - 1) * angle_op)
        angles_svg += f'<line class="clay" x1="{x+22}" y1="210" x2="{x+22}" y2="{ay}" stroke-width="2" opacity="{angle_op:.2f}"/>\n'
        angles_svg += f'<text x="{x+22}" y="235" text-anchor="middle" class="tok" style="font-size:11px" opacity="{angle_op:.2f}">&#952;={0.1+i*0.3:.1f}</text>\n'
    return f'''
{pipeline_bar(2)}
<g>
  <text x="60" y="80" class="label">Discrete topology, then continuous angles.</text>
  <text x="60" y="108" class="cap">Stage 1: Transformer emits operator sequence</text>
  {seq_svg}
  <path class="ink" d="M 270 142 L 330 142" stroke-dasharray="3,3" opacity="{angle_op:.2f}"/>
  <text x="300" y="135" text-anchor="middle" class="cap" opacity="{angle_op:.2f}">L-BFGS-B</text>
  <text x="350" y="108" class="cap" opacity="{angle_op:.2f}">Stage 2: Classical angle optimization</text>
  {angles_svg}
  <path class="ink" d="M 270 230 L 330 230" stroke-dasharray="3,3" opacity="{energy_op:.2f}"/>
  <text x="300" y="223" text-anchor="middle" class="cap" opacity="{energy_op:.2f}">CUDA-Q</text>
  <g opacity="{energy_op:.2f}" transform="translate(350,250)">
    <rect class="card" x="0" y="0" width="160" height="60"/>
    <text x="80" y="25" text-anchor="middle" class="cap" style="font-weight:bold">Energy E = &#10216;&#968;&#8320;|U&#8224;HU|&#968;&#8320;&#10217;</text>
    <text x="80" y="48" text-anchor="middle" class="tok" style="font-size:12px">-1.137 Ha</text>
  </g>
  <text x="60" y="425" class="cap">Two-stage: discrete structure (classical)  →  continuous angles (classical)  →  quantum energy</text>
</g>'''


def scene4(frac):
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
{pipeline_bar(1)}
<g>
  <text x="60" y="80" class="label">Collapse, avoided.</text>

  <text x="60" y="135" class="cap">Z-only — commutes, zero gradient, traps at Hartree-Fock</text>
  <g transform="translate(60,155)">
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

  <text x="60" y="285" class="cap">UCCSD pool — every operator carries X/Y, entangles by construction</text>
  <g transform="translate(60,305)">
    {row2}
    <text x="440" y="30" class="tok clay-fill" opacity="{check_op:.2f}">&#10003;</text>
  </g>
  <text x="60" y="425" class="cap">UCCSD pool + commutator penalty + force_entanglement  →  zero Z-only by construction</text>
</g>'''


def scene5(frac):
    """RL feedback loop: energy → reward → DAPO + MAP-Elites → policy update → back to decoder."""
    loop_op = min(1.0, frac * 2)
    # MAP-Elites grid filling
    grid_n = 5
    fill_order = [0, 6, 12, 18, 24, 3, 9, 15, 21, 5, 11, 17, 23, 1, 7, 13, 19, 4, 10, 16, 22, 2, 8, 14, 20]
    n_filled = int(round(frac * len(fill_order)))
    filled = set(fill_order[:n_filled])
    cell = 28
    cells = ""
    for r in range(grid_n):
        for c in range(grid_n):
            idx = r * grid_n + c
            x = c * cell
            y = r * cell
            if idx in filled:
                cells += f'<rect x="{x}" y="{y}" width="{cell-3}" height="{cell-3}" fill="#cc785c" opacity="0.7"/>\n'
            else:
                cells += f'<rect class="card" x="{x}" y="{y}" width="{cell-3}" height="{cell-3}"/>\n'
    coverage = int(100 * len(filled) / (grid_n * grid_n))
    return f'''
{pipeline_bar(3)}
<g>
  <text x="60" y="80" class="label">Quality-diversity RL closes the loop.</text>
  <g opacity="{loop_op:.2f}">
    <rect class="card" x="40" y="110" width="100" height="50"/>
    <text x="90" y="130" text-anchor="middle" class="cap" style="font-weight:bold">Energy</text>
    <text x="90" y="148" text-anchor="middle" class="tok" style="font-size:11px">-1.137</text>
    <path class="ink" d="M 140 135 L 200 135" stroke-dasharray="3,3"/>
    <rect class="card" x="200" y="110" width="100" height="50"/>
    <text x="250" y="130" text-anchor="middle" class="cap" style="font-weight:bold">Reward</text>
    <text x="250" y="148" text-anchor="middle" class="cap">w&#8321;(-E) + w&#8322;(ent)</text>
    <path class="ink" d="M 300 135 L 360 135" stroke-dasharray="3,3"/>
    <g transform="translate(360,100)">
      {cells}
    </g>
    <text x="360" y="250" class="cap">MAP-Elites  coverage: {coverage}%</text>
    <path class="ink" d="M 430 135 C 430 80 200 80 200 110" stroke-dasharray="3,3" opacity="{loop_op:.2f}"/>
    <text x="315" y="75" text-anchor="middle" class="cap" opacity="{loop_op:.2f}">DAPO policy update</text>
  </g>
  <text x="60" y="425" class="cap">Energy &#8594; reward &#8594; DAPO + MAP-Elites &#8594; gradient back to decoder  (feedback loop)</text>
</g>'''


def scene6(frac):
    """Energy convergence to chemical accuracy + multi-vendor QPU deployment."""
    dash = 1000 * frac
    gap = 1000 - dash
    dot_op = 1.0 if frac >= 0.95 else 0.0
    # QPU badges
    qpu_op = min(1.0, max(0.0, (frac - 0.4) * 2))
    vendors = ["Rigetti", "IonQ", "IQM"]
    qpu_svg = ""
    for i, name in enumerate(vendors):
        x = 380 + i * 120
        qpu_svg += f'<rect class="card" x="{x}" y="280" width="100" height="50" opacity="{qpu_op:.2f}"/>\n'
        qpu_svg += f'<text x="{x+50}" y="300" text-anchor="middle" class="cap" opacity="{qpu_op:.2f}">{name}</text>\n'
        qpu_svg += f'<text x="{x+50}" y="318" text-anchor="middle" class="cap" style="font-size:10px" opacity="{qpu_op:.2f}">qBraid</text>\n'
    return f'''
{pipeline_bar(4)}
<g>
  <text x="60" y="80" class="label">Energy finds its floor.</text>
  <g transform="translate(80,260)">
    <line class="ink" x1="0" y1="0" x2="280" y2="0"/>
    <line class="ink" x1="0" y1="0" x2="0" y2="-140"/>
    <text x="-10" y="-145" class="cap" text-anchor="end">&#x27E8;H&#x27E9;</text>
    <text x="280" y="18" class="cap" text-anchor="end">RL steps</text>
    <path class="ink" pathLength="1000" stroke-dasharray="{dash:.1f},{gap:.1f}"
          d="M0,-20 C 40,-40 80,-90 140,-115 C 180,-128 220,-132 280,-132"/>
    <circle class="clay-fill" cx="280" cy="-132" r="5" opacity="{dot_op}"/>
    <line class="clay" x1="0" y1="-132" x2="280" y2="-132" stroke-dasharray="2,4" opacity="0.5"/>
    <text x="290" y="-130" class="cap" style="font-size:10px">E_FCI</text>
  </g>
  {qpu_svg}
  <text x="60" y="425" class="cap">&#8804; 1.6 mHa chemical accuracy  ·  QPU validation via qBraid (Rigetti, IonQ, IQM)</text>
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
