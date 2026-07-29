#!/usr/bin/env python3
"""Render the GIC 2026 journey loop to an animated GIF.

Approach: since headless-browser/ffmpeg tooling isn't available on this
machine, we synthesize N static SVG frames (one per animation step),
rasterize each with `rsvg-convert`, then assemble into a looping GIF with
Pillow. No SMIL/JS execution required at render time.
"""
import math
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
    .card { fill: #ffffff; stroke: #191919; stroke-width: 1.2; }
    .clay { stroke: #cc785c; fill: none; stroke-width: 1.8; stroke-linecap: round; }
    .clay-fill { fill: #cc785c; }
    .headline { font-family: Georgia, 'DejaVu Serif', serif; font-size: 22px; fill: #191919; }
    .label { font-family: 'DejaVu Sans', Inter, system-ui, sans-serif; font-size: 12.5px; fill: #191919; }
    .foot { font-family: 'DejaVu Sans', Inter, system-ui, sans-serif; font-size: 12.5px; fill: #737373; }
    .mono { font-family: 'DejaVu Sans Mono', monospace; font-size: 13px; fill: #191919; }
"""

DEFS = ('<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6.5" markerHeight="6.5" orient="auto">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#191919"/></marker></defs>\n')


def svg_header():
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
            f'<style>{STYLE}</style>\n{DEFS}'
            f'<rect class="canvas" width="{W}" height="{H}"/>\n')


# Layout grid: 64px margins, shared baselines for every scene.
MX = 64          # left margin
HEAD_Y = 100     # headline baseline
FOOT_Y = 414     # footnote baseline

PIPELINE_STAGES = ["Chemistry", "AI Synthesis", "Energy Eval", "RL Update", "Deploy"]


def pipeline_bar(active):
    """Quiet 5-dot progress indicator centered at the top."""
    x0, gap, cy = 100, 150, 34
    s = "<g>"
    for i, name in enumerate(PIPELINE_STAGES):
        cx = x0 + i * gap
        if i < len(PIPELINE_STAGES) - 1:
            done = i < active
            col = "#cc785c" if done else "#737373"
            s += f'<line x1="{cx + 10}" y1="{cy}" x2="{cx + gap - 10}" y2="{cy}" stroke="{col}" stroke-width="1" opacity="{0.5 if done else 0.25}"/>\n'
        if i == active:
            s += f'<circle cx="{cx}" cy="{cy}" r="5" class="clay-fill"/>\n'
            s += f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" class="label" style="font-weight:bold">{name}</text>\n'
        elif i < active:
            s += f'<circle cx="{cx}" cy="{cy}" r="4" class="clay-fill" opacity="0.45"/>\n'
            s += f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" class="foot">{name}</text>\n'
        else:
            s += f'<circle cx="{cx}" cy="{cy}" r="4" fill="#f0eee6" stroke="#737373" stroke-width="1" opacity="0.8"/>\n'
            s += f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" class="foot" opacity="0.55">{name}</text>\n'
    return s + "</g>"


def headline(text):
    return f'<text x="{MX}" y="{HEAD_Y}" class="headline">{text}</text>\n'


def footnote(text):
    return f'<text x="{W // 2}" y="{FOOT_Y}" text-anchor="middle" class="foot">{text}</text>\n'


def card(x, y, w, h, title, sub=None, op=1.0):
    """White rounded card with vertically centered title (and optional subtitle)."""
    cx, cy = x + w / 2, y + h / 2
    s = f'<g opacity="{op:.2f}"><rect class="card" x="{x}" y="{y}" width="{w}" height="{h}" rx="4"/>\n'
    if sub:
        s += f'<text x="{cx:.0f}" y="{cy - 8:.0f}" text-anchor="middle" dominant-baseline="central" class="label" style="font-weight:bold">{title}</text>\n'
        s += f'<text x="{cx:.0f}" y="{cy + 10:.0f}" text-anchor="middle" dominant-baseline="central" class="foot">{sub}</text>\n'
    else:
        s += f'<text x="{cx:.0f}" y="{cy:.0f}" text-anchor="middle" dominant-baseline="central" class="label" style="font-weight:bold">{title}</text>\n'
    return s + "</g>"


def harrow(x1, x2, y, label=None, op=1.0):
    """Horizontal dashed arrow with an optional label centered above it."""
    s = f'<g opacity="{op:.2f}"><line class="ink" x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke-dasharray="4,4" marker-end="url(#arr)"/>\n'
    if label:
        s += f'<text x="{(x1 + x2) // 2}" y="{y - 10}" text-anchor="middle" class="foot">{label}</text>\n'
    return s + "</g>"


def token(x, y, letter, active=False, dim=False, size=46, op=1.0):
    """Single Pauli token box; active = clay outline, dim = muted letter."""
    stroke = "#cc785c" if active else "#191919"
    sw = 2 if active else 1.2
    fill = "#cc785c" if active else ("#737373" if dim else "#191919")
    return (f'<g opacity="{op:.2f}">'
            f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="4" fill="#ffffff" stroke="{stroke}" stroke-width="{sw}"/>'
            f'<text x="{x + size / 2:.0f}" y="{y + size / 2:.0f}" text-anchor="middle" dominant-baseline="central" class="mono" style="fill:{fill}">{letter}</text>'
            f"</g>\n")


def hex_points(cx, cy, r=58):
    pts = []
    for k in range(6):
        a = math.pi / 2 + k * math.pi / 3
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    return pts


def scene1(frac):
    """Chemistry input: molecule graph + Hamiltonian panel (two parallel views)."""
    op = min(1.0, 0.4 + frac * 3)
    ham_op = min(1.0, max(0.0, (frac - 0.3) * 2))
    cx, cy = 190, 255
    pts = hex_points(cx, cy)
    edges, nodes = "", ""
    for i in range(6):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 6]
        edges += f'<line class="ink" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>\n'
    for (x, y) in pts:
        nodes += f'<circle class="card" cx="{x:.1f}" cy="{y:.1f}" r="7"/>\n'
    p = frac * 6
    i0, t = int(p) % 6, p - int(p)
    i1 = (i0 + 1) % 6
    px = pts[i0][0] + (pts[i1][0] - pts[i0][0]) * t
    py = pts[i0][1] + (pts[i1][1] - pts[i0][1]) * t
    terms = ["YZXI  +0.12", "XZYI  -0.08", "ZZII  +0.34", "IYZX  +0.05"]
    rows = "".join(
        f'<text x="560" y="{238 + i * 20}" text-anchor="middle" class="mono" style="font-size:12px">{term}</text>\n'
        for i, term in enumerate(terms))
    return f'''
{pipeline_bar(0)}
{headline("Every molecule becomes a graph.")}
<g opacity="{op:.2f}">
  {edges}{nodes}
  <circle class="clay-fill" cx="{px:.1f}" cy="{py:.1f}" r="5"/>
</g>
{harrow(268, 420, 255, "encode", ham_op)}
<g opacity="{ham_op:.2f}">
  <rect class="card" x="440" y="170" width="240" height="170" rx="4"/>
  <text x="560" y="196" text-anchor="middle" class="label" style="font-weight:bold">Hamiltonian</text>
  <text x="560" y="216" text-anchor="middle" class="foot">H = &#931; h&#8202;<tspan baseline-shift="sub" font-size="9">l</tspan> P&#8202;<tspan baseline-shift="sub" font-size="9">l</tspan> &#183; Jordan-Wigner</text>
  {rows}
</g>
{footnote("Atom graph and Pauli Hamiltonian — two views of the same molecule")}
'''


def scene2(frac):
    """AI synthesis: GNN prefix + Hamiltonian cross-attention -> decoder -> tokens."""
    enc_op = min(1.0, frac * 3)
    active_tok = min(3, int(frac * 4))
    toks = "".join(
        token(530 + i * 54, 222, t, active=(i == active_tok), size=44)
        for i, t in enumerate(["Y", "Z", "X", "I"]))
    return f'''
{pipeline_bar(1)}
{headline("Conditioned generation.")}
{card(64, 160, 150, 64, "Chemistry GNN", "soft prefix", enc_op)}
{card(64, 264, 150, 64, "Hamiltonian enc.", "K, V memory", enc_op)}
<path class="ink" d="M 214 192 C 268 192 278 236 322 240" stroke-dasharray="4,4" marker-end="url(#arr)"/>
<text x="266" y="186" text-anchor="middle" class="foot">prefix</text>
<path class="ink" d="M 214 296 C 268 296 278 252 322 248" stroke-dasharray="4,4" marker-end="url(#arr)"/>
<text x="266" y="312" text-anchor="middle" class="foot">cross-attn</text>
{card(330, 212, 130, 64, "Decoder", "autoregressive")}
{harrow(468, 522, 244)}
{toks}
{footnote("GNN prefix + Hamiltonian cross-attention &#8594; autoregressive operator tokens")}
'''


def scene3(frac):
    """Two-stage optimization: operator sequence -> L-BFGS-B angles -> CUDA-Q energy."""
    angle_op = min(1.0, max(0.0, (frac - 0.15) * 3))
    energy_op = min(1.0, max(0.0, (frac - 0.5) * 3))
    toks = "".join(token(64 + i * 52, 228, t, size=44) for i, t in enumerate(["Y", "Z", "X", "I"]))
    return f'''
{pipeline_bar(2)}
{headline("Structure first, angles second.")}
<text x="{MX}" y="202" class="foot">operator sequence</text>
{toks}
{harrow(288, 340, 250, "L-BFGS-B", angle_op)}
{card(348, 206, 140, 88, "Angles", "&#952;&#8321;&#8230;&#952;&#8342; &#183; 3&#8211;5 iters", angle_op)}
{harrow(504, 556, 250, "CUDA-Q", energy_op)}
<g opacity="{energy_op:.2f}">
  <rect class="card" x="564" y="206" width="172" height="88" rx="4"/>
  <text x="650" y="236" text-anchor="middle" dominant-baseline="central" class="label" style="font-weight:bold">Energy</text>
  <text x="650" y="260" text-anchor="middle" dominant-baseline="central" class="mono">-1.137 Ha</text>
  <text x="650" y="281" text-anchor="middle" dominant-baseline="central" class="foot" style="font-size:11px">&#10216;&#968;&#8320;|U&#8224;HU|&#968;&#8320;&#10217;</text>
</g>
{footnote("Two-stage optimization — discrete topology, classical angles, quantum energy")}
'''


def scene4(frac):
    """Diagonal-collapse mitigation: Z-only struck out vs entangling UCCSD row."""
    n_show = min(4, int(round(frac * 4)))
    row_bad = "".join(token(64 + i * 58, 186, "Z", dim=True) for i in range(4))
    row_good = "".join(
        token(64 + i * 58, 298, t, active=(i == n_show - 1), op=1.0 if i < n_show else 0.18)
        for i, t in enumerate(["X", "Y", "Y", "X"]))
    check_op = 1.0 if n_show >= 4 else 0.0
    return f'''
{pipeline_bar(1)}
{headline("Collapse, avoided.")}
<text x="{MX}" y="170" class="foot">Z-only — commutes with H, zero gradient</text>
{row_bad}
<line class="clay" x1="56" y1="180" x2="292" y2="238"/>
<text x="{MX}" y="282" class="foot">UCCSD pool — every operator entangles</text>
{row_good}
<text x="316" y="321" dominant-baseline="central" class="mono" style="fill:#cc785c;font-size:18px" opacity="{check_op:.2f}">&#10003;</text>
{footnote("Commutator penalty + force_entanglement &#8594; zero Z-only circuits by construction")}
'''


def scene5(frac):
    """RL feedback loop: energy -> reward -> MAP-Elites -> DAPO back to decoder."""
    fill_order = [0, 6, 12, 18, 24, 3, 9, 15, 21, 5, 11, 17, 23, 1, 7, 13, 19, 4, 10, 16, 22, 2, 8, 14, 20]
    n_filled = int(round(frac * len(fill_order)))
    filled = set(fill_order[:n_filled])
    cell, gx, gy = 26, 470, 140
    cells = ""
    for r in range(5):
        for c in range(5):
            idx = r * 5 + c
            x, y = gx + c * cell, gy + r * cell
            if idx in filled:
                cells += f'<rect x="{x}" y="{y}" width="{cell - 3}" height="{cell - 3}" rx="2" class="clay-fill" opacity="0.75"/>\n'
            else:
                cells += f'<rect x="{x}" y="{y}" width="{cell - 3}" height="{cell - 3}" rx="2" class="card"/>\n'
    coverage = int(100 * len(filled) / 25)
    fb_op = min(1.0, max(0.0, (frac - 0.4) * 2.5))
    return f'''
{pipeline_bar(3)}
{headline("Quality-diversity RL closes the loop.")}
{card(64, 170, 140, 70, "Energy", "-1.137 Ha")}
{harrow(212, 246, 205)}
{card(254, 170, 150, 70, "Reward", "-E &#183; entropy &#183; novelty")}
{harrow(412, 462, 205)}
{cells}
<text x="{gx}" y="{gy - 12}" class="foot">MAP-Elites — coverage {coverage}%</text>
{card(64, 300, 140, 56, "Decoder", "&#8711;&#952; policy")}
<path class="ink" d="M {gx + 65} {gy + 5 * cell} V 328 H 212" stroke-dasharray="4,4" marker-end="url(#arr)" opacity="{fb_op:.2f}"/>
<text x="370" y="318" text-anchor="middle" class="foot" opacity="{fb_op:.2f}">DAPO update</text>
{footnote("Energy &#8594; reward &#8594; MAP-Elites archive &#8594; gradient back to decoder")}
'''


def scene6(frac):
    """Energy convergence to chemical accuracy + multi-vendor QPU deployment."""
    dash = 1000 * frac
    gap = 1000 - dash
    dot_op = 1.0 if frac >= 0.95 else 0.0
    qpu_op = min(1.0, max(0.0, (frac - 0.35) * 2.5))
    vendors = "".join(
        card(470, 154 + i * 60, 200, 46, name, op=qpu_op)
        for i, name in enumerate(["Rigetti", "IonQ", "IQM"]))
    return f'''
{pipeline_bar(4)}
{headline("Energy finds its floor.")}
<g transform="translate(110,310)">
  <line class="ink" x1="0" y1="0" x2="260" y2="0"/>
  <line class="ink" x1="0" y1="0" x2="0" y2="-160"/>
  <text x="-12" y="-166" text-anchor="end" class="foot">&#10216;H&#10217;</text>
  <text x="260" y="22" text-anchor="end" class="foot">RL steps</text>
  <line x1="0" y1="-140" x2="260" y2="-140" stroke="#cc785c" stroke-width="1" stroke-dasharray="2,4" opacity="0.6"/>
  <text x="268" y="-140" dominant-baseline="central" class="foot">E<tspan baseline-shift="sub" font-size="9">FCI</tspan></text>
  <path class="ink" pathLength="1000" stroke-dasharray="{dash:.1f},{gap:.1f}"
        d="M0,-16 C 40,-36 80,-92 140,-118 C 180,-132 220,-138 260,-139"/>
  <circle class="clay-fill" cx="260" cy="-139" r="5" opacity="{dot_op}"/>
</g>
<text x="470" y="136" class="foot" opacity="{qpu_op:.2f}">QPU validation — qBraid</text>
{vendors}
{footnote("&#8804; 1.6 mHa chemical accuracy &#183; Rigetti, IonQ, IQM via qBraid")}
'''


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
