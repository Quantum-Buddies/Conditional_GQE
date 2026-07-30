#!/usr/bin/env python3
"""Generate 4 animated GIFs for the GIC 2026 submission.

Reuses the design system from build_gif.py:
- 800x450 SVG frames -> rsvg-convert -> Pillow GIF
- Clay/ink palette, Georgia headlines, DejaVu Sans labels

GIFs produced:
1. rl_training_loop.gif       — DAPO/GRPO training cycle
2. transformer_architecture.gif — GNN + Hamiltonian encoder + decoder
3. vqe_vs_gqe.gif             — Traditional VQE vs H-cGQE comparison
4. hpc_qpu_workflow.gif       — HPC -> QWC manifest -> qBraid -> QPU
"""
import math
import subprocess
import shutil
from pathlib import Path
from PIL import Image

OUT_DIR = Path("/scratch/kcwp264/Conditional-GQE_materials/docs")
FRAME_DIR = OUT_DIR / "_gif_frames_suite"
FRAME_DIR.mkdir(exist_ok=True)

W, H = 800, 450
FRAMES_PER_SCENE = 7
FRAME_MS = 260

STYLE = """
    .canvas { fill: #f0eee6; }
    .ink { stroke: #191919; fill: none; stroke-width: 1.4; stroke-linecap: round; stroke-linejoin: round; }
    .card { fill: #ffffff; stroke: #191919; stroke-width: 1.2; }
    .clay { stroke: #cc785c; fill: none; stroke-width: 1.8; stroke-linecap: round; }
    .clay-fill { fill: #cc785c; }
    .muted { stroke: #737373; fill: none; stroke-width: 1.0; stroke-linecap: round; }
    .muted-fill { fill: #737373; }
    .headline { font-family: Georgia, 'DejaVu Serif', serif; font-size: 22px; fill: #191919; }
    .label { font-family: 'DejaVu Sans', Inter, system-ui, sans-serif; font-size: 12.5px; fill: #191919; }
    .foot { font-family: 'DejaVu Sans', Inter, system-ui, sans-serif; font-size: 12.5px; fill: #737373; }
    .mono { font-family: 'DejaVu Sans Mono', monospace; font-size: 13px; fill: #191919; }
    .small { font-family: 'DejaVu Sans', Inter, system-ui, sans-serif; font-size: 10px; fill: #737373; }
"""

DEFS = ('<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6.5" markerHeight="6.5" orient="auto">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#191919"/></marker>'
        '<marker id="arrc" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6.5" markerHeight="6.5" orient="auto">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#cc785c"/></marker></defs>\n')

MX = 64
HEAD_Y = 80
FOOT_Y = 396


def svg_header():
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
            f'<style>{STYLE}</style>\n{DEFS}'
            f'<rect class="canvas" width="{W}" height="{H}"/>\n')


def headline(text):
    return f'<text x="{MX}" y="{HEAD_Y}" class="headline">{text}</text>\n'


def footnote(main, sub=None):
    s = f'<text x="{MX}" y="{FOOT_Y}" class="foot">{main}</text>\n'
    if sub:
        s += f'<text x="{MX}" y="{FOOT_Y + 20}" class="foot" opacity="0.7">{sub}</text>\n'
    return s


def card(x, y, w, h, title, sub=None, op=1.0):
    cx, cy = x + w / 2, y + h / 2
    s = f'<g opacity="{op:.2f}"><rect class="card" x="{x}" y="{y}" width="{w}" height="{h}" rx="4"/>\n'
    if sub:
        s += f'<text x="{cx:.0f}" y="{cy - 8:.0f}" text-anchor="middle" dominant-baseline="central" class="label" style="font-weight:bold">{title}</text>\n'
        s += f'<text x="{cx:.0f}" y="{cy + 10:.0f}" text-anchor="middle" dominant-baseline="central" class="foot">{sub}</text>\n'
    else:
        s += f'<text x="{cx:.0f}" y="{cy:.0f}" text-anchor="middle" dominant-baseline="central" class="label" style="font-weight:bold">{title}</text>\n'
    return s + "</g>"


def harrow(x1, x2, y, label=None, op=1.0, clay=False):
    cls = "clay" if clay else "ink"
    mid = (x1 + x2) // 2
    s = f'<g opacity="{op:.2f}"><line class="{cls}" x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke-dasharray="4,4" marker-end="url(#{"arrc" if clay else "arr"})"/>\n'
    if label:
        s += f'<text x="{mid}" y="{y - 10}" text-anchor="middle" class="foot">{label}</text>\n'
    return s + "</g>"


def varrow(x, y1, y2, label=None, op=1.0, clay=False):
    cls = "clay" if clay else "ink"
    mid = (y1 + y2) // 2
    s = f'<g opacity="{op:.2f}"><line class="{cls}" x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke-dasharray="4,4" marker-end="url(#{"arrc" if clay else "arr"})"/>\n'
    if label:
        s += f'<text x="{x + 8}" y="{mid}" class="foot">{label}</text>\n'
    return s + "</g>"


def token(x, y, letter, active=False, dim=False, size=46, op=1.0):
    stroke = "#cc785c" if active else "#191919"
    sw = 2 if active else 1.2
    fill = "#cc785c" if active else ("#737373" if dim else "#191919")
    return (f'<g opacity="{op:.2f}">'
            f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="4" fill="#ffffff" stroke="{stroke}" stroke-width="{sw}"/>'
            f'<text x="{x + size / 2:.0f}" y="{y + size / 2:.0f}" text-anchor="middle" dominant-baseline="central" class="mono" style="fill:{fill}">{letter}</text>'
            f"</g>\n")


def status_badge(x, y, status, op=1.0):
    colors = {"pending": "#737373", "running": "#cc785c", "completed": "#2d7d46", "queued": "#5b8def"}
    c = colors.get(status, "#737373")
    return (f'<g opacity="{op:.2f}">'
            f'<circle cx="{x}" cy="{y}" r="6" fill="{c}"/>'
            f'<text x="{x + 12}" y="{y + 1}" dominant-baseline="central" class="small" style="fill:{c}">{status}</text>'
            f"</g>\n")


def layer_block(x, y, w, h, label, sub=None, op=1.0, clay=False):
    stroke_c = "#cc785c" if clay else "#191919"
    sw = 1.6 if clay else 1.0
    s = f'<g opacity="{op:.2f}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="#ffffff" stroke="{stroke_c}" stroke-width="{sw}"/>\n'
    s += f'<text x="{x + w / 2:.0f}" y="{y + h / 2 - 4:.0f}" text-anchor="middle" dominant-baseline="central" class="small" style="font-weight:bold">{label}</text>\n'
    if sub:
        s += f'<text x="{x + w / 2:.0f}" y="{y + h / 2 + 8:.0f}" text-anchor="middle" dominant-baseline="central" class="small" style="fill:#737373">{sub}</text>\n'
    return s + "</g>"


def circuit_gate(x, y, label, w=28, h=28, op=1.0, clay=False):
    stroke_c = "#cc785c" if clay else "#191919"
    return (f'<g opacity="{op:.2f}">'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="#ffffff" stroke="{stroke_c}" stroke-width="1.0"/>'
            f'<text x="{x + w / 2:.0f}" y="{y + h / 2:.0f}" text-anchor="middle" dominant-baseline="central" class="mono" style="font-size:10px;fill:{stroke_c}">{label}</text>'
            f"</g>\n")


def gate_row(x, y, gates, op=1.0, clay=False):
    return "".join(circuit_gate(x + i * 32, y, g, op=op, clay=clay) for i, g in enumerate(gates))


def split_divider(x, op=1.0):
    return f'<line x1="{x}" y1="100" x2="{x}" y2="380" stroke="#191919" stroke-width="1.0" stroke-dasharray="3,5" opacity="{op:.2f}"/>\n'


def side_label(x, y, text, clay=False):
    c = "#cc785c" if clay else "#737373"
    return f'<text x="{x}" y="{y}" class="label" style="font-weight:bold;fill:{c}">{text}</text>\n'


# ============================================================
# GIF 1: RL TRAINING LOOP (7 scenes)
# ============================================================

def rl_s1(frac):
    """Sample circuits from the policy."""
    op = min(1.0, 0.3 + frac * 2)
    n_show = min(4, int(round(frac * 4)))
    rows = ""
    seqs = [["Y","Z","X","I"],["X","Y","Y","X"],["Z","X","Y","Z"],["Y","X","Z","Y"]]
    for r, seq in enumerate(seqs):
        ry = 150 + r * 52
        for i, t in enumerate(seq):
            act = (r == 0 and i == min(i, n_show - 1))
            vis = 1.0 if (r < n_show or r == 0) else 0.15
            rows += token(200 + i * 50, ry, t, active=act, size=40, op=vis)
    return f'''
{headline("Sample circuits from the policy.")}
{card(64, 170, 110, 60, "Decoder", "N=16 samples", op)}
{harrow(180, 196, 200, op=op)}
{rows}
{footnote("Autoregressive decoder generates N=16 operator sequences per molecule", "Top-p sampling (p=0.9, temp=1.0) with force_entanglement masking")}
'''


def rl_s2(frac):
    """Evaluate energies on CUDA-Q."""
    gpu_op = min(1.0, frac * 2.5)
    energy_op = min(1.0, max(0.0, (frac - 0.3) * 2.5))
    gpus = "".join(card(280 + i * 80, 160, 64, 56, f"L40S", f"GPU {i+1}", op=gpu_op) for i in range(3))
    energies = ["-1.137", "-1.089", "-1.062", "-1.145"]
    e_text = "".join(f'<text x="530" y="{200 + i * 28}" class="mono" opacity="{energy_op:.2f}">{e}</text>\n' for i, e in enumerate(energies))
    return f'''
{headline("Evaluate energies on CUDA-Q.")}
{card(64, 170, 110, 56, "Circuits", "16 sequences", op=gpu_op)}
{harrow(180, 276, 198, "nvidia-mqpu", op=gpu_op)}
{gpus}
{harrow(440, 510, 198, op=energy_op)}
<text x="530" y="180" class="foot" opacity="{energy_op:.2f}">E (Ha)</text>
{e_text}
{footnote("CUDA-Q statevector evaluation across 3x L40S GPUs (nvidia-mqpu)", "SQLite cache: 24k+ entries &#183; L-BFGS-B angle optimization (3-5 iters)")}
'''


def rl_s3(frac):
    """Compute multi-component reward."""
    comps = [("Energy","w1=1.0","-E/|E_ref|"),("Entanglement","w2=0.1","frac(X/Y)"),("Depth","w3=0.05","-depth/max"),("Non-commuting","w4=0.05","frac([Ai,Aj])")]
    n_show = min(4, int(round(frac * 4)))
    cards_s = "".join(card(64 + i * 170, 160, 150, 70, t, f"{w} &#183; {d}", op=(1.0 if i < n_show else 0.15)) for i, (t, w, d) in enumerate(comps))
    sum_op = min(1.0, max(0.0, (frac - 0.5) * 2.5))
    return f'''
{headline("Multi-component reward.")}
{cards_s}
{harrow(380, 430, 280, "R = &#931; wi &#183; ri", op=sum_op, clay=True)}
{card(430, 250, 200, 60, "Total Reward", "R = -0.847", op=sum_op)}
{footnote("Reward = w1(-E/|E_ref|) + w2(ent) + w3(-depth) + w4(non-comm)", "Auxiliary rewards gated on HF improvement (--energy-improvement-threshold)")}
'''


def rl_s4(frac):
    """GRPO group-relative advantages."""
    dot_op = min(1.0, frac * 2)
    mean_op = min(1.0, max(0.0, (frac - 0.2) * 2.5))
    bar_op = min(1.0, max(0.0, (frac - 0.4) * 2.5))
    rewards = [-0.3, 0.5, -0.8, 0.9, -0.1, 0.3, -0.5, 0.7]
    dots = ""
    bars = ""
    for i, r in enumerate(rewards):
        x = 100 + i * 70
        y = 230 - r * 50
        dots += f'<circle cx="{x}" cy="{y:.0f}" r="5" class="clay-fill" opacity="{dot_op:.2f}"/>\n'
        h_bar = abs(r) * 50
        if r >= 0:
            bars += f'<rect x="{x - 8}" y="{y:.0f}" width="16" height="{h_bar:.0f}" rx="1" class="clay-fill" opacity="{bar_op:.2f}"/>\n'
        else:
            bars += f'<rect x="{x - 8}" y="230" width="16" height="{h_bar:.0f}" rx="1" fill="#737373" opacity="{bar_op:.2f}"/>\n'
    return f'''
{headline("GRPO group-relative advantages.")}
{dots}
<line x1="80" y1="230" x2="660" y2="230" stroke="#191919" stroke-width="1" stroke-dasharray="3,3" opacity="{mean_op:.2f}"/>
<text x="670" y="234" class="foot" opacity="{mean_op:.2f}">&#956;_R</text>
{bars}
<text x="300" y="320" class="foot" opacity="{bar_op:.2f}">A_i = (R_i - &#956;_R) / (&#963;_R + &#949;)</text>
{footnote("GRPO: group-relative normalization &#183; Dynamic sampling filter: skip if std(R) &lt; 1e-8", "Zero advantage = zero gradient &#183; skips identical-reward batches")}
'''


def rl_s5(frac):
    """MAP-Elites archive fills."""
    fill_order = [0,6,12,18,24,3,9,15,21,5,11,17,23,1,7,13,19,4,10,16,22,2,8,14,20]
    n_filled = int(round(frac * len(fill_order)))
    filled = set(fill_order[:n_filled])
    cell, gx, gy = 30, 280, 150
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
    return f'''
{headline("MAP-Elites archive fills.")}
{card(64, 180, 160, 60, "Elite circuits", "quality-diversity")}
{harrow(230, 276, 210, op=min(1.0, frac * 3))}
{cells}
<text x="{gx}" y="{gy - 12}" class="foot">Entanglement &#215; Depth &#183; coverage {coverage}%</text>
<text x="{gx}" y="{gy + 5 * cell + 16}" class="foot">10&#215;10 grid &#183; novelty bonus &#955;=1.0&#8594;0.1</text>
{footnote("MAP-Elites: 10x10 grid of entanglement x depth &#183; novelty bonus decays", "Elite circuits selected for QPU deployment and QSCI scaling")}
'''


def rl_s6(frac):
    """DAPO asymmetric clip update."""
    show_op = min(1.0, frac * 2)
    clip_op = min(1.0, max(0.0, (frac - 0.2) * 2.5))
    grad_op = min(1.0, max(0.0, (frac - 0.5) * 2.5))
    return f'''
{headline("DAPO asymmetric clip update.")}
<g opacity="{show_op:.2f}">
  <line x1="100" y1="280" x2="500" y2="280" class="ink"/>
  <text x="510" y="284" class="foot">r = &#960;_&#952; / &#960;_old</text>
  <line x1="200" y1="265" x2="200" y2="295" stroke="#737373" stroke-width="1.5" opacity="{clip_op:.2f}"/>
  <text x="200" y="310" text-anchor="middle" class="small" opacity="{clip_op:.2f}">&#949;_lo=0.2</text>
  <line x1="380" y1="265" x2="380" y2="295" stroke="#cc785c" stroke-width="2" opacity="{clip_op:.2f}"/>
  <text x="380" y="310" text-anchor="middle" class="small" style="fill:#cc785c" opacity="{clip_op:.2f}">&#949;_hi=0.28</text>
  <rect x="200" y="270" width="180" height="20" fill="#cc785c" opacity="{clip_op * 0.15:.2f}"/>
</g>
<text x="150" y="230" class="foot" opacity="{clip_op:.2f}">L = -min(r&#183;A, clip(r, &#949;_lo, &#949;_hi)&#183;A)</text>
{card(64, 160, 130, 50, "Token-level", "loss", op=clip_op)}
{card(64, 330, 130, 50, "Decoder", "&#8711;&#952; policy", op=grad_op)}
<path class="clay" d="M 500 280 C 560 280 580 200 640 190" stroke-dasharray="4,4" marker-end="url(#arrc)" opacity="{grad_op:.2f}"/>
<text x="560" y="250" class="foot" style="fill:#cc785c" opacity="{grad_op:.2f}">gradient</text>
{footnote("DAPO: asymmetric clipping prevents entropy collapse &#183; token-level loss", "Clip-higher (0.28 > 0.2) encourages exploration &#183; no KL penalty needed")}
'''


def rl_s7(frac):
    """Replay buffer + next epoch."""
    buf_op = min(1.0, frac * 2)
    mix_op = min(1.0, max(0.0, (frac - 0.2) * 2.5))
    loop_op = min(1.0, max(0.0, (frac - 0.5) * 2.5))
    return f'''
{headline("Replay buffer closes the loop.")}
{card(64, 170, 160, 70, "Replay Buffer", "FIFO, size=2000", op=buf_op)}
{card(64, 280, 160, 50, "Pretrain mix", "80% &#8594; 0%", op=mix_op)}
<rect x="300" y="170" width="200" height="160" rx="8" class="card" opacity="{buf_op:.2f}"/>
<text x="400" y="195" text-anchor="middle" class="label" style="font-weight:bold" opacity="{buf_op:.2f}">FIFO Buffer</text>
<circle cx="340" cy="230" r="8" class="clay-fill" opacity="0.7"/>
<circle cx="370" cy="250" r="8" class="clay-fill" opacity="0.5"/>
<circle cx="410" cy="220" r="8" class="clay-fill" opacity="0.6"/>
<circle cx="450" cy="260" r="8" class="clay-fill" opacity="0.4"/>
<circle cx="390" cy="290" r="8" class="clay-fill" opacity="0.3"/>
<path class="clay" d="M 500 250 C 580 250 600 120 400 90 C 200 90 180 150 130 170" stroke-dasharray="4,4" marker-end="url(#arrc)" opacity="{loop_op:.2f}"/>
<text x="380" y="60" text-anchor="middle" class="foot" style="fill:#cc785c" opacity="{loop_op:.2f}">next epoch</text>
{footnote("FIFO replay buffer with pretrain sample mixing (80% &#8594; 0% decay)", "Gradient accumulation (4 micro-batches) &#183; Adam lr=1e-5")}
'''


SCENES_RL = [rl_s1, rl_s2, rl_s3, rl_s4, rl_s5, rl_s6, rl_s7]


# ============================================================
# GIF 2: TRANSFORMER ARCHITECTURE (6 scenes)
# ============================================================

def tx_s1(frac):
    """Chemistry input features."""
    n_show = min(3, int(round(frac * 3)))
    features = [("Atom Features","Z, charge, hybridization, valence"),("Bond Features","bond order, R_ij, conjugation"),("Global Features","N_q, N_e, spin 2S+1, CAS size")]
    cards_s = "".join(card(200, 130 + i * 80, 400, 60, t, d, op=(1.0 if i < n_show else 0.15)) for i, (t, d) in enumerate(features))
    return f'''
{headline("Chemistry input features.")}
{cards_s}
{footnote("Three feature streams: atom-level, bond-level, and global molecular features", "Fed into the Chemistry GNN encoder for graph-level conditioning")}
'''


def tx_s2(frac):
    """GNN encoder - message passing."""
    n_layers = min(3, int(round(frac * 3)))
    layers = "".join(layer_block(250, 130 + i * 55, 200, 42, f"MessageBlock {i+1}", "edge-weighted agg.", op=(1.0 if i < n_layers else 0.15), clay=(i == n_layers - 1)) for i in range(3))
    readout_op = min(1.0, max(0.0, (frac - 0.5) * 2.5))
    prefix_op = min(1.0, max(0.0, (frac - 0.7) * 3.5))
    return f'''
{headline("GNN encoder &#8212; message passing.")}
{layers}
{harrow(350, 350, 280, "readout", op=readout_op)}
{card(250, 290, 200, 50, "Graph Readout", "mean + max pool", op=readout_op)}
{harrow(350, 350, 340, op=prefix_op)}
{card(250, 350, 200, 40, "Prefix Projection", "128-dim + LayerNorm", op=prefix_op)}
{footnote("3-layer edge-aware MPNN &#183; residual + LayerNorm between blocks", "Graph readout: mean + max pool &#8594; prefix projection to conditioning_dim=128")}
'''


def tx_s3(frac):
    """Hamiltonian encoder - 4-layer transformer."""
    n_layers = min(4, int(round(frac * 4)))
    layers = "".join(layer_block(250, 120 + i * 42, 200, 36, f"Encoder {i+1}", "8-head self-attn", op=(1.0 if i < n_layers else 0.12)) for i in range(4))
    mem_op = min(1.0, max(0.0, (frac - 0.6) * 2.5))
    return f'''
{headline("Hamiltonian encoder &#8212; 4 layers.")}
{card(64, 150, 150, 50, "Pauli Terms", "vocab &#8594; d=256", op=min(1.0, frac * 3))}
{harrow(220, 250, 175, op=min(1.0, frac * 3))}
{layers}
{harrow(350, 350, 300, "K, V memory", op=mem_op, clay=True)}
{card(250, 310, 200, 40, "Encoder Memory", "R^(M&#215;256)", op=mem_op)}
{footnote("4-layer Transformer encoder &#183; 8-head self-attention + FFN=1024", "Pauli term IDs + coefficients &#8594; encoder memory (K, V for cross-attention)")}
'''


def tx_s4(frac):
    """Cross-attention decoder - 6 layers."""
    n_layers = min(6, int(round(frac * 6)))
    layers = "".join(layer_block(350, 110 + i * 35, 180, 30, f"Dec {i+1}", "self+cross attn", op=(1.0 if i < n_layers else 0.1)) for i in range(6))
    kv_op = min(1.0, max(0.0, (frac - 0.4) * 2))
    return f'''
{headline("Cross-attention decoder &#8212; 6 layers.")}
{card(64, 150, 130, 50, "Prefix", "soft prompt", op=min(1.0, frac * 3))}
{harrow(200, 340, 175, op=min(1.0, frac * 3))}
{card(64, 220, 130, 50, "BOS", "start token", op=min(1.0, max(0.0, (frac - 0.1) * 3)))}
{harrow(200, 340, 245, op=min(1.0, max(0.0, (frac - 0.1) * 3)))}
{layers}
<path class="clay" d="M 250 330 C 300 330 320 200 350 180" stroke-dasharray="4,4" marker-end="url(#arrc)" opacity="{kv_op:.2f}"/>
<text x="280" y="300" class="foot" style="fill:#cc785c" opacity="{kv_op:.2f}">K,V cross-attn</text>
{footnote("6-layer Transformer decoder &#183; 8-head self-attn + 8-head cross-attn from encoder", "Prefix tokens + BOS &#8594; autoregressive operator token generation")}
'''


def tx_s5(frac):
    """Constrained sampling."""
    funnel_op = min(1.0, frac * 2)
    mask_op = min(1.0, max(0.0, (frac - 0.2) * 2.5))
    tok_op = min(1.0, max(0.0, (frac - 0.5) * 2.5))
    return f'''
{headline("Constrained sampling.")}
<g opacity="{funnel_op:.2f}">
  <polygon points="200,130 600,130 500,200 300,200" fill="#ffffff" stroke="#191919" stroke-width="1.2"/>
  <text x="400" y="160" text-anchor="middle" class="label" style="font-weight:bold">Output Logits</text>
  <text x="400" y="178" text-anchor="middle" class="foot">R^L per step</text>
</g>
<g opacity="{mask_op:.2f}">
  <rect x="280" y="215" width="240" height="40" rx="4" class="card"/>
  <text x="400" y="237" text-anchor="middle" class="label" style="font-weight:bold">Top-p (p=0.9) + Z-mask + Length mask</text>
</g>
{harrow(400, 400, 270, op=mask_op)}
{token(300, 290, "Y", active=True, size=40, op=tok_op)}
{token(350, 290, "Z", dim=True, size=40, op=tok_op)}
{token(400, 290, "X", active=True, size=40, op=tok_op)}
{token(450, 290, "I", size=40, op=tok_op)}
<text x="350" y="350" class="foot" style="fill:#cc785c" opacity="{mask_op:.2f}">force_entanglement: Z-only blocked</text>
{footnote("Top-p sampling with constrained decoding &#183; Z-only token masking", "force_entanglement=True prevents diagonal collapse &#183; length mask filters by n_qubits")}
'''


def tx_s6(frac):
    """Autoregressive generation."""
    n_tok = min(6, int(round(frac * 6)))
    seq = ["Y", "Z", "X", "I", "Y", "X"]
    toks = "".join(token(120 + i * 52, 220, t, active=(i == n_tok - 1), size=42, op=(1.0 if i < n_tok else 0.1)) for i, t in enumerate(seq))
    loop_op = min(1.0, max(0.0, (frac - 0.4) * 2))
    last_x = 120 + max(0, n_tok - 1) * 52 + 21
    return f'''
{headline("Autoregressive generation.")}
{card(64, 140, 100, 40, "Decoder", op=min(1.0, frac * 3))}
{harrow(170, 280, 160, op=min(1.0, frac * 3))}
<text x="120" y="200" class="foot">operator sequence</text>
{toks}
<path class="clay" d="M {last_x} 220 C {last_x + 60} 220 {last_x + 80} 120 400 110 C 200 100 150 120 120 140" stroke-dasharray="4,4" marker-end="url(#arrc)" opacity="{loop_op:.2f}"/>
<text x="380" y="100" text-anchor="middle" class="foot" style="fill:#cc785c" opacity="{loop_op:.2f}">autoregressive feedback</text>
{footnote("Each token fed back into decoder for next-step prediction", "Sequence: A1, A2, ..., Ak &#183; UCCSD operator pool &#183; 0% Z-only by construction")}
'''


SCENES_TX = [tx_s1, tx_s2, tx_s3, tx_s4, tx_s5, tx_s6]


# ============================================================
# GIF 3: VQE vs GQE COMPARISON (6 scenes)
# ============================================================

def cmp_s1(frac):
    """The VQE problem - split screen setup."""
    op = min(1.0, frac * 2)
    return f'''
{headline("VQE vs H-cGQE: two philosophies.")}
{split_divider(400, op=op)}
{side_label(80, 120, "Traditional VQE")}
{side_label(480, 120, "H-cGQE (Ours)", clay=True)}
{card(80, 160, 280, 200, "Fixed ansatz", "Human-designed circuit", op=op)}
{card(440, 160, 280, 200, "AI-generated", "Conditioned on molecule", op=op)}
{footnote("Left: manually designed quantum circuits &#183; Right: AI proposes circuit structure", "Same goal: minimize &#10216;H&#10217; &#183; fundamentally different approach")}
'''


def cmp_s2(frac):
    """Ansatz design."""
    left_op = min(1.0, frac * 2)
    right_op = min(1.0, max(0.0, (frac - 0.2) * 2.5))
    left_gates = gate_row(90, 180, ["H","CX","RZ","CX","H","RZ","CX","H"], op=left_op)
    right_toks = "".join(token(460 + i * 48, 180, t, active=(i == 2), size=38, op=right_op) for i, t in enumerate(["Y","Z","X","I"]))
    return f'''
{headline("Ansatz design.")}
{split_divider(400)}
{side_label(80, 120, "Traditional VQE")}
{side_label(480, 120, "H-cGQE", clay=True)}
<text x="90" y="165" class="foot" opacity="{left_op:.2f}">UCCSD &#183; deep, many gates</text>
{left_gates}
<text x="90" y="230" class="foot" opacity="{left_op:.2f}">Human-designed, fixed topology</text>
<text x="460" y="165" class="foot" opacity="{right_op:.2f}">AI tokens &#183; compact sequence</text>
{right_toks}
<text x="460" y="230" class="foot" style="fill:#cc785c" opacity="{right_op:.2f}">Conditioned, adaptive structure</text>
{footnote("Left: fixed UCCSD ansatz with many gates &#183; Right: AI-generated compact operator sequence", "Same target energy, dramatically different circuit complexity")}
'''


def cmp_s3(frac):
    """Parameter optimization."""
    left_op = min(1.0, frac * 2)
    right_op = min(1.0, max(0.0, (frac - 0.2) * 2.5))
    return f'''
{headline("Parameter optimization.")}
{split_divider(400)}
{side_label(80, 120, "Traditional VQE")}
{side_label(480, 120, "H-cGQE", clay=True)}
<text x="90" y="160" class="foot" opacity="{left_op:.2f}">k continuous params on QPU</text>
<text x="90" y="190" class="mono" opacity="{left_op:.2f}">&#952; = (&#952;&#8321;, &#952;&#8322;, ..., &#952;&#8342;)</text>
<text x="90" y="220" class="foot" opacity="{left_op:.2f}">Parameter-shift gradient:</text>
<text x="90" y="240" class="mono" opacity="{left_op:.2f}">2k circuit evals / step</text>
{card(90, 260, 200, 50, "On quantum device", "expensive", op=left_op)}
<text x="460" y="160" class="foot" opacity="{right_op:.2f}">L-BFGS-B on k angles</text>
<text x="460" y="190" class="mono" opacity="{right_op:.2f}">3-5 iterations</text>
<text x="460" y="220" class="foot" opacity="{right_op:.2f}">Classical optimization:</text>
<text x="460" y="240" class="mono" opacity="{right_op:.2f}">fast, no QPU needed</text>
{card(460, 260, 200, 50, "On classical CPU", "cheap", op=right_op)}
{footnote("Left: all k parameters optimized on quantum hardware &#183; Right: classical L-BFGS-B", "Discrete structure chosen by AI &#183; only continuous angles need optimization")}
'''


def cmp_s4(frac):
    """Barren plateaus."""
    left_op = min(1.0, frac * 2)
    right_op = min(1.0, max(0.0, (frac - 0.2) * 2.5))
    dash_l = 1000 * frac
    gap_l = 1000 - dash_l
    dash_r = 1000 * max(0, frac - 0.2)
    gap_r = 1000 - dash_r
    return f'''
{headline("Barren plateaus.")}
{split_divider(400)}
{side_label(80, 120, "Traditional VQE")}
{side_label(480, 120, "H-cGQE", clay=True)}
<g transform="translate(90,300)" opacity="{left_op:.2f}">
  <line class="ink" x1="0" y1="0" x2="200" y2="0"/>
  <line class="ink" x1="0" y1="0" x2="0" y2="-100"/>
  <text x="-8" y="-106" text-anchor="end" class="small">dE/d&#952;</text>
  <path class="ink" pathLength="1000" stroke-dasharray="{dash_l:.1f},{gap_l:.1f}" d="M0,-10 C 40,-12 80,-14 120,-15 C 160,-16 200,-16 200,-16"/>
  <text x="100" y="-50" text-anchor="middle" class="small" style="fill:#737373">flat &#8594; vanishing gradient</text>
</g>
<text x="90" y="340" class="foot" style="fill:#737373" opacity="{left_op:.2f}">Barren plateau: dE/d&#952; &#8594; 0</text>
<g transform="translate(460,300)" opacity="{right_op:.2f}">
  <line class="ink" x1="0" y1="0" x2="200" y2="0"/>
  <line class="ink" x1="0" y1="0" x2="0" y2="-100"/>
  <text x="-8" y="-106" text-anchor="end" class="small">E</text>
  <path class="clay" pathLength="1000" stroke-dasharray="{dash_r:.1f},{gap_r:.1f}" d="M0,-10 C 30,-30 60,-70 100,-85 C 140,-92 180,-95 200,-96"/>
  <circle class="clay-fill" cx="200" cy="-96" r="4" opacity="{right_op:.2f}"/>
  <text x="100" y="-50" text-anchor="middle" class="small" style="fill:#cc785c">smooth descent</text>
</g>
<text x="460" y="340" class="foot" style="fill:#cc785c" opacity="{right_op:.2f}">No gradient issue: discrete + classical</text>
{footnote("Left: barren plateaus &#8212; gradient vanishes exponentially with system size", "Right: AI chooses structure, classical optimizer handles angles &#183; no barren plateaus")}
'''


def cmp_s5(frac):
    """Diagonal collapse."""
    left_op = min(1.0, frac * 2)
    right_op = min(1.0, max(0.0, (frac - 0.2) * 2.5))
    bad_toks = "".join(token(90 + i * 48, 170, "Z", dim=True, size=38, op=left_op) for i in range(4))
    good_toks = "".join(token(460 + i * 48, 170, t, active=(i == 1), size=38, op=right_op) for i, t in enumerate(["X","Y","Y","X"]))
    return f'''
{headline("Diagonal collapse, avoided.")}
{split_divider(400)}
{side_label(80, 120, "Naive GQE")}
{side_label(480, 120, "H-cGQE", clay=True)}
<text x="90" y="155" class="foot" opacity="{left_op:.2f}">Z-only operators &#183; commute with H</text>
{bad_toks}
<line class="clay" x1="82" y1="165" x2="290" y2="215" opacity="{left_op:.2f}"/>
<text x="90" y="250" class="foot" style="fill:#737373" opacity="{left_op:.2f}">Zero gradient &#183; trapped at HF energy</text>
<text x="460" y="155" class="foot" style="fill:#cc785c" opacity="{right_op:.2f}">UCCSD pool &#183; every operator entangles</text>
{good_toks}
<text x="460" y="250" class="foot" style="fill:#cc785c" opacity="{right_op:.2f}">force_entanglement &#183; 0% Z-only</text>
<text x="660" y="195" class="mono" style="fill:#2d7d46;font-size:18px" opacity="{right_op:.2f}">&#10003;</text>
{footnote("Left: Z-only sequences commute with H &#183; zero gradient &#183; trapped at Hartree-Fock", "Right: UCCSD pool + force_entanglement masking &#183; every operator carries X/Y")}
'''


def cmp_s6(frac):
    """Result comparison."""
    left_op = min(1.0, frac * 2)
    right_op = min(1.0, max(0.0, (frac - 0.2) * 2.5))
    bar_l_h = 30
    bar_r_h = 30 + int(120 * min(1.0, max(0.0, (frac - 0.3) * 2)))
    return f'''
{headline("The result.")}
{split_divider(400)}
{side_label(80, 120, "Traditional VQE")}
{side_label(480, 120, "H-cGQE", clay=True)}
<text x="90" y="160" class="foot" opacity="{left_op:.2f}">Trapped at HF energy</text>
<rect x="120" y="{300 - bar_l_h}" width="80" height="{bar_l_h}" fill="#737373" opacity="{left_op:.2f}" rx="2"/>
<text x="160" y="320" text-anchor="middle" class="mono" opacity="{left_op:.2f}">-0.98 Ha</text>
<text x="90" y="350" class="foot" style="fill:#737373" opacity="{left_op:.2f}">No improvement</text>
<text x="460" y="160" class="foot" style="fill:#cc785c" opacity="{right_op:.2f}">Chemical accuracy &#8804; 1.6 mHa</text>
<line x1="440" y1="170" x2="720" y2="170" stroke="#cc785c" stroke-width="1" stroke-dasharray="2,4" opacity="{right_op:.2f}"/>
<text x="730" y="174" class="small" style="fill:#cc785c" opacity="{right_op:.2f}">E_FCI</text>
<rect x="490" y="{300 - bar_r_h}" width="80" height="{bar_r_h}" class="clay-fill" opacity="{right_op:.2f}" rx="2"/>
<text x="530" y="320" text-anchor="middle" class="mono" opacity="{right_op:.2f}">-1.137 Ha</text>
<text x="460" y="350" class="foot" style="fill:#cc785c" opacity="{right_op:.2f}">&#10003; Sub-chemical accuracy</text>
{footnote("Left: VQE trapped at Hartree-Fock energy on large molecules", "Right: H-cGQE achieves &#8804; 1.6 mHa chemical accuracy &#183; CH3I: 0.63 mHa")}
'''


SCENES_CMP = [cmp_s1, cmp_s2, cmp_s3, cmp_s4, cmp_s5, cmp_s6]


# ============================================================
# GIF 4: HPC <-> QPU ASYNC WORKFLOW (6 scenes)
# ============================================================

def hq_s1(frac):
    """HPC does the heavy lifting."""
    op = min(1.0, frac * 2)
    gpus = "".join(card(200 + i * 120, 160, 100, 70, "L40S", f"GPU {i+1}", op=op) for i in range(3))
    return f'''
{headline("HPC does the heavy lifting.")}
{card(64, 170, 120, 60, "AIRE Cluster", "3x L40S", op=op)}
{harrow(190, 200, 200, op=op)}
{gpus}
<text x="400" y="270" text-anchor="middle" class="label" style="font-weight:bold" opacity="{op:.2f}">CUDA-Q</text>
<text x="400" y="290" text-anchor="middle" class="foot" opacity="{op:.2f}">nvidia-mqpu target</text>
<text x="400" y="330" text-anchor="middle" class="foot" opacity="{op:.2f}">RL training &#183; circuit synthesis &#183; L-BFGS-B optimization</text>
{footnote("All classical + simulation compute runs on HPC before QPU submission", "RL training, circuit synthesis, and angle optimization all on 3x L40S")}
'''


def hq_s2(frac):
    """QWC Pauli term grouping."""
    group_op = min(1.0, frac * 2)
    reduce_op = min(1.0, max(0.0, (frac - 0.3) * 2.5))
    terms = ""
    for i in range(20):
        x = 80 + (i % 10) * 30
        y = 150 + (i // 10) * 30
        terms += f'<rect x="{x}" y="{y}" width="24" height="24" rx="2" fill="#737373" opacity="0.4"/>\n'
    groups = ""
    colors = ["#cc785c", "#5b8def", "#2d7d46", "#c4a747", "#cc785c", "#5b8def"]
    for i in range(6):
        x = 420 + (i % 3) * 80
        y = 150 + (i // 3) * 60
        groups += f'<rect x="{x}" y="{y}" width="60" height="40" rx="3" fill="{colors[i]}" opacity="{group_op * 0.6:.2f}"/>\n'
    return f'''
{headline("QWC Pauli term grouping.")}
<text x="80" y="135" class="foot">631 terms (LiH)</text>
{terms}
{harrow(380, 410, 200, "group", op=group_op)}
<text x="420" y="135" class="foot" opacity="{group_op:.2f}">180 circuits</text>
{groups}
<text x="400" y="280" text-anchor="middle" class="label" style="font-weight:bold;fill:#cc785c" opacity="{reduce_op:.2f}">3.5x reduction</text>
{footnote("Qubit-wise commuting terms grouped into shared measurement circuits", "631 Pauli terms &#8594; 180 QWC groups &#183; same basis measurement per group")}
'''


def hq_s3(frac):
    """Export QWC manifest."""
    op = min(1.0, frac * 2)
    fields = [("operators", "A1, A2, ..., Ak"),("thetas", "&#952;1, &#952;2, ..., &#952;k"),("groups", "180 QWC groups"),("QASM", "OpenQASM 2.0")]
    n_show = min(4, int(round(frac * 4)))
    field_text = ""
    for i, (k, v) in enumerate(fields):
        o = 1.0 if i < n_show else 0.15
        field_text += f'<text x="300" y="{170 + i * 35}" class="mono" opacity="{o:.2f}">{k}: {v}</text>\n'
    return f'''
{headline("Export QWC manifest.")}
{card(64, 170, 180, 50, "Optimize", "L-BFGS-B", op=op)}
{harrow(250, 290, 195, "export", op=op)}
<rect x="290" y="140" width="380" height="180" rx="6" class="card" opacity="{op:.2f}"/>
<text x="480" y="130" text-anchor="middle" class="label" style="font-weight:bold" opacity="{op:.2f}">QWC Manifest</text>
{field_text}
{footnote("Portable artifact: operators, optimized angles, QWC groups, QASM circuits", "Decouples HPC compute from QPU queue time &#183; manifest is self-contained")}
'''


def hq_s4(frac):
    """Async submit to qBraid."""
    op = min(1.0, frac * 2)
    vendor_op = min(1.0, max(0.0, (frac - 0.2) * 2.5))
    vendors = "".join(card(500, 150 + i * 70, 200, 50, name, desc, op=vendor_op) for i, (name, desc) in enumerate([("Rigetti","superconducting"),("IonQ","trapped ion"),("IQM","superconducting")]))
    return f'''
{headline("Async submit to qBraid.")}
{card(64, 170, 160, 60, "Manifest", "QWC + QASM", op=op)}
{harrow(230, 320, 200, op=op)}
{card(320, 170, 140, 60, "qBraid", "routing hub", op=op)}
{harrow(460, 500, 180, "submit", op=vendor_op)}
{vendors}
{footnote("qBraid routes manifest to multiple QPU vendors simultaneously", "Rigetti (superconducting) &#183; IonQ (trapped ion) &#183; IQM (superconducting)")}
'''


def hq_s5(frac):
    """QPU execution + queue."""
    op = min(1.0, frac * 2)
    statuses = ["completed", "running", "pending"]
    n_done = min(3, int(round(frac * 3)))
    badges = ""
    for i, s in enumerate(statuses):
        actual = "completed" if i < n_done else ("running" if i == n_done else "pending")
        badges += status_badge(520, 165 + i * 70, actual, op=op)
    vendors = "".join(card(320, 150 + i * 70, 180, 50, name, desc, op=op) for i, (name, desc) in enumerate([("Rigetti","Ankaa-3"),("IonQ","Aria"),("IQM","Emerald")]))
    return f'''
{headline("QPU execution + queue.")}
{vendors}
{badges}
<text x="320" y="130" class="foot" opacity="{op:.2f}">Vendor &#183; QPU &#183; Status</text>
{footnote("Jobs queue independently at each vendor &#183; status tracked asynchronously", "HPC compute already complete &#183; only waiting for QPU queue + execution")}
'''


def hq_s6(frac):
    """Retrieve + merge results."""
    op = min(1.0, frac * 2)
    merge_op = min(1.0, max(0.0, (frac - 0.3) * 2.5))
    result_op = min(1.0, max(0.0, (frac - 0.5) * 2.5))
    return f'''
{headline("Retrieve + merge results.")}
{card(64, 160, 130, 50, "Rigetti", "results", op=op)}
{card(64, 230, 130, 50, "IonQ", "results", op=op)}
{card(64, 300, 130, 50, "IQM", "results", op=op)}
{harrow(200, 300, 220, "retrieve", op=op, clay=True)}
{card(300, 220, 160, 70, "Merge", "QWC parse", op=merge_op)}
{harrow(460, 560, 255, op=result_op, clay=True)}
{card(560, 210, 180, 90, "Energy", "-1.118 Ha", op=result_op)}
<text x="560" y="320" class="foot" style="fill:#cc785c" opacity="{result_op:.2f}">H2: 1.48 mHa vs GPU</text>
{footnote("Results retrieved asynchronously &#183; merged via QWC group parsing", "Decoupled from queue time &#183; HPC compute + QPU execution run independently")}
'''


SCENES_HQ = [hq_s1, hq_s2, hq_s3, hq_s4, hq_s5, hq_s6]


# ============================================================
# BUILD & ASSEMBLE
# ============================================================

def build_and_save(name, scenes):
    """Build frames for a set of scenes and assemble into a GIF."""
    idx = 0
    paths = []
    sub_dir = FRAME_DIR / name
    sub_dir.mkdir(exist_ok=True)
    for scene_fn in scenes:
        for step in range(FRAMES_PER_SCENE):
            frac = step / (FRAMES_PER_SCENE - 1)
            svg = svg_header() + scene_fn(frac) + "\n</svg>"
            svg_path = sub_dir / f"frame_{idx:03d}.svg"
            svg_path.write_text(svg)
            png_path = sub_dir / f"frame_{idx:03d}.png"
            subprocess.run(
                ["rsvg-convert", "-w", str(W), "-h", str(H), "-o", str(png_path), str(svg_path)],
                check=True,
            )
            paths.append(png_path)
            idx += 1
    out_path = OUT_DIR / f"{name}.gif"
    frames = [Image.open(p).convert("RGB") for p in paths]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=False,
    )
    shutil.rmtree(sub_dir)
    size_kb = out_path.stat().st_size / 1024
    print(f"  Wrote {out_path.name} ({size_kb:.1f} KB, {len(paths)} frames)")
    return out_path


if __name__ == "__main__":
    print("Generating GIC 2026 GIF suite...")
    build_and_save("rl_training_loop", SCENES_RL)
    build_and_save("transformer_architecture", SCENES_TX)
    build_and_save("vqe_vs_gqe", SCENES_CMP)
    build_and_save("hpc_qpu_workflow", SCENES_HQ)
    # cleanup
    if FRAME_DIR.exists():
        shutil.rmtree(FRAME_DIR)
    print("Done. All 4 GIFs generated in docs/")
