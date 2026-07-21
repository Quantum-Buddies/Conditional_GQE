#!/usr/bin/env python3
"""Generate publication-quality visualizations of the RL proxy vs converged energy problem."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_PATH = Path("results/eval/verify_rl_proxy_iodobenzene.json")
OUT_DIR = Path("results/eval/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.size": 13,
    "font.family": "serif",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Load data
with RESULTS_PATH.open() as f:
    data = json.load(f)

proxy = np.array([d["proxy_energy"] for d in data["data"]])
final = np.array([d["final_energy"] for d in data["data"]])
operators = [d["operators"] for d in data["data"]]
rho = data["spearman_rho"]
pval = data["p_value"]

# Shift to mHa relative to min for readability
proxy_mHa = (proxy - proxy.min()) * 1000  # mHa
final_mHa = (final - final.min()) * 1000  # mHa
final_spread = (final.max() - final.min()) * 1000
proxy_spread = (proxy.max() - proxy.min()) * 1000

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: The core problem — scatter plot with annotations
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

# Shift to mHa from HF baseline for interpretability
hf_baseline = proxy[0]  # all proxy values are ~HF
proxy_rel = (proxy - hf_baseline) * 1e6  # micro-Ha for proxy (since spread is tiny)
final_rel = (final - hf_baseline) * 1000  # mHa for final

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(proxy)))
ax.scatter(proxy_rel, final_rel, c=colors, s=120, zorder=5, edgecolors="black", linewidth=0.5)

# Add jitter to proxy for visibility since they're all nearly identical
proxy_jittered = proxy_rel + np.random.RandomState(42).uniform(-0.5, 0.5, len(proxy_rel))
ax.clear()
ax.scatter(proxy_jittered, final_rel, c=colors, s=120, zorder=5, edgecolors="black", linewidth=0.5)

# Horizontal line at median final energy
median_final = np.median(final_rel)
ax.axhline(median_final, color="red", linestyle="--", alpha=0.5, label=f"Median $E^\\star$ = {median_final:.2f} mHa")

ax.set_xlabel("RL Proxy Energy ($E_{\\theta=0.01}$, µHa from HF)", fontsize=14)
ax.set_ylabel("Converged Multi-Start Energy ($E^\\star$, mHa from HF)", fontsize=14)
ax.set_title(f"RL Proxy vs Converged Energy — Iodobenzene (8q)\n"
             f"Spearman $\\rho$ = {rho:.3f} (p = {pval:.3f}) — "
             f"Proxy spread: {proxy_spread*1e6:.1f} µHa | Final spread: {final_spread:.2f} mHa",
             fontsize=13)

# Annotate the problem
ax.annotate(
    "All proxy energies\nare nearly identical\n→ RL sees flat reward",
    xy=(proxy_jittered[0], final_rel.max()),
    xytext=(proxy_jittered.mean() + 2, final_rel.max() - 1),
    fontsize=11, fontweight="bold", color="red",
    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="red", alpha=0.8),
)

ax.annotate(
    "But converged energies\nspan {0:.1f} mHa!".format(final_spread),
    xy=(proxy_jittered[5], final_rel.min()),
    xytext=(proxy_jittered.mean() + 2, final_rel.min() + 0.5),
    fontsize=11, fontweight="bold", color="darkblue",
    arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="darkblue", alpha=0.8),
)

ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.2)
fig.savefig(OUT_DIR / "01_proxy_vs_converged_scatter.png")
plt.close(fig)
print(f"Saved 01_proxy_vs_converged_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Bar chart — proxy is flat, final varies
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

x = np.arange(len(proxy))
labels = [f"C{i+1}\n({len(ops)}g)" for i, ops in enumerate(operators)]

# Top: Proxy energies (in µHa from min)
ax1.bar(x, proxy_mHa * 1e6, color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.5)
ax1.set_ylabel("Proxy Energy\n(µHa from min)", fontsize=12)
ax1.set_title("What the RL Policy Sees (Fixed $\\theta=0.01$)", fontsize=14, color="#e74c3c")
ax1.axhline(0, color="black", linewidth=0.5)
ax1.set_ylim(-0.5, max(proxy_mHa * 1e6) + 1)
ax1.grid(axis="y", alpha=0.2)

# Annotate
ax1.text(0.98, 0.95, f"Spread: {proxy_spread*1e6:.2f} µHa\n(Essentially FLAT)",
         transform=ax1.transAxes, ha="right", va="top", fontsize=11,
         fontweight="bold", color="#e74c3c",
         bbox=dict(facecolor="white", edgecolor="#e74c3c", alpha=0.9))

# Bottom: Final converged energies (in mHa from min)
bars = ax2.bar(x, final_mHa, color="#2980b9", alpha=0.8, edgecolor="black", linewidth=0.5)
ax2.set_ylabel("Converged $E^\\star$\n(mHa from min)", fontsize=12)
ax2.set_title("What L-BFGS-B Multi-Start Actually Produces (5 restarts)", fontsize=14, color="#2980b9")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
ax2.grid(axis="y", alpha=0.2)

# Highlight best and worst
best_idx = np.argmin(final_mHa)
worst_idx = np.argmax(final_mHa)
bars[best_idx].set_color("#27ae60")
bars[worst_idx].set_color("#c0392b")

ax2.text(0.98, 0.95, f"Spread: {final_spread:.2f} mHa\n(Real signal!)",
         transform=ax2.transAxes, ha="right", va="top", fontsize=11,
         fontweight="bold", color="#2980b9",
         bbox=dict(facecolor="white", edgecolor="#2980b9", alpha=0.9))

# Annotate best/worst
ax2.annotate(f"Best\n({final_mHa[best_idx]:.2f} mHa)",
             xy=(best_idx, final_mHa[best_idx]),
             xytext=(best_idx, final_mHa[best_idx] - 1.5),
             fontsize=9, fontweight="bold", color="#27ae60",
             ha="center", arrowprops=dict(arrowstyle="->", color="#27ae60"))
ax2.annotate(f"Worst\n({final_mHa[worst_idx]:.2f} mHa)",
             xy=(worst_idx, final_mHa[worst_idx]),
             xytext=(worst_idx, final_mHa[worst_idx] + 1),
             fontsize=9, fontweight="bold", color="#c0392b",
             ha="center", arrowprops=dict(arrowstyle="->", color="#c0392b"))

fig.suptitle("The Physicist's Objection: Proxy $\\neq$ Converged Energy",
             fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "02_proxy_flat_vs_final_varied.png")
plt.close(fig)
print(f"Saved 02_proxy_flat_vs_final_varied.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Conceptual diagram — what the RL landscape looks like
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# Panel A: What RL thinks the landscape looks like (flat)
ax = axes[0]
circuit_ids = np.arange(15)
proxy_vals = proxy_mHa * 1e6  # µHa
ax.plot(circuit_ids, proxy_vals, "o-", color="#e74c3c", markersize=8, linewidth=2)
ax.fill_between(circuit_ids, proxy_vals - 0.01, proxy_vals + 0.01, alpha=0.3, color="#e74c3c")
ax.set_xlabel("Circuit Index (generated by policy)", fontsize=12)
ax.set_ylabel("Proxy Energy (µHa from min)", fontsize=12)
ax.set_title("(A) RL Reward Landscape\n(What the policy optimizes)", fontsize=13, color="#e74c3c")
ax.set_ylim(-0.1, 0.1)
ax.grid(True, alpha=0.2)
ax.text(0.5, 0.02, "FLAT → No gradient signal\n→ Policy learns noise",
        transform=ax.transAxes, ha="center", fontsize=11, fontweight="bold",
        color="#e74c3c", bbox=dict(facecolor="lightyellow", edgecolor="#e74c3c", alpha=0.9))

# Panel B: What the real landscape looks like (varied)
ax = axes[1]
final_vals = final_mHa
ax.plot(circuit_ids, final_vals, "o-", color="#2980b9", markersize=8, linewidth=2)
ax.fill_between(circuit_ids, final_vals, final_vals.min() - 0.5, alpha=0.15, color="#2980b9")
ax.set_xlabel("Circuit Index (generated by policy)", fontsize=12)
ax.set_ylabel("Converged $E^\\star$ (mHa from min)", fontsize=12)
ax.set_title("(B) True Energy Landscape\n(After multi-start L-BFGS-B)", fontsize=13, color="#2980b9")
ax.grid(True, alpha=0.2)
# Mark best
best_i = np.argmin(final_vals)
ax.plot(best_i, final_vals[best_i], "*", color="#27ae60", markersize=15, zorder=5)
ax.annotate("Best circuit", xy=(best_i, final_vals[best_i]),
            xytext=(best_i + 2, final_vals[best_i] - 1),
            fontsize=10, color="#27ae60", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#27ae60"))
ax.text(0.5, 0.02, f"Spread: {final_spread:.1f} mHa\n→ Real structure matters!",
        transform=ax.transAxes, ha="center", fontsize=11, fontweight="bold",
        color="#2980b9", bbox=dict(facecolor="lightblue", edgecolor="#2980b9", alpha=0.9))

# Panel C: The ranking mismatch
ax = axes[2]
# Rank by proxy (best to worst by proxy)
proxy_rank = np.argsort(proxy_vals)
# Show what the final energy looks like in that proxy-ranking order
final_in_proxy_order = final_vals[proxy_rank]
colors_rank = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(final_in_proxy_order)))
ax.barh(circuit_ids, final_in_proxy_order, color=colors_rank, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Converged $E^\\star$ (mHa from min)", fontsize=12)
ax.set_ylabel("Circuits ranked by\nRL proxy (best → worst)", fontsize=12)
ax.set_title("(C) Ranking Mismatch\n(If proxy were good, bars would\nincrease left→right)", fontsize=12)
ax.invert_yaxis()  # best proxy at top
ax.grid(axis="x", alpha=0.2)

# Add trend line
z = np.polyfit(circuit_ids, final_in_proxy_order, 1)
p = np.poly1d(z)
ax.plot(p(circuit_ids), circuit_ids, "k--", alpha=0.5, linewidth=2, label=f"Trend slope: {z[0]:.3f}")
ax.legend(fontsize=10)
ax.text(0.98, 0.02, f"Spearman $\\rho$ = {rho:.3f}\n→ No correlation",
        transform=ax.transAxes, ha="right", fontsize=11, fontweight="bold",
        color="red", bbox=dict(facecolor="lightyellow", edgecolor="red", alpha=0.9))

fig.suptitle("Why Fixed-$\\theta$ RL Proxy Fails: The Physicist's Verification",
             fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "03_conceptual_landscape.png")
plt.close(fig)
print(f"Saved 03_conceptual_landscape.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Energy convergence curves (simulated from the data we have)
# Shows how circuits that look identical at theta=0.01 diverge after optimization
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

# We don't have per-iteration data, but we can show the concept:
# All circuits start at the same proxy point, then L-BFGS-B sends them to different places
n_circuits = len(proxy)
hf = proxy[0]  # all proxy ≈ HF

# Plot starting point
ax.axhline(0, color="gray", linestyle=":", alpha=0.5, label="HF Baseline (proxy $\\theta=0.01$)")

# For each circuit, draw a line from (0, 0) to (100, final_mHa[i])
# This represents the L-BFGS-B optimization trajectory (conceptual)
for i in range(n_circuits):
    color = plt.cm.coolwarm(i / n_circuits)
    alpha = 0.7 if i in [best_idx, worst_idx] else 0.3
    lw = 2.5 if i in [best_idx, worst_idx] else 1.0
    label = None
    if i == best_idx:
        label = f"Best circuit (C{best_idx+1})"
        color = "#27ae60"
    elif i == worst_idx:
        label = f"Worst circuit (C{worst_idx+1})"
        color = "#c0392b"
    
    # Simulated convergence curve (exponential decay)
    iters = np.linspace(0, 100, 50)
    # Start at 0 (proxy = HF), converge to final_mHa[i]
    target = final_mHa[i]
    curve = target * (1 - np.exp(-iters / 20))
    ax.plot(iters, curve, color=color, alpha=alpha, linewidth=lw, label=label)

ax.set_xlabel("L-BFGS-B Iteration (conceptual)", fontsize=14)
ax.set_ylabel("Energy Improvement over HF (mHa)", fontsize=14)
ax.set_title("All circuits start at the same proxy point,\n"
             "but L-BFGS-B reveals their true quality differences",
             fontsize=13)
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.2)

# Annotate
ax.annotate("All circuits are\nindistinguishable here",
            xy=(1, 0.01), xytext=(15, final_mHa.max() * 0.6),
            fontsize=11, fontweight="bold", color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
            bbox=dict(facecolor="lightyellow", edgecolor="gray", alpha=0.8))

ax.annotate(f"Spread after\noptimization:\n{final_spread:.1f} mHa",
            xy=(100, final_mHa.max()), xytext=(70, final_mHa.max() + 0.5),
            fontsize=11, fontweight="bold", color="darkblue",
            arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.5),
            bbox=dict(facecolor="lightblue", edgecolor="darkblue", alpha=0.8))

fig.savefig(OUT_DIR / "04_convergence_conceptual.png")
plt.close(fig)
print(f"Saved 04_convergence_conceptual.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VISUALIZATION SUMMARY")
print("=" * 60)
print(f"Proxy energy spread:  {proxy_spread*1e6:.2f} µHa (FLAT)")
print(f"Final energy spread:  {final_spread:.2f} mHa (REAL SIGNAL)")
print(f"Spearman correlation: {rho:.4f} (p={pval:.4f}) — UNCORRELATED")
print(f"\nFigures saved to: {OUT_DIR}/")
print("  01_proxy_vs_converged_scatter.png  — Core scatter plot")
print("  02_proxy_flat_vs_final_varied.png  — Side-by-side bar charts")
print("  03_conceptual_landscape.png        — 3-panel conceptual diagram")
print("  04_convergence_conceptual.png      — How circuits diverge after optimization")
