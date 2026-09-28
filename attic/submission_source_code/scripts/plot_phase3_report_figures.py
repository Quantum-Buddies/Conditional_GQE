#!/usr/bin/env python3
"""Generate publication-quality figures for the GIC 2026 Phase 3 report.

Reads from existing JSON results under results/phase3_final/ and results/qpu/.
Writes PNGs to results/phase3_final/figures/.

Usage:
    python scripts/plot_phase3_report_figures.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "phase3_final" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLOR_HCGQE = "#2196F3"
COLOR_GQE = "#4CAF50"
COLOR_QPU = "#FF9800"
COLOR_ERROR = "#F44336"
COLOR_CHEM = "#2E7D32"


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        print(f"Warning: missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def save(fig: plt.Figure, name: str) -> Path:
    out = OUT_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_gpu_benchmark_bar() -> None:
    """Grouped bar chart: H-cGQE vs CUDA-Q GQE error vs FCI for 17 molecules."""
    data = load_json(ROOT / "results" / "phase3_final" / "consolidated_results_gic2026.json")
    if data is None:
        return
    rows = data.get("gpu_benchmark", [])

    names = []
    hcgqe_err = []
    gqe_err = []
    for r in rows:
        mol = r["molecule"].replace("_", " ")
        names.append(mol)
        hcgqe_err.append(r["error_vs_reference_mha"])
        ref = r["reference_energy"]
        gqe = r["cudaq_gqe_energy"]
        gqe_err.append(abs(ref - gqe) * 1000)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, hcgqe_err, width, label="H-cGQE", color=COLOR_HCGQE, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, gqe_err, width, label="CUDA-Q GQE", color=COLOR_GQE, edgecolor="black", linewidth=0.5)
    ax.axhline(1.6, color=COLOR_CHEM, linestyle="--", linewidth=1.5, label="Chemical accuracy (1.6 mHa)")

    ax.set_ylabel("Energy error vs FCI (mHa)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Molecule", fontsize=11, fontweight="bold")
    ax.set_title("GPU benchmark: H-cGQE vs CUDA-Q GQE", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    save(fig, "fig_gpu_benchmark_bar.png")


def plot_error_vs_qubits() -> None:
    """Scatter: energy error vs qubit count, color by method."""
    data = load_json(ROOT / "results" / "phase3_final" / "consolidated_results_gic2026.json")
    if data is None:
        return
    rows = data.get("gpu_benchmark", [])

    hcgqe_nq, hcgqe_err = [], []
    gqe_nq, gqe_err = [], []
    for r in rows:
        nq = r["n_qubits"]
        hcgqe_nq.append(nq)
        hcgqe_err.append(r["error_vs_reference_mha"])
        ref = r["reference_energy"]
        gqe = r["cudaq_gqe_energy"]
        gqe_nq.append(nq)
        gqe_err.append(abs(ref - gqe) * 1000)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(hcgqe_nq, hcgqe_err, s=80, c=COLOR_HCGQE, marker="o", edgecolors="black", linewidth=0.5, label="H-cGQE", zorder=3)
    ax.scatter(gqe_nq, gqe_err, s=80, c=COLOR_GQE, marker="^", edgecolors="black", linewidth=0.5, label="CUDA-Q GQE", zorder=3)
    ax.axhline(1.6, color=COLOR_CHEM, linestyle="--", linewidth=1.5, label="Chemical accuracy (1.6 mHa)")

    ax.set_xlabel("Number of qubits", fontsize=11, fontweight="bold")
    ax.set_ylabel("Energy error vs FCI (mHa)", fontsize=11, fontweight="bold")
    ax.set_title("Energy error grows with system size", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    save(fig, "fig_error_vs_qubits.png")


def plot_qpu_heatmap() -> None:
    """Heatmap of QPU SQD error and particle-number preservation."""
    data = load_json(ROOT / "results" / "qpu" / "cepheus_sqd_energies.json")
    if data is None:
        data = load_json(ROOT / "results" / "phase3_final" / "consolidated_results_gic2026.json")
        if data is None:
            return
        rows = data.get("qpu_results", [])
    else:
        rows = data.get("results", [])

    names = []
    errors = []
    pn = []
    for r in rows:
        err = r.get("error_mHa_vs_fci") if "error_mHa_vs_fci" in r else r.get("error_mHa")
        if err is None:
            continue
        names.append(r["molecule"].replace("_", " "))
        errors.append(err)
        pn.append(r.get("correct_pn_pct", 0))

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(names))
    width = 0.35

    ax.barh(y - width / 2, errors, width, color=COLOR_QPU, edgecolor="black", linewidth=0.5, label="SQD error vs FCI (mHa)")
    ax2 = ax.twinx()
    ax2.barh(y + width / 2, pn, width, color=COLOR_HCGQE, edgecolor="black", linewidth=0.5, label="Correct particle number (%)")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("SQD error (mHa)", fontsize=11, fontweight="bold", color=COLOR_QPU)
    ax2.set_xlabel("Correct particle number (%)", fontsize=11, fontweight="bold", color=COLOR_HCGQE)
    ax.set_title("Rigetti Cepheus QPU validation: SQD error and particle-number preservation", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelcolor=COLOR_QPU)
    ax2.tick_params(axis="x", labelcolor=COLOR_HCGQE)
    ax.grid(axis="x", alpha=0.3)

    ax.legend(loc="lower right", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)

    save(fig, "fig_qpu_validation.png")


def plot_sft_vs_rl_ablation() -> None:
    """Paired bar chart: SFT error vs RL error for 8 molecules."""
    data = load_json(ROOT / "results" / "phase3_final" / "ablation_sft_vs_rl.json")
    if data is None:
        return
    comps = data.get("comparisons", [])

    names = [c["molecule"].replace("_", " ") for c in comps]
    sft_err = [c["sft_error_mha"] for c in comps]
    rl_err = [c["rl_error_mha"] for c in comps]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, sft_err, width, label="SFT only", color="#9E9E9E", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, rl_err, width, label="SFT + DAPO RL", color=COLOR_HCGQE, edgecolor="black", linewidth=0.5)
    ax.axhline(1.6, color=COLOR_CHEM, linestyle="--", linewidth=1.5, label="Chemical accuracy (1.6 mHa)")

    ax.set_ylabel("Energy error (mHa)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Molecule", fontsize=11, fontweight="bold")
    ax.set_title("SFT vs SFT + DAPO RL: RL helps small molecules, not large ones", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    save(fig, "fig_sft_vs_rl_ablation.png")


def plot_qwc_reduction() -> None:
    """Bar chart showing QWC grouping reduction."""
    data = load_json(ROOT / "results" / "phase3_final" / "noise_mitigation_summary.json")
    if data is None:
        return
    qwc = data.get("qwc_grouping", {})

    molecules = []
    terms = []
    groups = []
    for mol, info in qwc.items():
        molecules.append(mol.upper())
        terms.append(info["n_terms"])
        groups.append(info["n_groups"])

    x = np.arange(len(molecules))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, terms, width, label="Pauli terms", color=COLOR_ERROR, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, groups, width, label="QWC groups (circuits)", color=COLOR_HCGQE, edgecolor="black", linewidth=0.5)

    for i, (t, g) in enumerate(zip(terms, groups)):
        reduction = t / g if g else 0
        ax.annotate(f"{reduction:.1f}×", xy=(x[i], max(t, g)), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax.set_xlabel("Molecule", fontsize=11, fontweight="bold")
    ax.set_title("QWC Pauli grouping reduces circuit count 2–3.5×", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(molecules)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    save(fig, "fig_qwc_reduction.png")


def plot_gpu_scaling_ladder() -> None:
    """Staircase chart: GPU max statevector qubits vs hardware."""
    hardware = [
        ("AIRE L40S", 24, 48, "PCIe, no NVLink"),
        ("qBraid H200", 30, 141, "NVLink"),
        ("qBraid B200", 32, 192, "NVLink"),
        ("qBraid 4×B200", 36, 768, "NVLink pooled"),
    ]

    names = [h[0] for h in hardware]
    sv_qubits = [h[1] for h in hardware]
    vram = [h[2] for h in hardware]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    width = 0.4

    bars = ax1.bar(x, sv_qubits, width, color=COLOR_HCGQE, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Max statevector qubits", fontsize=11, fontweight="bold", color=COLOR_HCGQE)
    ax1.set_xlabel("GPU platform", fontsize=11, fontweight="bold")
    ax1.set_title("GPU scaling ladder: from L40S to Blackwell B200", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax1.tick_params(axis="y", labelcolor=COLOR_HCGQE)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, vram, color=COLOR_QPU, marker="s", linewidth=2, markersize=8, label="VRAM (GB)")
    ax2.set_ylabel("VRAM (GB)", fontsize=11, fontweight="bold", color=COLOR_QPU)
    ax2.tick_params(axis="y", labelcolor=COLOR_QPU)
    ax2.legend(loc="upper left", fontsize=9)

    for bar, q in zip(bars, sv_qubits):
        ax1.annotate(f"{q}q", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    save(fig, "fig_gpu_scaling_ladder.png")


def plot_classical_vs_quantum() -> None:
    """Grouped bar chart: HF vs HEA-VQE vs ADAPT-VQE vs CUDA-Q GQE vs H-cGQE error vs FCI."""
    # Load all baseline data
    consolidated = load_json(ROOT / "results" / "phase3_final" / "classical_baseline_comparison.json")
    gic_bench = load_json(ROOT / "results" / "eval" / "gic_benchmark_consolidated.json")
    vqe_data = load_json(ROOT / "results" / "baselines" / "cudaq_vqe.json")
    adapt_data = load_json(ROOT / "results" / "baselines" / "adapt_vqe_h2.json")
    ch3i_vqe = load_json(ROOT / "results" / "phase3_final" / "baselines" / "benchmark_ch3i_he_vqe.json")

    if consolidated is None:
        return

    # Build lookup from gic_benchmark for HF energies
    hf_lookup = {}
    if gic_bench:
        for row in gic_bench.get("rows", []):
            hf_lookup[row["molecule"]] = {
                "hf": row.get("hf_energy_ha"),
                "fci": row.get("fci_energy_ha"),
            }

    # Build VQE lookup
    vqe_lookup = {}
    if vqe_data:
        for r in vqe_data.get("results", []):
            vqe_lookup[r["system"]] = abs(r["delta_energy"]) * 1000 if r["delta_energy"] else None
    if ch3i_vqe:
        for r in ch3i_vqe.get("results", []):
            vqe_lookup[r["system"]] = abs(r["delta_energy"]) * 1000 if r["delta_energy"] else None

    # Build ADAPT-VQE lookup
    adapt_lookup = {}
    if adapt_data:
        for r in adapt_data.get("results", []):
            adapt_lookup[r["system"]] = abs(r["delta_energy"]) * 1000 if r["delta_energy"] else None

    # Build comparison table from consolidated data
    molecules = consolidated.get("molecules", [])
    names = []
    hf_err = []
    hea_vqe_err = []
    adapt_vqe_err = []
    gqe_err = []
    hcgqe_err = []

    for m in molecules:
        mol = m["molecule"]
        fci = m["fci_energy"]
        # Skip if no FCI reference
        if fci is None:
            continue

        # Map molecule names
        display_name = mol.replace("_", " ")
        if mol == "methyl_iodide":
            display_name = "CH3I"
        elif mol == "imeph":
            display_name = "IMePh"
        elif mol == "iodobenzene":
            display_name = "iodobenzene"

        names.append(display_name)

        # HF error
        hf_info = hf_lookup.get(mol, {})
        hf_e = hf_info.get("hf") if hf_info else None
        if hf_e is None:
            # Try from consolidated if available
            hf_e = m.get("hf_energy")
        hf_err.append(abs(fci - hf_e) * 1000 if hf_e else None)

        # HEA-VQE error
        vqe_key = mol.split("_")[0] if "_" in mol else mol
        hea_vqe_err.append(vqe_lookup.get(vqe_key) or vqe_lookup.get(mol))

        # ADAPT-VQE error
        adapt_vqe_err.append(adapt_lookup.get(vqe_key) or adapt_lookup.get(mol))

        # CUDA-Q GQE error
        gqe_err.append(m.get("cudaq_gqe_error_mha"))

        # H-cGQE error
        hcgqe_err.append(m.get("hcgqe_error_mha"))

    # Only plot molecules with at least 3 methods
    has_data = []
    for i, n in enumerate(names):
        count = sum(1 for x in [hf_err[i], hea_vqe_err[i], adapt_vqe_err[i], gqe_err[i], hcgqe_err[i]] if x is not None)
        if count >= 3:
            has_data.append(i)

    if not has_data:
        print("Warning: no molecules with enough baseline data for comparison plot")
        return

    names = [names[i] for i in has_data]
    hf_err = [hf_err[i] for i in has_data]
    hea_vqe_err = [hea_vqe_err[i] for i in has_data]
    adapt_vqe_err = [adapt_vqe_err[i] for i in has_data]
    gqe_err = [gqe_err[i] for i in has_data]
    hcgqe_err = [hcgqe_err[i] for i in has_data]

    n_mols = len(names)
    x = np.arange(n_mols)
    n_methods = 5
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["#FF7043", "#FFB74D", "#81C784", "#4CAF50", "#2196F3"]
    labels = ["HF", "HEA-VQE", "ADAPT-VQE", "CUDA-Q GQE", "H-cGQE"]
    datasets = [hf_err, hea_vqe_err, adapt_vqe_err, gqe_err, hcgqe_err]

    for i, (data, color, label) in enumerate(zip(datasets, colors, labels)):
        offsets = x + (i - n_methods / 2 + 0.5) * width
        vals = [v if v is not None else 0 for v in data]
        bars = ax.bar(offsets, vals, width, label=label, color=color, edgecolor="black", linewidth=0.4)
        # Mark missing data
        for j, v in enumerate(data):
            if v is None:
                bars[j].set_alpha(0.15)
                bars[j].set_hatch("//")

    ax.axhline(1.6, color=COLOR_CHEM, linestyle="--", linewidth=1.5, label="Chemical accuracy (1.6 mHa)")
    ax.set_ylabel("Energy error vs FCI (mHa)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Molecule", fontsize=11, fontweight="bold")
    ax.set_title("Classical and quantum baselines vs H-cGQE", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9, framealpha=0.9, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 10000)

    save(fig, "fig_classical_vs_quantum.png")


def copy_proxy_figures() -> None:
    """Copy existing proxy-verification figures into report figures dir."""
    src_dir = ROOT / "results" / "eval" / "figures"
    for name in ["01_proxy_vs_converged_scatter.png", "02_proxy_flat_vs_final_varied.png"]:
        src = src_dir / name
        if src.exists():
            dst = OUT_DIR / f"fig_{name}"
            shutil.copy(src, dst)
            print(f"Copied: {dst}")
        else:
            print(f"Warning: {src} not found")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_gpu_benchmark_bar()
    plot_error_vs_qubits()
    plot_qpu_heatmap()
    plot_sft_vs_rl_ablation()
    plot_qwc_reduction()
    plot_gpu_scaling_ladder()
    plot_classical_vs_quantum()
    copy_proxy_figures()
    print("\nAll Phase 3 report figures generated.")


if __name__ == "__main__":
    main()
