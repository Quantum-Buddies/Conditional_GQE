#!/usr/bin/env python3
"""Bar chart comparing QPU/simulator energies vs GPU energies.

Also generates a scaling error plot (energy error vs qubit count)
if benchmark data is available.

Usage:
    python scripts/plot_qpu_vs_gpu.py \
        --benchmark results/eval/gic_benchmark_consolidated.json \
        --out-dir results/eval/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--benchmark", type=Path, required=True,
                   help="Consolidated benchmark JSON")
    p.add_argument("--out-dir", type=Path, default=Path("results/eval/figures"))
    return p.parse_args()


def plot_qpu_vs_gpu(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Bar chart: QPU vs GPU energy for molecules with both."""
    qpu_rows = [r for r in rows if r.get("h_cgqe_qpu_ha") is not None and r.get("h_cgqe_gpu_ha") is not None]
    if not qpu_rows:
        print("No molecules with both QPU and GPU energies - skipping QPU vs GPU plot")
        return

    names = [r["molecule"] for r in qpu_rows]
    gpu = [r["h_cgqe_gpu_ha"] for r in qpu_rows]
    qpu = [r["h_cgqe_qpu_ha"] for r in qpu_rows]
    fci = [r.get("fci_energy_ha") for r in qpu_rows]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_gpu = ax.bar(x - width, gpu, width, label="GPU (CUDA-Q)", color="#2196F3", edgecolor="black", linewidth=0.5)
    bars_qpu = ax.bar(x, qpu, width, label="QPU/Sim", color="#FF9800", edgecolor="black", linewidth=0.5)
    if all(f is not None for f in fci):
        bars_fci = ax.bar(x + width, fci, width, label="FCI", color="#4CAF50", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Energy (Hartree)", fontsize=12, fontweight="bold")
    ax.set_title("QPU/Simulator vs GPU Energy Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # Add delta annotations
    for i, r in enumerate(qpu_rows):
        delta = r.get("qpu_gpu_delta_mha")
        if delta is not None:
            ax.annotate(f"{delta:.1f} mHa", xy=(x[i], max(gpu[i], qpu[i])),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color="red")

    plt.tight_layout()
    out_path = out_dir / "qpu_vs_gpu.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"QPU vs GPU plot -> {out_path}")


def plot_scaling_error(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Scatter plot: energy error (mHa) vs qubit count."""
    err_rows = [r for r in rows if r.get("err_vs_fci_mha") is not None and r.get("n_qubits") is not None]
    if not err_rows:
        print("No molecules with error data - skipping scaling plot")
        return

    n_qubits = [r["n_qubits"] for r in err_rows]
    errors = [r["err_vs_fci_mha"] for r in err_rows]
    names = [r["molecule"] for r in err_rows]
    chem_acc = [r.get("chemical_accuracy", False) for r in err_rows]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Chemical accuracy threshold
    ax.axhline(y=1.6, color="green", linestyle="--", linewidth=1.5, label="Chemical accuracy (1.6 mHa)")

    # Plot points
    for nq, err, name, ca in zip(n_qubits, errors, names, chem_acc):
        color = "#4CAF50" if ca else "#F44336"
        marker = "o" if ca else "s"
        ax.scatter(nq, err, c=color, marker=marker, s=60, edgecolors="black", linewidth=0.5, zorder=5)
        if err > 200 or ca:
            ax.annotate(name, xy=(nq, err), xytext=(4, 4), textcoords="offset points",
                        fontsize=6, alpha=0.8)

    ax.set_xlabel("Number of Qubits", fontsize=12, fontweight="bold")
    ax.set_ylabel("Energy Error vs FCI (mHa)", fontsize=12, fontweight="bold")
    ax.set_title("H-cGQE Scaling: Energy Error vs Qubit Count", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.01)

    plt.tight_layout()
    out_path = out_dir / "scaling_error.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Scaling error plot -> {out_path}")


def plot_error_distribution(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Bar chart: error per molecule sorted by qubit count."""
    err_rows = [r for r in rows if r.get("err_vs_fci_mha") is not None]
    if not err_rows:
        return

    err_rows.sort(key=lambda r: (r.get("n_qubits", 0), r["molecule"]))

    names = [r["molecule"] for r in err_rows]
    errors = [r["err_vs_fci_mha"] for r in err_rows]
    colors = ["#4CAF50" if r.get("chemical_accuracy") else "#F44336" for r in err_rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), errors, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(y=1.6, color="green", linestyle="--", linewidth=1.2, label="Chemical accuracy (1.6 mHa)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Error vs FCI (mHa)", fontsize=12, fontweight="bold")
    ax.set_title("H-cGQE Energy Error per Molecule", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    out_path = out_dir / "error_per_molecule.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Error distribution plot -> {out_path}")


def plot_sqd_convergence(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Plot SQD energy vs subspace size for molecules with SQD data.

    If nested subspace energies are available in the benchmark rows,
    plot them. Otherwise, plot SQD raw vs recovered energy comparison.
    """
    sqd_rows = [r for r in rows if r.get("sqd_energy_ha") is not None]
    if not sqd_rows:
        print("No molecules with SQD data - skipping SQD convergence plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot raw vs recovered SQD energy
    names = [r["molecule"] for r in sqd_rows]
    raw = [r["sqd_energy_ha"] for r in sqd_rows]
    recovered = [r.get("sqd_recovered_energy_ha") or r["sqd_energy_ha"] for r in sqd_rows]
    fci = [r.get("fci_energy_ha") for r in sqd_rows]

    x = np.arange(len(names))
    width = 0.25

    ax.bar(x - width, raw, width, label="SQD (raw counts)", color="#2196F3", edgecolor="black", linewidth=0.5)
    ax.bar(x, recovered, width, label="SQD (recovered)", color="#FF9800", edgecolor="black", linewidth=0.5)
    if all(f is not None for f in fci):
        ax.bar(x + width, fci, width, label="FCI", color="#4CAF50", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Energy (Hartree)", fontsize=12, fontweight="bold")
    ax.set_title("SQD Convergence: Raw vs Recovered vs FCI", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "sqd_convergence.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"SQD convergence plot -> {out_path}")


def plot_selector_control(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Plot H-cGQE vs GQE vs VQE vs ADAPT-VQE vs SQD error comparison.

    Grouped bar chart showing error vs FCI (mHa) for each method,
    for molecules that have multiple baseline results.
    """
    # Collect molecules with at least 2 methods
    method_keys = [
        ("H-cGQE", "err_vs_fci_mha"),
        ("GQE", "gqe_err_vs_fci_mha"),
        ("VQE", "vqe_err_vs_fci_mha"),
        ("ADAPT-VQE", "adapt_vqe_err_vs_fci_mha"),
        ("SQD", "sqd_err_vs_fci_mha"),
    ]

    multi_rows = []
    for r in rows:
        n_methods = sum(1 for _, key in method_keys if r.get(key) is not None)
        if n_methods >= 2:
            multi_rows.append(r)

    if not multi_rows:
        print("No molecules with multiple method errors - skipping selector control plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    names = [r["molecule"] for r in multi_rows]
    x = np.arange(len(names))
    n_methods = len(method_keys)
    width = 0.8 / n_methods

    colors = ["#2196F3", "#FF9800", "#9C27B0", "#F44336", "#4CAF50"]
    for i, (label, key) in enumerate(method_keys):
        vals = [r.get(key) for r in multi_rows]
        vals = [v if v is not None else 0 for v in vals]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=colors[i],
                      edgecolor="black", linewidth=0.3, alpha=0.85)

    ax.axhline(y=1.6, color="green", linestyle="--", linewidth=1.2, label="Chemical accuracy (1.6 mHa)")
    ax.set_ylabel("Error vs FCI (mHa)", fontsize=12, fontweight="bold")
    ax.set_title("Method Comparison: H-cGQE vs Baselines", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=9, framealpha=0.9, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.01)

    plt.tight_layout()
    out_path = out_dir / "selector_control.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Selector control plot -> {out_path}")


def plot_vqe_comparison(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Scatter plot: H-cGQE error vs VQE/ADAPT-VQE error per molecule."""
    vqe_rows = [r for r in rows if r.get("vqe_err_vs_fci_mha") is not None and r.get("err_vs_fci_mha") is not None]
    adapt_rows = [r for r in rows if r.get("adapt_vqe_err_vs_fci_mha") is not None and r.get("err_vs_fci_mha") is not None]

    if not vqe_rows and not adapt_rows:
        print("No molecules with VQE/ADAPT-VQE data - skipping VQE comparison plot")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    # Diagonal line (equal performance)
    max_err = 0
    if vqe_rows:
        max_err = max(max_err, max(r["vqe_err_vs_fci_mha"] for r in vqe_rows),
                      max(r["err_vs_fci_mha"] for r in vqe_rows))
    if adapt_rows:
        max_err = max(max_err, max(r["adapt_vqe_err_vs_fci_mha"] for r in adapt_rows),
                      max(r["err_vs_fci_mha"] for r in adapt_rows))
    max_err = max(max_err * 1.1, 10)

    ax.plot([0, max_err], [0, max_err], "k--", linewidth=1, alpha=0.5, label="Equal performance")
    ax.fill_between([0, max_err], [0, max_err], [max_err, max_err], alpha=0.1, color="green",
                    label="H-cGQE wins (below diagonal)")

    if vqe_rows:
        ax.scatter([r["vqe_err_vs_fci_mha"] for r in vqe_rows],
                   [r["err_vs_fci_mha"] for r in vqe_rows],
                   c="#9C27B0", marker="o", s=60, edgecolors="black", linewidth=0.5,
                   label="vs UCCSD-VQE", zorder=5)
        for r in vqe_rows:
            ax.annotate(r["molecule"], xy=(r["vqe_err_vs_fci_mha"], r["err_vs_fci_mha"]),
                        xytext=(3, 3), textcoords="offset points", fontsize=6, alpha=0.8)

    if adapt_rows:
        ax.scatter([r["adapt_vqe_err_vs_fci_mha"] for r in adapt_rows],
                   [r["err_vs_fci_mha"] for r in adapt_rows],
                   c="#F44336", marker="s", s=60, edgecolors="black", linewidth=0.5,
                   label="vs ADAPT-VQE", zorder=5)
        for r in adapt_rows:
            ax.annotate(r["molecule"], xy=(r["adapt_vqe_err_vs_fci_mha"], r["err_vs_fci_mha"]),
                        xytext=(3, 3), textcoords="offset points", fontsize=6, alpha=0.8)

    ax.set_xlabel("Baseline Error vs FCI (mHa)", fontsize=12, fontweight="bold")
    ax.set_ylabel("H-cGQE Error vs FCI (mHa)", fontsize=12, fontweight="bold")
    ax.set_title("H-cGQE vs VQE Baselines", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max_err)
    ax.set_ylim(0, max_err)

    plt.tight_layout()
    out_path = out_dir / "vqe_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"VQE comparison plot -> {out_path}")


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = json.loads(args.benchmark.read_text())
    rows = benchmark.get("rows", [])

    plot_qpu_vs_gpu(rows, args.out_dir)
    plot_scaling_error(rows, args.out_dir)
    plot_error_distribution(rows, args.out_dir)
    plot_sqd_convergence(rows, args.out_dir)
    plot_selector_control(rows, args.out_dir)
    plot_vqe_comparison(rows, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
