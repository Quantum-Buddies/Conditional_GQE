"""Plot scalability benchmark results for GIC Mitsubishi Challenge.

Generates plots showing:
1. Energy accuracy vs molecule size (qubits)
2. Wall-clock time vs molecule size
3. Improvement over GQE baseline vs molecule size

Usage:
    python scripts/plot_scalability.py --report results/scaling_benchmark/scalability_report.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scalability benchmark results")
    parser.add_argument("--report", type=Path, required=True, help="Path to scalability_report.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots")
    args = parser.parse_args()

    with args.report.open("r") as f:
        data = json.load(f)

    results = data["results"]
    results.sort(key=lambda r: r["n_qubits"])

    molecules = [r["molecule"] for r in results]
    n_qubits = [r["n_qubits"] for r in results]
    best_energies = [r["best_energy"] for r in results]
    gqe_energies = [r.get("gqe_baseline_energy") for r in results]
    ref_energies = [r.get("reference_energy") for r in results]
    errors_mHa = [r.get("error_vs_ref_mHa") for r in results]
    improvements_mHa = [
        r.get("improvement_over_gqe", 0) * 1000 if r.get("improvement_over_gqe") is not None else None
        for r in results
    ]
    times = [r["total_time_s"] for r in results]
    infer_times = [r["infer_time_s"] for r in results]
    opt_times = [r["optimize_time_s"] for r in results]

    out_dir = args.out_dir or args.report.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Energy accuracy vs qubit count ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(molecules))
    width = 0.25

    bars_hcgqe = ax.bar(x - width, best_energies, width, label="H-cGQE (RL)", color="#2196F3", alpha=0.85)
    if any(g is not None for g in gqe_energies):
        gqe_vals = [g if g is not None else 0 for g in gqe_energies]
        bars_gqe = ax.bar(x, gqe_vals, width, label="GQE Baseline", color="#4CAF50", alpha=0.85)
    if any(r is not None for r in ref_energies):
        ref_vals = [r if r is not None else 0 for r in ref_energies]
        bars_ref = ax.bar(x + width, ref_vals, width, label="FCI Reference", color="#FF5722", alpha=0.85)

    ax.set_xlabel("Molecule", fontsize=13)
    ax.set_ylabel("Energy (Hartree)", fontsize=13)
    ax.set_title("H-cGQE vs GQE Baseline vs FCI Reference\n(GIC Scalability Benchmark)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({nq}q)" for m, nq in zip(molecules, n_qubits)], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "scalability_energy.png", dpi=150)
    print(f"Saved: {out_dir / 'scalability_energy.png'}")
    plt.close(fig)

    # --- Plot 2: Error vs FCI (mHa) vs qubit count ---
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_errors = [(nq, e) for nq, e in zip(n_qubits, errors_mHa) if e is not None]
    if valid_errors:
        x_err, y_err = zip(*valid_errors)
        ax.bar(x_err, y_err, color="#FF5722", alpha=0.85, width=1.5)
        ax.axhline(y=1.6, color="blue", linestyle="--", linewidth=2, label="Chemical accuracy (1.6 mHa)")
        ax.set_xlabel("Number of Qubits", fontsize=13)
        ax.set_ylabel("Error vs FCI (mHa)", fontsize=13)
        ax.set_title("Energy Error vs FCI Reference by Molecule Size", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for nq, e in zip(x_err, y_err):
            ax.text(nq, e + 0.5, f"{e:.1f}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "scalability_error.png", dpi=150)
    print(f"Saved: {out_dir / 'scalability_error.png'}")
    plt.close(fig)

    # --- Plot 3: Wall-clock time vs qubit count ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(molecules))
    width = 0.35
    ax.bar(x - width/2, infer_times, width, label="Inference", color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, opt_times, width, label="L-BFGS-B Optimization", color="#FF9800", alpha=0.85)
    ax.set_xlabel("Molecule", fontsize=13)
    ax.set_ylabel("Wall-clock Time (s)", fontsize=13)
    ax.set_title("Compute Time vs Molecule Size\n(3x L40S GPUs, nvidia-mqpu)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({nq}q)" for m, nq in zip(molecules, n_qubits)], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "scalability_time.png", dpi=150)
    print(f"Saved: {out_dir / 'scalability_time.png'}")
    plt.close(fig)

    # --- Plot 4: Improvement over GQE baseline ---
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_imprv = [(nq, i) for nq, i in zip(n_qubits, improvements_mHa) if i is not None]
    if valid_imprv:
        x_imp, y_imp = zip(*valid_imprv)
        colors = ["#4CAF50" if v > 0 else "#F44336" for v in y_imp]
        ax.bar(x_imp, y_imp, color=colors, alpha=0.85, width=1.5)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xlabel("Number of Qubits", fontsize=13)
        ax.set_ylabel("Improvement over GQE (mHa)", fontsize=13)
        ax.set_title("H-cGQE Improvement over GQE Baseline by Molecule Size", fontsize=14)
        ax.grid(axis="y", alpha=0.3)
        for nq, i in zip(x_imp, y_imp):
            ax.text(nq, i + 5 if i > 0 else i - 15, f"{i:.0f}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "scalability_improvement.png", dpi=150)
    print(f"Saved: {out_dir / 'scalability_improvement.png'}")
    plt.close(fig)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
