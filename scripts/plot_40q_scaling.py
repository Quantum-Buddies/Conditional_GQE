#!/usr/bin/env python3
"""Generate publication-quality cross-platform scaling plots.

Compares GPU statevector, GPU MPS, and QPU results across qubit counts
from 4q to 40q. Produces:

1. Energy convergence vs qubit count (SV + MPS + QPU)
2. MPS bond dimension convergence at 32q and 40q
3. QPU vs GPU measurement distribution comparison
4. Circuit depth / gate count vs qubit count
5. Execution time scaling

Usage:
    python scripts/plot_40q_scaling.py \
        --results results/scaling_40q/scaling_report.json \
        --out results/scaling_40q/plots/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def _load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_energies(results: dict) -> dict[str, dict]:
    """Extract per-molecule energies from each stage."""
    molecules = {}

    # From optimization
    opt = results.get("stages", {}).get("optimization", [])
    if isinstance(opt, list):
        for mol in opt:
            name = mol.get("molecule", "?")
            nq = mol.get("n_qubits", 0)
            best_e = mol.get("best_optimized_energy", mol.get("best_energy"))
            ops = mol.get("best_sequence", {}).get("operators", [])
            thetas = mol.get("best_sequence", {}).get("thetas", [])
            if name not in molecules:
                molecules[name] = {"n_qubits": nq, "operators": ops, "thetas": thetas}
            molecules[name]["optimized_energy"] = best_e

    # From statevector
    sv = results.get("stages", {}).get("statevector", {})
    sv_results = sv.get("results", sv) if isinstance(sv, dict) else []
    if isinstance(sv_results, list):
        for mol in sv_results:
            name = mol.get("molecule", mol.get("system", "?"))
            e = mol.get("h_cgqe_energy", mol.get("energy"))
            if e is not None and name in molecules:
                molecules[name]["sv_energy"] = e

    # From MPS
    mps = results.get("stages", {}).get("mps", {})
    mps_results = mps.get("results", []) if isinstance(mps, dict) else []
    if isinstance(mps_results, list):
        for mol in mps_results:
            name = mol.get("molecule", mol.get("system", "?"))
            if name not in molecules:
                nq = mol.get("n_qubits", 0)
                molecules[name] = {"n_qubits": nq}
            bond_results = {}
            for bd in mol.get("bond_dimension_results", []):
                bd_val = bd.get("bond_dimension", 0)
                bd_energy = bd.get("energy")
                if bd_energy is not None:
                    bond_results[bd_val] = bd_energy
            molecules[name]["mps_energies"] = bond_results
            molecules[name]["mps_converged"] = mol.get("converged", False)

    # From QPU
    qpu = results.get("stages", {}).get("qpu", {})
    for key, val in qpu.items():
        if isinstance(val, dict):
            mol_name = val.get("molecule", key.split("_")[0])
            if mol_name in molecules:
                molecules[mol_name].setdefault("qpu_results", {})[key] = {
                    "device": val.get("device_id", key),
                    "energy": val.get("energy"),
                    "shots": val.get("shots", 0),
                    "fidelity": val.get("fidelity"),
                }

    return molecules


def plot_energy_vs_qubits(molecules: dict, out_dir: Path) -> None:
    """Plot energy error vs qubit count across platforms."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    names = sorted(molecules.keys(), key=lambda n: molecules[n].get("n_qubits", 0))
    qubits = [molecules[n].get("n_qubits", 0) for n in names]

    # SV energies
    sv_energies = [molecules[n].get("sv_energy") for n in names]
    sv_valid = [(q, e) for q, e in zip(qubits, sv_energies) if e is not None]
    if sv_valid:
        ax.plot([q for q, _ in sv_valid], [e for _, e in sv_valid],
                "o-", color="#2196F3", linewidth=2, markersize=8,
                label="GPU Statevector (exact)", zorder=5)

    # MPS best (highest bond dim)
    mps_best = []
    mps_qubits = []
    for n in names:
        mps_e = molecules[n].get("mps_energies", {})
        if mps_e:
            best_bd = max(mps_e.keys())
            mps_best.append(mps_e[best_bd])
            mps_qubits.append(molecules[n].get("n_qubits", 0))
    if mps_best:
        ax.plot(mps_qubits, mps_best, "s-", color="#4CAF50", linewidth=2, markersize=8,
                label="GPU MPS (D=256, approximate)", zorder=4)

    # Optimized energies
    opt_energies = [molecules[n].get("optimized_energy") for n in names]
    opt_valid = [(q, e) for q, e in zip(qubits, opt_energies) if e is not None]
    if opt_valid:
        ax.plot([q for q, _ in opt_valid], [e for _, e in opt_valid],
                "^--", color="#FF9800", linewidth=1.5, markersize=7,
                label="AI-optimized (L-BFGS-B)", zorder=3)

    # QPU energies
    qpu_energies = []
    qpu_qubits = []
    for n in names:
        qpu_r = molecules[n].get("qpu_results", {})
        for key, val in qpu_r.items():
            if val.get("energy") is not None:
                qpu_energies.append(val["energy"])
                qpu_qubits.append(molecules[n].get("n_qubits", 0))
    if qpu_energies:
        ax.scatter(qpu_qubits, qpu_energies, marker="*", s=200, color="#E91E63",
                   zorder=6, label="QPU (IQM/Rigetti)", edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Number of Qubits", fontsize=14, fontweight="bold")
    ax.set_ylabel("Energy (Hartree)", fontsize=14, fontweight="bold")
    ax.set_title("GPU-AI-QPU Convergence: 4q → 40q Quantum Chemistry",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

    fig.tight_layout()
    fig.savefig(out_dir / "energy_vs_qubits.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: energy_vs_qubits.png")


def plot_mps_convergence(molecules: dict, out_dir: Path) -> None:
    """Plot MPS bond dimension convergence for large molecules."""
    large_mols = {n: m for n, m in molecules.items()
                  if m.get("n_qubits", 0) >= 28 and m.get("mps_energies")}

    if not large_mols:
        print("  Skipping MPS convergence plot — no MPS data")
        return

    fig, axes = plt.subplots(1, len(large_mols), figsize=(6 * len(large_mols), 5),
                             dpi=150, squeeze=False)
    axes = axes[0]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for idx, (name, mol) in enumerate(sorted(large_mols.items(),
                                              key=lambda x: x[1].get("n_qubits", 0))):
        ax = axes[idx]
        mps_e = mol["mps_energies"]
        bds = sorted(mps_e.keys())
        energies = [mps_e[bd] for bd in bds]

        ax.plot(bds, energies, "o-", color=colors[idx % len(colors)],
                linewidth=2, markersize=8)
        ax.set_xlabel("Bond Dimension", fontsize=12, fontweight="bold")
        ax.set_ylabel("Energy (Ha)", fontsize=12, fontweight="bold")
        nq = mol.get("n_qubits", 0)
        ax.set_title(f"{name} ({nq}q)", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Mark convergence
        if len(energies) >= 2:
            delta = abs(energies[-1] - energies[-2])
            ax.annotate(f"ΔE(D={bds[-2]}→{bds[-1]})\n= {delta:.4f} Ha",
                        xy=(bds[-1], energies[-1]), xytext=(bds[-1] - 20, energies[-1]),
                        fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"),
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

    fig.suptitle("MPS Bond Dimension Convergence (28q → 40q)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "mps_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mps_convergence.png")


def plot_circuit_complexity(molecules: dict, out_dir: Path) -> None:
    """Plot circuit depth and operator count vs qubit count."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150)

    names = sorted(molecules.keys(), key=lambda n: molecules[n].get("n_qubits", 0))
    qubits = [molecules[n].get("n_qubits", 0) for n in names]
    n_ops = [len(molecules[n].get("operators", [])) for n in names]

    # Operator count
    ax1.bar(qubits, n_ops, color="#2196F3", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Qubits", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Operators", fontsize=12, fontweight="bold")
    ax1.set_title("GQE Circuit Compactness: AI-Generated Ansatz Size",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    for q, n in zip(qubits, n_ops):
        ax1.annotate(str(n), (q, n), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=9)

    # Estimated CNOT count (each operator: 2*(n_active-1) CNOTs)
    est_cnots = []
    for n in names:
        ops = molecules[n].get("operators", [])
        nq = molecules[n].get("n_qubits", 0)
        total_cnots = 0
        for op in ops:
            n_active = sum(1 for c in op if c != "I") if op else 0
            total_cnots += 2 * max(0, n_active - 1)
        est_cnots.append(total_cnots)

    colors = ["#4CAF50" if c < 100 else "#FF9800" if c < 500 else "#E91E63"
              for c in est_cnots]
    ax2.bar(qubits, est_cnots, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Qubits", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Estimated CNOT Count", fontsize=12, fontweight="bold")
    ax2.set_title("Circuit Depth: Why GQE Circuits Fit on Current QPUs",
                  fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="~100 CNOTs (QPU-feasible)")
    ax2.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "circuit_complexity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: circuit_complexity.png")


def plot_platform_comparison(molecules: dict, out_dir: Path) -> None:
    """Plot energy error comparison: GPU vs QPU at each qubit count."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    names = sorted(molecules.keys(), key=lambda n: molecules[n].get("n_qubits", 0))

    gpu_errors = []
    qpu_errors = []
    qubit_labels = []

    for n in names:
        nq = molecules[n].get("n_qubits", 0)
        opt_e = molecules[n].get("optimized_energy")
        sv_e = molecules[n].get("sv_energy")
        mps_e = molecules[n].get("mps_energies", {})
        mps_best = mps_e.get(max(mps_e.keys())) if mps_e else None
        gpu_e = sv_e or mps_best or opt_e

        qpu_r = molecules[n].get("qpu_results", {})
        qpu_e = None
        for val in qpu_r.values():
            if val.get("energy") is not None:
                qpu_e = val["energy"]
                break

        if gpu_e is not None and opt_e is not None:
            gpu_err = abs(gpu_e - opt_e)
            gpu_errors.append(gpu_err)
        else:
            gpu_errors.append(None)

        if qpu_e is not None and gpu_e is not None:
            qpu_err = abs(qpu_e - gpu_e)
            qpu_errors.append(qpu_err)
        else:
            qpu_errors.append(None)

        qubit_labels.append(f"{n}\n({nq}q)")

    x = np.arange(len(qubit_labels))
    width = 0.35

    gpu_valid = [e if e is not None else 0 for e in gpu_errors]
    qpu_valid = [e if e is not None else 0 for e in qpu_errors]

    ax.bar(x - width/2, gpu_valid, width, color="#2196F3", alpha=0.8,
           label="GPU (SV/MPS) vs AI-optimized", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, qpu_valid, width, color="#E91E63", alpha=0.8,
           label="QPU vs GPU", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Molecule (qubits)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Energy Error (Ha)", fontsize=12, fontweight="bold")
    ax.set_title("Cross-Platform Energy Agreement: GPU vs QPU",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(qubit_labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_dir / "platform_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: platform_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 40q scaling results")
    parser.add_argument("--results", type=Path, required=True,
                        help="Path to scaling_report.json")
    parser.add_argument("--out", type=Path, default=Path("results/scaling_40q/plots"),
                        help="Output directory for plots")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    results = _load_results(args.results)
    molecules = _extract_energies(results)

    print(f"Generating plots for {len(molecules)} molecules...")

    plot_energy_vs_qubits(molecules, args.out)
    plot_mps_convergence(molecules, args.out)
    plot_circuit_complexity(molecules, args.out)
    plot_platform_comparison(molecules, args.out)

    print(f"\nAll plots saved to {args.out}/")


if __name__ == "__main__":
    main()
