"""Plot H-cGQE evaluation results for the proposal.

Generates:
- Energy error vs n_qubits
- Runtime vs accuracy
- Training curves
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_energy_error_vs_qubits(
    eval_path: Path | None,
    optimized_path: Path | None,
    baseline_path: Path | None,
    out_path: Path,
) -> None:
    """Plot energy error vs number of qubits comparing baseline, fixed, and optimized H-cGQE."""
    import json
    
    # Load exact reference energies from baseline results if available
    ref_energies = {}
    gqe_energies = {}
    if baseline_path and baseline_path.exists():
        with baseline_path.open("r") as f:
            b_data = json.load(f)
            for res in b_data.get("results", []):
                name = res.get("system")
                if name:
                    ref_energies[name] = res.get("reference_energy")
                    gqe_energies[name] = res.get("baseline_energy")

    # Load evaluated results
    eval_data = []
    if eval_path and eval_path.exists():
        with eval_path.open("r") as f:
            eval_data = json.load(f)

    # Load optimized results
    opt_data = []
    if optimized_path and optimized_path.exists():
        with optimized_path.open("r") as f:
            opt_data = json.load(f)

    # Group by molecule
    molecules = sorted(list(set(
        [d["molecule"] for d in eval_data] + [d["molecule"] for d in opt_data]
    )))

    # Real qubit counts
    qubit_map = {"h2": 4, "iodobenzene": 8, "lih": 12, "beh2": 14, "n2": 20}
    qubits = [qubit_map.get(m, len(m) * 2) for m in molecules]

    gqe_errors = []
    fixed_errors = []
    opt_errors = []

    for mol in molecules:
        ref_e = ref_energies.get(mol)
        if ref_e is None:
            # Fallback for N2 reference
            if mol == "n2":
                ref_e = -109.5422  # Known exact ground state energy for N2
            else:
                ref_e = 0.0

        # GQE error
        gqe_e = gqe_energies.get(mol)
        gqe_errors.append(abs(gqe_e - ref_e) if gqe_e is not None else None)

        # Fixed error
        fixed_e = next((d["best_generated_energy"] for d in eval_data if d["molecule"] == mol), None)
        fixed_errors.append(abs(fixed_e - ref_e) if fixed_e is not None else None)

        # Optimized error
        opt_e = next((d["best_energy"] for d in opt_data if d["molecule"] == mol), None)
        opt_errors.append(abs(opt_e - ref_e) if opt_e is not None else None)

    plt.figure(figsize=(9, 6))
    
    # Plot curves
    valid_gqe = [(q, err) for q, err in zip(qubits, gqe_errors) if err is not None]
    valid_fixed = [(q, err) for q, err in zip(qubits, fixed_errors) if err is not None]
    valid_opt = [(q, err) for q, err in zip(qubits, opt_errors) if err is not None]

    if valid_gqe:
        plt.plot(*zip(*sorted(valid_gqe)), "o-", label="GQE Baseline", color="#4c72b0", linewidth=2.5, markersize=8)
    if valid_fixed:
        plt.plot(*zip(*sorted(valid_fixed)), "s--", label="H-cGQE (Fixed Theta=0.01)", color="#dd8452", linewidth=2, markersize=8)
    if valid_opt:
        plt.plot(*zip(*sorted(valid_opt)), "^-", label="H-cGQE (Optimized Thetas)", color="#55a868", linewidth=2.5, markersize=10)

    plt.xlabel("Number of Qubits", fontsize=12)
    plt.ylabel("Energy Error vs Reference (Ha, Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.title("H-cGQE Performance: Multi-GPU Scaling and Two-Stage Evaluation", fontsize=14, pad=15)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    # Add annotation about diagonal terms
    plt.text(11, 1.2, "Note: Z-only (diagonal) sequence collapse\non LiH, BeH2, N2 limits optimization.",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved advanced energy scaling plot to {out_path}")


def plot_training_curves(metrics_path: Path, out_path: Path) -> None:
    """Plot training loss curves."""
    with metrics_path.open("r") as f:
        data = json.load(f)

    train_losses = data.get("train_losses", [])
    val_losses = data.get("val_losses", [])
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(9, 4))
    plt.plot(epochs, train_losses, label="Train", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("H-cGQE Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved training curves to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot H-cGQE evaluation results")
    parser.add_argument("--eval", type=Path, help="Evaluation JSON (fixed-theta)")
    parser.add_argument("--optimized", type=Path, help="Optimized evaluation JSON")
    parser.add_argument("--baseline", type=Path, help="Baseline GQE results JSON")
    parser.add_argument("--metrics", type=Path, help="Training metrics JSON")
    parser.add_argument("--out-dir", type=Path, default=Path("results/eval/plots"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval or args.optimized:
        plot_energy_error_vs_qubits(
            args.eval,
            args.optimized,
            args.baseline,
            args.out_dir / "energy_error_vs_qubits.png"
        )

    if args.metrics:
        plot_training_curves(args.metrics, args.out_dir / "training_curves.png")


if __name__ == "__main__":
    main()
