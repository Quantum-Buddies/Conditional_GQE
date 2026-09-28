#!/usr/bin/env python3
"""Plot H2 dissociation curve: GPU vs QPU vs FCI vs HF.

Once QPU results are available from submit_remaining_qpu.py, this script
generates a publication-quality dissociation curve figure.

Usage:
    python scripts/plot_h2_dissociation.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results/phase3_final/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load GPU results
    consolidated_path = ROOT / "results/phase3_final/consolidated_results_gic2026.json"
    with consolidated_path.open() as f:
        consolidated = json.load(f)

    gpu_results = {r["molecule"]: r for r in consolidated.get("gpu_benchmark", [])}

    # Load classical baselines
    baseline_path = ROOT / "results/phase3_final/classical_baseline_comparison.json"
    with baseline_path.open() as f:
        baselines = json.load(f)
    baseline_map = {m["molecule"]: m for m in baselines.get("molecules", [])}

    # H2 bond distances
    h2_molecules = ["h2_0.5", "h2_0.74", "h2_1.0", "h2_1.5", "h2_2.0"]
    bond_lengths = [0.5, 0.74, 1.0, 1.5, 2.0]

    # Extract energies
    fci_energies = []
    hf_energies = []
    gpu_energies = []
    gpu_errors = []
    qpu_energies = []
    qpu_errors = []

    for name in h2_molecules:
        gpu = gpu_results.get(name, {})
        base = baseline_map.get(name, {})

        fci = base.get("fci_energy") or gpu.get("reference_energy")
        hf = base.get("hf_energy")
        gpu_e = gpu.get("h_cgqe_optimized_energy")

        fci_energies.append(fci)
        hf_energies.append(hf)
        gpu_energies.append(gpu_e)
        gpu_errors.append(gpu.get("error_vs_reference_mha", 0))

    # Try to load QPU results
    qpu_path = ROOT / "results/qpu/remaining_submissions.json"
    qpu_results_path = ROOT / "results/qpu/h2_qpu_energies.json"
    has_qpu = False

    if qpu_results_path.exists():
        with qpu_results_path.open() as f:
            qpu_data = json.load(f)
        qpu_map = {}
        for r in qpu_data.get("results", []):
            qpu_map[r["molecule"]] = r
        for name in h2_molecules:
            r = qpu_map.get(name)
            if r:
                qpu_energies.append(r.get("sqd_energy") or r.get("energy"))
                qpu_errors.append(r.get("error_mha") or r.get("sqd_error_mha"))
                has_qpu = True
            else:
                qpu_energies.append(None)
                qpu_errors.append(None)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Top panel: Energy vs bond length
    ax1.plot(bond_lengths, fci_energies, "k-", linewidth=2, label="FCI (exact)", marker="o", markersize=4)
    if all(hf_energies):
        ax1.plot(bond_lengths, hf_energies, "b--", linewidth=1.5, label="Hartree-Fock", marker="s", markersize=4)
    ax1.plot(bond_lengths, gpu_energies, "r-", linewidth=2, label="H-cGQE (GPU)", marker="D", markersize=5)
    if has_qpu:
        qpu_valid = [(bl, e) for bl, e in zip(bond_lengths, qpu_energies) if e is not None]
        if qpu_valid:
            bls, qes = zip(*qpu_valid)
            ax1.plot(bls, qes, "g^--", linewidth=1.5, label="H-cGQE (QPU + SQD)", markersize=7)

    ax1.set_xlabel("H–H Bond Length (Å)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Energy (Ha)", fontsize=12, fontweight="bold")
    ax1.set_title("H₂ Dissociation Curve: GPU vs QPU vs Classical", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Error vs bond length
    ax2.axhline(y=1.6, color="gray", linestyle=":", linewidth=1, label="Chemical accuracy (1.6 mHa)")
    ax2.plot(bond_lengths, gpu_errors, "r-", linewidth=2, label="H-cGQE (GPU)", marker="D", markersize=5)
    hf_errors = [abs(h - f) * 1000 if h and f else None for h, f in zip(hf_energies, fci_energies)]
    if all(hf_errors):
        ax2.plot(bond_lengths, hf_errors, "b--", linewidth=1.5, label="HF error", marker="s", markersize=4)
    if has_qpu:
        qpu_valid = [(bl, e) for bl, e in zip(bond_lengths, qpu_errors) if e is not None]
        if qpu_valid:
            bls, qes = zip(*qpu_valid)
            ax2.plot(bls, qes, "g^--", linewidth=1.5, label="QPU + SQD error", markersize=7)

    ax2.set_xlabel("H–H Bond Length (Å)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Error vs FCI (mHa)", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIG_DIR / "fig_h2_dissociation_qpu.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # Print summary table
    print(f"\n{'Bond(Å)':>8s} {'FCI':>12s} {'HF':>12s} {'GPU':>12s} {'GPU err':>10s}")
    print("-" * 58)
    for bl, fci, hf, gpu_e, gpu_err in zip(bond_lengths, fci_energies, hf_energies, gpu_energies, gpu_errors):
        hf_str = f"{hf:.6f}" if hf else "N/A"
        print(f"{bl:>8.2f} {fci:>12.6f} {hf_str:>12s} {gpu_e:>12.6f} {gpu_err:>8.2f} mHa")
    if has_qpu:
        print(f"\n{'Bond(Å)':>8s} {'QPU':>12s} {'QPU err':>10s}")
        print("-" * 35)
        for bl, qpu_e, qpu_err in zip(bond_lengths, qpu_energies, qpu_errors):
            if qpu_e is not None:
                print(f"{bl:>8.2f} {qpu_e:>12.6f} {qpu_err:>8.2f} mHa")
    else:
        print("\n⚠ No QPU results yet. Run submit_remaining_qpu.py from qBraid first.")


if __name__ == "__main__":
    main()
