#!/usr/bin/env python3
"""Comprehensive Chemeleon2 results visualization: Stage 2 energies + training curves + diversity comparison."""

import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_stage2_results(project_root: Path) -> dict:
    """Load Stage 2 optimization results for supervised and Chemeleon2."""
    results = {}
    for label, fname in [("Supervised", "supervised_stage2_optimized.json"),
                         ("Chemeleon2", "chemeleon2_stage2_optimized.json")]:
        path = project_root / "results" / "eval" / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            results[label] = {r["molecule"]: r for r in data}
    return results

def load_training_metrics(project_root: Path) -> list:
    """Load Chemeleon2 RL training metrics."""
    path = project_root / "results" / "train" / "h_cgqe_rl_chemeleon2_1gpu_rl_metrics.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("train_log", [])

def load_diversity_results(project_root: Path) -> dict:
    """Load diversity analysis from generated circuit JSONs."""
    results = {}
    for label, fname in [("Supervised", "supervised_sampled.json"),
                         ("Chemeleon2", "chemeleon2_sampled.json")]:
        path = project_root / "results" / "inference" / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        mol_data = {}
        for mol_entry in data:
            mol = mol_entry["molecule"]
            seqs = [tuple(s["operators"]) for s in mol_entry["generated_sequences"]]
            n_unique = len(set(seqs))
            diversity = n_unique / len(seqs) if seqs else 0
            lengths = [len(s) for s in seqs]
            mol_data[mol] = {
                "diversity": diversity,
                "n_unique": n_unique,
                "n_total": len(seqs),
                "mean_length": float(np.mean(lengths)) if lengths else 0,
                "std_length": float(np.std(lengths)) if lengths else 0,
            }
        results[label] = mol_data
    return results

def plot_stage2_energies(stage2: dict, output_dir: Path):
    """Bar chart comparing best optimized energies."""
    molecules = ["h2", "lih", "beh2", "n2"]
    labels = ["H2", "LiH", "BeH2", "N2"]
    x = np.arange(len(molecules))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Best energies
    ax = axes[0]
    sup_best = [stage2.get("Supervised", {}).get(m, {}).get("best_energy", 0) for m in molecules]
    chm_best = [stage2.get("Chemeleon2", {}).get(m, {}).get("best_energy", 0) for m in molecules]

    bars1 = ax.bar(x - width/2, sup_best, width, label='Supervised', color='#2196F3', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, chm_best, width, label='Chemeleon2 RL', color='#FF6F00', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Molecule', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Energy (Hartree)', fontsize=12, fontweight='bold')
    ax.set_title('Stage 2: Best Optimized Energies', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Mean energies with error bars
    ax = axes[1]
    sup_means = []
    sup_stds = []
    chm_means = []
    chm_stds = []
    for m in molecules:
        sup_all = stage2.get("Supervised", {}).get(m, {}).get("all_optimized_energies", [])
        chm_all = stage2.get("Chemeleon2", {}).get(m, {}).get("all_optimized_energies", [])
        sup_clean = [e for e in sup_all if e != float('inf')]
        chm_clean = [e for e in chm_all if e != float('inf')]
        sup_means.append(np.mean(sup_clean) if sup_clean else 0)
        sup_stds.append(np.std(sup_clean) if sup_clean else 0)
        chm_means.append(np.mean(chm_clean) if chm_clean else 0)
        chm_stds.append(np.std(chm_clean) if chm_clean else 0)

    bars1 = ax.bar(x - width/2, sup_means, width, yerr=sup_stds, label='Supervised',
                   color='#2196F3', edgecolor='black', linewidth=0.5, capsize=3)
    bars2 = ax.bar(x + width/2, chm_means, width, yerr=chm_stds, label='Chemeleon2 RL',
                   color='#FF6F00', edgecolor='black', linewidth=0.5, capsize=3)
    ax.set_xlabel('Molecule', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Energy (Hartree)', fontsize=12, fontweight='bold')
    ax.set_title('Stage 2: Mean Optimized Energies (±std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = output_dir / "stage2_energy_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

def plot_energy_delta(stage2: dict, output_dir: Path):
    """Plot energy improvement delta (Chemeleon2 - Supervised)."""
    molecules = ["h2", "lih", "beh2", "n2"]
    labels = ["H2", "LiH", "BeH2", "N2"]
    deltas = []
    for m in molecules:
        sup = stage2.get("Supervised", {}).get(m, {}).get("best_energy", 0)
        chm = stage2.get("Chemeleon2", {}).get(m, {}).get("best_energy", 0)
        deltas.append(chm - sup)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#4CAF50' if d < 0 else '#F44336' for d in deltas]
    bars = ax.bar(labels, deltas, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Molecule', fontsize=12, fontweight='bold')
    ax.set_ylabel('Δ Energy (Chemeleon2 − Supervised, Hartree)', fontsize=12, fontweight='bold')
    ax.set_title('Chemeleon2 Energy Improvement vs Supervised', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, d in zip(bars, deltas):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., y + (0.0005 if y >= 0 else -0.0005),
                f'{d:+.4f}', ha='center', va='bottom' if y >= 0 else 'top', fontsize=11, fontweight='bold')

    ax.text(0.02, 0.95, 'Green = Chemeleon2 better\n(lower energy)', transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out = output_dir / "stage2_energy_delta.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

def plot_diversity_comparison(diversity: dict, output_dir: Path):
    """Bar chart comparing diversity metrics."""
    molecules = ["h2", "lih", "beh2", "n2"]
    labels = ["H2", "LiH", "BeH2", "N2"]
    x = np.arange(len(molecules))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Diversity
    ax = axes[0]
    sup_div = [diversity.get("Supervised", {}).get(m, {}).get("diversity", 0) for m in molecules]
    chm_div = [diversity.get("Chemeleon2", {}).get(m, {}).get("diversity", 0) for m in molecules]

    bars1 = ax.bar(x - width/2, sup_div, width, label='Supervised', color='#2196F3', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, chm_div, width, label='Chemeleon2 RL', color='#FF6F00', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Molecule', fontsize=12, fontweight='bold')
    ax.set_ylabel('Diversity (unique/total)', fontsize=12, fontweight='bold')
    ax.set_title('Circuit Diversity Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # Mean sequence length
    ax = axes[1]
    sup_len = [diversity.get("Supervised", {}).get(m, {}).get("mean_length", 0) for m in molecules]
    chm_len = [diversity.get("Chemeleon2", {}).get(m, {}).get("mean_length", 0) for m in molecules]
    sup_std = [diversity.get("Supervised", {}).get(m, {}).get("std_length", 0) for m in molecules]
    chm_std = [diversity.get("Chemeleon2", {}).get(m, {}).get("std_length", 0) for m in molecules]

    ax.bar(x - width/2, sup_len, width, yerr=sup_std, label='Supervised',
           color='#2196F3', edgecolor='black', linewidth=0.5, capsize=3)
    ax.bar(x + width/2, chm_len, width, yerr=chm_std, label='Chemeleon2 RL',
           color='#FF6F00', edgecolor='black', linewidth=0.5, capsize=3)
    ax.set_xlabel('Molecule', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Sequence Length', fontsize=12, fontweight='bold')
    ax.set_title('Circuit Length Comparison (±std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = output_dir / "diversity_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

def plot_training_curves(metrics: list, output_dir: Path):
    """Plot RL training curves from metrics."""
    if not metrics:
        print("No training metrics found, skipping training curves")
        return

    epochs = [m["epoch"] for m in metrics]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Chemeleon2 RL Training Curves (1-GPU, 200 epochs)", fontsize=16, fontweight='bold')

    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, [m["mean_loss"] for m in metrics], 'b-', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('DAPO Loss'); ax.grid(True, alpha=0.3)

    # 2. Energy
    ax = axes[0, 1]
    ax.plot(epochs, [m["mean_energy"] for m in metrics], 'g-', label='Mean', linewidth=2)
    ax.plot(epochs, [m["min_energy"] for m in metrics], 'r--', label='Min', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Energy (Hartree)')
    ax.set_title('Energy Progression'); ax.legend(); ax.grid(True, alpha=0.3)

    # 3. Reward
    ax = axes[0, 2]
    ax.plot(epochs, [m["mean_reward"] for m in metrics], 'purple', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Reward')
    ax.set_title('Mean Reward'); ax.grid(True, alpha=0.3)

    # 4. Entropy
    ax = axes[1, 0]
    ax.plot(epochs, [m["mean_entropy"] for m in metrics], 'orange', linewidth=2)
    ax.axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label='Target (1.5)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Entropy (nats)')
    ax.set_title('Policy Entropy'); ax.legend(); ax.grid(True, alpha=0.3)

    # 5. mSUN
    ax = axes[1, 1]
    ax.plot(epochs, [m["msun"] for m in metrics], 'k-', label='mSUN', linewidth=2)
    ax.plot(epochs, [m["msun_converged"] for m in metrics], 'g--', label='Converged', linewidth=1.5)
    ax.plot(epochs, [m["msun_unique"] for m in metrics], 'b--', label='Unique', linewidth=1.5)
    ax.plot(epochs, [m["msun_novel"] for m in metrics], 'r--', label='Novel', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Fraction')
    ax.set_title('mSUN Metrics'); ax.legend(); ax.set_ylim([0, 1.05]); ax.grid(True, alpha=0.3)

    # 6. Best energies per molecule
    ax = axes[1, 2]
    molecules = ["h2", "lih", "beh2", "n2"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for mol, color in zip(molecules, colors):
        energies = [m["best_energies"][mol] for m in metrics]
        ax.plot(epochs, energies, color=color, label=mol.upper(), linewidth=2, marker='o', markersize=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Best Energy (Hartree)')
    ax.set_title('Best Energy per Molecule'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "chemeleon2_training_curves.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

def plot_summary_table(stage2: dict, diversity: dict, metrics: list, output_dir: Path):
    """Create a summary table figure."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    molecules = ["h2", "lih", "beh2", "n2"]
    rows = []
    for m in molecules:
        sup_e = stage2.get("Supervised", {}).get(m, {}).get("best_energy", 0)
        chm_e = stage2.get("Chemeleon2", {}).get(m, {}).get("best_energy", 0)
        delta = chm_e - sup_e
        sup_div = diversity.get("Supervised", {}).get(m, {}).get("diversity", 0)
        chm_div = diversity.get("Chemeleon2", {}).get(m, {}).get("diversity", 0)
        div_improvement = (chm_div - sup_div) / sup_div * 100 if sup_div > 0 else 0
        rows.append([m.upper(), f"{sup_e:.4f}", f"{chm_e:.4f}", f"{delta:+.4f}",
                      f"{sup_div:.3f}", f"{chm_div:.3f}", f"+{div_improvement:.0f}%"])

    # Add header
    header = ["Molecule", "Sup E (Ha)", "Chm E (Ha)", "Δ Energy",
              "Sup Div", "Chm Div", "Div Improvement"]
    rows.insert(0, header)

    table = ax.table(cellText=rows, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(header)):
        table[0, j].set_facecolor('#4CAF50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color code delta column
    for i in range(1, len(rows)):
        delta_val = float(rows[i][3])
        if delta_val < 0:
            table[i, 3].set_facecolor('#C8E6C9')
        else:
            table[i, 3].set_facecolor('#FFCDD2')

    ax.set_title("Chemeleon2 RL vs Supervised: Complete Results Summary",
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    out = output_dir / "results_summary_table.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default="/scratch/kcwp264/Conditional-GQE_materials")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    stage2 = load_stage2_results(project_root)
    metrics = load_training_metrics(project_root)
    diversity = load_diversity_results(project_root)

    print(f"  Stage 2 results: {list(stage2.keys())}")
    print(f"  Training metrics: {len(metrics)} epochs")
    print(f"  Diversity data: {list(diversity.keys())}")

    print("\nGenerating plots...")
    if stage2:
        plot_stage2_energies(stage2, output_dir)
        plot_energy_delta(stage2, output_dir)
    if diversity:
        plot_diversity_comparison(diversity, output_dir)
    if metrics:
        plot_training_curves(metrics, output_dir)
    if stage2 and diversity:
        plot_summary_table(stage2, diversity, metrics, output_dir)

    # Print text summary
    print("\n" + "=" * 80)
    print("COMPLETE RESULTS SUMMARY")
    print("=" * 80)
    if stage2:
        print("\nStage 2 Optimized Energies:")
        print(f"  {'Molecule':10s} {'Supervised':>14s} {'Chemeleon2':>14s} {'Delta':>10s}")
        for m in ["h2", "lih", "beh2", "n2"]:
            sup = stage2.get("Supervised", {}).get(m, {}).get("best_energy", 0)
            chm = stage2.get("Chemeleon2", {}).get(m, {}).get("best_energy", 0)
            print(f"  {m.upper():10s} {sup:14.6f} {chm:14.6f} {chm-sup:+10.6f}")

    if diversity:
        print("\nCircuit Diversity:")
        print(f"  {'Molecule':10s} {'Supervised':>14s} {'Chemeleon2':>14s} {'Improvement':>12s}")
        for m in ["h2", "lih", "beh2", "n2"]:
            sup = diversity.get("Supervised", {}).get(m, {}).get("diversity", 0)
            chm = diversity.get("Chemeleon2", {}).get(m, {}).get("diversity", 0)
            imp = (chm - sup) / sup * 100 if sup > 0 else 0
            print(f"  {m.upper():10s} {sup:14.3f} {chm:14.3f} {imp:+11.1f}%")

    if metrics:
        final = metrics[-1]
        print(f"\nTraining Final Epoch ({final['epoch']}):")
        print(f"  Loss: {final['mean_loss']:.4f}")
        print(f"  Entropy: {final['mean_entropy']:.4f}")
        print(f"  mSUN: {final['msun']:.3f}")
        print(f"  Best energies: {final['best_energies']}")

    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
