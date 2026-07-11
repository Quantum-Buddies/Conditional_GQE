#!/usr/bin/env python3
"""GIC 2026 Scaling Visualization: Energy vs qubits, diversity vs molecule size,
generalization to unseen molecules, and EUV photoresist focus plots."""

import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_json(path: str) -> dict | list | None:
    if not path or not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_hamiltonian_qubits(ham_path: str) -> dict:
    """Return {molecule_name: (n_qubits, n_pauli_terms, split)}"""
    data = load_json(ham_path)
    if data is None:
        return {}
    records = data.get('records', data) if isinstance(data, dict) else data
    result = {}
    for r in records:
        result[r['name']] = (r['n_qubits'], r['n_pauli_terms'], r.get('split', 'unknown'))
    return result


def is_entangling(word: str) -> bool:
    return any(c in word for c in "XY")


def compute_diversity(infer_path: str) -> dict:
    """Return {molecule: {diversity, mean_length, std_length, n_unique, n_total}}"""
    data = load_json(infer_path)
    if data is None:
        return {}
    result = {}
    for mol_entry in data:
        mol = mol_entry['molecule']
        seqs = [tuple(s['operators']) for s in mol_entry['generated_sequences']]
        n_unique = len(set(seqs))
        lengths = [len(s) for s in seqs]
        result[mol] = {
            'diversity': n_unique / len(seqs) if seqs else 0,
            'n_unique': n_unique,
            'n_total': len(seqs),
            'mean_length': float(np.mean(lengths)) if lengths else 0,
            'std_length': float(np.std(lengths)) if lengths else 0,
        }
    return result


def plot_energy_vs_qubits(sup_opt, chm_opt, gqe_base, ham_info, output_dir):
    """Plot best energy vs qubit count for all three methods."""
    fig, ax = plt.subplots(figsize=(12, 7))

    methods = [
        ('Supervised', sup_opt, '#2196F3', 'o'),
        ('Chemeleon2 RL', chm_opt, '#FF6F00', 's'),
        ('CUDA-Q GQE', gqe_base, '#4CAF50', '^'),
    ]

    for label, data, color, marker in methods:
        if data is None:
            continue
        records = data if isinstance(data, list) else data.get('results', [])
        qubits_list = []
        energies = []
        for r in records:
            mol = r.get('molecule', '')
            best_e = r.get('best_energy')
            if best_e is None or best_e == float('inf'):
                continue
            info = ham_info.get(mol)
            if info is None:
                continue
            nq = info[0]
            qubits_list.append(nq)
            energies.append(best_e)
        if qubits_list:
            ax.scatter(qubits_list, energies, c=color, marker=marker, s=100,
                      label=label, edgecolors='black', linewidth=0.5, zorder=3)

    ax.set_xlabel('Number of Qubits', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Energy (Hartree)', fontsize=13, fontweight='bold')
    ax.set_title('GIC 2026: Energy vs System Size', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xmargin(0.05)

    plt.tight_layout()
    out = output_dir / 'scaling_energy_vs_qubits.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_energy_error_vs_qubits(sup_opt, chm_opt, gqe_base, ham_info, output_dir):
    """Plot energy error (relative to best known) vs qubit count."""
    # Collect all energies per molecule to find reference
    all_energies = {}
    for data in [sup_opt, chm_opt, gqe_base]:
        if data is None:
            continue
        records = data if isinstance(data, list) else data.get('results', [])
        for r in records:
            mol = r.get('molecule', '')
            best_e = r.get('best_energy')
            if best_e is None or best_e == float('inf'):
                continue
            if mol not in all_energies or best_e < all_energies[mol]:
                all_energies[mol] = best_e

    fig, ax = plt.subplots(figsize=(12, 7))

    methods = [
        ('Supervised', sup_opt, '#2196F3', 'o'),
        ('Chemeleon2 RL', chm_opt, '#FF6F00', 's'),
        ('CUDA-Q GQE', gqe_base, '#4CAF50', '^'),
    ]

    for label, data, color, marker in methods:
        if data is None:
            continue
        records = data if isinstance(data, list) else data.get('results', [])
        qubits_list = []
        errors = []
        for r in records:
            mol = r.get('molecule', '')
            best_e = r.get('best_energy')
            if best_e is None or best_e == float('inf'):
                continue
            info = ham_info.get(mol)
            if info is None:
                continue
            ref = all_energies.get(mol, best_e)
            if ref != 0:
                error_mha = abs(best_e - ref) * 1000  # Convert to mHa
                qubits_list.append(info[0])
                errors.append(error_mha)
        if qubits_list:
            ax.scatter(qubits_list, errors, c=color, marker=marker, s=100,
                      label=label, edgecolors='black', linewidth=0.5, zorder=3)

    ax.axhline(y=1.6, color='red', linestyle='--', linewidth=1.5, label='Chemical accuracy (1.6 mHa)')
    ax.set_xlabel('Number of Qubits', fontsize=13, fontweight='bold')
    ax.set_ylabel('Energy Error (mHa)', fontsize=13, fontweight='bold')
    ax.set_title('GIC 2026: Energy Error vs System Size', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.yscale('log')
    ax.set_xmargin(0.05)

    plt.tight_layout()
    out = output_dir / 'scaling_energy_error_vs_qubits.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_diversity_vs_qubits(sup_div, chm_div, ham_info, output_dir):
    """Plot circuit diversity vs qubit count."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for label, div_data, color, marker in [
        ('Supervised', sup_div, '#2196F3', 'o'),
        ('Chemeleon2 RL', chm_div, '#FF6F00', 's'),
    ]:
        if not div_data:
            continue
        qubits = []
        diversities = []
        for mol, stats in div_data.items():
            info = ham_info.get(mol)
            if info is None:
                continue
            qubits.append(info[0])
            diversities.append(stats['diversity'])
        if qubits:
            ax.scatter(qubits, diversities, c=color, marker=marker, s=100,
                      label=label, edgecolors='black', linewidth=0.5, zorder=3)

    ax.set_xlabel('Number of Qubits', fontsize=13, fontweight='bold')
    ax.set_ylabel('Circuit Diversity (unique/total)', fontsize=13, fontweight='bold')
    ax.set_title('GIC 2026: Circuit Diversity vs System Size', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xmargin(0.05)

    plt.tight_layout()
    out = output_dir / 'scaling_diversity_vs_qubits.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_generalization(sup_opt, chm_opt, ham_info, output_dir):
    """Compare performance on train vs val vs test splits."""
    splits = {'train': [], 'val': [], 'test': []}

    for label, data, color in [
        ('Supervised', sup_opt, '#2196F3'),
        ('Chemeleon2', chm_opt, '#FF6F00'),
    ]:
        if data is None:
            continue
        records = data if isinstance(data, list) else data.get('results', [])
        for r in records:
            mol = r.get('molecule', '')
            best_e = r.get('best_energy')
            if best_e is None or best_e == float('inf'):
                continue
            info = ham_info.get(mol)
            if info is None:
                continue
            split = info[2]
            if split in splits:
                splits[split].append((mol, best_e, label, color))

    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = 0
    x_labels = []
    x_positions = {'Supervised': {}, 'Chemeleon2': {}}
    bar_width = 0.35

    for split in ['train', 'val', 'test']:
        split_data = splits[split]
        if not split_data:
            continue
        # Sort by molecule name
        split_data.sort(key=lambda x: x[0])
        for i, (mol, energy, label, color) in enumerate(split_data):
            offset = 0 if label == 'Supervised' else bar_width
            pos = x_pos + i * 1.0 + offset
            ax.bar(pos, energy, bar_width, color=color, edgecolor='black', linewidth=0.3)
            x_positions[label][mol] = pos

        x_labels.append((x_pos + len(split_data) * 0.5, split.upper()))
        x_pos += len(split_data) + 1.5

    ax.set_ylabel('Best Energy (Hartree)', fontsize=13, fontweight='bold')
    ax.set_title('GIC 2026: Generalization Across Data Splits', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add split labels
    for pos, label in x_labels:
        ax.text(pos, ax.get_ylim()[0], label, ha='center', va='top',
               fontsize=12, fontweight='bold', transform=ax.get_xaxis_transform())

    # Legend
    legend_elements = [
        Patch(facecolor='#2196F3', edgecolor='black', label='Supervised'),
        Patch(facecolor='#FF6F00', edgecolor='black', label='Chemeleon2 RL'),
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc='upper right')

    plt.tight_layout()
    out = output_dir / 'generalization_train_val_test.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_euv_focus(sup_opt, chm_opt, gqe_base, ham_info, output_dir):
    """Focus plot on EUV photoresist molecules."""
    euv_molecules = ['imeph_cas12', 'iodobenzene_cas12', 'methyl_iodide_cas12',
                     'phenol_cas12', 'ocresol_cas12', 'anisole_cas12',
                     'benzene_cas12', 'toluene_cas12']
    euv_labels = ['IMePh', 'Iodobenzene', 'Methyl iodide', 'Phenol',
                  'o-Cresol', 'Anisole', 'Benzene', 'Toluene']

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(euv_molecules))
    width = 0.25

    methods = [
        ('Supervised', sup_opt, '#2196F3', -width),
        ('Chemeleon2 RL', chm_opt, '#FF6F00', 0),
        ('CUDA-Q GQE', gqe_base, '#4CAF50', width),
    ]

    for label, data, color, offset in methods:
        if data is None:
            continue
        records = data if isinstance(data, list) else data.get('results', [])
        energy_map = {r['molecule']: r.get('best_energy') for r in records
                      if r.get('best_energy') is not None}
        energies = [energy_map.get(m, 0) for m in euv_molecules]
        bars = ax.bar(x + offset, energies, width, label=label, color=color,
                     edgecolor='black', linewidth=0.5)
        for bar, e in zip(bars, energies):
            if e != 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{e:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('EUV Photoresist Molecule', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Energy (Hartree)', fontsize=13, fontweight='bold')
    ax.set_title('GIC 2026: EUV Photoresist Materials (Mitsubishi Chemical Use Case)',
                fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(euv_labels, fontsize=10, rotation=15, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = output_dir / 'euv_photoresist_focus.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_training_curves(rl_metrics, output_dir):
    """Plot RL training curves."""
    if not rl_metrics:
        return
    log = rl_metrics.get('train_log', [])
    if not log:
        return

    epochs = [m['epoch'] for m in log]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('GIC 2026: Chemeleon2 RL Training (All Molecules)', fontsize=16, fontweight='bold')

    # Loss
    axes[0,0].plot(epochs, [m['mean_loss'] for m in log], 'b-', linewidth=2)
    axes[0,0].set_title('DAPO Loss'); axes[0,0].set_xlabel('Epoch'); axes[0,0].grid(True, alpha=0.3)

    # Energy
    axes[0,1].plot(epochs, [m['mean_energy'] for m in log], 'g-', label='Mean', linewidth=2)
    axes[0,1].plot(epochs, [m['min_energy'] for m in log], 'r--', label='Min', linewidth=2)
    axes[0,1].set_title('Energy'); axes[0,1].set_xlabel('Epoch'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    # Reward
    axes[0,2].plot(epochs, [m['mean_reward'] for m in log], 'purple', linewidth=2)
    axes[0,2].set_title('Mean Reward'); axes[0,2].set_xlabel('Epoch'); axes[0,2].grid(True, alpha=0.3)

    # Entropy
    axes[1,0].plot(epochs, [m['mean_entropy'] for m in log], 'orange', linewidth=2)
    axes[1,0].set_title('Policy Entropy'); axes[1,0].set_xlabel('Epoch'); axes[1,0].grid(True, alpha=0.3)

    # mSUN
    axes[1,1].plot(epochs, [m['msun'] for m in log], 'k-', label='mSUN', linewidth=2)
    axes[1,1].plot(epochs, [m['msun_converged'] for m in log], 'g--', label='Conv', linewidth=1.5)
    axes[1,1].plot(epochs, [m['msun_unique'] for m in log], 'b--', label='Unique', linewidth=1.5)
    axes[1,1].plot(epochs, [m['msun_novel'] for m in log], 'r--', label='Novel', linewidth=1.5)
    axes[1,1].set_title('mSUN Metrics'); axes[1,1].set_xlabel('Epoch'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    # Buffer
    axes[1,2].plot(epochs, [m['buffer_size'] for m in log], 'brown', linewidth=2)
    axes[1,2].set_title('Replay Buffer Size'); axes[1,2].set_xlabel('Epoch'); axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / 'gic2026_training_curves.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def print_summary(sup_opt, chm_opt, gqe_base, sup_div, chm_div, ham_info, output_dir):
    """Print comprehensive text summary."""
    print("\n" + "=" * 100)
    print("GIC 2026 SCALING RESULTS SUMMARY")
    print("=" * 100)

    # Collect all results
    all_mols = sorted(ham_info.keys())
    print(f"\n{'Molecule':30s} {'Qubits':>6s} {'Terms':>6s} {'Split':>6s} "
          f"{'Sup E':>12s} {'Chm E':>12s} {'GQE E':>12s} {'Sup Div':>8s} {'Chm Div':>8s}")
    print("-" * 110)

    for mol in all_mols:
        nq, nt, split = ham_info[mol]
        sup_e = chm_e = gqe_e = "—"
        sup_d = chm_d = "—"

        if sup_opt:
            for r in (sup_opt if isinstance(sup_opt, list) else sup_opt.get('results', [])):
                if r.get('molecule') == mol and r.get('best_energy') is not None:
                    sup_e = f"{r['best_energy']:.4f}"
                    break
        if chm_opt:
            for r in (chm_opt if isinstance(chm_opt, list) else chm_opt.get('results', [])):
                if r.get('molecule') == mol and r.get('best_energy') is not None:
                    chm_e = f"{r['best_energy']:.4f}"
                    break
        if gqe_base:
            for r in (gqe_base if isinstance(gqe_base, list) else gqe_base.get('results', [])):
                if r.get('molecule') == mol and r.get('best_energy') is not None:
                    gqe_e = f"{r['best_energy']:.4f}"
                    break
        if mol in sup_div:
            sup_d = f"{sup_div[mol]['diversity']:.3f}"
        if mol in chm_div:
            chm_d = f"{chm_div[mol]['diversity']:.3f}"

        print(f"{mol:30s} {nq:6d} {nt:6d} {split:6s} {sup_e:>12s} {chm_e:>12s} {gqe_e:>12s} {sup_d:>8s} {chm_d:>8s}")

    # EUV focus
    euv_mols = ['imeph_cas12', 'iodobenzene_cas12', 'methyl_iodide_cas12', 'phenol_cas12']
    print(f"\nEUV Photoresist Focus:")
    for mol in euv_mols:
        if mol in ham_info:
            nq = ham_info[mol][0]
            print(f"  {mol:25s} ({nq}q): trained on={'YES' if ham_info[mol][2]=='train' else 'NO (generalization)'}")


def main():
    parser = argparse.ArgumentParser(description='GIC 2026 Scaling Plots')
    parser.add_argument('--supervised-opt', type=str, required=True)
    parser.add_argument('--chemeleon2-opt', type=str, required=True)
    parser.add_argument('--gqe-baseline', type=str, default=None)
    parser.add_argument('--rl-metrics', type=str, default=None)
    parser.add_argument('--supervised-infer', type=str, default=None)
    parser.add_argument('--chemeleon2-infer', type=str, default=None)
    parser.add_argument('--hamiltonians', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results/plots/gic2026')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    sup_opt = load_json(args.supervised_opt)
    chm_opt = load_json(args.chemeleon2_opt)
    gqe_base = load_json(args.gqe_baseline)
    rl_metrics = load_json(args.rl_metrics)
    ham_info = load_hamiltonian_qubits(args.hamiltonians)
    sup_div = compute_diversity(args.supervised_infer) if args.supervised_infer else {}
    chm_div = compute_diversity(args.chemeleon2_infer) if args.chemeleon2_infer else {}

    print(f"  Hamiltonians: {len(ham_info)} molecules")
    print(f"  Supervised opt: {len(sup_opt) if sup_opt else 0} results")
    print(f"  Chemeleon2 opt: {len(chm_opt) if chm_opt else 0} results")
    print(f"  GQE baseline: {len(gqe_base) if gqe_base else 0} results")

    print("\nGenerating plots...")
    plot_energy_vs_qubits(sup_opt, chm_opt, gqe_base, ham_info, output_dir)
    plot_energy_error_vs_qubits(sup_opt, chm_opt, gqe_base, ham_info, output_dir)
    plot_diversity_vs_qubits(sup_div, chm_div, ham_info, output_dir)
    plot_generalization(sup_opt, chm_opt, ham_info, output_dir)
    plot_euv_focus(sup_opt, chm_opt, gqe_base, ham_info, output_dir)
    if rl_metrics:
        plot_training_curves(rl_metrics, output_dir)

    print_summary(sup_opt, chm_opt, gqe_base, sup_div, chm_div, ham_info, output_dir)
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
