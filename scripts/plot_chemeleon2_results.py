#!/usr/bin/env python3
"""Plot Chemeleon2 RL training results from metrics JSON."""

import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster

def load_metrics(metrics_path: str) -> tuple[list[dict[str, Any]], int]:
    """Load training metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data['train_log'], data.get('final_buffer_size', 0)

def plot_training_curves(metrics: list[dict[str, Any]], output_dir: str):
    """Plot training curves: loss, energy, reward, entropy, mSUN."""
    epochs = [m['epoch'] for m in metrics]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Chemeleon2 RL Training Curves', fontsize=16, fontweight='bold')
    
    # 1. Loss
    losses = [m['mean_loss'] for m in metrics]
    axes[0, 0].plot(epochs, losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('DAPO Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mean Energy
    mean_energies = [m['mean_energy'] for m in metrics]
    min_energies = [m['min_energy'] for m in metrics]
    axes[0, 1].plot(epochs, mean_energies, 'g-', label='Mean', linewidth=2)
    axes[0, 1].plot(epochs, min_energies, 'r--', label='Min', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Energy (Hartree)')
    axes[0, 1].set_title('Energy Progression')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Reward
    rewards = [m['mean_reward'] for m in metrics]
    axes[0, 2].plot(epochs, rewards, 'purple', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Reward')
    axes[0, 2].set_title('Mean Reward')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Entropy
    entropies = [m['mean_entropy'] for m in metrics]
    axes[1, 0].plot(epochs, entropies, 'orange', linewidth=2)
    axes[1, 0].axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label='Target (1.5)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Entropy (nats)')
    axes[1, 0].set_title('Policy Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. mSUN Metrics
    msun = [m['msun'] for m in metrics]
    converged = [m['msun_converged'] for m in metrics]
    unique = [m['msun_unique'] for m in metrics]
    novel = [m['msun_novel'] for m in metrics]
    
    axes[1, 1].plot(epochs, msun, 'k-', label='mSUN', linewidth=2)
    axes[1, 1].plot(epochs, converged, 'g--', label='Converged', linewidth=1.5)
    axes[1, 1].plot(epochs, unique, 'b--', label='Unique', linewidth=1.5)
    axes[1, 1].plot(epochs, novel, 'r--', label='Novel', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Fraction')
    axes[1, 1].set_title('mSUN Metrics (Chemeleon2)')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Buffer Size
    buffer_sizes = [m['buffer_size'] for m in metrics]
    axes[1, 2].plot(epochs, buffer_sizes, 'brown', linewidth=2)
    axes[1, 2].axhline(y=1000, color='r', linestyle='--', alpha=0.5, label='Max (1000)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Buffer Size')
    axes[1, 2].set_title('Replay Buffer Size')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'chemeleon2_training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    plt.close()

def plot_per_molecule_energies(metrics: list[dict[str, Any]], output_dir: str):
    """Plot best energy progression per molecule."""
    molecules = ['h2', 'lih', 'beh2', 'n2']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mol, color in zip(molecules, colors):
        energies = [m['best_energies'][mol] for m in metrics]
        ax.plot([m['epoch'] for m in metrics], energies, 
                color=color, label=mol.upper(), linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Best Energy (Hartree)')
    ax.set_title('Best Energy per Molecule (Chemeleon2 RL)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'chemeleon2_per_molecule_energies.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-molecule energies to {output_path}")
    plt.close()

def plot_curriculum_stages(metrics: list[dict[str, Any]], output_dir: str):
    """Plot curriculum learning stages."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    epochs = [m['epoch'] for m in metrics]
    mean_energies = [m['mean_energy'] for m in metrics]
    
    ax.plot(epochs, mean_energies, 'b-', linewidth=2, label='Mean Energy')
    
    # Mark curriculum stages
    warmup = 30
    ax.axvline(x=warmup, color='r', linestyle='--', alpha=0.7, label='Stage 0→1')
    ax.axvline(x=2*warmup, color='g', linestyle='--', alpha=0.7, label='Stage 1→2')
    
    # Annotate stages
    ax.text(warmup/2, ax.get_ylim()[0], 'Stage 0\n(H2 only)', 
            ha='center', va='bottom', fontsize=9, alpha=0.7)
    ax.text(1.5*warmup, ax.get_ylim()[0], 'Stage 1\n(H2 + LiH)', 
            ha='center', va='bottom', fontsize=9, alpha=0.7)
    ax.text(2.5*warmup, ax.get_ylim()[0], 'Stage 2\n(All molecules)', 
            ha='center', va='bottom', fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Energy (Hartree)')
    ax.set_title('Curriculum Learning Stages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'chemeleon2_curriculum.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved curriculum plot to {output_path}")
    plt.close()

def print_summary(metrics: list[dict[str, Any]], final_buffer_size: int):
    """Print training summary statistics."""
    final = metrics[-1]
    print("\n" + "="*60)
    print("CHEMELEON2 RL TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs: {len(metrics)}")
    print(f"Final loss: {final['mean_loss']:.4f}")
    print(f"Final mean energy: {final['mean_energy']:.4f} Ha")
    print(f"Final min energy: {final['min_energy']:.4f} Ha")
    print(f"Final reward: {final['mean_reward']:.4f}")
    print(f"Final entropy: {final['mean_entropy']:.4f} nats")
    print(f"Final mSUN: {final['msun']:.3f}")
    print(f"  - Converged: {final['msun_converged']:.3f}")
    print(f"  - Unique: {final['msun_unique']:.3f}")
    print(f"  - Novel: {final['msun_novel']:.3f}")
    print(f"Replay buffer: {final['buffer_size']}/{final_buffer_size}")
    print(f"Skipped batches: {final['n_skipped']}")
    print("\nBest energies per molecule:")
    for mol, e in final['best_energies'].items():
        print(f"  {mol.upper():6s}: {e:.6f} Ha")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Plot Chemeleon2 RL training results')
    parser.add_argument('--metrics', type=str, 
                        default='/scratch/kcwp264/Conditional-GQE_materials/results/train/h_cgqe_rl_chemeleon2_1gpu_rl_metrics.json',
                        help='Path to metrics JSON file')
    parser.add_argument('--output-dir', type=str,
                        default='/scratch/kcwp264/Conditional-GQE_materials/results/plots',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Load metrics
    metrics, final_buffer_size = load_metrics(args.metrics)
    print(f"Loaded {len(metrics)} epochs of training data")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_training_curves(metrics, args.output_dir)
    plot_per_molecule_energies(metrics, args.output_dir)
    plot_curriculum_stages(metrics, args.output_dir)
    
    # Print summary
    print_summary(metrics, final_buffer_size)
    
    print(f"\nAll plots saved to {args.output_dir}")

if __name__ == '__main__':
    main()
