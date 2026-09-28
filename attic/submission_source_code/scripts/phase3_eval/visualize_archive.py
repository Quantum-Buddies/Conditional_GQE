"""Visualize MAP-Elites archive as publication-quality heatmap.

Generates a 2D heatmap of the MAP-Elites archive showing elite energies
across the entanglement_density × circuit_depth feature space. This is
the key figure for the paper: it *is* the illumination of the quantum
circuit fitness landscape.

Usage:
    python scripts/phase3_eval/visualize_archive.py \
        --archive results/train/h_cgqe_rl_dapo_model_map_elites.json \
        --hf-energy -7.431 \
        --fci-energy -7.478 \
        --out results/eval/map_elites_heatmap.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def load_archive(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_heatmap_data(
    archive_data: dict,
    n_bins_e: int,
    n_bins_d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 2D arrays for heatmap: energies and occupancy mask."""
    energies = np.full((n_bins_e, n_bins_d), np.nan)
    occupied = np.zeros((n_bins_e, n_bins_d), dtype=bool)

    grid = archive_data.get("grid", {})
    for cell_key, entry in grid.items():
        e_bin, d_bin = map(int, cell_key.split("_"))
        if 0 <= e_bin < n_bins_e and 0 <= d_bin < n_bins_d:
            energies[e_bin, d_bin] = entry["energy"]
            occupied[e_bin, d_bin] = True

    return energies, occupied


def plot_archive_heatmap(
    archive_data: dict,
    hf_energy: float | None = None,
    fci_energy: float | None = None,
    molecule_name: str = "",
    out_path: str = "map_elites_heatmap.png",
) -> None:
    """Generate publication-quality MAP-Elites heatmap."""
    n_bins_e = archive_data.get("n_bins_e", 10)
    n_bins_d = archive_data.get("n_bins_d", 10)
    energies, occupied = build_heatmap_data(archive_data, n_bins_e, n_bins_d)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 1]})

    # --- Panel 1: Energy heatmap ---
    ax = axes[0]

    # Mask unoccupied cells
    masked_energies = np.ma.array(energies, mask=~occupied)

    # Color normalization: center on HF energy if available
    if hf_energy is not None and fci_energy is not None:
        vmin = min(fci_energy, np.nanmin(energies)) - 0.01
        vmax = max(hf_energy, np.nanmax(energies)) + 0.01
    else:
        valid = energies[occupied]
        vmin, vmax = valid.min(), valid.max()

    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="#f0f0f0")  # light gray for unoccupied cells

    im = ax.imshow(
        masked_energies, cmap=cmap, aspect="auto",
        origin="lower", vmin=vmin, vmax=vmax,
        extent=[0, 1, 0, 1],
    )

    # Add energy values in occupied cells
    for e_bin in range(n_bins_e):
        for d_bin in range(n_bins_d):
            if occupied[e_bin, d_bin]:
                e_val = energies[e_bin, d_bin]
                x = (d_bin + 0.5) / n_bins_d
                y = (e_bin + 0.5) / n_bins_e
                color = "white" if abs(e_val - vmin) > abs(e_val - vmax) * 0.5 else "black"
                ax.text(x, y, f"{e_val:.3f}", ha="center", va="center",
                        fontsize=7, fontweight="bold", color=color)

    # Reference energy lines drawn on colorbar after it's created (below)
    ref_lines = []
    if hf_energy is not None:
        ref_lines.append(("HF", hf_energy, "blue"))
    if fci_energy is not None:
        ref_lines.append(("FCI", fci_energy, "green"))

    ax.set_xlabel("Circuit Depth (normalized)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Entanglement Density", fontsize=14, fontweight="bold")
    title = "MAP-Elites Archive: Elite Circuit Energies"
    if molecule_name:
        title += f" ({molecule_name})"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Energy (Hartree)", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # Draw HF/FCI reference lines on the colorbar (energy axis, not entanglement axis)
    for label, ref_e, color in ref_lines:
        if vmin <= ref_e <= vmax:
            cbar.ax.axhline(y=ref_e, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
            cbar.ax.text(vmax + 0.002, ref_e, f"{label}={ref_e:.3f}",
                         fontsize=8, va="center", color=color)

    # --- Panel 2: Archive statistics ---
    ax2 = axes[1]
    ax2.axis("off")

    summary = archive_data.get("summary", {})
    n_elites = summary.get("n_elites", int(occupied.sum()))
    coverage = summary.get("coverage", occupied.sum() / (n_bins_e * n_bins_d))
    best_e = summary.get("best_energy", np.nanmin(energies))
    mean_e = summary.get("mean_energy", np.nanmean(energies))
    qd_score = summary.get("qd_score", 0.0)
    mean_ent = summary.get("mean_entanglement", 0.0)
    mean_depth = summary.get("mean_depth", 0.0)

    stats_text = (
        f"Archive Statistics\n"
        f"{'─' * 30}\n\n"
        f"Grid: {n_bins_e} × {n_bins_d}\n"
        f"Total cells: {n_bins_e * n_bins_d}\n"
        f"Occupied cells: {n_elites}\n"
        f"Coverage: {coverage:.1%}\n\n"
        f"Best energy: {best_e:.6f}\n"
        f"Mean energy: {mean_e:.6f}\n"
        f"QD-Score: {qd_score:.4f}\n\n"
        f"Mean entanglement: {mean_ent:.3f}\n"
        f"Mean depth: {mean_depth:.3f}\n"
    )

    if hf_energy is not None:
        err_best = abs(best_e - hf_energy) * 1000
        stats_text += f"\nHF energy: {hf_energy:.6f}\n"
        stats_text += f"Best vs HF: {err_best:.2f} mHa\n"
    if fci_energy is not None:
        err_fci = abs(best_e - fci_energy) * 1000
        stats_text += f"FCI energy: {fci_energy:.6f}\n"
        stats_text += f"Best vs FCI: {err_fci:.2f} mHa\n"

    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"))

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to: {out_path}")
    plt.close(fig)


def plot_archive_evolution(
    metrics_path: str,
    out_path: str = "archive_evolution.png",
) -> None:
    """Plot archive coverage and QD-score evolution over training epochs."""
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    train_log = metrics.get("train_log", [])
    if not train_log:
        print("No training log found in metrics file.")
        return

    epochs = [e["epoch"] for e in train_log]
    coverages = [e.get("qd_coverage", 0.0) for e in train_log]
    sizes = [e.get("qd_archive_size", 0) for e in train_log]
    lambdas = [e.get("qd_lambda", 0.0) for e in train_log]
    cache_rates = [e.get("qd_cache_hit_rate", 0.0) for e in train_log]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Coverage
    ax = axes[0, 0]
    ax.plot(epochs, coverages, "b-", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Archive Coverage", fontsize=12, fontweight="bold")
    ax.set_title("MAP-Elites Coverage Over Training", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Archive size
    ax = axes[0, 1]
    ax.plot(epochs, sizes, "g-", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Elite Circuits", fontsize=12, fontweight="bold")
    ax.set_title("Archive Size Over Training", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Adaptive lambda
    ax = axes[1, 0]
    ax.plot(epochs, lambdas, "r-", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Novelty Weight (λ)", fontsize=12, fontweight="bold")
    ax.set_title("Adaptive λ Schedule", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Cache hit rate
    ax = axes[1, 1]
    ax.plot(epochs, cache_rates, "m-", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cache Hit Rate", fontsize=12, fontweight="bold")
    ax.set_title("Dedup Cache Hit Rate", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.suptitle("QD-GRPO Training Dynamics", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Evolution plot saved to: {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MAP-Elites archive")
    parser.add_argument("--archive", type=str, required=True,
                        help="Path to MAP-Elites archive JSON file or directory of per-molecule archives")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Path to RL metrics JSON (for evolution plot)")
    parser.add_argument("--hf-energy", type=float, default=None,
                        help="Hartree-Fock reference energy")
    parser.add_argument("--fci-energy", type=float, default=None,
                        help="FCI reference energy")
    parser.add_argument("--molecule", type=str, default="",
                        help="Molecule name for title (used with single archive file)")
    parser.add_argument("--out", type=str, default="results/eval/map_elites_heatmap.png",
                        help="Output heatmap path (or directory for per-molecule archives)")
    parser.add_argument("--out-evolution", type=str, default=None,
                        help="Output evolution plot path (requires --metrics)")
    args = parser.parse_args()

    archive_path = Path(args.archive)

    # Support both single archive file and directory of per-molecule archives
    if archive_path.is_dir():
        archive_files = sorted(archive_path.glob("map_elites_*.json"))
        if not archive_files:
            print(f"No map_elites_*.json files found in {archive_path}")
            sys.exit(1)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for af in archive_files:
            mol_name = af.stem.replace("map_elites_", "")
            archive_data = load_archive(str(af))
            out_path = str(out_dir / f"heatmap_{mol_name}.png")
            plot_archive_heatmap(
                archive_data,
                hf_energy=args.hf_energy,
                fci_energy=args.fci_energy,
                molecule_name=mol_name,
                out_path=out_path,
            )
    else:
        archive_data = load_archive(str(archive_path))
        plot_archive_heatmap(
            archive_data,
            hf_energy=args.hf_energy,
            fci_energy=args.fci_energy,
            molecule_name=args.molecule,
            out_path=args.out,
        )

    if args.metrics and Path(args.metrics).exists():
        out_evo = args.out_evolution or str(Path(args.out).parent / "archive_evolution.png")
        plot_archive_evolution(args.metrics, out_path=out_evo)


if __name__ == "__main__":
    main()
