from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _plot_curves(summary: dict[str, Any], out_path: Path) -> None:
    graph = summary.get("graph", {})
    flat = summary.get("flat", {})
    graph_train = [float(x) for x in graph.get("train_loss_history", [])]
    flat_train = [float(x) for x in flat.get("train_loss_history", [])]
    graph_val = [float(x) for x in graph.get("val_loss_history", [])]
    flat_val = [float(x) for x in flat.get("val_loss_history", [])]

    plt.figure(figsize=(9, 4))
    if graph_train:
        plt.plot(range(1, len(graph_train) + 1), graph_train, marker="o", label="graph train", color="#4c72b0")
    if flat_train:
        plt.plot(range(1, len(flat_train) + 1), flat_train, marker="o", label="flat train", color="#dd8452")
    if graph_val:
        plt.plot(range(1, len(graph_val) + 1), graph_val, marker="s", linestyle="--", label="graph val", color="#1f77b4")
    if flat_val:
        plt.plot(range(1, len(flat_val) + 1), flat_val, marker="s", linestyle="--", label="flat val", color="#ff7f0e")
    plt.title("Chemistry-conditioning ablation: training trajectories")
    plt.xlabel("epoch")
    plt.ylabel("standardized regression loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_final_metrics(summary: dict[str, Any], out_path: Path) -> None:
    graph = summary.get("graph", {})
    flat = summary.get("flat", {})
    comparison = summary.get("comparison", {})
    metrics = {
        "graph val MAE": float(graph.get("final_val_mae", 0.0) or 0.0),
        "flat val MAE": float(flat.get("final_val_mae", 0.0) or 0.0),
        "graph val MSE": float(graph.get("final_val_mse", 0.0) or 0.0),
        "flat val MSE": float(flat.get("final_val_mse", 0.0) or 0.0),
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#4c72b0", "#dd8452", "#1f77b4", "#ff7f0e"]

    plt.figure(figsize=(9, 4))
    bars = plt.bar(labels, values, color=colors)
    plt.title("Chemistry-conditioning ablation: final validation metrics")
    plt.ylabel("metric value")
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    if comparison:
        plt.figtext(
            0.5,
            -0.02,
            f"graph-minus-flat delta: val MAE={comparison.get('final_val_mae_delta'):.3g}, val MSE={comparison.get('final_val_mse_delta'):.3g}",
            ha="center",
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot graph-vs-flat conditioning ablation results.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Comparison JSON produced by compare_conditioning_ablation.py")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for plots")
    parser.add_argument("--manifest", type=Path, required=True, help="Output manifest JSON")
    args = parser.parse_args()

    summary = _read_json(args.summary_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curve_plot = args.out_dir / "conditioning_ablation_curves.png"
    metric_plot = args.out_dir / "conditioning_ablation_final_metrics.png"
    _plot_curves(summary, curve_plot)
    _plot_final_metrics(summary, metric_plot)

    manifest = [
        {"kind": "curves", "path": str(curve_plot.name)},
        {"kind": "final_metrics", "path": str(metric_plot.name)},
    ]
    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote conditioning ablation plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
