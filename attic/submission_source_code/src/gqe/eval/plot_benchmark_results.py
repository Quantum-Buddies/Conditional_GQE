#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dataset_rows(rows: list[dict]) -> tuple[list[str], list[float]]:
    systems = []
    values = []
    for rec in rows:
        if rec.get("section") != "dataset":
            continue
        v = _to_float(rec.get("value"))
        if v is None:
            continue
        systems.append(rec.get("system", ""))
        values.append(v)
    return systems, values


def _baseline_rows(rows: list[dict]) -> dict[str, tuple[list[str], list[float]]]:
    values_by_method: dict[str, tuple[list[str], list[float]]] = {}
    for rec in rows:
        if rec.get("section") != "baseline":
            continue
        method = str(rec.get("metric", "baseline")).replace("_delta_energy", "")
        v = _to_float(rec.get("value"))
        if v is None:
            continue
        systems, vals = values_by_method.get(method, ([], []))
        systems.append(rec.get("system", ""))
        vals.append(v)
        values_by_method[method] = (systems, vals)
    return values_by_method


def _to_train_stats(train: dict) -> list[float]:
    if not train:
        return []
    if isinstance(train.get("epoch_losses"), list) and train["epoch_losses"]:
        return [float(x) for x in train["epoch_losses"]]
    if isinstance(train.get("batch_losses"), list) and train["batch_losses"]:
        return [float(x) for x in train["batch_losses"]]
    final_loss = _to_float(train.get("final_loss"))
    return [final_loss] if final_loss is not None else []


def _plot_bar(values_by_system: tuple[list[str], list[float]], out_path: Path, title: str, ylabel: str) -> None:
    systems, values = values_by_system
    if not systems:
        return
    plt.figure(figsize=(8, 4))
    plt.bar(systems, values, color="#4c72b0")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("system")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_grouped_baseline(values_by_method: dict[str, tuple[list[str], list[float]]], out_path: Path) -> None:
    if not values_by_method:
        return
    all_methods = list(values_by_method.keys())
    systems = values_by_method[all_methods[0]][0]
    n_systems = len(systems)
    width = 0.35
    x = range(n_systems)
    plt.figure(figsize=(8, 4))
    for i, method in enumerate(all_methods):
        _, values = values_by_method[method]
        offset = width * i - (width * (len(all_methods) - 1)) / 2
        plt.bar([ix + offset for ix in x], values, width=width, label=method)
    plt.title("Baseline delta energy")
    plt.ylabel("abs(E_model - E_ref)")
    plt.xticks(list(x), systems, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_baseline_scatter(values_by_method: dict[str, tuple[list[str], list[float]]], out_path: Path) -> None:
    if not values_by_method:
        return
    plt.figure(figsize=(8, 4))
    for method, (systems, values) in values_by_method.items():
        x = list(range(len(systems)))
        plt.scatter(x, values, label=method, s=35)
        # label points
        for x_i, s, v in zip(x, systems, values):
            plt.text(x_i, v, str(s), fontsize=7, va="bottom", ha="left")
    plt.title("Baseline delta energy scatter")
    plt.ylabel("abs(E_model - E_ref)")
    plt.xlabel("system index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_training(losses: list[float], out_path: Path) -> None:
    if not losses:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.title("Training loss trajectory")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark summary to PNG.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_csv(args.summary_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    dataset_plot = args.out_dir / "benchmark_dataset_n_pauli_terms.png"
    systems, values = _dataset_rows(rows)
    _plot_bar((systems, values), dataset_plot, "Hamiltonian Pauli term count", "n_pauli_terms")
    if systems:
        manifest.append({"kind": "dataset", "path": str(dataset_plot.name)})

    baseline_by_method = _baseline_rows(rows)
    bar_plot = args.out_dir / "benchmark_baseline_delta_energy_bars.png"
    scatter_plot = args.out_dir / "benchmark_baseline_delta_energy_scatter.png"
    _plot_grouped_baseline(baseline_by_method, bar_plot)
    _plot_baseline_scatter(baseline_by_method, scatter_plot)
    if baseline_by_method:
        manifest.append({"kind": "baseline_bar", "path": str(bar_plot.name)})
        manifest.append({"kind": "baseline_scatter", "path": str(scatter_plot.name)})

    if args.train_json is not None and args.train_json.exists():
        with args.train_json.open("r", encoding="utf-8") as f:
            train = json.load(f)
        losses = _to_train_stats(train)
        train_plot = args.out_dir / "benchmark_training_loss_curve.png"
        _plot_training(losses, train_plot)
        if losses:
            manifest.append({"kind": "training", "path": str(train_plot.name)})

    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote benchmark plots and manifest to: {args.out_dir}")


if __name__ == "__main__":
    main()

