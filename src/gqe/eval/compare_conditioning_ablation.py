from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_summary(payload: dict[str, Any], mode_label: str) -> dict[str, Any]:
    return {
        "mode": mode_label,
        "dataset": payload.get("dataset"),
        "num_samples": payload.get("num_samples"),
        "final_train_mse": payload.get("final_train_mse"),
        "final_train_mae": payload.get("final_train_mae"),
        "final_val_mse": payload.get("final_val_mse"),
        "final_val_mae": payload.get("final_val_mae"),
        "target_names": payload.get("target_names", []),
        "train_loss_history": payload.get("train_loss_history", []),
        "val_loss_history": payload.get("val_loss_history", []),
    }


def _delta(graph_value: Any, flat_value: Any) -> Any:
    if graph_value is None or flat_value is None:
        return None
    return float(flat_value) - float(graph_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare graph-conditioned and flat conditioning ablations.")
    parser.add_argument("--graph-metrics", type=Path, required=True, help="Metrics JSON from graph conditioning.")
    parser.add_argument("--flat-metrics", type=Path, required=True, help="Metrics JSON from flat conditioning.")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV summary path.")
    args = parser.parse_args()

    graph_payload = _read_json(args.graph_metrics)
    flat_payload = _read_json(args.flat_metrics)
    graph = _extract_summary(graph_payload, "graph")
    flat = _extract_summary(flat_payload, "flat")
    comparison = {
        "mode": "graph_minus_flat",
        "dataset": graph.get("dataset") or flat.get("dataset"),
        "num_samples": graph.get("num_samples") or flat.get("num_samples"),
        "final_train_mse_delta": _delta(graph.get("final_train_mse"), flat.get("final_train_mse")),
        "final_train_mae_delta": _delta(graph.get("final_train_mae"), flat.get("final_train_mae")),
        "final_val_mse_delta": _delta(graph.get("final_val_mse"), flat.get("final_val_mse")),
        "final_val_mae_delta": _delta(graph.get("final_val_mae"), flat.get("final_val_mae")),
        "interpretation": "Positive deltas mean the graph-conditioned model is lower error than the flat baseline.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "dataset",
                "num_samples",
                "final_train_mse",
                "final_train_mae",
                "final_val_mse",
                "final_val_mae",
                "final_train_mse_delta",
                "final_train_mae_delta",
                "final_val_mse_delta",
                "final_val_mae_delta",
                "interpretation",
            ],
        )
        writer.writeheader()
        for row in (graph, flat, comparison):
            writer.writerow(
                {
                    "mode": row.get("mode"),
                    "dataset": row.get("dataset"),
                    "num_samples": row.get("num_samples"),
                    "final_train_mse": row.get("final_train_mse"),
                    "final_train_mae": row.get("final_train_mae"),
                    "final_val_mse": row.get("final_val_mse"),
                    "final_val_mae": row.get("final_val_mae"),
                    "final_train_mse_delta": row.get("final_train_mse_delta"),
                    "final_train_mae_delta": row.get("final_train_mae_delta"),
                    "final_val_mse_delta": row.get("final_val_mse_delta"),
                    "final_val_mae_delta": row.get("final_val_mae_delta"),
                    "interpretation": row.get("interpretation"),
                }
            )

    summary_path = args.out.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"graph": graph, "flat": flat, "comparison": comparison}, f, indent=2)

    print(f"Wrote conditioning comparison CSV: {args.out}")
    print(f"Wrote conditioning comparison JSON: {summary_path}")


if __name__ == "__main__":
    main()
