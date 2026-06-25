"""Compare baseline vs conditioned GQE results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs conditioned GQE results.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline GQE JSON")
    parser.add_argument("--conditioned", type=Path, nargs="+", required=True, help="One or more conditioned GQE JSONs")
    parser.add_argument("--labels", type=str, nargs="+", default=["conditioned"], help="Labels for each conditioned JSON")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    if len(args.conditioned) != len(args.labels):
        raise ValueError("--conditioned and --labels must have same length")

    baseline = {r["system"]: r for r in _load_results(args.baseline)}
    conditioned_maps = {label: {r["system"]: r for r in _load_results(path)} for label, path in zip(args.labels, args.conditioned)}

    systems = sorted(set(baseline.keys()))
    for cmap in conditioned_maps.values():
        systems = sorted(set(systems) & set(cmap.keys()))

    rows: List[Dict[str, Any]] = []
    for sys in systems:
        b = baseline[sys]
        b_delta = b.get("delta_energy")
        ref = b.get("reference_energy")
        row: Dict[str, Any] = {
            "system": sys,
            "n_qubits": b.get("n_spin_orbitals"),
            "reference_energy": ref,
            "baseline_delta_Ha": b_delta,
        }
        best_label = "baseline"
        best_delta = b_delta
        for label, cmap in conditioned_maps.items():
            c = cmap[sys]
            c_delta = c.get("delta_energy")
            c_energy = c.get("baseline_energy")
            row[f"{label}_delta_Ha"] = c_delta
            row[f"{label}_energy"] = c_energy
            if c_delta is not None and best_delta is not None:
                if c_delta < best_delta:
                    best_delta = c_delta
                    best_label = label
        row["best"] = best_label
        rows.append(row)

    fieldnames = ["system", "n_qubits", "reference_energy", "baseline_delta_Ha"]
    for label in args.labels:
        fieldnames.extend([f"{label}_delta_Ha", f"{label}_energy"])
    fieldnames.append("best")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = args.out.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"comparison": rows}, f, indent=2)

    print(f"Wrote GQE comparison CSV: {args.out}")
    print(f"Wrote GQE comparison JSON: {json_path}")

    print(f"\n=== GQE Baseline vs {', '.join(args.labels)} Summary ===")
    for row in rows:
        name = row["system"]
        b_str = f"{row['baseline_delta_Ha']:.6f}" if row['baseline_delta_Ha'] is not None else "N/A"
        parts = [f"  {name:12s}  baseline ΔE={b_str:>10s}"]
        for label in args.labels:
            val = row.get(f"{label}_delta_Ha")
            v_str = f"{val:.6f}" if val is not None else "N/A"
            parts.append(f"{label} ΔE={v_str:>10s}")
        parts.append(f"best={row['best']}")
        print("  ".join(parts))


if __name__ == "__main__":
    main()
