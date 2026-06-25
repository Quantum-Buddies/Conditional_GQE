import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _append_dataset_rows(records: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    for rec in records:
        system = rec["name"]
        rows.append({"section": "dataset", "system": system, "metric": "n_qubits", "value": rec["n_qubits"]})
        rows.append({"section": "dataset", "system": system, "metric": "n_pauli_terms", "value": rec["n_pauli_terms"]})


def _append_baseline_rows(payload: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    for rec in payload.get("results", []):
        system = rec.get("system", "unknown")
        baseline = rec.get("baseline", "baseline")

        def _add(metric_key: str) -> None:
            value = rec.get(metric_key)
            if value is None:
                return
            rows.append(
                {
                    "section": "baseline",
                    "system": system,
                    "metric": f"{baseline}_{metric_key}",
                    "value": value,
                }
            )

        for key in ("reference_energy", "baseline_energy", "delta_energy"):
            _add(key)


def _append_reference_rows(payload: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    for rec in payload.get("results", []):
        system = rec.get("system", "unknown")
        if rec.get("reference_energy") is not None:
            rows.append(
                {
                    "section": "reference",
                    "system": system,
                    "metric": "exact_reference_energy",
                    "value": rec["reference_energy"],
                }
            )
        if rec.get("energy_gap") is not None:
            rows.append(
                {
                    "section": "reference",
                    "system": system,
                    "metric": "exact_energy_gap",
                    "value": rec["energy_gap"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment outputs into CSV.")
    parser.add_argument("--ham", type=Path, required=True, help="Hamiltonian JSON path")
    parser.add_argument("--baseline", type=Path, required=True, help="Primary baseline JSON path")
    parser.add_argument("--cudaq-baseline", type=Path, default=None, help="Optional CUDA-Q baseline JSON path")
    parser.add_argument(
        "--gqe-baseline",
        type=Path,
        default=None,
        help="Optional CUDA-Q GQE baseline JSON path",
    )
    parser.add_argument("--reference", type=Path, default=None, help="Optional exact diagonalization JSON")
    parser.add_argument("--train", type=Path, required=True, help="Training metrics JSON path")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    ham = _read_json(args.ham)
    baseline = _read_json(args.baseline)
    train = _read_json(args.train)

    rows: List[Dict[str, Any]] = []
    _append_dataset_rows(ham["records"], rows)
    _append_baseline_rows(baseline, rows)
    if args.cudaq_baseline is not None and args.cudaq_baseline.exists():
        cudaq_baseline = _read_json(args.cudaq_baseline)
        _append_baseline_rows(cudaq_baseline, rows)
    if args.gqe_baseline is not None and args.gqe_baseline.exists():
        gqe_baseline = _read_json(args.gqe_baseline)
        _append_baseline_rows(gqe_baseline, rows)
    if args.reference is not None and args.reference.exists():
        reference_payload = _read_json(args.reference)
        _append_reference_rows(reference_payload, rows)

    rows.append(
        {
            "section": "training",
            "system": "seq_model",
            "metric": "final_loss",
            "value": train.get("final_loss"),
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "system", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote aggregated CSV: {args.out}")


if __name__ == "__main__":
    main()

