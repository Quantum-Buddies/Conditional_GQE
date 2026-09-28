#!/usr/bin/env python3
"""Prepare validated, submission-free SQD manifests for Rigetti Cepheus."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit.qasm2 import loads

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import (
    get_active_electron_count,
    load_hamiltonian_records,
)
from src.gqe.eval.qbraid_backend import export_sqd_sampling_circuit


TARGETS = [
    "methyl_iodide_cas12", "benzene_cas12", "phenol_cas12",
    "anisole_cas12", "toluene_cas12", "imeph_cas12",
    "iodobenzene_cas12", "ocresol_cas12", "diarylethene_frag_cas12",
    "hf", "h2o", "nh3", "ch4", "n2", "co",
    "h2_0.5", "h2_1.0", "h2_1.5", "h2_2.0",
    "lih_1.2", "lih_2.0", "lih_3.0", "lih_1.6_631g",
    "beh2_1.0", "beh2_1.6", "n2_1.8", "n2_2.5",
    "n2_1.1_631g_cas8", "h2o_1.0_631g_cas8",
]
DEVICE = "aws:rigetti:qpu:cepheus-1-108q"
SHOTS = 4096
COST_PER_JOB = 204.08
DEFAULT_MAX_RAW_DEPTH = 500
DEFAULT_MAX_TRANSPILED_DEPTH = 500


def _load_best_circuits(path: Path) -> tuple[dict[str, dict[str, Any]], float]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    default_theta = 0.01
    if isinstance(data, dict):
        default_theta = float(data.get("theta", default_theta))
        circuits = data.get("best_circuits", {})
        if circuits:
            return circuits, default_theta
    return {
        entry["molecule"]: entry["generated_sequences"][0]
        for entry in data
        if entry.get("generated_sequences")
    }, default_theta


def _preflight(
    name: str,
    record: dict[str, Any] | None,
    circuit: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if record is None:
        return ["missing Hamiltonian"], warnings
    n_qubits = int(record.get("n_qubits", 0))
    n_electrons = get_active_electron_count(record)
    if not record.get("terms"):
        errors.append("missing Hamiltonian terms")
    if not 0 < n_electrons <= n_qubits:
        errors.append(f"invalid electron sector {n_electrons}/{n_qubits}")
    operators = circuit.get("operators", []) if circuit else []
    if not operators:
        errors.append("missing operators")
        return errors, warnings
    malformed = [
        op for op in operators
        if not isinstance(op, str)
        or len(op) != n_qubits
        or set(op) - set("IXYZ")
    ]
    if malformed:
        errors.append(
            f"{len(malformed)} malformed/padded Pauli words "
            f"(all operators must be explicit {n_qubits}-character IXYZ strings)"
        )
    if not any(
        ("X" in op or "Y" in op) and sum(pauli != "I" for pauli in op) >= 2
        for op in operators
    ):
        errors.append("no entangling-capable X/Y operators")
    fci = record.get("fci_energy")
    tracked = circuit.get("energy") if circuit else None
    if fci is not None and tracked is not None and float(tracked) < float(fci) - 1e-7:
        errors.append(
            f"tracked energy {float(tracked):.9f} is below same-record "
            f"FCI reference {float(fci):.9f}"
        )
    if fci is None:
        warnings.append("FCI reference unavailable; HF is the only stored reference")
    if n_qubits >= 20:
        warnings.append(
            "20-22q SQD is likely shot-limited at 4096 shots; symmetry-filtered "
            "determinant support may be too sparse"
        )
    if len(operators) <= 3:
        warnings.append(
            "very shallow operator sequence may provide inadequate determinant support"
        )
    return errors, warnings


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "molecule", "status", "n_qubits", "n_electrons", "n_operators",
        "raw_depth", "transpiled_depth", "raw_two_qubit_gates",
        "transpiled_two_qubit_gates", "shots", "estimated_cost",
        "reference_type", "reference_energy", "manifest_path", "warnings",
        "rejection_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hamiltonians",
        type=Path,
        default=Path("results/data/hamiltonians_gic2026/hamiltonians.json"),
    )
    parser.add_argument(
        "--circuits",
        type=Path,
        default=Path("results/eval/rl_best_circuits_converted.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/qpu/cepheus_scaling_29_dryrun_v2"),
    )
    parser.add_argument("--max-raw-depth", type=int, default=DEFAULT_MAX_RAW_DEPTH)
    parser.add_argument(
        "--max-transpiled-depth",
        type=int,
        default=DEFAULT_MAX_TRANSPILED_DEPTH,
    )
    args = parser.parse_args()

    started = time.perf_counter()
    records = {
        record["name"]: record
        for record in load_hamiltonian_records(args.hamiltonians)
    }
    circuits, default_theta = _load_best_circuits(args.circuits)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest_dir = args.out_dir / "manifests"
    manifest_dir.mkdir()

    rows: list[dict[str, Any]] = []
    accepted_manifests: list[str] = []
    for name in TARGETS:
        record = records.get(name)
        circuit = circuits.get(name)
        errors, warnings = _preflight(name, record, circuit)
        row: dict[str, Any] = {
            "molecule": name,
            "status": "rejected" if errors else "accepted",
            "n_qubits": int(record["n_qubits"]) if record else None,
            "n_electrons": get_active_electron_count(record) if record else None,
            "n_operators": len(circuit.get("operators", [])) if circuit else 0,
            "shots": SHOTS,
            "estimated_cost": 0.0 if errors else COST_PER_JOB,
            "reference_type": (
                "stored_fci" if record and record.get("fci_energy") is not None
                else "stored_hf" if record and record.get("hf_energy") is not None
                else "none"
            ),
            "reference_energy": (
                record.get("fci_energy", record.get("hf_energy")) if record else None
            ),
            "warnings": "; ".join(warnings),
            "rejection_reasons": "; ".join(errors),
            "manifest_path": "",
        }
        if not errors and record and circuit:
            operators = circuit["operators"]
            theta_values = np.full(len(operators), default_theta, dtype=float)
            manifest_path = manifest_dir / f"{name}_sqd_cepheus_manifest.json"
            manifest = export_sqd_sampling_circuit(
                record,
                operators,
                theta_values,
                DEVICE,
                SHOTS,
                manifest_path,
            )
            raw_circuit = loads(manifest["circuit_qasm"])
            transpiled = transpile(
                raw_circuit,
                basis_gates=["rz", "sx", "x", "cz"],
                optimization_level=3,
                seed_transpiler=42,
            )
            raw_gates = manifest["circuit_gates"]
            transpiled_gates = dict(transpiled.count_ops())
            row.update({
                "raw_depth": int(manifest["circuit_depth"]),
                "transpiled_depth": int(transpiled.depth()),
                "raw_two_qubit_gates": int(raw_gates.get("cx", 0) + raw_gates.get("cz", 0)),
                "transpiled_two_qubit_gates": int(
                    transpiled_gates.get("cx", 0) + transpiled_gates.get("cz", 0)
                ),
                "manifest_path": str(manifest_path),
            })
            if (
                row["raw_depth"] > args.max_raw_depth
                or row["transpiled_depth"] > args.max_transpiled_depth
            ):
                manifest_path.unlink()
                row["status"] = "rejected"
                row["estimated_cost"] = 0.0
                row["manifest_path"] = ""
                row["rejection_reasons"] = (
                    f"depth exceeds thresholds: raw={row['raw_depth']}/"
                    f"{args.max_raw_depth}, transpiled={row['transpiled_depth']}/"
                    f"{args.max_transpiled_depth}"
                )
            else:
                if warnings:
                    row["status"] = "warning"
                accepted_manifests.append(str(manifest_path))
        rows.append(row)

    accepted = [row for row in rows if row["status"] in {"accepted", "warning"}]
    bundle = {
        "dry_run": True,
        "device": DEVICE,
        "shots_per_job": SHOTS,
        "cost_per_job_credits": COST_PER_JOB,
        "accepted_count": len(accepted),
        "projected_cost_credits": round(len(accepted) * COST_PER_JOB, 2),
        "max_raw_depth": args.max_raw_depth,
        "max_transpiled_depth": args.max_transpiled_depth,
        "source_hamiltonians": str(args.hamiltonians),
        "source_circuits": str(args.circuits),
        "theta_source": f"tracked fixed evaluation theta={default_theta}",
        "checkpoint_present": False,
        "manifests": accepted_manifests,
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    with (args.out_dir / "dry_run_bundle.json").open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)
    _write_csv(args.out_dir / "dry_run_summary.csv", rows)
    print(json.dumps({
        "accepted": len([r for r in rows if r["status"] == "accepted"]),
        "warning": len([r for r in rows if r["status"] == "warning"]),
        "rejected": len([r for r in rows if r["status"] == "rejected"]),
        "projected_cost_credits": bundle["projected_cost_credits"],
        "runtime_seconds": bundle["runtime_seconds"],
        "bundle": str(args.out_dir / "dry_run_bundle.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
