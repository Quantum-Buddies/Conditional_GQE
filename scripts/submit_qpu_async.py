#!/usr/bin/env python3
"""Export optimized circuits + QWC groups and submit asynchronously to qBraid.

This script runs on AIRE HPC after the full pipeline (RL training → inference →
L-BFGS-B optimization) has produced best operators and thetas.  It:

1. Loads Hamiltonian records and optimized results
2. Builds QWC-grouped measurement circuits
3. Exports a self-contained manifest (operators, thetas, groups, metadata)
4. Submits to the specified qBraid device with submit_only=True
5. Saves job metadata for later retrieval via retrieve_qbraid_job

The manifest can also be used to submit from a different machine later,
decoupling HPC compute from QPU submission.

Usage:
    # Submit to IonQ simulator (free, 29q)
    python scripts/submit_qpu_async.py --device ionq:ionq:sim:simulator --shots 4096

    # Submit to a 40q QPU (when available)
    python scripts/submit_qpu_async.py --device aws:rigetti:qpu:cepheus-1-108q --shots 8192

    # Export manifest only (no submission)
    python scripts/submit_qpu_async.py --export-only --out results/qpu/manifest.json

    # Retrieve results from a previous submission
    python scripts/submit_qpu_async.py --retrieve results/qpu/h2_submission_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import load_hamiltonian_records, find_record_by_name, iter_terms
from src.gqe.eval.qbraid_backend import (
    _build_ansatz_circuit,
    _group_qwc_terms,
    evaluate_energy_qbraid_batched,
    retrieve_qbraid_job,
)
from qiskit import QuantumCircuit


def _circuit_to_qasm(circ: QuantumCircuit) -> str:
    """Export circuit to QASM 2.0 string (Qiskit 1.x and 2.x compatible)."""
    try:
        from qiskit.qasm2 import dumps as qasm2_dumps
        return qasm2_dumps(circ)
    except ImportError:
        return circ.qasm()


def export_manifest(
    record: dict,
    operators: list[str],
    thetas: list[float],
    out_path: Path,
) -> dict:
    """Export a self-contained QWC manifest without submitting to any device."""
    n_qubits = int(record["n_qubits"])
    n_electrons = int(record.get("n_electrons", n_qubits // 2))

    circuit, params, param_objs = _build_ansatz_circuit(n_qubits, n_electrons, operators)
    bound = circuit.assign_parameters({t: float(v) for t, v in zip(param_objs, thetas)})

    active = [("".join(ops), coeff.real) for ops, coeff in iter_terms(record)]
    groups = _group_qwc_terms(active)

    group_data = []
    for gi, group_indices in enumerate(groups):
        group_base = ["I"] * n_qubits
        terms_in_group = []
        for ti in group_indices:
            word = active[ti][0]
            padded = word + "I" * (n_qubits - len(word)) if len(word) < n_qubits else word
            for q in range(n_qubits):
                if padded[q] != "I" and group_base[q] == "I":
                    group_base[q] = padded[q]
            terms_in_group.append({"term": active[ti][0], "coeff": active[ti][1]})

        meas = QuantumCircuit(n_qubits)
        meas.compose(bound, inplace=True)
        for q in range(n_qubits):
            q_qiskit = n_qubits - 1 - q
            if group_base[q] == "X":
                meas.h(q_qiskit)
            elif group_base[q] == "Y":
                meas.sdg(q_qiskit)
                meas.h(q_qiskit)
        meas.measure_all()

        group_data.append({
            "group_index": gi,
            "measurement_basis": "".join(group_base),
            "terms": terms_in_group,
            "qasm": _circuit_to_qasm(meas),
        })

    manifest = {
        "molecule": record["name"],
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "operators": operators,
        "thetas": thetas,
        "n_hamiltonian_terms": len(active),
        "n_groups": len(groups),
        "groups": group_data,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest exported: {out_path}")
    print(f"  {len(active)} terms -> {len(groups)} QWC groups ({len(active)/len(groups):.1f}x reduction)")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Async QPU submission workflow")
    parser.add_argument("--device", default="ionq:ionq:sim:simulator",
                        help="qBraid device ID (default: free IonQ sim)")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--molecules", nargs="*", default=["h2_0.74"],
                        help="Molecule names to submit")
    parser.add_argument("--hamiltonians", default="results/data/hamiltonians_merged.json")
    parser.add_argument("--optimized", default="results/eval/h_cgqe_uccsd_optimized.json")
    parser.add_argument("--out-dir", default="results/qpu", help="Output directory")
    parser.add_argument("--export-only", action="store_true",
                        help="Export manifest without submitting")
    parser.add_argument("--retrieve", metavar="META_FILE",
                        help="Retrieve results from a previous submission")
    args = parser.parse_args()

    if args.retrieve:
        meta_path = Path(args.retrieve)
        out_path = meta_path.parent / (meta_path.stem + "_result.json")
        print(f"Retrieving results from {meta_path}...")
        retrieve_qbraid_job(meta_path, out_path)
        print(f"Results saved to {out_path}")
        return

    ham_path = ROOT / args.hamiltonians
    opt_path = ROOT / args.optimized
    out_dir = ROOT / args.out_dir

    records = load_hamiltonian_records(ham_path)
    with open(opt_path) as f:
        optimized = json.load(f)

    for mol_name in args.molecules:
        print(f"\n=== {mol_name} ===")
        try:
            record = find_record_by_name(records, mol_name)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        mol_opt = next((e for e in optimized if e.get("molecule") == mol_name), None)
        if mol_opt is None:
            print(f"  SKIP: No optimized data")
            continue

        operators = mol_opt.get("best_operators", [])
        thetas = mol_opt.get("best_thetas", [])
        gpu_energy = mol_opt.get("best_energy")
        n_qubits = int(record["n_qubits"])

        print(f"  Qubits: {n_qubits}, GPU energy: {gpu_energy:.6f} Ha")

        if args.export_only:
            manifest_path = out_dir / f"{mol_name}_manifest.json"
            export_manifest(record, operators, thetas, manifest_path)
            continue

        # Submit asynchronously
        meta_path = out_dir / f"{mol_name}_submission_meta.json"
        print(f"  Submitting to {args.device} ({args.shots} shots)...")
        job_ids = evaluate_energy_qbraid_batched(
            record,
            operators,
            theta_values=np.asarray(thetas),
            device=args.device,
            shots=args.shots,
            submit_only=True,
            metadata_out_path=meta_path,
        )
        print(f"  Job IDs: {job_ids}")
        print(f"  Metadata: {meta_path}")
        print(f"  Retrieve later with: python scripts/submit_qpu_async.py --retrieve {meta_path}")


if __name__ == "__main__":
    main()
