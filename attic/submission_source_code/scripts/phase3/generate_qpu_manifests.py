#!/usr/bin/env python3
"""Generate QPU manifests for H2 and LiH on Rigetti Cepheus.

Exports QWC-grouped measurement manifests for the Rigetti Cepheus QPU
(108q, native gate set: CZ + RX/RY/RZ). Each manifest contains:
  - Optimized operators and thetas from H-cGQE inference
  - QWC-grouped measurement circuits as QASM 2.0
  - Metadata for qBraid submission (device, shots, cost estimate)

Usage:
    # Generate manifests for both H2 and LiH
    python scripts/phase3/generate_qpu_manifests.py

    # Generate for specific molecules only
    python scripts/phase3/generate_qpu_manifests.py --molecules h2_0.74

    # With custom optimized results
    python scripts/phase3/generate_qpu_manifests.py \
        --optimized results/eval/h_cgqe_uccsd_optimized.json \
        --hamiltonians results/data/hamiltonians_merged.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import load_hamiltonian_records, find_record_by_name, iter_terms
from src.gqe.eval.qbraid_backend import _build_ansatz_circuit, _group_qwc_terms
from qiskit import QuantumCircuit


# Rigetti Cepheus QPU specifications
RIGETTI_CEPHEUS = {
    "device_id": "aws:rigetti:qpu:cepheus-1-108q",
    "n_qubits": 108,
    "native_gates": ["RX", "RY", "RZ", "CZ"],
    "topology": "octagonal lattice",
    "cost_per_shot": 0.0425,  # qBraid credits
    "cost_per_task": 30.0,    # qBraid credits
}

# Default molecules for Phase 3 QPU validation
DEFAULT_MOLECULES = ["h2_0.74", "lih_1.6_full"]


def _circuit_to_qasm(circ: QuantumCircuit) -> str:
    """Export circuit to QASM 2.0 string."""
    try:
        from qiskit.qasm2 import dumps as qasm2_dumps
        return qasm2_dumps(circ)
    except ImportError:
        return circ.qasm()


def estimate_qpu_cost(n_groups: int, shots: int, device_info: dict) -> dict:
    """Estimate qBraid credit cost for QPU execution."""
    shot_cost = n_groups * shots * device_info["cost_per_shot"]
    task_cost = device_info["cost_per_task"]
    total = shot_cost + task_cost
    return {
        "shot_cost": round(shot_cost, 2),
        "task_cost": task_cost,
        "total_cost": round(total, 2),
        "n_circuits": n_groups,
        "shots_per_circuit": shots,
    }


def export_manifest(
    record: dict,
    operators: list[str],
    thetas: list[float],
    out_path: Path,
    shots: int = 4096,
    device_info: dict | None = None,
) -> dict:
    """Export a self-contained QWC manifest for QPU submission."""
    if device_info is None:
        device_info = RIGETTI_CEPHEUS

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

    cost = estimate_qpu_cost(len(groups), shots, device_info)

    manifest = {
        "molecule": record["name"],
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "operators": operators,
        "thetas": thetas,
        "n_hamiltonian_terms": len(active),
        "n_groups": len(groups),
        "groups": group_data,
        "device": device_info,
        "cost_estimate": cost,
        "shots": shots,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {out_path}")
    print(f"    {len(active)} terms -> {len(groups)} QWC groups ({len(active)/len(groups):.1f}x reduction)")
    print(f"    Estimated cost: {cost['total_cost']:.2f} qBraid credits")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate QPU manifests for Rigetti Cepheus")
    parser.add_argument("--molecules", nargs="*", default=DEFAULT_MOLECULES,
                        help="Molecule names to generate manifests for")
    parser.add_argument("--hamiltonians", default="results/data/hamiltonians_merged.json",
                        help="Path to Hamiltonians JSON file")
    parser.add_argument("--optimized", default="results/eval/h_cgqe_uccsd_optimized.json",
                        help="Path to optimized results JSON file")
    parser.add_argument("--out-dir", default="results/qpu/manifests",
                        help="Output directory for manifests")
    parser.add_argument("--shots", type=int, default=4096,
                        help="Shots per measurement circuit")
    args = parser.parse_args()

    ham_path = ROOT / args.hamiltonians
    opt_path = ROOT / args.optimized
    out_dir = ROOT / args.out_dir

    print(f"QPU Manifest Generation for Rigetti Cepheus")
    print(f"  Device: {RIGETTI_CEPHEUS['device_id']} ({RIGETTI_CEPHEUS['n_qubits']}q)")
    print(f"  Molecules: {args.molecules}")
    print(f"  Shots per circuit: {args.shots}")
    print()

    records = load_hamiltonian_records(ham_path)
    with open(opt_path) as f:
        optimized = json.load(f)

    manifests = []
    total_cost = 0.0
    for mol_name in args.molecules:
        print(f"=== {mol_name} ===")
        try:
            record = find_record_by_name(records, mol_name)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        mol_opt = next((e for e in optimized if e.get("molecule") == mol_name), None)
        if mol_opt is None:
            print(f"  SKIP: No optimized data found")
            continue

        operators = mol_opt.get("best_operators", [])
        thetas = mol_opt.get("best_thetas", [])
        gpu_energy = mol_opt.get("best_energy")
        n_qubits = int(record["n_qubits"])

        print(f"  Qubits: {n_qubits}, GPU energy: {gpu_energy:.6f} Ha")
        print(f"  Operators: {operators}")
        print(f"  Thetas: {[f'{t:.6f}' for t in thetas]}")

        manifest_path = out_dir / f"{mol_name}_cepheus_manifest.json"
        manifest = export_manifest(record, operators, thetas, manifest_path, args.shots)
        manifests.append(manifest)
        total_cost += manifest["cost_estimate"]["total_cost"]
        print()

    # Save summary
    summary_path = out_dir / "manifest_summary.json"
    summary = {
        "device": RIGETTI_CEPHEUS,
        "molecules": [m["molecule"] for m in manifests],
        "total_circuits": sum(m["n_groups"] for m in manifests),
        "total_cost_estimate": round(total_cost, 2),
        "shots_per_circuit": args.shots,
        "manifests": [str(out_dir / f"{m['molecule']}_cepheus_manifest.json") for m in manifests],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'=' * 60}")
    print(f"Manifest Summary")
    print(f"  Molecules: {len(manifests)}")
    print(f"  Total circuits: {summary['total_circuits']}")
    print(f"  Total estimated cost: {total_cost:.2f} qBraid credits")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
