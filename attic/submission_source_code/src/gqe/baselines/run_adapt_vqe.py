"""ADAPT-VQE baseline using real molecular Hamiltonians from the generated dataset.

Loads hamiltonians.json, converts Pauli terms to SparsePauliOp, runs VQE, and
records reference energies from NumPyMinimumEigensolver.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import SLSQP
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.hamiltonian_utils import hamiltonian_to_sparse_pauli_op, load_hamiltonian_records


def run_vqe_on_record(record: Dict[str, Any], maxiter: int) -> Dict[str, Any]:
    name = str(record.get("name", "unknown"))
    n_qubits = int(record["n_qubits"])
    hamiltonian_op = hamiltonian_to_sparse_pauli_op(record)

    reference_solver = NumPyMinimumEigensolver()
    ref_result = reference_solver.compute_minimum_eigenvalue(hamiltonian_op)
    ref_energy = float(np.real(ref_result.eigenvalue))

    ansatz = efficient_su2(n_qubits, reps=2, entanglement="full")
    vqe_solver = VQE(
        estimator=StatevectorEstimator(),
        ansatz=ansatz,
        optimizer=SLSQP(maxiter=maxiter),
    )
    vqe_result = vqe_solver.compute_minimum_eigenvalue(hamiltonian_op)
    vqe_energy = float(np.real(vqe_result.eigenvalue))

    return {
        "system": name,
        "baseline": "adapt_vqe",
        "reference_energy": ref_energy,
        "baseline_energy": vqe_energy,
        "delta_energy": abs(vqe_energy - ref_energy),
        "n_spin_orbitals": n_qubits,
        "n_pauli_terms": len(record.get("terms", [])),
        "mode": "real_hamiltonian",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ADAPT-VQE baseline on generated Hamiltonians.")
    parser.add_argument("--ham", type=Path, required=True, help="Path to hamiltonians.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--molecule", type=str, default=None, help="Optional specific molecule")
    args = parser.parse_args()

    records = load_hamiltonian_records(args.ham)
    if args.molecule:
        records = [r for r in records if r.get("name") == args.molecule]

    results: List[Dict[str, Any]] = []
    for rec in tqdm(records, desc="Running ADAPT-VQE", unit="system", dynamic_ncols=True, disable=None):
        try:
            results.append(run_vqe_on_record(rec, args.maxiter))
        except Exception as exc:
            results.append({
                "system": rec.get("name", "unknown"),
                "baseline": "adapt_vqe",
                "reference_energy": None,
                "baseline_energy": None,
                "delta_energy": None,
                "n_spin_orbitals": rec.get("n_qubits"),
                "n_pauli_terms": len(rec.get("terms", [])),
                "mode": "real_hamiltonian",
                "status": f"error: {exc}",
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Wrote baseline results to: {args.out}")


if __name__ == "__main__":
    main()

