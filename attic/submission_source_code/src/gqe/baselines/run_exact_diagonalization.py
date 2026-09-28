"""Exact diagonalization reference for small molecular Hamiltonians."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.hamiltonian_utils import (
    exact_diagonalize_hamiltonian,
    load_hamiltonian_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact diagonalization of molecular Hamiltonians.")
    parser.add_argument("--ham", type=Path, required=True, help="Path to hamiltonians.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--molecule", type=str, default=None, help="Optional specific molecule")
    args = parser.parse_args()

    records = load_hamiltonian_records(args.ham)
    if args.molecule:
        records = [r for r in records if r.get("name") == args.molecule]

    results: List[Dict[str, Any]] = []
    for rec in tqdm(records, desc="Exact diagonalization", unit="system", dynamic_ncols=True):
        n_qubits = int(rec["n_qubits"])
        name = str(rec.get("name", "unknown"))
        try:
            if n_qubits > 14:
                ground_energy = None
                energy_gap = None
                status = "skipped_too_large"
            else:
                ground_energy, energy_gap = exact_diagonalize_hamiltonian(rec)
                status = "success"
        except Exception as exc:
            ground_energy = None
            energy_gap = None
            status = f"error: {exc}"

        results.append({
            "system": name,
            "baseline": "exact_diagonalization",
            "n_spin_orbitals": n_qubits,
            "n_pauli_terms": rec.get("n_pauli_terms"),
            "reference_energy": ground_energy,
            "energy_gap": energy_gap,
            "status": status,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Wrote exact diagonalization results to: {args.out}")


if __name__ == "__main__":
    main()
