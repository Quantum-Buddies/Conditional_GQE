#!/usr/bin/env python3
"""Run SQD post-processing on retrieved QPU counts.

Uses hf_energy/fci_energy from Hamiltonian records when available,
falls back to exact diagonalization only for small systems (<=12q).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
    get_active_electron_count,
)
from src.gqe.eval.sqd import run_sqd, exact_diagonalize, sqd_energy_from_bitstrings, reverse_bitstrings_in_counts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=Path, required=True, help="Retrieved counts JSON")
    parser.add_argument("--hamiltonians", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-ed-qubits", type=int, default=12, help="Max qubits for exact diagonalization")
    parser.add_argument("--reverse-bits", action="store_true", default=False,
                        help="Reverse bitstring bit order (for Rigetti QPU where qubit 0 is leftmost).")
    args = parser.parse_args()

    records = load_hamiltonian_records(args.hamiltonians)
    with open(args.counts) as f:
        counts_data = json.load(f)

    results = {}
    for mol, mol_data in counts_data.items():
        if mol_data.get("status") != "completed":
            print(f"\n{mol}: skipping (status={mol_data.get('status')})")
            continue

        counts = mol_data["counts"]
        record = find_record_by_name(records, mol)
        if record is None:
            print(f"\n{mol}: no Hamiltonian record found")
            continue

        n_qubits = int(record["n_qubits"])
        n_electrons = get_active_electron_count(record)
        print(f"\n=== {mol} ({n_qubits}q, {n_electrons}e) ===")
        print(f"  Counts: {len(counts)} unique, {sum(counts.values())} total shots")

        # FCI energy
        fci_energy = None
        if n_qubits <= args.max_ed_qubits:
            print(f"  Computing FCI (exact diagonalization, {n_qubits}q)...")
            t0 = time.time()
            try:
                fci_energy = exact_diagonalize(record)
                print(f"  FCI energy: {fci_energy:.6f} Ha ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  FCI failed: {e}")
        else:
            # Use hf_energy from record if available
            fci_energy = record.get("fci_energy")
            if fci_energy:
                print(f"  FCI energy (from record): {fci_energy:.6f} Ha")
            else:
                print(f"  FCI energy: skipped (n_qubits={n_qubits} > {args.max_ed_qubits})")

        # HF energy
        hf_bs = format((1 << n_electrons) - 1, f"0{n_qubits}b")
        hf_energy = sqd_energy_from_bitstrings(record, [hf_bs])
        print(f"  HF energy: {hf_energy:.6f} Ha")

        # SQD
        print(f"  Running SQD...")
        t0 = time.time()
        sqd_result = run_sqd(
            record,
            counts,
            n_electrons=n_electrons,
            subspace_size=None,
            particle_number_tol=0,
            spin_parity=0,
            n_recovered=0,
            return_details=True,
            reverse_bit_order=args.reverse_bits,
        )
        sqd_energy = sqd_result["energy"]
        print(f"  SQD energy: {sqd_energy:.6f} Ha ({time.time()-t0:.1f}s)")

        result_entry = {
            "status": "completed",
            "molecule": mol,
            "n_qubits": n_qubits,
            "n_electrons": n_electrons,
            "n_shots": sum(counts.values()),
            "n_unique_bitstrings": len(counts),
            "hf_energy": hf_energy,
            "sqd_energy": sqd_energy,
            "fci_energy": fci_energy,
            "error_vs_fci_mha": abs(sqd_energy - fci_energy) * 1000 if fci_energy else None,
            "improvement_over_hf_mha": (hf_energy - sqd_energy) * 1000,
            "n_symmetry_filtered": sqd_result.get("n_bitstrings", 0),
            "variational_bound_satisfied": sqd_result.get("variational_bound_satisfied", True),
        }
        results[mol] = result_entry

        if fci_energy:
            print(f"  Error vs FCI: {abs(sqd_energy - fci_energy)*1000:.3f} mHa")
        print(f"  Improvement over HF: {(hf_energy - sqd_energy)*1000:.3f} mHa")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()
