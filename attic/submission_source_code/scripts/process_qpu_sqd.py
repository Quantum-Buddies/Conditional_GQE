#!/usr/bin/env python3
"""Process QPU counts into SQD energies using pre-computed FCI where available.

Reads:
  - results/qpu/lbfgs_cepheus_counts.json  (raw bitstring counts from QPU)
  - results/qpu/lbfgs_cepheus_submission_meta.json  (molecule metadata)
  - Hamiltonian records for each molecule

Writes:
  - results/qpu/lbfgs_cepheus_sqd_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import load_hamiltonian_records, iter_terms
from src.gqe.eval.qsci_postprocess import qsci_energy_from_bitstrings
from src.gqe.eval.sqd import reverse_bitstrings_in_counts


def load_hamiltonian(name: str) -> dict | None:
    """Load Hamiltonian record for a molecule from any hamiltonian file."""
    ham_files = [
        ROOT / "results/data/hamiltonians_gic2026/hamiltonians.json",
        ROOT / "results/data/hamiltonians_40plus.json/hamiltonians.json",
        ROOT / "results/data/hamiltonians.json",
        ROOT / "results/data/hamiltonians_merged.json",
    ]
    for hf in ham_files:
        if not hf.exists():
            continue
        records = load_hamiltonian_records(hf)
        for r in records:
            if r["name"] == name:
                return r
    return None


# Pre-computed FCI energies (avoid expensive diagonalization)
FCI_ENERGIES = {
    "h2": -1.1372838344885021,
    "lih": -7.861865,
    "beh2": -15.561278,
    "methyl_iodide_cas12": -6889.840354306438,
    "iodobenzene": None,  # 8q, can diagonalize
}

# HF energies from RL best circuits
HF_ENERGIES = {
    "h2": -1.1167593073964253,
    "lih": -7.861850625814101,
    "beh2": -15.561205654944462,
    "methyl_iodide_cas12": -6889.839231932083,
}


def main():
    counts_path = ROOT / "results/qpu/lbfgs_cepheus_counts.json"
    meta_path = ROOT / "results/qpu/lbfgs_cepheus_submission_meta.json"
    out_path = ROOT / "results/qpu/lbfgs_cepheus_sqd_results.json"

    with counts_path.open() as f:
        all_counts = json.load(f)

    with meta_path.open() as f:
        meta = json.load(f)

    sqd_jobs = meta.get("sqd_jobs", {})
    results = {}

    for name, counts_dict in all_counts.items():
        job_meta = sqd_jobs.get(name, {})
        mol = job_meta.get("molecule", name)
        nq = job_meta.get("n_qubits", 0)
        label = job_meta.get("label", "")
        n_ops = job_meta.get("n_operators", 0)
        thetas = job_meta.get("thetas", [])

        print(f"\n=== {name} ({mol}, {nq}q, {label}) ===")

        # Convert counts to bitstring list (weighted by count)
        # Reverse bitstrings for Rigetti QPU (qubit 0 is leftmost, our SQD expects rightmost)
        reversed_counts = reverse_bitstrings_in_counts(counts_dict, nq) if nq > 0 else counts_dict
        bitstrings = list(reversed_counts.keys())
        total_shots = sum(reversed_counts.values())
        print(f"  {len(bitstrings)} unique bitstrings, {total_shots} total shots")

        # Load Hamiltonian
        record = load_hamiltonian(mol)
        if record is None:
            # Try iodobenzene variants
            for variant in ["iodobenzene_cas12", "iodobenzene"]:
                record = load_hamiltonian(variant)
                if record:
                    break
        if record is None:
            print(f"  WARNING: No Hamiltonian found for {mol}, skipping")
            results[name] = {
                "molecule": mol,
                "n_qubits": nq,
                "label": label,
                "error": f"No Hamiltonian found for {mol}",
                "n_unique_bitstrings": len(bitstrings),
                "total_shots": total_shots,
            }
            continue

        n_qubits_ham = int(record["n_qubits"])
        print(f"  Hamiltonian: {record['name']}, {n_qubits_ham}q, {len(record.get('terms', []))} terms")

        # Pad bitstrings if needed (QPU might have different qubit ordering)
        padded_bs = []
        for bs in bitstrings:
            if len(bs) < n_qubits_ham:
                bs = "0" * (n_qubits_ham - len(bs)) + bs
            elif len(bs) > n_qubits_ham:
                bs = bs[-n_qubits_ham:]
            padded_bs.append(bs)

        # Compute SQD energy
        t0 = time.time()
        try:
            sqd_energy = qsci_energy_from_bitstrings(record, padded_bs)
            elapsed = time.time() - t0
            print(f"  SQD energy: {sqd_energy:.6f} Ha ({elapsed:.2f}s)")
        except Exception as e:
            print(f"  SQD failed: {e}")
            sqd_energy = None
            elapsed = time.time() - t0

        # Get reference energies
        fci = FCI_ENERGIES.get(mol)
        hf = HF_ENERGIES.get(mol)

        # For small systems, compute exact if not pre-computed
        if fci is None and n_qubits_ham <= 12:
            print(f"  Computing exact FCI for {mol} ({n_qubits_ham}q)...")
            try:
                from src.gqe.eval.run_fmo2 import exact_energy_from_hamiltonian
                fci = exact_energy_from_hamiltonian(record)
                print(f"  FCI: {fci:.6f} Ha")
            except Exception as e:
                print(f"  FCI computation failed: {e}")

        # Compute error
        error_mha = None
        if sqd_energy is not None and fci is not None:
            error_mha = abs(sqd_energy - fci) * 1000  # mHa
            print(f"  Error vs FCI: {error_mha:.2f} mHa")
        elif sqd_energy is not None and hf is not None:
            error_mha = abs(sqd_energy - hf) * 1000
            print(f"  Error vs HF: {error_mha:.2f} mHa")

        results[name] = {
            "molecule": mol,
            "n_qubits": nq,
            "n_qubits_hamiltonian": n_qubits_ham,
            "label": label,
            "n_operators": n_ops,
            "thetas_nonzero": any(abs(t) > 0.001 for t in thetas) if thetas else False,
            "n_unique_bitstrings": len(bitstrings),
            "total_shots": total_shots,
            "sqd_energy": sqd_energy,
            "fci_energy": fci,
            "hf_energy": hf,
            "error_mha": error_mha,
            "diag_time_seconds": elapsed,
        }

    # Write results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Saved to: {out_path}")
    for name, r in results.items():
        e = r.get("sqd_energy")
        err = r.get("error_mha")
        if e is not None:
            print(f"  {name}: E={e:.6f}, error={err:.2f} mHa" if err else f"  {name}: E={e:.6f}")
        else:
            print(f"  {name}: FAILED ({r.get('error', 'unknown')})")


if __name__ == "__main__":
    main()
