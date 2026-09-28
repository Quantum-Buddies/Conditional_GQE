#!/usr/bin/env python3
"""Retrieve qBraid job results (SV1 simulator or Cepheus QPU) and run SQD post-processing.

Usage:
    # Retrieve SV1 simulator results
    python scripts/retrieve_and_sqd.py --meta results/qpu/h2_sv1_submission.json/h2_submission_meta.json \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --out results/qpu/h2_sv1_sqd_results.json

    # Retrieve Cepheus QPU results
    python scripts/retrieve_and_sqd.py --meta results/qpu/cepheus_submission_meta.json \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --out results/qpu/cepheus_sqd_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qbraid import QbraidProvider
from qbraid.runtime import load_job

from src.gqe.common.hamiltonian_utils import load_hamiltonian_records, find_record_by_name, get_active_electron_count
from src.gqe.eval.sqd import run_sqd, exact_diagonalize, sqd_energy_from_bitstrings


def retrieve_job_counts(job_id: str) -> dict[str, int] | None:
    """Retrieve counts from a completed qBraid job."""
    try:
        job = load_job(job_id)
        status = job.status()
        print(f"    {job_id}: {status}")
        if "COMPLETED" in str(status).upper():
            result = job.result()
            try:
                counts = result.data.get_counts()
            except Exception:
                counts = result.measurement_counts()
            return {str(k): int(v) for k, v in counts.items()}
        else:
            return None
    except Exception as e:
        print(f"    {job_id}: ERROR: {e}")
        return None


def run_sqd_on_counts(record: dict, counts: dict[str, int], n_electrons: int) -> dict:
    """Run SQD pipeline on retrieved counts."""
    fci_energy = exact_diagonalize(record)
    hf_bs = format((1 << n_electrons) - 1, f"0{int(record['n_qubits'])}b")
    hf_energy = sqd_energy_from_bitstrings(record, [hf_bs])

    result = run_sqd(
        record,
        counts,
        n_electrons=n_electrons,
        subspace_size=None,
        particle_number_tol=0,
        spin_parity=0,
        n_recovered=0,
        return_details=True,
    )
    return {
        "fci_energy": fci_energy,
        "hf_energy": hf_energy,
        "sqd_energy": result["energy"],
        "error_vs_fci_mha": abs(result["energy"] - fci_energy) * 1000,
        "variational_bound_satisfied": result.get("variational_bound_satisfied", result["energy"] >= fci_energy - 1e-10),
        "n_symmetry_filtered": result.get("n_bitstrings", 0),
        "n_unique_raw": result.get("n_unique_raw", 0),
        "monotonicity_ok": result.get("monotonicity_ok", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve qBraid results and run SQD")
    parser.add_argument("--meta", type=Path, required=True, help="Submission metadata JSON")
    parser.add_argument("--hamiltonians", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "sqd", "qwc"],
                        help="SQD: single circuit Z-basis. QWC: multi-circuit grouped.")
    args = parser.parse_args()

    provider = QbraidProvider()
    records = load_hamiltonian_records(args.hamiltonians)
    record_map = {r["name"]: r for r in records}

    with open(args.meta) as f:
        meta = json.load(f)

    # Determine mode from metadata structure
    has_sqd = "sqd_jobs" in meta or "job_ids" in meta
    has_qwc = "qwc_jobs" in meta

    all_results = {}

    # --- SQD mode: one circuit per molecule, Z-basis counts ---
    sqd_jobs = meta.get("sqd_jobs", {})
    if not sqd_jobs and "job_ids" in meta:
        # SV1 submission format: single molecule
        mol = meta.get("molecule", "h2")
        sqd_jobs = {mol: meta["job_ids"]}

    for mol, job_id in sqd_jobs.items():
        if job_id is None:
            continue
        print(f"\n=== {mol} SQD ===")
        if isinstance(job_id, list):
            # Multiple jobs — merge counts
            merged_counts = {}
            for jid in job_id:
                counts = retrieve_job_counts(jid)
                if counts:
                    for k, v in counts.items():
                        merged_counts[k] = merged_counts.get(k, 0) + v
            counts = merged_counts if merged_counts else None
        else:
            counts = retrieve_job_counts(job_id)

        if counts is None:
            print(f"  No counts retrieved for {mol}")
            all_results[mol] = {"status": "pending"}
            continue

        print(f"  Retrieved {len(counts)} bitstrings, {sum(counts.values())} total shots")

        record = record_map.get(mol)
        if record is None:
            # Try variations
            for r in records:
                if r["name"].startswith(mol):
                    record = r
                    break
        if record is None:
            print(f"  No Hamiltonian record for {mol}")
            all_results[mol] = {"status": "no_hamiltonian"}
            continue

        n_electrons = get_active_electron_count(record)
        sqd_result = run_sqd_on_counts(record, counts, n_electrons)
        all_results[mol] = {
            "status": "completed",
            "counts": counts,
            "sqd_analysis": sqd_result,
        }
        print(f"  SQD energy: {sqd_result['sqd_energy']:.6f} Ha")
        print(f"  FCI energy: {sqd_result['fci_energy']:.6f} Ha")
        print(f"  Error: {sqd_result['error_vs_fci_mha']:.3f} mHa")
        print(f"  Variational bound: {sqd_result['variational_bound_satisfied']}")

    # --- QWC mode: multiple circuits per molecule, parse grouped expectations ---
    qwc_jobs = meta.get("qwc_jobs", {})
    for mol, job_ids in qwc_jobs.items():
        if not job_ids:
            continue
        print(f"\n=== {mol} QWC ===")
        group_counts = []
        for jid in job_ids:
            counts = retrieve_job_counts(jid)
            group_counts.append(counts)

        if any(c is None for c in group_counts):
            print(f"  Some QWC jobs still pending for {mol}")
            all_results[f"{mol}_qwc"] = {"status": "pending"}
            continue

        # For QWC, we compute energy from grouped expectations
        # Need the manifest to parse groups
        manifest_path = ROOT / f"results/qpu/{mol}_0.74_manifest.json"
        if not manifest_path.exists():
            # Try other patterns
            for p in (ROOT / "results/qpu").glob(f"{mol}*manifest*.json"):
                manifest_path = p
                break

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            groups = manifest.get("groups", [])
            n_qubits = manifest["n_qubits"]

            total_energy = 0.0
            for gi, (counts, group) in enumerate(zip(group_counts, groups)):
                if counts is None:
                    continue
                terms = group["terms"]
                basis = group["measurement_basis"]
                for term_info in terms:
                    word = term_info["term"]
                    coeff = term_info["coeff"]
                    # Compute expectation from counts
                    parity = 0
                    total = sum(counts.values())
                    if total == 0:
                        continue
                    for bitstring, count in counts.items():
                        # bit ordering: qiskit bitstring is q_{n-1}...q_0
                        # Pauli position q maps to bitstring index q
                        sign = 1
                        for q, op in enumerate(word):
                            if op in "XY":
                                bit = int(bitstring[q])
                                if bit == 1:
                                    sign *= -1
                        parity += sign * count
                    exp_val = parity / total
                    total_energy += coeff * exp_val

            record = record_map.get(mol)
            if record is None:
                for r in records:
                    if r["name"].startswith(mol):
                        record = r
                        break

            if record:
                fci = exact_diagonalize(record)
                all_results[f"{mol}_qwc"] = {
                    "status": "completed",
                    "qwc_energy": total_energy,
                    "fci_energy": fci,
                    "error_vs_fci_mha": abs(total_energy - fci) * 1000,
                }
                print(f"  QWC energy: {total_energy:.6f} Ha")
                print(f"  FCI energy: {fci:.6f} Ha")
                print(f"  Error: {abs(total_energy - fci) * 1000:.3f} mHa")
        else:
            print(f"  No manifest found for {mol} QWC")

    # Save results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()
