#!/usr/bin/env python3
"""Retrieve qBraid QPU job counts and save raw results.

Lightweight version of retrieve_and_sqd.py — just fetches counts,
no SQD post-processing or exact diagonalization.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qbraid.runtime import load_job

from src.gqe.eval.sqd import reverse_bitstrings_in_counts


def retrieve_counts(job_id: str, max_retries: int = 6, reverse_bits: bool = False, n_qubits: int = 0) -> dict | None:
    """Retrieve counts from a qBraid job with retry.

    Args:
        job_id: qBraid job ID.
        reverse_bits: If True, reverse bitstring keys (for Rigetti QPU where
            qubit 0 is leftmost instead of rightmost).
        n_qubits: Number of qubits (needed for zero-padding during reversal).
    """
    for attempt in range(max_retries):
        try:
            job = load_job(job_id)
            status = job.status()
            print(f"  Attempt {attempt+1}: {job_id} -> {status}")
            if "COMPLETED" in str(status).upper():
                result = job.result()
                try:
                    counts = result.data.get_counts()
                except Exception:
                    counts = result.measurement_counts()
                counts = {str(k): int(v) for k, v in counts.items()}
                if reverse_bits and n_qubits > 0:
                    counts = reverse_bitstrings_in_counts(counts, n_qubits)
                return counts
            elif "FAILED" in str(status).upper() or "CANCELLED" in str(status).upper():
                print(f"  Job {job_id} terminal status: {status}")
                return None
            else:
                time.sleep(10)
        except Exception as e:
            print(f"  Attempt {attempt+1} error: {e}")
            time.sleep(10 * (attempt + 1))
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reverse-bits", action="store_true", default=False,
                        help="Reverse bitstring bit order (for Rigetti QPU where qubit 0 is leftmost).")
    args = parser.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)

    sqd_jobs = meta.get("sqd_jobs", {})
    all_results = {}

    for mol, job_id in sqd_jobs.items():
        if job_id is None:
            continue
        print(f"\n=== {mol} ===")
        counts = retrieve_counts(job_id, reverse_bits=args.reverse_bits, n_qubits=n_qubits if 'n_qubits' in dir() else 0)
        if counts:
            print(f"  Retrieved {len(counts)} bitstrings, {sum(counts.values())} total shots")
            all_results[mol] = {
                "status": "completed",
                "job_id": job_id,
                "counts": counts,
                "n_shots": sum(counts.values()),
                "n_unique": len(counts),
            }
        else:
            all_results[mol] = {"status": "pending", "job_id": job_id}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()
