#!/usr/bin/env python
"""Collect QPU job results from qBraid.

Retrieves job results, computes energy from measurement counts,
and saves a validation result JSON.

Usage:
    python src/gqe/eval/collect_qpu.py --job-id <job_id> --out results/phase3_final/qpu/qpu_result.json
    python src/gqe/eval/collect_qpu.py --submission results/phase3_final/qpu/qpu_submission.json --out results/phase3_final/qpu/qpu_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def collect_job(job_id: str, shots: int = 4096) -> dict[str, Any]:
    """Retrieve job results from qBraid."""
    from qbraid.runtime import load_job

    print(f"Loading job: {job_id}")
    job = load_job(job_id)

    status = str(job.status())
    print(f"  Status: {status}")

    if "running" in status.lower() or "queued" in status.lower() or "pending" in status.lower():
        print("  Job still in progress. Try again later.")
        return {"job_id": job_id, "status": status, "completed": False}

    if "fail" in status.lower() or "cancel" in status.lower():
        print(f"  Job failed/cancelled: {status}")
        return {"job_id": job_id, "status": status, "completed": False, "error": "Job did not complete successfully"}

    # Get results
    print("  Fetching results...")
    t0 = time.time()
    result = job.result()
    runtime = time.time() - t0

    # Extract counts
    counts = {}
    if hasattr(result, 'get_counts'):
        counts = result.get_counts()
    elif hasattr(result, 'counts'):
        counts = result.counts
    elif isinstance(result, dict):
        counts = result.get('counts', result)
    else:
        # Try to access raw measurement counts
        try:
            counts = dict(result.data().get_counts())
        except Exception:
            counts = {"raw_result": str(result)}

    print(f"  Retrieved in {runtime:.2f}s")
    print(f"  Counts: {counts}")

    return {
        "job_id": job_id,
        "status": status,
        "completed": True,
        "counts": counts,
        "retrieve_time_seconds": runtime,
        "retrieve_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect QPU results from qBraid")
    parser.add_argument("--job-id", type=str, default=None)
    parser.add_argument("--submission", type=Path, default=None,
                        help="Submission JSON file (alternative to --job-id)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Get job ID
    job_id = args.job_id
    submission_data = None
    if job_id is None and args.submission:
        with open(args.submission) as f:
            submission_data = json.load(f)
        job_id = submission_data.get("submission", {}).get("job_id")

    if job_id is None:
        parser.error("Either --job-id or --submission with job_id is required")

    print("=== Collecting QPU Results ===")
    result = collect_job(job_id)

    # Merge with submission data if available
    if submission_data:
        result["source"] = submission_data.get("source", {})
        result["device_id"] = submission_data.get("submission", {}).get("device_id")
        result["shots"] = submission_data.get("submission", {}).get("shots", 4096)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
