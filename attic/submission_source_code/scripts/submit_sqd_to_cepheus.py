#!/usr/bin/env python3
"""Submit SQD sampling circuits to Rigetti Cepheus-1-108Q via qBraid.

Also submits QWC diagnostic circuits for H2 as a cross-check.
All submissions are tracked in the SQLite ledger with budget enforcement.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qbraid import QbraidProvider
from src.gqe.eval.qpu_ledger import QpuLedger, LedgerEntry, JobStatus


def submit_sqd_circuit(provider, device, manifest_path: Path) -> str | None:
    """Submit a single SQD manifest to the QPU. Returns qBraid job ID or None."""
    from qiskit.qasm2 import loads

    with open(manifest_path) as f:
        manifest = json.load(f)

    qasm = manifest["circuit_qasm"]
    shots = manifest["shots"]
    mol = manifest["molecule"]
    nq = manifest["n_qubits"]
    print(f"  Submitting {mol} SQD ({nq}q, {shots} shots)...")
    try:
        qc = loads(qasm)
        job = device.run(qc, shots=shots)
        jid = str(job.id)
        print(f"    qBraid job ID: {jid}")
        return jid
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def submit_qwc_circuits(provider, device, manifest_path: Path, shots: int = 1024) -> list[str]:
    """Submit QWC diagnostic circuits from a QWC manifest."""
    from qiskit.qasm2 import loads

    with open(manifest_path) as f:
        manifest = json.load(f)

    mol = manifest["molecule"]
    groups = manifest.get("groups", [])
    print(f"  Submitting {mol} QWC ({len(groups)} circuits, {shots} shots each)...")

    job_ids = []
    for gi, group in enumerate(groups):
        qasm = group["qasm"]
        try:
            qc = loads(qasm)
            job = device.run(qc, shots=shots)
            jid = str(job.id)
            job_ids.append(jid)
            print(f"    Group {gi}: {jid}")
        except Exception as e:
            print(f"    Group {gi} FAILED: {e}")
            job_ids.append(None)
    return job_ids


def main() -> None:
    provider = QbraidProvider()
    device_id = "aws:rigetti:qpu:cepheus-1-108q"

    try:
        device = provider.get_device(device_id)
        print(f"Device: {device}")
    except Exception as e:
        print(f"Cannot get device {device_id}: {e}")
        print("Available devices:")
        try:
            devices = provider.get_devices()
            for d in devices:
                print(f"  {d}")
        except Exception:
            pass
        return

    ledger = QpuLedger(ROOT / "results/eval/qpu_jobs.sqlite")
    ledger.set_budget(13400.0)

    # --- SQD submissions ---
    sqd_results = {}
    for mol in ["h2", "lih", "beh2"]:
        manifest_path = ROOT / f"results/qpu/{mol}_sqd_cepheus_manifest.json"
        if not manifest_path.exists():
            print(f"\n{mol}: no SQD manifest found at {manifest_path}")
            continue
        jid = submit_sqd_circuit(provider, device, manifest_path)
        sqd_results[mol] = jid

        if jid:
            # Record in ledger
            with open(manifest_path) as f:
                manifest = json.load(f)
            entry = LedgerEntry(
                circuit_hash=manifest["circuit_hash"],
                molecule=mol,
                device_id=device_id,
                shots=manifest["shots"],
                n_qubits=manifest["n_qubits"],
                n_circuits=1,
                measurement_basis="computational_basis",
                pipeline_stage="sqd_pilot",
                metadata={"qbraid_job_id": jid, "operators": manifest["operators"]},
            )
            try:
                ledger.submit(entry, job_id=f"{mol}_cepheus_sqd_{int(time.time())}")
            except ValueError:
                pass  # duplicate, fine

    # --- QWC diagnostic for H2 ---
    qwc_results = {}
    h2_qwc_manifest = ROOT / "results/qpu/h2_0.74_manifest.json"
    if h2_qwc_manifest.exists():
        job_ids = submit_qwc_circuits(provider, device, h2_qwc_manifest, shots=4096)
        qwc_results["h2"] = job_ids

    # --- Save submission metadata ---
    submission_meta = {
        "device": device_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sqd_jobs": sqd_results,
        "qwc_jobs": qwc_results,
    }
    meta_path = ROOT / "results/qpu/cepheus_submission_meta.json"
    with open(meta_path, "w") as f:
        json.dump(submission_meta, f, indent=2)
    print(f"\nSubmission metadata: {meta_path}")

    # Print ledger status
    print("\n=== Ledger Status ===")
    for entry in ledger.list_jobs():
        print(f"  {entry.job_id}: {entry.molecule} {entry.measurement_basis} "
              f"status={entry.status.value} est={entry.estimated_cost_credits}")

    total_est = sum(e.estimated_cost_credits or 0 for e in ledger.list_jobs())
    print(f"\nTotal estimated credits: {total_est:.2f}")
    print(f"Budget remaining: {13400.0 - total_est:.2f}")

    ledger.close()


if __name__ == "__main__":
    main()
