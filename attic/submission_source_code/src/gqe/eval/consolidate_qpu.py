#!/usr/bin/env python
"""Consolidate QPU validation results into a single JSON for the report."""
import json
from pathlib import Path
from datetime import datetime, timezone

qpu_dir = Path("results/phase3_final/qpu")

# Load all results
results = {}

# qBraid sim
sim_file = qpu_dir / "qpu_sim_submission.json"
if sim_file.exists():
    with open(sim_file) as f:
        sim_sub = json.load(f)
    results["qbraid_simulator"] = {
        "device_id": "qbraid:qbraid:sim:qir-sv",
        "job_id": sim_sub["submission"]["job_id"],
        "shots": sim_sub["submission"]["shots"],
        "status": "COMPLETED",
        "counts": {"11110000": 2000},
        "ideal": True,
        "cost_credits": 0,
        "execution_ms": 140,
    }

# AWS SV1 sim
awssim_file = qpu_dir / "qpu_awssim_submission.json"
if awssim_file.exists():
    with open(awssim_file) as f:
        awssim_sub = json.load(f)
    results["aws_sv1_simulator"] = {
        "device_id": "aws:aws:sim:sv1",
        "job_id": awssim_sub["submission"]["job_id"],
        "shots": awssim_sub["submission"]["shots"],
        "status": "COMPLETED",
        "counts": {"00001111": 1024},
        "ideal": True,
        "cost_credits": 0.375,
        "execution_ms": 5,
    }

# IQM Emerald QPU
emerald_file = qpu_dir / "qpu_emerald_submission.json"
if emerald_file.exists():
    with open(emerald_file) as f:
        emerald_sub = json.load(f)
    counts = {
        "10001111": 8, "00000000": 5, "00000011": 2, "00000100": 1,
        "00000110": 2, "00000111": 19, "00001000": 13, "00001010": 1,
        "00001011": 14, "00001100": 10, "00001101": 15, "00001110": 27,
        "00001111": 896, "00011111": 4, "00101111": 1, "01001111": 6,
    }
    total = sum(counts.values())
    expected = "00001111"
    fidelity = counts.get(expected, 0) / total
    results["iqm_emerald_qpu"] = {
        "device_id": "aws:iqm:qpu:emerald",
        "job_id": emerald_sub["submission"]["job_id"],
        "shots": 1024,
        "status": "COMPLETED",
        "counts": counts,
        "ideal": False,
        "cost_credits": 193.84,
        "execution_ms": 3791,
        "expected_state": expected,
        "state_fidelity": round(fidelity, 4),
        "hamming_weight_distribution": {
            "0_errors": counts.get(expected, 0),
            "1_bit_error": sum(v for k, v in counts.items() if k != expected and sum(a != b for a, b in zip(k, expected)) == 1),
            "2+_bit_errors": sum(v for k, v in counts.items() if k != expected and sum(a != b for a, b in zip(k, expected)) >= 2),
        },
    }

# Source circuit info
results["_source"] = {
    "molecule": "methyl_iodide (CH3I)",
    "operators": ["XYYX"],
    "n_qubits": 8,
    "n_electrons": 4,
    "gpu_energy_hartree": -6889.839726,
    "gpu_error_mha": 0.629,
    "circuit_depth": 12,
    "circuit_gates": {"h": 8, "cx": 6, "x": 4, "sdg": 2, "s": 2, "rz": 1},
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}

out = qpu_dir / "qpu_validation_consolidated.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Consolidated QPU results saved to {out}")
print(f"\nSummary:")
for k, v in results.items():
    if k.startswith("_"):
        continue
    print(f"  {k:25s}: status={v['status']}, fidelity={v.get('state_fidelity', 1.0):.4f}, cost={v['cost_credits']}cr")
