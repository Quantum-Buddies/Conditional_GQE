#!/usr/bin/env python3
"""QPU preflight check for GIC Phase 3.

Enumerates online qBraid QPUs, prints capabilities/pricing/queue metadata,
creates a cost estimate, and saves a sanitized device snapshot.

Never prints or stores API key values.

Usage:
    python scripts/qpu_preflight.py [--out results/gic2026/manifests/preflight.json]
    python scripts/qpu_preflight.py --dry-run
    python scripts/qpu_preflight.py --device aws:rigetti:qpu:cepheus-1-108q --shots 1024 --n-circuits 15
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.common.run_manifest import create_run_manifest, save_run_manifest


# Known pricing (credits) — update from qBraid dashboard as needed
KNOWN_PRICING = {
    "aws:rigetti:qpu:cepheus-1-108q": {"per_task": 30, "per_shot": 0.0425, "max_qubits": 108},
    "aws:iqm:qpu:garnet": {"per_task": 30, "per_shot": 0.145, "max_qubits": 20},
    "aws:iqm:qpu:emerald": {"per_task": 30, "per_shot": 0.16, "max_qubits": 54},
    "ionq:ionq:qpu.aria-1": {"per_task": 30, "per_shot": 3.0, "max_qubits": 25},
    "ionq:ionq:qpu.forte-1": {"per_task": 30, "per_shot": 8.0, "max_qubits": 30},
    "aqt:aqt:qpu.ibex_q1": {"per_task": 30, "per_shot": 2.35, "max_qubits": 24},
    "qbraid:qbraid:sim:qir-sv": {"per_task": 0, "per_shot": 0, "max_qubits": 30},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_devices() -> list[dict]:
    """List available qBraid devices with sanitized metadata."""
    try:
        from qbraid import QbraidProvider
    except ImportError:
        print("qBraid SDK not installed. Install with: pip install qbraid")
        return []

    if not os.environ.get("QBRAID_API_KEY"):
        print("QBRAID_API_KEY not set. Cannot list remote devices.")
        return []

    provider = QbraidProvider()
    devices = []
    try:
        raw_devices = provider.get_devices()
        for d in raw_devices:
            entry = {
                "id": d.id,
                "status": str(d.status()),
                "num_qubits": getattr(d, "num_qubits", None),
                "provider": getattr(d, "provider", None),
                "pricing": KNOWN_PRICING.get(d.id, {}),
            }
            devices.append(entry)
    except Exception as e:
        print(f"Error listing devices: {e}")
    return devices


def estimate_cost(device_id: str, shots: int, n_circuits: int) -> dict:
    """Estimate qBraid credit cost for a batch submission."""
    pricing = KNOWN_PRICING.get(device_id, {})
    per_task = pricing.get("per_task", 0)
    per_shot = pricing.get("per_shot", 0)

    # Batch: 1 task fee + n_circuits * shots * per_shot
    batch_cost = per_task + n_circuits * shots * per_shot
    # Individual: n_circuits * (per_task + shots * per_shot)
    individual_cost = n_circuits * (per_task + shots * per_shot)
    savings = individual_cost - batch_cost

    return {
        "device": device_id,
        "shots": shots,
        "n_circuits": n_circuits,
        "batch_cost_credits": round(batch_cost, 2),
        "individual_cost_credits": round(individual_cost, 2),
        "savings_credits": round(savings, 2),
        "per_task_credits": per_task,
        "per_shot_credits": per_shot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QPU preflight for GIC Phase 3")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "gic2026" / "manifests" / "preflight.json")
    parser.add_argument("--device", type=str, default=None, help="Specific device ID to check")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per circuit for cost estimate")
    parser.add_argument("--n-circuits", type=int, default=15, help="Number of circuits for cost estimate")
    parser.add_argument(
        "--max-credits",
        type=float,
        default=None,
        help="Refuse to proceed if the estimated batch cost exceeds this credit budget",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip live API calls; use known pricing only")
    args = parser.parse_args()

    print("=== QPU Preflight for GIC Phase 3 ===")
    print(f"Timestamp: {_utc_now()}")
    print()

    # Check API key presence
    has_key = bool(os.environ.get("QBRAID_API_KEY"))
    print(f"QBRAID_API_KEY: {'set (value not shown)' if has_key else 'NOT SET'}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping live API calls.")
        devices = []
    else:
        print("\nQuerying available devices...")
        devices = list_devices()

    if not devices and not args.dry_run:
        print("No devices retrieved. Check API key and network.")
        print("Falling back to known pricing table for cost estimates.")
    elif devices:
        print(f"\nFound {len(devices)} device(s):")
        for d in devices:
            status_marker = "ONLINE" if "online" in d["status"].lower() else d["status"]
            pricing = d.get("pricing", {})
            nq = d.get('num_qubits', '?')
            nq_str = str(nq) if nq is not None else '?'
            print(f"  {d['id']:45s} | {status_marker:10s} | {nq_str:>4s}q | "
                  f"task={pricing.get('per_task', '?')}cr shot={pricing.get('per_shot', '?')}cr")

    # Cost estimate
    target_device = args.device or "aws:rigetti:qpu:cepheus-1-108q"
    cost = estimate_cost(target_device, args.shots, args.n_circuits)
    print(f"\nCost estimate for {target_device}:")
    print(f"  Shots per circuit: {args.shots}")
    print(f"  Number of circuits: {args.n_circuits}")
    print(f"  Batch cost:   {cost['batch_cost_credits']} credits")
    print(f"  Individual:   {cost['individual_cost_credits']} credits")
    print(f"  Savings:      {cost['savings_credits']} credits")

    budget_exceeded = (
        args.max_credits is not None
        and cost["batch_cost_credits"] > args.max_credits
    )
    if budget_exceeded:
        print(
            f"\n[REFUSED] Estimated batch cost {cost['batch_cost_credits']} credits "
            f"exceeds budget {args.max_credits} credits."
        )

    # Build preflight manifest
    manifest = create_run_manifest(
        command=f"python scripts/qpu_preflight.py --device {target_device} --shots {args.shots} --n-circuits {args.n_circuits}"
            + (" --dry-run" if args.dry_run else ""),
        extra={
            "preflight_type": "qpu_cost_estimate",
            "has_qbraid_api_key": has_key,
            "devices": devices,
            "cost_estimate": cost,
            "max_credits": args.max_credits,
            "budget_exceeded": budget_exceeded,
        },
    )

    out = save_run_manifest(manifest, args.out)
    print(f"\nPreflight manifest saved to: {out}")
    print("=== Preflight complete ===")
    if budget_exceeded:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
