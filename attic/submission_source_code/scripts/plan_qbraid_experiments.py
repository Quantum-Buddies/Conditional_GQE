#!/usr/bin/env python3
"""Plan qBraid GPU and QPU experiments without submitting jobs.

This keeps credit usage explicit before spending the remaining qBraid allocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GPU_CREDITS_PER_HOUR = {
    "gpu-l40s": 228,
    "gpu-gh200": 287,
    "gpu-h200": 549,
    "gpu-b200": 874,
    "gpu-b200-4x": 3395,
}

QPU_PRICING = {
    "aws:rigetti:qpu:cepheus-1-108q": {"per_task": 30, "per_shot": 0.0425},
    "aws:iqm:qpu:emerald": {"per_task": 30, "per_shot": 0.16},
    "aws:iqm:qpu:garnet": {"per_task": 30, "per_shot": 0.145},
    "ionq:ionq:qpu.aria-1": {"per_task": 30, "per_shot": 3.0},
    "qbraid:qbraid:sim:qir-sv": {"per_task": 0, "per_shot": 0.0},
}


def qpu_cost(device: str, circuits: int, shots: int) -> float:
    pricing = QPU_PRICING[device]
    return pricing["per_task"] + circuits * shots * pricing["per_shot"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, required=True, help="Available qBraid credits")
    parser.add_argument("--gpu-instance", default="gpu-h200")
    parser.add_argument("--gpu-hours", type=float, default=4.0)
    parser.add_argument("--qpu-device", default="aws:rigetti:qpu:cepheus-1-108q")
    parser.add_argument("--qpu-circuits", type=int, default=8)
    parser.add_argument("--qpu-shots", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.gpu_instance not in GPU_CREDITS_PER_HOUR:
        raise SystemExit(f"Unknown GPU instance: {args.gpu_instance}")
    if args.qpu_device not in QPU_PRICING:
        raise SystemExit(f"Unknown QPU device: {args.qpu_device}")

    gpu_cost = GPU_CREDITS_PER_HOUR[args.gpu_instance] * args.gpu_hours
    qpu = qpu_cost(args.qpu_device, args.qpu_circuits, args.qpu_shots)
    total = gpu_cost + qpu
    plan = {
        "budget_credits": args.budget,
        "gpu": {
            "instance": args.gpu_instance,
            "hours": args.gpu_hours,
            "estimated_credits": round(gpu_cost, 2),
        },
        "qpu": {
            "device": args.qpu_device,
            "circuits": args.qpu_circuits,
            "shots_per_circuit": args.qpu_shots,
            "estimated_credits": round(qpu, 2),
        },
        "estimated_total_credits": round(total, 2),
        "remaining_credits": round(args.budget - total, 2),
        "within_budget": total <= args.budget,
    }

    print(json.dumps(plan, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"Plan written to {args.out}")
    if not plan["within_budget"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
