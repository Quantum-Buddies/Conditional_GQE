#!/usr/bin/env python3
"""Submit only preflight-accepted SQD manifests after explicit approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qbraid import QbraidProvider
from qiskit.qasm2 import loads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required to make paid provider calls; omission is a safe dry-run.",
    )
    parser.add_argument(
        "--approve-cost",
        type=float,
        default=None,
        help="Must exactly match the projected bundle cost.",
    )
    args = parser.parse_args()

    with args.bundle.open(encoding="utf-8") as handle:
        bundle = json.load(handle)
    manifests = [Path(path) for path in bundle["manifests"]]
    projected = float(bundle["projected_cost_credits"])
    print(
        f"{len(manifests)} accepted SQD jobs; projected cost "
        f"{projected:.2f} credits on {bundle['device']}"
    )
    if not args.execute:
        print("DRY RUN: no qBraid provider was created and no jobs were submitted.")
        return
    if args.approve_cost is None or abs(args.approve_cost - projected) > 1e-9:
        raise SystemExit(
            f"Refusing submission: --approve-cost must equal {projected:.2f}"
        )

    provider = QbraidProvider()
    device = provider.get_device(bundle["device"])
    job_ids: dict[str, str] = {}
    for manifest_path in manifests:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("measurement_basis") != "computational_basis":
            raise ValueError(f"Non-SQD manifest refused: {manifest_path}")
        circuit = loads(manifest["circuit_qasm"])
        job = device.run(circuit, shots=int(manifest["shots"]))
        job_ids[manifest["molecule"]] = str(job.id)
        print(f"{manifest['molecule']}: {job.id}")
    print(json.dumps({"device": bundle["device"], "job_ids": job_ids}, indent=2))


if __name__ == "__main__":
    main()
