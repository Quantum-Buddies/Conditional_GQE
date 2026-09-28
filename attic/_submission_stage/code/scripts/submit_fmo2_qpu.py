#!/usr/bin/env python3
"""Export SQD manifests for 8q dimer circuits and submit to Cepheus QPU.

This is the FMO2 scaling QPU validation: submit the 3 dimer circuits (8q each)
to the Rigetti Cepheus-1-108Q QPU. Monomers (4q) are too small to be interesting
on QPU — the dimers are the largest sub-circuits in the FMO2 expansion.

Usage:
    # Export only (no QPU credits spent):
    python scripts/submit_fmo2_qpu.py --export-only

    # Export + submit to Cepheus:
    python scripts/submit_fmo2_qpu.py --submit

    # Use a different device:
    python scripts/submit_fmo2_qpu.py --submit --device aws:aws:sim:sv1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.eval.qbraid_backend import export_sqd_sampling_circuit

DEVICE = "aws:rigetti:qpu:cepheus-1-108q"
DEFAULT_SHOTS = 4096


def main():
    parser = argparse.ArgumentParser(description="Submit FMO2 dimer circuits to QPU")
    parser.add_argument("--export-only", action="store_true", help="Export manifests without submitting")
    parser.add_argument("--submit", action="store_true", help="Submit to QPU after export")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    args = parser.parse_args()

    if not args.export_only and not args.submit:
        args.export_only = True  # default to export-only

    # Load dimer Hamiltonians
    with open(ROOT / "results/data/fragments/dimers.json") as f:
        dimers = json.load(f)["records"]

    # Load RL best circuits to get operator sequences
    with open(ROOT / "results/train/h_cgqe_model_qbraid_rl_best_circuits.json") as f:
        rl_data = json.load(f)
    bc = rl_data.get("best_circuits", rl_data)

    # Find RL circuit with matching qubit count for 8q dimers
    # Use the best-energy circuit from any 8q molecule
    rl_circuits_8q = []
    for mol_name, info in bc.items():
        if isinstance(info, dict) and "operators" in info:
            nq = info.get("n_qubits", 0)
            if nq == 8:
                rl_circuits_8q.append((mol_name, info))

    if rl_circuits_8q:
        rl_circuits_8q.sort(key=lambda x: x[1].get("energy", 1e9))
        best_8q = rl_circuits_8q[0][1]
        print(f"Using RL circuit from {rl_circuits_8q[0][0]} (E={best_8q['energy']:.6f})")
        operators = best_8q["operators"]
    else:
        # Fallback: use h2 circuit (4q) — pad/truncate for 8q
        print("No 8q RL circuit found, using h2 operators as fallback")
        operators = bc["h2"]["operators"]

    # Also find 4q circuits for monomers
    rl_circuits_4q = []
    for mol_name, info in bc.items():
        if isinstance(info, dict) and "operators" in info:
            nq = info.get("n_qubits", 0)
            if nq == 4:
                rl_circuits_4q.append((mol_name, info))
    best_4q = None
    if rl_circuits_4q:
        rl_circuits_4q.sort(key=lambda x: x[1].get("energy", 1e9))
        best_4q = rl_circuits_4q[0][1]
        print(f"Using 4q RL circuit from {rl_circuits_4q[0][0]} (E={best_4q['energy']:.6f})")

    # Load monomers too
    with open(ROOT / "results/data/fragments/monomers.json") as f:
        monomers = json.load(f)["records"]

    print(f"\nDevice: {args.device}")
    print(f"Shots: {args.shots}")
    print(f"Dimers: {len(dimers)} × 8q")
    print(f"Monomers: {len(monomers)} × 4q")

    # Cost estimate
    n_circuits = len(dimers) + len(monomers)
    cost_per = 30 + args.shots * 0.0425
    total_est = n_circuits * cost_per
    print(f"Estimated cost: {total_est:.0f} credits ({n_circuits} tasks)")

    manifests = []

    # Export dimer manifests
    print("\n--- Exporting dimer manifests ---")
    for r in dimers:
        mol_name = r["name"]
        nq = r["n_qubits"]
        ops = operators if nq == 8 else (best_4q["operators"] if best_4q else operators)
        # Use zero thetas — RL model already optimized the operator sequence
        thetas = np.zeros(len(ops))

        manifest_path = ROOT / f"results/qpu/fmo2_{mol_name}_sqd_manifest.json"
        try:
            manifest = export_sqd_sampling_circuit(
                r, ops, thetas, args.device, args.shots, manifest_path
            )
            manifests.append((mol_name, manifest_path))
            print(f"  {mol_name}: {nq}q, {len(ops)} ops, depth={manifest['circuit_depth']}")
        except Exception as e:
            print(f"  {mol_name}: FAILED - {e}")

    # Export monomer manifests
    if best_4q:
        print("\n--- Exporting monomer manifests ---")
        for r in monomers:
            mol_name = r["name"]
            nq = r["n_qubits"]
            ops = best_4q["operators"]
            thetas = np.zeros(len(ops))

            manifest_path = ROOT / f"results/qpu/fmo2_{mol_name}_sqd_manifest.json"
            try:
                manifest = export_sqd_sampling_circuit(
                    r, ops, thetas, args.device, args.shots, manifest_path
                )
                manifests.append((mol_name, manifest_path))
                print(f"  {mol_name}: {nq}q, {len(ops)} ops, depth={manifest['circuit_depth']}")
            except Exception as e:
                print(f"  {mol_name}: FAILED - {e}")

    print(f"\nExported {len(manifests)} manifests to results/qpu/")

    if args.export_only and not args.submit:
        print("\nExport-only mode. Submit later with:")
        print(f"  python scripts/submit_fmo2_qpu.py --submit --device {args.device}")
        return

    # Submit to QPU
    print(f"\n{'='*60}")
    print(f"Submitting {len(manifests)} circuits to {args.device}")
    print(f"{'='*60}\n")

    try:
        from qbraid import QbraidProvider
        from qiskit.qasm2 import loads
    except ImportError:
        print("ERROR: qbraid-sdk or qiskit not installed. Manifests exported for later submission.")
        return

    provider = QbraidProvider()
    device_obj = provider.get_device(args.device)

    submission_meta = {
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shots": args.shots,
        "fmo2_context": "3-fragment iodobenzene scaling",
        "sqd_jobs": {},
    }

    for mol_name, manifest_path in manifests:
        with open(manifest_path) as f:
            manifest = json.load(f)

        qasm = manifest["circuit_qasm"]
        nq = manifest["n_qubits"]
        print(f"  {mol_name} ({nq}q)...")

        try:
            qc = loads(qasm)
            job = device_obj.run(qc, shots=args.shots)
            jid = str(job.id)
            print(f"    Job ID: {jid}")
            submission_meta["sqd_jobs"][mol_name] = {
                "job_id": jid,
                "shots": args.shots,
                "n_qubits": nq,
                "manifest": str(manifest_path),
            }
        except Exception as e:
            print(f"    FAILED: {e}")
            submission_meta["sqd_jobs"][mol_name] = {"error": str(e)}

    meta_path = ROOT / "results/qpu/fmo2_cepheus_submission_meta.json"
    with open(meta_path, "w") as f:
        json.dump(submission_meta, f, indent=2)
    print(f"\nSubmission metadata: {meta_path}")
    print(f"\nRetrieve with:")
    print(f"  python scripts/retrieve_and_sqd.py --meta {meta_path} --hamiltonians results/data/fragments/dimers.json --out results/qpu/fmo2_cepheus_sqd_results.json")


if __name__ == "__main__":
    main()
