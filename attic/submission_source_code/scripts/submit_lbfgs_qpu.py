#!/usr/bin/env python3
"""Submit L-BFGS-B optimized H-cGQE circuits to Cepheus QPU.

Wave 1: h2 (4q), lih (12q), beh2 (14q) — with L-BFGS-B optimized thetas
         (bi-level pipeline: RL topology → L-BFGS-B angles → QPU → SQD)
Wave 2: methyl_iodide_cas12 (12q), iodobenzene 8q — GIC benchmark molecules

Also submits zero-theta versions for direct comparison on the same QPU.

Usage:
    # Export only:
    python scripts/submit_lbfgs_qpu.py --export-only

    # Submit to Cepheus:
    python scripts/submit_lbfgs_qpu.py --submit
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


def load_hamiltonian(name: str) -> dict | None:
    """Load a Hamiltonian record by molecule name from available datasets."""
    for path in [
        ROOT / "results/data/hamiltonians_gic2026/hamiltonians.json",
        ROOT / "results/data/hamiltonians_merged.json",
        ROOT / "results/data/hamiltonians.json",
    ]:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        records = data if isinstance(data, list) else data.get("records", data.get("hamiltonians", []))
        for r in records:
            if r.get("name") == name:
                return r
    return None


def main():
    parser = argparse.ArgumentParser(description="Submit L-BFGS-B optimized circuits to QPU")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--include-zero-theta", action="store_true",
                        help="Also submit zero-theta versions for comparison")
    args = parser.parse_args()

    if not args.export_only and not args.submit:
        args.export_only = True

    # Load L-BFGS-B optimized circuits (with thetas)
    with open(ROOT / "results/eval/h_cgqe_rl_optimized.json") as f:
        rl_opt = json.load(f)

    # Load RL best circuits (for operator sequences of molecules not in rl_opt)
    with open(ROOT / "results/train/h_cgqe_model_qbraid_rl_best_circuits.json") as f:
        bc = json.load(f).get("best_circuits", {})

    # Build submission plan
    submissions = []

    # Wave 1: L-BFGS-B optimized h2, lih, beh2
    print("=== Wave 1: L-BFGS-B Optimized Circuits ===")
    for entry in rl_opt:
        mol = entry["molecule"]
        nq = entry["n_qubits"]
        thetas = entry.get("best_thetas", [])
        operators = entry.get("best_operators", [])
        ham = load_hamiltonian(mol)
        if ham is None:
            print(f"  {mol}: Hamiltonian not found, skipping")
            continue
        if not operators:
            # Fall back to RL best circuits
            if mol in bc and "operators" in bc[mol]:
                operators = bc[mol]["operators"]
                print(f"  {mol}: Using RL operators (no L-BFGS-B operators saved)")
            else:
                print(f"  {mol}: No operators found, skipping")
                continue
        if len(thetas) != len(operators):
            print(f"  {mol}: thetas({len(thetas)}) != operators({len(operators)}), using zero thetas")
            thetas = [0.0] * len(operators)

        submissions.append({
            "name": f"{mol}_lbfgs",
            "molecule": mol,
            "n_qubits": nq,
            "hamiltonian": ham,
            "operators": operators,
            "thetas": thetas,
            "wave": 1,
            "label": "L-BFGS-B optimized",
        })
        print(f"  {mol}: {nq}q, {len(operators)} ops, {len(thetas)} thetas")

        # Also add zero-theta version for comparison
        if args.include_zero_theta:
            submissions.append({
                "name": f"{mol}_zero_theta",
                "molecule": mol,
                "n_qubits": nq,
                "hamiltonian": ham,
                "operators": operators,
                "thetas": [0.0] * len(operators),
                "wave": 1,
                "label": "zero-theta baseline",
            })

    # Wave 2: GIC benchmark molecules
    print("\n=== Wave 2: GIC Benchmark Molecules ===")
    wave2_mols = ["methyl_iodide_cas12", "iodobenzene"]
    for mol in wave2_mols:
        ham = load_hamiltonian(mol)
        if ham is None:
            print(f"  {mol}: Hamiltonian not found, skipping")
            continue
        nq = ham["n_qubits"]

        # Get operators from RL best circuits
        if mol in bc and "operators" in bc[mol]:
            operators = bc[mol]["operators"]
            print(f"  {mol}: {nq}q, {len(operators)} RL operators")
        elif mol == "iodobenzene" and nq == 8:
            # No 8q RL circuit — use h2 operators (padded)
            operators = bc.get("h2", {}).get("operators", [])
            print(f"  {mol}: {nq}q, using h2 operators (padded to 8q)")
        else:
            print(f"  {mol}: No RL circuit found, skipping")
            continue

        if not operators:
            print(f"  {mol}: Empty operators, skipping")
            continue

        thetas = [0.0] * len(operators)
        submissions.append({
            "name": f"{mol}_rl",
            "molecule": mol,
            "n_qubits": nq,
            "hamiltonian": ham,
            "operators": operators,
            "thetas": thetas,
            "wave": 2,
            "label": "RL topology (zero theta)",
        })

    print(f"\nTotal circuits: {len(submissions)}")
    print(f"Device: {args.device}")
    print(f"Shots: {args.shots}")

    # Cost estimate
    cost_per = 30 + args.shots * 0.0425
    total_est = len(submissions) * cost_per
    print(f"Estimated cost: {total_est:.0f} credits ({len(submissions)} tasks)")

    # Export manifests
    print(f"\n--- Exporting manifests ---")
    manifests = []
    for sub in submissions:
        name = sub["name"]
        nq = sub["n_qubits"]
        ops = sub["operators"]
        thetas = np.array(sub["thetas"], dtype=float)
        ham = sub["hamiltonian"]

        manifest_path = ROOT / f"results/qpu/lbfgs_{name}_sqd_manifest.json"
        try:
            manifest = export_sqd_sampling_circuit(
                ham, ops, thetas, args.device, args.shots, manifest_path
            )
            manifests.append((sub, manifest_path))
            print(f"  {name}: {nq}q, {len(ops)} ops, depth={manifest['circuit_depth']}, thetas_max={max(abs(t) for t in thetas):.4f}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    print(f"\nExported {len(manifests)} manifests")

    if args.export_only and not args.submit:
        print("\nExport-only mode. Submit with:")
        print(f"  python scripts/submit_lbfgs_qpu.py --submit --device {args.device}")
        return

    # Submit to QPU
    print(f"\n{'='*60}")
    print(f"Submitting {len(manifests)} circuits to {args.device}")
    print(f"{'='*60}\n")

    try:
        from qbraid import QbraidProvider
        from qiskit.qasm2 import loads
    except ImportError:
        print("ERROR: qbraid-sdk or qiskit not installed.")
        return

    provider = QbraidProvider()
    device_obj = provider.get_device(args.device)

    submission_meta = {
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shots": args.shots,
        "description": "Bi-level pipeline QPU validation: RL topology + L-BFGS-B angles",
        "sqd_jobs": {},
    }

    for sub, manifest_path in manifests:
        name = sub["name"]
        with open(manifest_path) as f:
            manifest = json.load(f)

        qasm = manifest["circuit_qasm"]
        nq = manifest["n_qubits"]
        print(f"  {name} ({nq}q, {sub['label']})...")

        try:
            qc = loads(qasm)
            job = device_obj.run(qc, shots=args.shots)
            jid = str(job.id)
            print(f"    Job ID: {jid}")
            submission_meta["sqd_jobs"][name] = {
                "job_id": jid,
                "shots": args.shots,
                "n_qubits": nq,
                "molecule": sub["molecule"],
                "wave": sub["wave"],
                "label": sub["label"],
                "manifest": str(manifest_path),
                "n_operators": len(sub["operators"]),
                "thetas": sub["thetas"],
            }
        except Exception as e:
            print(f"    FAILED: {e}")
            submission_meta["sqd_jobs"][name] = {"error": str(e)}

        time.sleep(2)  # Rate limit

    meta_path = ROOT / "results/qpu/lbfgs_cepheus_submission_meta.json"
    with open(meta_path, "w") as f:
        json.dump(submission_meta, f, indent=2)
    print(f"\nSubmission metadata: {meta_path}")


if __name__ == "__main__":
    main()
