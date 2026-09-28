#!/usr/bin/env python3
"""Optimize RL circuits for all available molecules and submit to Cepheus QPU.

Extends rl_optimize_and_submit.py to handle ALL molecules with RL circuits,
not just h2/lih/beh2. Filters by statevector limit (<=24q) for L-BFGS-B.

Usage:
    python scripts/submit_more_qpu.py --molecules all
    python scripts/submit_more_qpu.py --molecules methyl_iodide_cas12 benzene_cas12 n2
    python scripts/submit_more_qpu.py --molecules all --export-only  # skip QPU submission
    python scripts/submit_more_qpu.py --molecules all --shots 8192
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import cudaq
cudaq.set_target("nvidia")

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
    get_active_electron_count,
    hamiltonian_to_spin_operator,
)
from src.gqe.eval.optimize_h_cgqe_coefficients import (
    _build_kernel_for_sequence,
    _evaluate_energy,
    _pad_pauli_word,
)
from src.gqe.eval.qbraid_backend import export_sqd_sampling_circuit
from scipy.optimize import minimize


MAX_QUBITS_SV = 24  # statevector limit on L40S
DEFAULT_SHOTS = 4096
DEVICE = "aws:rigetti:qpu:cepheus-1-108q"


def optimize_single_molecule(
    record: dict,
    operators: list[str],
    max_iter: int = 50,
    n_starts: int = 4,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Run L-BFGS-B optimization for a single molecule."""
    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)
    spin_ham = hamiltonian_to_spin_operator(record)

    padded = [_pad_pauli_word(w, n_qubits) for w in operators]
    pauli_words = [cudaq.pauli_word(w) for w in padded]

    kernel, _ = _build_kernel_for_sequence(n_qubits, n_electrons, operators)

    rng = np.random.default_rng(seed)
    best_energy = 1e9
    best_thetas = None

    for start in range(n_starts):
        initial = rng.uniform(-0.1, 0.1, size=len(operators))

        def cost_fn(thetas):
            return _evaluate_energy(thetas, kernel, spin_ham, n_qubits, n_electrons, pauli_words)

        result = minimize(
            cost_fn,
            initial,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-8},
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_thetas = result.x

    return best_energy, best_thetas


def main():
    parser = argparse.ArgumentParser(description="Optimize and submit more molecules to QPU")
    parser.add_argument("--molecules", nargs="+", default=["all"],
                        help="Molecule names or 'all' for all available")
    parser.add_argument("--export-only", action="store_true",
                        help="Export manifests without submitting to QPU")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--max-qubits", type=int, default=MAX_QUBITS_SV,
                        help="Max qubits for statevector optimization")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip molecules that already have manifests")
    args = parser.parse_args()

    # Load RL best circuits
    with open(ROOT / "results/train/h_cgqe_model_qbraid_rl_best_circuits.json") as f:
        rl_data = json.load(f)

    bc = rl_data.get("best_circuits", rl_data)
    records = load_hamiltonian_records(ROOT / "results/data/hamiltonians_gic2026/hamiltonians.json")

    # Determine which molecules to process
    if args.molecules == ["all"]:
        mol_names = list(bc.keys())
    else:
        mol_names = args.molecules

    # Filter by qubit count and available records
    to_process = []
    skipped = []
    for mol in mol_names:
        if mol not in bc:
            print(f"  SKIP {mol}: no RL circuit found")
            skipped.append(mol)
            continue
        try:
            record = find_record_by_name(records, mol)
        except (ValueError, KeyError):
            print(f"  SKIP {mol}: no Hamiltonian record found")
            skipped.append(mol)
            continue
        nq = int(record["n_qubits"])
        if nq > args.max_qubits:
            print(f"  SKIP {mol}: {nq}q > {args.max_qubits}q statevector limit")
            skipped.append(mol)
            continue
        if args.skip_existing:
            manifest_path = ROOT / f"results/qpu/{mol}_rl_sqd_cepheus_manifest.json"
            if manifest_path.exists():
                print(f"  SKIP {mol}: manifest already exists")
                skipped.append(mol)
                continue
        to_process.append((mol, record, bc[mol]))

    print(f"\n{'='*70}")
    print(f"Molecules to process: {len(to_process)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Device: {args.device}")
    print(f"Shots: {args.shots}")
    print(f"Export only: {args.export_only}")
    print(f"{'='*70}\n")

    if not to_process:
        print("Nothing to process. Exiting.")
        return

    # Estimate costs
    n_tasks = len(to_process)
    cost_per = 30 + args.shots * 0.0425  # task fee + shot fee (AWS Braket)
    total_est = n_tasks * cost_per
    print(f"Estimated QPU cost: {total_est:.0f} credits ({n_tasks} tasks × {cost_per:.1f} credits/task)")
    print()

    optimized_results = []
    manifests = []

    for mol, record, info in tqdm(to_process, desc="Optimizing", unit="mol"):
        ops = info["operators"]
        nq = int(record["n_qubits"])
        n_ent = sum(1 for o in ops if any(c in o for c in "XY"))
        rl_energy = info.get("energy", 0)

        tqdm.write(f"\n  {mol}: {nq}q, {len(ops)} ops ({n_ent} entangling), RL E={rl_energy:.6f}")

        # Run L-BFGS-B
        t0 = time.time()
        try:
            best_energy, best_thetas = optimize_single_molecule(
                record, ops, max_iter=args.max_iter, n_starts=args.n_starts, seed=42
            )
        except Exception as e:
            tqdm.write(f"    FAILED optimization: {e}")
            continue
        elapsed = time.time() - t0
        tqdm.write(f"    Optimized E={best_energy:.6f} ({elapsed:.1f}s)")

        # Save optimized result
        opt_entry = {
            "molecule": mol,
            "n_qubits": nq,
            "n_operators": len(ops),
            "n_entangling": n_ent,
            "rl_unoptimized_energy": rl_energy,
            "optimized_energy": best_energy,
            "best_operators": ops,
            "best_thetas": best_thetas.tolist(),
            "optimization_time_seconds": elapsed,
            "checkpoint": "results/train/h_cgqe_model_qbraid_rl.pt",
        }
        optimized_results.append(opt_entry)

        # Export SQD manifest
        manifest_path = ROOT / f"results/qpu/{mol}_rl_sqd_cepheus_manifest.json"
        try:
            manifest = export_sqd_sampling_circuit(
                record, ops, best_thetas.tolist(), args.device, args.shots, manifest_path
            )
            manifests.append(str(manifest_path))
            tqdm.write(f"    Manifest: {manifest_path}")
        except Exception as e:
            tqdm.write(f"    FAILED manifest export: {e}")
            continue

    # Save all optimized results
    opt_path = ROOT / "results/eval/h_cgqe_rl_optimized_all.json"
    with open(opt_path, "w") as f:
        json.dump(optimized_results, f, indent=2)
    print(f"\nOptimized results: {opt_path} ({len(optimized_results)} molecules)")

    if args.export_only:
        print(f"\nExport-only mode. {len(manifests)} manifests saved to results/qpu/")
        print("Submit later with:")
        print(f"  python scripts/submit_more_qpu.py --submit-only --manifests {' '.join(manifests)}")
        return

    # Submit to Cepheus
    print(f"\n{'='*70}")
    print(f"Submitting {len(manifests)} circuits to {args.device}")
    print(f"{'='*70}\n")

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
        "checkpoint": "rl",
        "shots": args.shots,
        "sqd_jobs": {},
    }

    for manifest_path in tqdm(manifests, desc="Submitting", unit="job"):
        with open(manifest_path) as f:
            manifest = json.load(f)

        mol = manifest.get("molecule", Path(manifest_path).stem)
        qasm = manifest["circuit_qasm"]
        nq = manifest["n_qubits"]
        tqdm.write(f"  {mol} ({nq}q)...")

        try:
            qc = loads(qasm)
            job = device_obj.run(qc, shots=args.shots)
            jid = str(job.id)
            tqdm.write(f"    Job ID: {jid}")
            submission_meta["sqd_jobs"][mol] = {
                "job_id": jid,
                "shots": args.shots,
                "n_qubits": nq,
                "manifest": str(manifest_path),
            }
        except Exception as e:
            tqdm.write(f"    FAILED: {e}")
            submission_meta["sqd_jobs"][mol] = {"error": str(e)}

    meta_path = ROOT / "results/qpu/cepheus_rl_submission_meta_all.json"
    with open(meta_path, "w") as f:
        json.dump(submission_meta, f, indent=2)
    print(f"\nSubmission metadata: {meta_path}")
    print(f"\nDone! Retrieve with:")
    print(f"  python scripts/retrieve_and_sqd.py --meta {meta_path} --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json --out results/qpu/cepheus_rl_sqd_results_all.json")


if __name__ == "__main__":
    main()
