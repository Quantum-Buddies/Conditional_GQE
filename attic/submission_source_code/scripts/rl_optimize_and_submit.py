#!/usr/bin/env python3
"""Run L-BFGS-B on RL checkpoint circuits, export SQD manifests, submit to Cepheus.

Per-molecule execution with timeout to prevent hangs on larger circuits.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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

    # Pad operators to n_qubits
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
        print(f"  Start {start+1}/{n_starts}: E={result.fun:.6f}")

    return best_energy, best_thetas


def main() -> None:
    # Load data
    with open(ROOT / "results/train/h_cgqe_model_qbraid_rl_best_circuits.json") as f:
        rl_data = json.load(f)

    bc = rl_data["best_circuits"]
    records = load_hamiltonian_records(ROOT / "results/data/hamiltonians_gic2026/hamiltonians.json")

    device = "aws:rigetti:qpu:cepheus-1-108q"
    shots = 4096
    optimized_results = []
    manifests = []

    for mol in ["h2", "lih", "beh2"]:
        info = bc[mol]
        ops = info["operators"]
        record = find_record_by_name(records, mol)
        nq = int(record["n_qubits"])
        print(f"\n{'='*60}")
        print(f"{mol}: {nq}q, {len(ops)} operators")
        print(f"  Entangling: {sum(1 for o in ops if any(c in o for c in 'XY'))}/{len(ops)}")
        print(f"  RL energy (unoptimized thetas): {info['energy']:.6f}")

        # Run L-BFGS-B
        print(f"  Running L-BFGS-B ({4} starts, {50} iters)...")
        t0 = time.time()
        best_energy, best_thetas = optimize_single_molecule(
            record, ops, max_iter=50, n_starts=4, seed=42
        )
        elapsed = time.time() - t0
        print(f"  Optimized energy: {best_energy:.6f} ({elapsed:.1f}s)")

        # Save optimized result
        opt_entry = {
            "molecule": mol,
            "n_qubits": nq,
            "n_operators": len(ops),
            "rl_unoptimized_energy": info["energy"],
            "optimized_energy": best_energy,
            "best_operators": ops,
            "best_thetas": best_thetas.tolist(),
            "optimization_time_seconds": elapsed,
            "checkpoint": "results/train/h_cgqe_model_qbraid_rl.pt",
        }
        optimized_results.append(opt_entry)

        # Export SQD manifest
        manifest_path = ROOT / f"results/qpu/{mol}_rl_sqd_cepheus_manifest.json"
        manifest = export_sqd_sampling_circuit(
            record, ops, best_thetas.tolist(), device, shots, manifest_path
        )
        manifests.append(manifest)
        print(f"  Manifest: {manifest_path}")

    # Save optimized results
    opt_path = ROOT / "results/eval/h_cgqe_rl_optimized.json"
    with open(opt_path, "w") as f:
        json.dump(optimized_results, f, indent=2)
    print(f"\nOptimized results: {opt_path}")

    # Submit to Cepheus
    from qbraid import QbraidProvider
    from qiskit.qasm2 import loads

    provider = QbraidProvider()
    device_obj = provider.get_device(device)
    print(f"\nSubmitting to {device}...")

    submission_meta = {
        "device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": "rl",
        "sqd_jobs": {},
    }

    for mol in ["h2", "lih", "beh2"]:
        manifest_path = ROOT / f"results/qpu/{mol}_rl_sqd_cepheus_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        qasm = manifest["circuit_qasm"]
        print(f"  {mol} ({manifest['n_qubits']}q)...")
        try:
            qc = loads(qasm)
            job = device_obj.run(qc, shots=shots)
            jid = str(job.id)
            print(f"    Job ID: {jid}")
            submission_meta["sqd_jobs"][mol] = jid
        except Exception as e:
            print(f"    FAILED: {e}")
            submission_meta["sqd_jobs"][mol] = None

    meta_path = ROOT / "results/qpu/cepheus_rl_submission_meta.json"
    with open(meta_path, "w") as f:
        json.dump(submission_meta, f, indent=2)
    print(f"\nSubmission metadata: {meta_path}")
    print("\nDone! Retrieve with:")
    print(f"  python scripts/retrieve_and_sqd.py --meta {meta_path} --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json --out results/qpu/cepheus_rl_sqd_results.json")


if __name__ == "__main__":
    main()
