#!/usr/bin/env python
"""Submit H-cGQE circuit to a qBraid QPU or simulator.

Reads the best circuit from Experiment 1, builds a Qiskit circuit,
and submits to a qBraid device. Supports both synchronous and async modes.

Usage:
    python src/gqe/eval/submit_qpu.py \
        --benchmark results/phase3_final/benchmark_ch3i_consolidated.json \
        --config configs/phase3_final/qpu_validation.yaml \
        --out results/phase3_final/qpu/qpu_submission.json

    # Submit to specific device
    python src/gqe/eval/submit_qpu.py \
        --benchmark results/phase3_final/benchmark_ch3i_consolidated.json \
        --device aws:rigetti:qpu:cepheus-1-108q \
        --shots 4096 \
        --out results/phase3_final/qpu/qpu_submission.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
    get_active_electron_count,
)
from src.gqe.common.run_manifest import create_run_manifest, save_run_manifest

try:
    from src.gqe.eval.mitigation import (
        calibrate_rem,
        apply_rem,
        fold_gates,
        zne_extrapolate,
        run_zne_experiment,
    )
except ImportError:
    calibrate_rem = None
    apply_rem = None
    fold_gates = None
    zne_extrapolate = None
    run_zne_experiment = None


def _load_best_circuit(benchmark_path: Path) -> dict[str, Any]:
    """Extract the best H-cGQE circuit from consolidated benchmark."""
    with open(benchmark_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    hcgqe = None
    for r in results:
        if "h_cgqe" in r.get("method", ""):
            hcgqe = r
            break

    if hcgqe is None:
        raise ValueError("No H-cGQE results found in benchmark file")

    return {
        "molecule": hcgqe.get("molecule", "methyl_iodide"),
        "operators": hcgqe.get("best_operators", []),
        "energy_hartree": hcgqe.get("energy_hartree"),
        "error_mha": hcgqe.get("error_mha"),
        "n_qubits": hcgqe.get("qubits", 8),
        "reference_energy": hcgqe.get("reference_energy_hartree"),
        "method": hcgqe.get("method"),
    }


def _build_qiskit_circuit(
    n_qubits: int,
    n_electrons: int,
    operators: list[str],
    thetas: list[float] | None = None,
):
    """Build a Qiskit circuit from H-cGQE operator sequence."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    import numpy as np

    qc = QuantumCircuit(n_qubits)

    # HF state: X on first n_electrons qubits
    for i in range(n_electrons):
        qc.x(i)

    # Apply operators as exp(i * theta * P) via decomposition
    if thetas is None:
        thetas = [0.01] * len(operators)

    for op_idx, (pauli_word, theta) in enumerate(zip(operators, thetas)):
        # Pad/truncate pauli_word to n_qubits
        if len(pauli_word) < n_qubits:
            pauli_word = pauli_word + "I" * (n_qubits - len(pauli_word))
        elif len(pauli_word) > n_qubits:
            pauli_word = pauli_word[:n_qubits]

        # Manual decomposition of exp(i * theta * P) into basic gates:
        # 1. Basis change: H for X, H+S for Y, nothing for Z/I
        # 2. CNOT chain to accumulate parity onto last non-I qubit
        # 3. RZ(2*theta) on last qubit
        # 4. Uncompute CNOT chain
        # 5. Uncompute basis change
        non_identity = [(i, p) for i, p in enumerate(pauli_word) if p != "I"]
        if not non_identity:
            continue  # Identity contributes only a global phase

        # Step 1: Basis change
        for i, p in non_identity:
            if p == "X":
                qc.h(i)
            elif p == "Y":
                qc.sdg(i)
                qc.h(i)

        # Step 2: CNOT chain
        for idx in range(len(non_identity) - 1):
            qc.cx(non_identity[idx][0], non_identity[idx + 1][0])

        # Step 3: RZ on last qubit
        last_q = non_identity[-1][0]
        qc.rz(2 * theta, last_q)

        # Step 4: Uncompute CNOT chain
        for idx in range(len(non_identity) - 2, -1, -1):
            qc.cx(non_identity[idx][0], non_identity[idx + 1][0])

        # Step 5: Uncompute basis change
        for i, p in non_identity:
            if p == "X":
                qc.h(i)
            elif p == "Y":
                qc.h(i)
                qc.s(i)

    return qc


def _circuit_complexity(circuit) -> dict[str, int]:
    decomposed = circuit.decompose(reps=3)
    two_qubit_gates = sum(
        1 for instruction, _, _ in decomposed.data if instruction.num_qubits == 2
    )
    return {
        "depth": int(decomposed.depth()),
        "two_qubit_gates": two_qubit_gates,
        "total_gates": len(decomposed.data),
    }


def _run_ideal_simulation(circuit, record: dict[str, Any]) -> float:
    """Run an exact statevector simulation and compute ``<psi|H|psi>``.

    The previous implementation returned ``P(0...0)``, which is not a
    molecular energy.  The Hamiltonian is a weighted sum of Pauli terms, so
    the reference must use the same observable as the QPU evaluation.
    """
    from qiskit.quantum_info import Statevector
    from src.gqe.common.hamiltonian_utils import hamiltonian_to_sparse_pauli_op

    decomposed = circuit.decompose().decompose()
    statevector = Statevector.from_instruction(decomposed)
    hamiltonian = hamiltonian_to_sparse_pauli_op(record)
    return float(np.real(statevector.expectation_value(hamiltonian)))


def submit_to_qbraid(
    circuit,
    device_id: str,
    shots: int,
    submit_only: bool = True,
) -> dict[str, Any]:
    """Submit circuit to qBraid device."""
    from qbraid import QbraidProvider

    provider = QbraidProvider()

    # Get device
    devices = provider.get_devices()
    qdevice = next((d for d in devices if d.id == device_id), None)

    if qdevice is None:
        # Try partial match
        qdevice = next((d for d in devices if device_id in d.id), None)

    if qdevice is None:
        available = [d.id for d in devices]
        raise ValueError(f"Device {device_id} not found. Available: {available[:10]}")

    print(f"  Device: {qdevice.id}")
    print(f"  Status: {qdevice.status()}")
    print(f"  Qubits: {getattr(qdevice, 'num_qubits', '?')}")

    # Use circuit as-is (basic gates: h, cx, x, sdg, s, rz)
    # qBraid's transpiler will handle native gate conversion
    decomposed = circuit

    # Add measurements if not present
    if not decomposed.clbits:
        decomposed.measure_all()

    # Submit
    print(f"  Submitting {decomposed.num_qubits}-qubit circuit with {shots} shots...")
    t0 = time.time()
    job = qdevice.run(decomposed, shots=shots)
    runtime = time.time() - t0

    job_id = job.id if hasattr(job, 'id') else str(job)
    print(f"  Job ID: {job_id}")
    print(f"  Submit time: {runtime:.2f}s")

    return {
        "job_id": job_id,
        "device_id": qdevice.id,
        "device_status": str(qdevice.status()),
        "shots": shots,
        "submit_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "submit_runtime_seconds": runtime,
        "circuit_qubits": decomposed.num_qubits,
        "circuit_depth": decomposed.depth(),
        "circuit_gates": dict(decomposed.count_ops()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit H-cGQE circuit to qBraid QPU")
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Consolidated benchmark JSON from Experiment 1")
    parser.add_argument("--config", type=Path, default=None,
                        help="QPU validation config YAML")
    parser.add_argument("--device", type=str, default=None,
                        help="qBraid device ID (overrides config)")
    parser.add_argument("--shots", type=int, default=None,
                        help="Number of shots (overrides config)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit-only", action="store_true", default=True,
                        help="Submit asynchronously and return job ID")
    parser.add_argument("--wait", action="store_true",
                        help="Wait synchronously for results")
    parser.add_argument("--mitigate", type=str, default=None,
                        help="Comma-separated mitigation methods: rem,zne")
    parser.add_argument("--zne-scales", type=str, default="1,2,3",
                        help="Comma-separated ZNE scale factors")
    parser.add_argument("--zne-method", type=str, default="richardson",
                        help="ZNE extrapolation method: linear, richardson, polynomial")
    parser.add_argument("--max-zne-two-qubit-gates", type=int, default=20,
                        help="Skip ZNE when the unfolded circuit exceeds this two-qubit-gate count")
    parser.add_argument("--max-rem-qubits", type=int, default=10,
                        help="Skip full-assignment REM calibration above this qubit count")
    args = parser.parse_args()

    # Load config
    config = {}
    if args.config and args.config.exists():
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Load best circuit from benchmark
    print("=== QPU Submission for Phase 3 ===")
    best = _load_best_circuit(args.benchmark)
    print(f"  Molecule: {best['molecule']}")
    print(f"  Operators: {best['operators']}")
    print(f"  GPU energy: {best['energy_hartree']:.6f} Ha")
    print(f"  GPU error: {best['error_mha']:.3f} mHa")
    print(f"  Qubits: {best['n_qubits']}")

    # Load Hamiltonian for energy evaluation
    ham_path = Path("results/data/hamiltonians_phase3.json/hamiltonians.json")
    if not ham_path.exists():
        ham_path = Path("results/data/hamiltonians_scaling.json/hamiltonians.json")

    records = load_hamiltonian_records(ham_path)
    mol_name = best["molecule"]
    record = find_record_by_name(records, mol_name)
    if record is None:
        # Try alternate names
        for alt in ["methyl_iodide", "ch3i", "methyl_iodide_cas12"]:
            record = find_record_by_name(records, alt)
            if record:
                break

    if record is None:
        print(f"  WARNING: Hamiltonian record not found for {mol_name}")
        print(f"  Available: {[r['name'] for r in records[:10]]}")
    else:
        print(f"  Hamiltonian: {record['name']}, {record['n_qubits']} qubits")

    # Build Qiskit circuit
    n_qubits = best["n_qubits"]
    n_electrons = 4  # CH3I CAS(4,4)
    if record:
        n_electrons = get_active_electron_count(record)

    circuit = _build_qiskit_circuit(n_qubits, n_electrons, best["operators"])
    circuit_complexity = _circuit_complexity(circuit)
    print(
        f"  Circuit: {circuit.num_qubits}q, depth={circuit_complexity['depth']}, "
        f"two-qubit gates={circuit_complexity['two_qubit_gates']}, "
        f"ops={circuit.count_ops()}"
    )

    # Ideal simulation reference
    try:
        ideal_energy = _run_ideal_simulation(circuit, record)
        print(f"  Ideal Hamiltonian expectation: {ideal_energy:.6f} Ha")
    except Exception as e:
        print(f"  Ideal sim failed: {e}")
        ideal_energy = None

    # Determine device and shots
    qpu_cfg = config.get("qpu", {})
    device_id = args.device or qpu_cfg.get("device", "qbraid:qbraid:sim:qir-sv")
    shots = args.shots or qpu_cfg.get("shots", 4096)

    # Parse mitigation options
    mitigate = args.mitigate.split(",") if args.mitigate else []
    zne_scales = [float(s) for s in args.zne_scales.split(",")] if "zne" in mitigate else []
    mitigation_results = {}

    # REM calibration (if requested)
    if "rem" in mitigate and calibrate_rem is not None:
        if n_qubits > args.max_rem_qubits:
            message = (
                f"full-assignment REM disabled for {n_qubits} qubits "
                f"(limit={args.max_rem_qubits})"
            )
            print(f"\n  REM skipped: {message}")
            mitigation_results["rem_calibrated"] = False
            mitigation_results["rem_skipped_reason"] = message
            cal_matrix = None
        else:
            print(f"\n  Calibrating REM on {device_id}...")
            try:
                cal_matrix = calibrate_rem(n_qubits, device_id, shots=min(shots, 1024))
                mitigation_results["rem_calibrated"] = True
                mitigation_results["rem_matrix_shape"] = list(cal_matrix.shape)
                print(f"    REM calibration matrix: {cal_matrix.shape}")
            except Exception as e:
                print(f"    REM calibration failed: {e}")
                mitigation_results["rem_calibrated"] = False
                mitigation_results["rem_error"] = str(e)
                cal_matrix = None
    else:
        cal_matrix = None

    # ZNE experiment (if requested)
    if "zne" in mitigate and run_zne_experiment is not None:
        if circuit_complexity["two_qubit_gates"] > args.max_zne_two_qubit_gates:
            message = (
                f"unfolded circuit has {circuit_complexity['two_qubit_gates']} two-qubit gates "
                f"(limit={args.max_zne_two_qubit_gates})"
            )
            print(f"\n  ZNE skipped: {message}")
            mitigation_results["zne_skipped_reason"] = message
        else:
            print(f"\n  Running ZNE with scales {zne_scales}...")
            try:
                from src.gqe.common.hamiltonian_utils import hamiltonian_to_sparse_pauli_op
                ham_op = hamiltonian_to_sparse_pauli_op(record) if record else None

                zne_result = run_zne_experiment(
                    circuit, device_id, ham_op,
                    scale_factors=zne_scales,
                    shots=shots,
                    extrapolation=args.zne_method,
                )
                mitigation_results["zne"] = zne_result
                print(f"    ZNE energy: {zne_result['zne_energy']:.6f} Ha")
            except Exception as e:
                print(f"    ZNE failed: {e}")
                mitigation_results["zne_error"] = str(e)

    # Submit to qBraid
    print(f"\n  Submitting to: {device_id}")
    print(f"  Shots: {shots}")

    submit_result = submit_to_qbraid(
        circuit, device_id, shots,
        submit_only=not args.wait,
    )

    # Build result
    result = {
        "experiment": "phase3_qpu_validation",
        "source": {
            "benchmark_file": str(args.benchmark),
            "molecule": best["molecule"],
            "operators": best["operators"],
            "gpu_energy_hartree": best["energy_hartree"],
            "gpu_error_mha": best["error_mha"],
            "ideal_sim_energy": ideal_energy,
        },
        "submission": submit_result,
        "circuit_complexity": circuit_complexity,
        "mitigation": mitigation_results if mitigate else None,
        "manifest": create_run_manifest(
            command=f"python src/gqe/eval/submit_qpu.py --benchmark {args.benchmark} --device {device_id} --shots {shots}" + (f" --mitigate {args.mitigate}" if args.mitigate else ""),
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to: {args.out}")
    print(f"  Job ID: {submit_result['job_id']}")
    print(f"\  Collect with: python src/gqe/eval/collect_qpu.py --job-id {submit_result['job_id']}")


if __name__ == "__main__":
    main()
