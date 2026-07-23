"""Evaluate H-cGQE generated circuits against GQE baseline.

Loads H-cGQE generated operator sequences, computes their energies via CUDA-Q,
and compares against the GQE baseline results.

Usage:
    python src/gqe/eval/evaluate_h_cgqe.py \
        --generated results/inference/h_cgqe_generated.json \
        --baseline results/baselines/cudaq_gqe_baseline.json \
        --hamiltonians results/data/hamiltonians.json \
        --out results/eval/h_cgqe_evaluation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import cudaq
except ImportError:
    cudaq = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    hamiltonian_to_spin_operator,
    find_record_by_name,
    get_active_electron_count,
)


def _ensure_cuda_context() -> None:
    """Create a CUDA context on the GPU assigned to this MPI rank.

    Open MPI's smcuda BTL needs each rank to have a CUDA context before
    MPI_Init() so it can set up GPU-buffer communication.
    """
    import ctypes
    import os

    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    libcudart = ctypes.CDLL(os.environ.get("CUDAQ_CUDART", "libcudart.so"))
    libcudart.cudaSetDevice(local_rank)
    d = ctypes.c_void_p()
    libcudart.cudaMalloc(ctypes.byref(d), 4)
    libcudart.cudaFree(d)


def _compute_circuit_energy(
    molecule_record: dict[str, Any],
    operators: list[str],
    device: str = "cpu",
) -> float:
    """Compute energy for a circuit defined by operator sequence using CUDA-Q.

    Args:
        molecule_record: Hamiltonian record for the molecule.
        operators: List of Pauli words predicted by the model.
        device: CUDA-Q target device.
    """
    if cudaq is None:
        # Fallback to mock if cudaq not installed
        return 1.0 / len(operators) if operators else 1.0

    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)

    spin_ham = hamiltonian_to_spin_operator(molecule_record)

    @cudaq.kernel
    def kernel(
        n_qubits: int,
        n_electrons: int,
        pauli_words: list[cudaq.pauli_word],
        thetas: list[float],
    ):
        q = cudaq.qvector(n_qubits)
        # Hartree-Fock state initialization
        for i in range(n_electrons):
            x(q[i])

        # Apply generated operators
        for i in range(len(pauli_words)):
            exp_pauli(thetas[i], q, pauli_words[i])

    # Convert operators to cudaq.pauli_word (pad/truncate to match n_qubits)
    # Using a small fixed theta (e.g. 0.01) as the model only predicts the word
    # In a full GQE, these would be selected from a pool with fixed scales
    padded_ops = []
    for w in operators:
        if len(w) < n_qubits:
            w = w + "I" * (n_qubits - len(w))
        elif len(w) > n_qubits:
            w = w[:n_qubits]
        padded_ops.append(w)
    pauli_words = [cudaq.pauli_word(w) for w in padded_ops]
    thetas = [0.01] * len(pauli_words)

    try:
        result = cudaq.observe(kernel, spin_ham, n_qubits, n_electrons, pauli_words, thetas)
        return float(result.expectation())
    except Exception as e:
        print(f"Error in CUDA-Q evaluation: {e}")
        return 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate H-cGQE circuits")
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hamiltonians", type=Path, default=Path("results/data/hamiltonians.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=str, default="qpp-cpu")
    parser.add_argument("--target-option", type=str, default=None)
    parser.add_argument("--parallel-gpus", type=int, default=None, help="Number of GPUs to parallelize across")
    parser.add_argument("--max-qubits", type=int, default=None, help="Skip molecules with more than this many qubits")
    args = parser.parse_args()

    is_mgpu = False
    if cudaq and args.target:
        try:
            needs_mpi = (args.target_option and "mgpu" in args.target_option) or args.target == "tensornet"
            if needs_mpi:
                is_mgpu = True
                if not cudaq.mpi.is_initialized():
                    _ensure_cuda_context()
                    cudaq.mpi.initialize()
                    print(f"MPI initialized: rank={cudaq.mpi.rank()}, num_ranks={cudaq.mpi.num_ranks()}")
            if args.target == "nvidia" and (args.target_option == "mqpu" or args.parallel_gpus):
                cudaq.set_target("nvidia", option="mqpu")
            elif args.target_option:
                cudaq.set_target(args.target, option=args.target_option)
            else:
                cudaq.set_target(args.target)
            print(f"Using CUDA-Q target: {args.target} (option: {args.target_option or 'default'})")
        except Exception as e:
            print(f"Warning: Could not set target {args.target}, error: {e}")

    # Load generated circuits
    with args.generated.open("r", encoding="utf-8") as f:
        generated_data = json.load(f)

    # Load baseline GQE results
    with args.baseline.open("r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    # Load Hamiltonian records
    ham_records = load_hamiltonian_records(args.hamiltonians)

    if args.max_qubits is not None:
        generated_data = [
            mol for mol in generated_data
            if find_record_by_name(ham_records, mol["molecule"])["n_qubits"] <= args.max_qubits
        ]
        print(f"Filtered to {len(generated_data)} molecules with <= {args.max_qubits} qubits")

    # Build comprehensive baseline lookup
    # Keys: molecule -> {reference_energy, baseline_energy, delta_energy}
    baseline_lookup: dict[str, dict[str, float | None]] = {}
    results_list = baseline_data.get("results") or baseline_data.get("comparison") or []
    if isinstance(baseline_data, list):
        results_list = baseline_data

    for result in results_list:
        name = result.get("system") or result.get("molecule")
        if not name:
            continue
        b_energy = result.get("baseline_energy")
        if b_energy is None:
            val = result.get("energy") or result.get("gqe_energy")
            if isinstance(val, (int, float)):
                b_energy = float(val)
        ref_e = result.get("reference_energy")
        delta_e = result.get("delta_energy") or result.get("baseline_delta_Ha")
        if b_energy is None and ref_e is not None and delta_e is not None:
            b_energy = ref_e + delta_e
        baseline_lookup[name] = {
            "reference_energy": ref_e,
            "baseline_energy": b_energy,
            "delta_energy": delta_e,
        }

    # Evaluate each molecule
    evaluation: list[dict[str, Any]] = []
    for mol_result in generated_data:
        molecule = mol_result["molecule"]
        baseline_info = baseline_lookup.get(molecule, {})
        reference_energy = baseline_info.get("reference_energy")
        baseline_energy = baseline_info.get("baseline_energy")
        baseline_delta = baseline_info.get("delta_energy")
        
        try:
            mol_record = find_record_by_name(ham_records, molecule)
        except ValueError:
            print(f"Warning: No Hamiltonian record for {molecule}")
            continue

        if baseline_energy is None:
            print(f"Warning: No baseline energy for {molecule}")

        print(f"Evaluating {molecule} ({len(mol_result['generated_sequences'])} samples)...")
        
        # Evaluate each generated circuit
        circuit_energies: list[float] = []
        
        # Determine number of QPUs/GPUs for parallel execution
        # mgpu pools all GPUs for one statevector — no per-QPU parallelism
        num_qpus = 1
        if is_mgpu:
            num_qpus = 1
        elif cudaq and args.parallel_gpus:
            num_qpus = args.parallel_gpus
        elif cudaq:
            try:
                target = cudaq.get_target()
                if hasattr(target, "num_qpus"):
                    num_qpus = target.num_qpus()
                else:
                    import torch
                    num_qpus = torch.cuda.device_count()
            except:
                num_qpus = 1

        if num_qpus > 1:
            print(f"  Parallelizing over {num_qpus} QPUs/GPUs...")
            futures = []
            spin_ham = hamiltonian_to_spin_operator(mol_record)
            n_qubits = int(mol_record["n_qubits"])
            n_electrons = get_active_electron_count(mol_record)
            
            @cudaq.kernel
            def kernel_parallel(
                n_qubits: int,
                n_electrons: int,
                pauli_words: list[cudaq.pauli_word],
                thetas: list[float],
            ):
                q = cudaq.qvector(n_qubits)
                for i in range(n_electrons):
                    x(q[i])
                for i in range(len(pauli_words)):
                    exp_pauli(thetas[i], q, pauli_words[i])

            for i, seq in enumerate(mol_result["generated_sequences"]):
                _ops = seq["operators"]
                _padded = []
                for w in _ops:
                    if len(w) < n_qubits:
                        w = w + "I" * (n_qubits - len(w))
                    elif len(w) > n_qubits:
                        w = w[:n_qubits]
                    _padded.append(w)
                pauli_words = [cudaq.pauli_word(w) for w in _padded]
                thetas = [0.01] * len(pauli_words)
                gpu_id = i % num_qpus
                
                future = cudaq.observe_async(
                    kernel_parallel, 
                    spin_ham, 
                    n_qubits, 
                    n_electrons, 
                    pauli_words, 
                    thetas, 
                    qpu_id=gpu_id
                )
                futures.append(future)
            
            print(f"  Synchronizing {len(futures)} futures...")
            circuit_energies = [f.get().expectation() for f in futures]
        else:
            for seq in mol_result["generated_sequences"]:
                energy = _compute_circuit_energy(mol_record, seq["operators"])
                circuit_energies.append(energy)

        best_energy = min(circuit_energies)
        avg_energy = float(np.mean(circuit_energies))

        # Compute proper error metrics
        # If we have a reference energy (exact diagonalization), compute error vs reference
        # Otherwise, compute error vs GQE baseline
        if reference_energy is not None:
            error_vs_ref = abs(best_energy - reference_energy)
            baseline_error = abs(baseline_energy - reference_energy) if baseline_energy is not None else None
        else:
            error_vs_ref = abs(best_energy - baseline_energy) if baseline_energy is not None else None
            baseline_error = None

        # Also compute improvement over baseline (if both exist)
        if baseline_energy is not None:
            improvement = baseline_energy - best_energy  # positive = better
        else:
            improvement = None

        ref_for_error = reference_energy if reference_energy is not None else baseline_energy
        energy_error_mha = float(abs(best_energy - ref_for_error) * 1000.0) if ref_for_error is not None else None

        evaluation.append({
            "molecule": molecule,
            "baseline_energy": baseline_energy,
            "best_generated_energy": best_energy,
            "avg_generated_energy": avg_energy,
            "energy_error": energy_error_mha,
            "reference_energy": reference_energy,
            "baseline_error_vs_reference": baseline_error,
            "improvement_over_baseline": improvement,
            "n_samples": len(circuit_energies),
        })

    # Save evaluation
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)

    # Print summary with proper columns
    print("\nEvaluation Summary:")
    print("-" * 100)
    print(f"{'Molecule':15s} {'Ref (Ha)':>12s} {'GQE (Ha)':>12s} {'H-cGQE (Ha)':>12s} {'Err(Ref)':>10s} {'Err(GQE)':>10s} {'Imprv':>8s}")
    print("-" * 100)
    for ev in evaluation:
        ref_str = f"{ev['reference_energy']:12.4f}" if ev['reference_energy'] is not None else "     N/A    "
        gqe_str = f"{ev['baseline_energy']:12.4f}" if ev['baseline_energy'] is not None else "     N/A    "
        err_ref = f"{ev['energy_error']:10.4f}" if ev['energy_error'] is not None else "   N/A   "
        err_gqe = f"{abs(ev['best_generated_energy'] - ev['baseline_energy']):10.4f}" if ev['baseline_energy'] is not None else "   N/A   "
        imprv = f"{ev['improvement_over_baseline']:8.4f}" if ev['improvement_over_baseline'] is not None else "  N/A  "
        print(f"{ev['molecule']:15s} {ref_str} {gqe_str} {ev['best_generated_energy']:12.4f} {err_ref} {err_gqe} {imprv}")

    print(f"\nSaved evaluation to {args.out}")
    print("\nNOTE: Generated circuits use fixed coefficient theta=0.01 for all operators.")
    print("      The model predicts operator identities, not optimal coefficients.")
    print("      For best results, the predicted operators should be fed into a GQE")
    print("      coefficient optimizer (like the one in run_cudaq_gqe.py).")


if __name__ == "__main__":
    main()
