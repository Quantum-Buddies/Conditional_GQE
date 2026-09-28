"""Two-stage pipeline: optimize coefficients for H-cGQE predicted operators.

Stage 1: H-cGQE Transformer predicts operator sequences (discrete structure).
Stage 2: Classical optimizer (L-BFGS-B) finds optimal rotation angles (continuous params).

Usage:
    python src/gqe/eval/optimize_h_cgqe_coefficients.py \
        --generated results/inference/h_cgqe_generated.json \
        --hamiltonians results/data/hamiltonians.json \
        --out results/eval/h_cgqe_optimized.json \
        --target nvidia \
        --parallel-gpus 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

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

# Apply CUDA-Q env tuning for gate fusion + mempool before any cudaq usage
try:
    from src.gqe.accel.cudaq_tuning import ensure_applied
    ensure_applied()
except Exception:
    pass


def _pad_pauli_word(word: str, n_qubits: int) -> str:
    """Pad or truncate a Pauli word to match n_qubits."""
    if len(word) == n_qubits:
        return word
    if len(word) < n_qubits:
        return word + "I" * (n_qubits - len(word))
    return word[:n_qubits]


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


def _build_kernel_for_sequence(
    n_qubits: int,
    n_electrons: int,
    operators: list[str],
) -> tuple[Any, Any]:
    """Build a CUDA-Q kernel for a fixed operator sequence with variable coefficients.
    
    Returns:
        kernel: cudaq.kernel function
        pauli_words: list of cudaq.pauli_word objects
    """
    padded = [_pad_pauli_word(w, n_qubits) for w in operators]
    pauli_words = [cudaq.pauli_word(w) for w in padded]
    
    @cudaq.kernel
    def kernel(
        n_qubits_k: int,
        n_electrons_k: int,
        thetas: list[float],
        words: list[cudaq.pauli_word],
    ):
        q = cudaq.qvector(n_qubits_k)
        for i in range(n_electrons_k):
            x(q[i])
        for i in range(len(words)):
            exp_pauli(thetas[i], q, words[i])
    
    return kernel, pauli_words


def _evaluate_energy(
    thetas: np.ndarray,
    kernel: Any,
    spin_ham: Any,
    n_qubits: int,
    n_electrons: int,
    pauli_words: list[Any],
) -> float:
    """Evaluate circuit energy for given theta parameters."""
    thetas_list = thetas.tolist()
    result = cudaq.observe(
        kernel,
        spin_ham,
        n_qubits,
        n_electrons,
        thetas_list,
        pauli_words,
    )
    return float(result.expectation())


def _evaluate_fixed_theta_energy(
    molecule_record: dict[str, Any],
    operators: list[str],
    theta: float = 0.01,
) -> float:
    """Evaluate a fixed-theta circuit on the currently configured CUDA-Q target.

    Unlike the evaluator's helper, this function does *not* force qpp-cpu.
    That makes it suitable for GPU-backed ranking in the coefficient optimizer.
    """
    if cudaq is None:
        raise RuntimeError("CUDA-Q not available")

    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    spin_ham = hamiltonian_to_spin_operator(molecule_record)
    kernel, pauli_words = _build_kernel_for_sequence(n_qubits, n_electrons, operators)
    thetas = np.full(len(operators), theta, dtype=float)
    return _evaluate_energy(thetas, kernel, spin_ham, n_qubits, n_electrons, pauli_words)


def _optimize_coefficients(
    molecule_record: dict[str, Any],
    operators: list[str],
    initial_thetas: np.ndarray | None = None,
    max_iter: int = 100,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Optimize rotation coefficients for a fixed operator sequence.
    
    Args:
        molecule_record: Hamiltonian record.
        operators: List of Pauli words (fixed by H-cGQE).
        initial_thetas: Initial guess for coefficients. If None, uses small random values.
        max_iter: Maximum optimization iterations.
        seed: Random seed for deterministic reproducibility.
    
    Returns:
        best_energy: Optimized energy value.
        best_thetas: Optimized coefficient array.
    """
    if cudaq is None:
        raise RuntimeError("CUDA-Q not available")
    
    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    spin_ham = hamiltonian_to_spin_operator(molecule_record)

    kernel, pauli_words = _build_kernel_for_sequence(n_qubits, n_electrons, operators)
    
    if initial_thetas is None:
        rng = np.random.default_rng(seed)
        initial_thetas = rng.uniform(-0.05, 0.05, size=len(operators))
    
    def cost_fn(thetas: np.ndarray) -> float:
        return _evaluate_energy(thetas, kernel, spin_ham, n_qubits, n_electrons, pauli_words)
    
    bounds = [(-np.pi, np.pi) for _ in range(len(operators))]
    
    result = minimize(
        cost_fn,
        initial_thetas,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )
    
    best_energy = float(result.fun)
    best_thetas = result.x
    
    return best_energy, best_thetas


def _optimize_coefficients_multistart(
    molecule_record: dict[str, Any],
    operators: list[str],
    max_iter: int = 100,
    n_starts: int = 4,
    seed: int = 42,
    n_gpus: int = 1,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Multi-start L-BFGS-B optimization with deterministic seeds.
    
    Runs n_starts independent optimizations from different initial points,
    all seeded deterministically for reproducibility. Returns the best result
    along with convergence metadata.
    
    Uses the accelerated batched optimizer (src.gqe.accel.batched_optimizer)
    when available for multi-GPU parallel observe_async, with fallback to
    the original sequential L-BFGS-B path.
    
    Args:
        molecule_record: Hamiltonian record.
        operators: List of Pauli words (fixed by H-cGQE).
        max_iter: Maximum optimization iterations per start.
        n_starts: Number of independent optimization starts.
        seed: Base random seed (each start uses seed + start_index).
        n_gpus: Number of GPUs for parallel evaluation.
    
    Returns:
        best_energy: Best optimized energy across all starts.
        best_thetas: Best coefficient array.
        metadata: Dict with per-start energies, convergence info, timing.
    """
    if cudaq is None:
        raise RuntimeError("CUDA-Q not available")
    
    # Try accelerated batched optimizer
    if n_gpus > 1:
        try:
            from src.gqe.accel.batched_optimizer import optimize_coefficients_batched
            return optimize_coefficients_batched(
                molecule_record, operators,
                max_iter=max_iter, n_starts=n_starts, seed=seed,
                n_gpus=n_gpus,
            )
        except Exception as e:
            print(f"  Batched optimizer unavailable ({e}), falling back to sequential")
    
    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    spin_ham = hamiltonian_to_spin_operator(molecule_record)
    kernel, pauli_words = _build_kernel_for_sequence(n_qubits, n_electrons, operators)
    
    all_energies = []
    all_thetas = []
    all_converged = []
    all_iters = []
    start_times = []
    
    for start_idx in range(n_starts):
        start_seed = seed + start_idx
        rng = np.random.default_rng(start_seed)
        initial_thetas = rng.uniform(-0.05, 0.05, size=len(operators))
        
        def cost_fn(thetas: np.ndarray) -> float:
            return _evaluate_energy(thetas, kernel, spin_ham, n_qubits, n_electrons, pauli_words)
        
        bounds = [(-np.pi, np.pi) for _ in range(len(operators))]
        
        t0 = time.time()
        result = minimize(
            cost_fn,
            initial_thetas,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter},
        )
        elapsed = time.time() - t0
        
        all_energies.append(float(result.fun))
        all_thetas.append(result.x)
        all_converged.append(result.success)
        all_iters.append(result.nit if hasattr(result, 'nit') else max_iter)
        start_times.append(elapsed)
    
    best_idx = int(np.argmin(all_energies))
    best_energy = all_energies[best_idx]
    best_thetas = all_thetas[best_idx]
    
    metadata = {
        "n_starts": n_starts,
        "seed_base": seed,
        "all_energies": all_energies,
        "all_converged": all_converged,
        "all_iters": all_iters,
        "all_times_seconds": start_times,
        "best_start_index": best_idx,
        "total_time_seconds": sum(start_times),
    }
    
    return best_energy, best_thetas, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize H-cGQE operator coefficients")
    parser.add_argument("--generated", type=Path, required=True, help="Generated sequences JSON")
    parser.add_argument("--hamiltonians", type=Path, default=Path("results/data/hamiltonians.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=str, default="qpp-cpu")
    parser.add_argument("--target-option", type=str, default=None)
    parser.add_argument("--parallel-gpus", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=100, help="Max optimization iterations per sequence")
    parser.add_argument("--top-k", type=int, default=10, help="Optimize top-k sequences per molecule (by heuristic)")
    parser.add_argument("--max-qubits", type=int, default=None, help="Skip molecules with more than this many qubits")
    parser.add_argument("--n-starts", type=int, default=4, help="Number of deterministic multi-starts per sequence")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for deterministic reproducibility")
    args = parser.parse_args()

    if cudaq and args.target:
        try:
            needs_mpi = (args.target_option and "mgpu" in args.target_option)
            if needs_mpi:
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

    # Load generated sequences
    with args.generated.open("r", encoding="utf-8") as f:
        generated_data = json.load(f)

    # Load Hamiltonian records
    ham_records = load_hamiltonian_records(args.hamiltonians)

    if args.max_qubits is not None:
        generated_data = [
            mol for mol in generated_data
            if find_record_by_name(ham_records, mol["molecule"])["n_qubits"] <= args.max_qubits
        ]
        print(f"Filtered to {len(generated_data)} molecules with <= {args.max_qubits} qubits")

    optimized_results: list[dict[str, Any]] = []

    for mol_result in generated_data:
        molecule = mol_result["molecule"]
        
        try:
            mol_record = find_record_by_name(ham_records, molecule)
        except ValueError:
            print(f"Warning: No Hamiltonian record for {molecule}")
            continue

        sequences = mol_result["generated_sequences"]
        print(f"\nOptimizing {molecule} ({len(sequences)} sequences, top-{args.top_k})...")

        # Stage 1: Quick heuristic evaluation with fixed theta=0.01 to rank sequences
        print("  Ranking sequences with fixed coefficients...")
        heuristic_energies = []
        for seq in sequences:
            try:
                energy = _evaluate_fixed_theta_energy(mol_record, seq["operators"], theta=0.01)
                heuristic_energies.append(energy)
            except Exception as e:
                print(f"    Ranking failed for seq ({len(seq['operators'])} ops): {e}")
                heuristic_energies.append(float("inf"))

        # If all heuristic evaluations failed, skip this molecule
        if all(e == float("inf") for e in heuristic_energies):
            print(f"  All heuristic evaluations failed for {molecule}, skipping optimization")
            optimized_results.append({
                "molecule": molecule,
                "n_qubits": int(mol_record["n_qubits"]),
                "n_sequences_evaluated": len(sequences),
                "n_sequences_optimized": 0,
                "best_energy": None,
                "error": "All heuristic energy evaluations failed (possible MPS backend error)",
            })
            continue
        
        # Select top-k sequences by lowest heuristic energy
        top_indices = np.argsort(heuristic_energies)[:args.top_k]
        top_sequences = [sequences[i] for i in top_indices]
        
        # Stage 2: Full coefficient optimization on top-k sequences
        print(f"  Optimizing coefficients for top-{args.top_k} sequences...")
        optimized_energies = []
        optimized_thetas_list = []
        optimized_metadata = []
        
        for i, seq in enumerate(top_sequences):
            print(f"    Sequence {i+1}/{args.top_k} ({len(seq['operators'])} ops)...", end=" ")
            try:
                energy, thetas, opt_meta = _optimize_coefficients_multistart(
                    mol_record,
                    seq["operators"],
                    max_iter=args.max_iter,
                    n_starts=args.n_starts,
                    seed=args.seed + i,
                    n_gpus=args.parallel_gpus or 1,
                )
                optimized_energies.append(energy)
                optimized_thetas_list.append(thetas.tolist())
                optimized_metadata.append(opt_meta)
                print(f"E = {energy:.6f} Ha ({opt_meta['total_time_seconds']:.1f}s, {opt_meta['n_starts']} starts)")
            except Exception as e:
                print(f"FAILED: {e}")
                optimized_energies.append(float("inf"))
                optimized_thetas_list.append(None)
                optimized_metadata.append(None)
        
        # Find best optimized result
        if optimized_energies:
            best_idx = int(np.argmin(optimized_energies))
            best_energy = optimized_energies[best_idx]
            best_thetas = optimized_thetas_list[best_idx]
            best_ops = top_sequences[best_idx]["operators"]
            
            print(f"  Best optimized energy: {best_energy:.6f} Ha")
            
            optimized_results.append({
                "molecule": molecule,
                "n_qubits": int(mol_record["n_qubits"]),
                "n_sequences_evaluated": len(sequences),
                "n_sequences_optimized": args.top_k,
                "best_energy": best_energy,
                "best_operators": best_ops,
                "best_thetas": best_thetas,
                "all_optimized_energies": optimized_energies,
                "optimization_metadata": optimized_metadata[best_idx] if optimized_metadata else None,
            })
        else:
            print(f"  No successful optimizations for {molecule}")

    # Save results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(optimized_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"{'Molecule':15s} {'Qubits':>6s} {'Best E (Ha)':>14s}")
    print("-" * 80)
    for res in optimized_results:
        print(f"{res['molecule']:15s} {res['n_qubits']:6d} {res['best_energy']:14.6f}")
    print(f"\nSaved optimized results to {args.out}")
    print("\nNOTE: This is a proper two-stage evaluation:")
    print("  Stage 1: H-cGQE predicts operator identities (discrete)")
    print("  Stage 2: Classical L-BFGS-B optimizes rotation angles (continuous)")


if __name__ == "__main__":
    main()
