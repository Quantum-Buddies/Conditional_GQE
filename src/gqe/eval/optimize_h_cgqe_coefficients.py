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


def _pad_pauli_word(word: str, n_qubits: int) -> str:
    """Pad or truncate a Pauli word to match n_qubits."""
    if len(word) == n_qubits:
        return word
    if len(word) < n_qubits:
        return word + "I" * (n_qubits - len(word))
    return word[:n_qubits]


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
) -> tuple[float, np.ndarray]:
    """Optimize rotation coefficients for a fixed operator sequence.
    
    Args:
        molecule_record: Hamiltonian record.
        operators: List of Pauli words (fixed by H-cGQE).
        initial_thetas: Initial guess for coefficients. If None, uses small random values.
        max_iter: Maximum optimization iterations.
    
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
        # Start with small values similar to GQE pool scales
        initial_thetas = np.random.uniform(-0.05, 0.05, size=len(operators))
    
    def cost_fn(thetas: np.ndarray) -> float:
        return _evaluate_energy(thetas, kernel, spin_ham, n_qubits, n_electrons, pauli_words)
    
    # Use L-BFGS-B for gradient-free bounded optimization
    # Bounds keep coefficients in a reasonable range
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
    args = parser.parse_args()

    if cudaq and args.target:
        try:
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
            # Use the currently selected CUDA-Q target; do not force CPU here.
            energy = _evaluate_fixed_theta_energy(mol_record, seq["operators"], theta=0.01)
            heuristic_energies.append(energy)
        
        # Select top-k sequences by lowest heuristic energy
        top_indices = np.argsort(heuristic_energies)[:args.top_k]
        top_sequences = [sequences[i] for i in top_indices]
        
        # Stage 2: Full coefficient optimization on top-k sequences
        print(f"  Optimizing coefficients for top-{args.top_k} sequences...")
        optimized_energies = []
        optimized_thetas_list = []
        
        for i, seq in enumerate(top_sequences):
            print(f"    Sequence {i+1}/{args.top_k} ({len(seq['operators'])} ops)...", end=" ")
            try:
                energy, thetas = _optimize_coefficients(
                    mol_record,
                    seq["operators"],
                    max_iter=args.max_iter,
                )
                optimized_energies.append(energy)
                optimized_thetas_list.append(thetas.tolist())
                print(f"E = {energy:.6f} Ha")
            except Exception as e:
                print(f"FAILED: {e}")
                optimized_energies.append(float("inf"))
                optimized_thetas_list.append(None)
        
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
