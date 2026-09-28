"""Tile H-cGQE operators to 4×n_qubits depth, optimize thetas with L-BFGS-B, re-run QSCI.

This script addresses the key weakness: our RL circuits are too short (5-21 ops)
vs the organizers' L=140 or gqex's 4×n_qubits default. By tiling the RL operator
sequences and optimizing the full tiled sequence with L-BFGS-B, we create deeper
circuits that produce more diverse bitstrings for QSCI subspace diagonalization.

Usage (from project root):
    python scripts/improve_qsci.py --molecules n2 formaldehyde ethylene benzene_cas20

    # With custom paths:
    python scripts/improve_qsci.py \
        --operators results/eval/h_cgqe_operators_for_qsci.json \
        --hamiltonians results/data/hamiltonians_40plus.json/hamiltonians.json \
        --molecules n2 formaldehyde \
        --out results/phase3_final/qsci/qsci_improved_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cudaq
except ImportError:
    cudaq = None
    print("WARNING: CUDA-Q not available, cannot run L-BFGS-B or QSCI sampling")

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    hamiltonian_to_spin_operator,
    get_active_electron_count,
    find_record_by_name,
    iter_terms,
)
from src.gqe.eval.qsci_postprocess import qsci_energy_from_bitstrings


def _pad_pauli_word(word: str, n_qubits: int) -> str:
    if len(word) == n_qubits:
        return word
    if len(word) < n_qubits:
        return word + "I" * (n_qubits - len(word))
    return word[:n_qubits]


def tile_operators(
    operators: list[str],
    thetas: list[float],
    target_depth: int,
) -> tuple[list[str], list[float]]:
    """Tile operator sequence to reach target depth.

    Repeats the operator sequence and assigns slightly perturbed thetas
    for each repetition to create diverse excitations.
    """
    n_base = len(operators)
    n_repeats = max(1, target_depth // n_base + (1 if target_depth % n_base else 0))

    tiled_ops = []
    tiled_thetas = []
    for rep in range(n_repeats):
        for i, op in enumerate(operators):
            tiled_ops.append(op)
            if rep == 0 and i < len(thetas):
                tiled_thetas.append(thetas[i])
            else:
                base_theta = thetas[i % len(thetas)] if thetas else 0.1
                perturbation = 0.02 * (rep + 1) * ((-1) ** i)
                tiled_thetas.append(base_theta + perturbation)

    return tiled_ops[:target_depth], tiled_thetas[:target_depth]


def build_kernel(n_qubits: int, n_electrons: int, operators: list[str]):
    """Build a CUDA-Q kernel for HF state + tiled operator sequence."""
    padded = [_pad_pauli_word(w, n_qubits) for w in operators]
    pauli_words = [cudaq.pauli_word(w) for w in padded]

    @cudaq.kernel
    def kernel(n_qubits_k: int, n_electrons_k: int,
               thetas: list[float], words: list[cudaq.pauli_word]):
        q = cudaq.qvector(n_qubits_k)
        for i in range(n_electrons_k):
            x(q[i])
        for i in range(len(words)):
            exp_pauli(thetas[i], q, words[i])

    return kernel, pauli_words


def optimize_thetas_lbfgsb(
    record: dict[str, Any],
    operators: list[str],
    initial_thetas: np.ndarray,
    max_iter: int = 200,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Run L-BFGS-B optimization on the tiled operator sequence."""
    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)
    spin_ham = hamiltonian_to_spin_operator(record)
    kernel, pauli_words = build_kernel(n_qubits, n_electrons, operators)

    cudaq.set_target("nvidia")

    def cost_fn(thetas: np.ndarray) -> float:
        result = cudaq.observe(
            kernel, spin_ham, n_qubits, n_electrons,
            thetas.tolist(), pauli_words,
        )
        return float(result.expectation())

    bounds = [(-np.pi, np.pi) for _ in range(len(operators))]

    print(f"  L-BFGS-B: {len(operators)} params, {n_qubits}q, max_iter={max_iter}")
    t0 = time.time()
    result = minimize(
        cost_fn, initial_thetas, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": max_iter, "ftol": 1e-8},
    )
    elapsed = time.time() - t0
    print(f"  L-BFGS-B done: E={result.fun:.6f} Ha, {result.nit} iters, {elapsed:.1f}s")

    return float(result.fun), result.x


def sample_bitstrings_mps(
    operators: list[str],
    thetas: list[float],
    n_qubits: int,
    n_electrons: int,
    n_shots: int = 8192,
    bond_dim: int = 256,
    seed: int = 42,
) -> list[str]:
    """Sample bitstrings using CUDA-Q MPS backend."""
    if seed is not None:
        cudaq.set_random_seed(seed)

    os.environ["CUDAQ_MPS_MAX_BOND"] = str(bond_dim)
    cudaq.set_target("tensornet-mps")

    padded = [_pad_pauli_word(w, n_qubits) for w in operators]
    pauli_words = [cudaq.pauli_word(w) for w in padded]

    @cudaq.kernel
    def kernel(n_qubits_k: int, n_electrons_k: int,
               thetas_k: list[float], words: list[cudaq.pauli_word]):
        q = cudaq.qvector(n_qubits_k)
        for i in range(n_electrons_k):
            x(q[i])
        for i in range(len(words)):
            exp_pauli(thetas_k[i], q, words[i])

    counts = cudaq.sample(
        kernel, n_qubits, n_electrons, thetas, pauli_words,
        shots_count=n_shots,
    )

    bitstring_counts = [(bs, int(count)) for bs, count in counts.items()]
    bitstring_counts.sort(key=lambda x: -x[1])
    return [bs for bs, _ in bitstring_counts]


def sample_bitstrings_sv(
    operators: list[str],
    thetas: list[float],
    n_qubits: int,
    n_electrons: int,
    n_shots: int = 8192,
    seed: int = 42,
) -> list[str]:
    """Sample bitstrings using CUDA-Q statevector backend (≤24q)."""
    if seed is not None:
        cudaq.set_random_seed(seed)

    cudaq.set_target("nvidia")

    padded = [_pad_pauli_word(w, n_qubits) for w in operators]
    pauli_words = [cudaq.pauli_word(w) for w in padded]

    @cudaq.kernel
    def kernel(n_qubits_k: int, n_electrons_k: int,
               thetas_k: list[float], words: list[cudaq.pauli_word]):
        q = cudaq.qvector(n_qubits_k)
        for i in range(n_electrons_k):
            x(q[i])
        for i in range(len(words)):
            exp_pauli(thetas_k[i], q, words[i])

    counts = cudaq.sample(
        kernel, n_qubits, n_electrons, thetas, pauli_words,
        shots_count=n_shots,
    )

    bitstring_counts = [(bs, int(count)) for bs, count in counts.items()]
    bitstring_counts.sort(key=lambda x: -x[1])
    return [bs for bs, _ in bitstring_counts]


def run_qsci(
    record: dict[str, Any],
    operators: list[str],
    thetas: list[float],
    n_shots: int = 8192,
    bond_dims: list[int] | None = None,
    n_samples_list: list[int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run QSCI for a single molecule with the improved operators."""
    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)

    if bond_dims is None:
        bond_dims = [128, 256]
    if n_samples_list is None:
        n_samples_list = [100, 500, 1000, 5000]

    use_sv = n_qubits <= 24

    # Compute HF energy
    hf_energy = None
    if use_sv:
        try:
            spin_ham = hamiltonian_to_spin_operator(record)
            cudaq.set_target("nvidia")

            @cudaq.kernel
            def hf_kern(n_q: int, n_e: int):
                q = cudaq.qvector(n_q)
                for i in range(n_e):
                    x(q[i])

            result = cudaq.observe(hf_kern, spin_ham, n_qubits, n_electrons)
            hf_energy = float(result.expectation())
        except Exception as e:
            print(f"    HF energy failed: {e}")
    else:
        hf_energy = record.get("hf_energy")

    results = {
        "molecule": record.get("name", "unknown"),
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "n_operators": len(operators),
        "hf_energy": hf_energy,
        "sweep_results": [],
    }

    for bond_dim in bond_dims:
        print(f"  Bond D={bond_dim}...")
        t0 = time.time()
        try:
            if use_sv:
                all_bs = sample_bitstrings_sv(
                    operators, thetas, n_qubits, n_electrons,
                    n_shots=n_shots, seed=seed,
                )
            else:
                all_bs = sample_bitstrings_mps(
                    operators, thetas, n_qubits, n_electrons,
                    n_shots=n_shots, bond_dim=bond_dim, seed=seed,
                )
            sample_time = time.time() - t0

            # Always include HF determinant
            hf_bs = format((1 << n_electrons) - 1, f"0{n_qubits}b")
            if hf_bs not in all_bs:
                all_bs.insert(0, hf_bs)

            print(f"    {len(all_bs)} unique bitstrings in {sample_time:.1f}s")
        except Exception as e:
            print(f"    Sampling failed: {e}")
            results["sweep_results"].append({
                "bond_dim": bond_dim, "n_unique_bitstrings": 0,
                "error": str(e), "qsci_energies": {},
            })
            continue

        for n_samples in n_samples_list:
            actual = min(n_samples, len(all_bs))
            if actual == 0:
                continue

            selected = all_bs[:actual]
            t0 = time.time()
            try:
                energy = qsci_energy_from_bitstrings(record, selected)
                diag_time = time.time() - t0
                err_vs_hf = abs(energy - hf_energy) * 1000 if hf_energy else None
                print(f"    QSCI N={actual}: E={energy:.6f} Ha ({diag_time:.1f}s, ΔHF={err_vs_hf:.3f} mHa)")
            except Exception as e:
                print(f"    QSCI N={actual} failed: {e}")
                energy = None
                diag_time = time.time() - t0
                err_vs_hf = None

            results["sweep_results"].append({
                "bond_dim": bond_dim,
                "n_samples_requested": n_samples,
                "n_samples_used": actual,
                "n_unique_bitstrings": len(all_bs),
                "n_shots": n_shots,
                "sample_time_seconds": sample_time,
                "diag_time_seconds": diag_time,
                "qsci_energy": energy,
                "error_vs_hf_mha": err_vs_hf,
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Tile operators, optimize with L-BFGS-B, re-run QSCI"
    )
    parser.add_argument(
        "--operators", type=Path,
        default=PROJECT_ROOT / "results" / "eval" / "h_cgqe_operators_for_qsci.json",
        help="Path to operator JSON file",
    )
    parser.add_argument(
        "--hamiltonians", type=Path,
        default=PROJECT_ROOT / "results" / "data" / "hamiltonians_40plus.json" / "hamiltonians.json",
        help="Path to Hamiltonian JSON file",
    )
    parser.add_argument(
        "--molecules", type=str, nargs="+",
        default=["n2", "formaldehyde", "ethylene", "benzene_cas20"],
        help="Target molecules to improve",
    )
    parser.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "results" / "phase3_final" / "qsci" / "qsci_improved_results.json",
        help="Output path for QSCI results",
    )
    parser.add_argument(
        "--operators-out", type=Path,
        default=PROJECT_ROOT / "results" / "eval" / "h_cgqe_operators_tiled.json",
        help="Output path for tiled operator JSON",
    )
    parser.add_argument(
        "--depth-multiplier", type=int, default=4,
        help="Circuit depth = depth_multiplier × n_qubits",
    )
    parser.add_argument(
        "--max-iter", type=int, default=200,
        help="Max L-BFGS-B iterations",
    )
    parser.add_argument(
        "--n-shots", type=int, default=8192,
        help="Number of sampling shots for QSCI",
    )
    parser.add_argument(
        "--max-opt-qubits", type=int, default=24,
        help="Skip L-BFGS-B for molecules with more qubits (use tiled default thetas)",
    )
    args = parser.parse_args()

    if cudaq is None:
        print("ERROR: CUDA-Q is required. Install with: pip install cudaq")
        sys.exit(1)

    # Load operator data
    print(f"Loading operators from {args.operators}")
    with args.operators.open("r") as f:
        op_data = json.load(f)
    op_map = {entry["molecule"]: entry for entry in op_data}

    # Load Hamiltonians
    print(f"Loading Hamiltonians from {args.hamiltonians}")
    ham_records = load_hamiltonian_records(args.hamiltonians)
    record_map = {r["name"]: r for r in ham_records}

    # Phase 1: Tile operators and optimize thetas
    print("\n" + "=" * 70)
    print(f"PHASE 1: Tile operators to {args.depth_multiplier}×n_qubits and optimize with L-BFGS-B")
    print("=" * 70)

    improved_operators = []
    for name in args.molecules:
        if name not in op_map:
            print(f"\n  {name}: No operator data found, skipping")
            continue
        if name not in record_map:
            print(f"\n  {name}: No Hamiltonian found, skipping")
            continue

        entry = op_map[name]
        record = record_map[name]
        n_qubits = int(record["n_qubits"])

        base_ops = entry["best_sequence"]["operators"]
        base_thetas = entry["best_sequence"]["thetas"]
        target_depth = args.depth_multiplier * n_qubits

        print(f"\n  {name} ({n_qubits}q): {len(base_ops)} ops → {target_depth} ops")

        tiled_ops, tiled_thetas = tile_operators(base_ops, base_thetas, target_depth)

        # Run L-BFGS-B if within qubit limit
        if n_qubits <= args.max_opt_qubits:
            print(f"  Optimizing thetas with L-BFGS-B...")
            energy, opt_thetas = optimize_thetas_lbfgsb(
                record, tiled_ops, np.array(tiled_thetas),
                max_iter=args.max_iter, seed=42,
            )
            tiled_thetas = opt_thetas.tolist()
            theta_source = "lbfgs_b_tiled"
        else:
            print(f"  Skipping L-BFGS-B ({n_qubits}q > {args.max_opt_qubits}q limit)")
            energy = entry["best_sequence"].get("energy", None)
            theta_source = "tiled_default"

        improved_operators.append({
            "molecule": name,
            "best_sequence": {
                "operators": tiled_ops,
                "thetas": tiled_thetas,
                "energy": energy,
            },
            "theta_source": theta_source,
            "n_base_operators": len(base_ops),
            "target_depth": target_depth,
            "depth_multiplier": args.depth_multiplier,
        })

    # Save tiled operators
    args.operators_out.parent.mkdir(parents=True, exist_ok=True)
    with args.operators_out.open("w") as f:
        json.dump(improved_operators, f, indent=2)
    print(f"\nSaved tiled operators to {args.operators_out}")

    # Phase 2: Run QSCI with improved operators
    print("\n" + "=" * 70)
    print("PHASE 2: Re-run QSCI with improved operators")
    print("=" * 70)

    all_results = []
    for entry in improved_operators:
        name = entry["molecule"]
        record = record_map[name]
        n_qubits = int(record["n_qubits"])

        print(f"\n  [{name}] ({n_qubits}q, {len(entry['best_sequence']['operators'])} ops)")

        mol_result = run_qsci(
            record,
            operators=entry["best_sequence"]["operators"],
            thetas=entry["best_sequence"]["thetas"],
            n_shots=args.n_shots,
        )
        mol_result["theta_source"] = entry["theta_source"]
        mol_result["n_operators"] = len(entry["best_sequence"]["operators"])
        all_results.append(mol_result)

        # Incremental save
        partial = {
            "experiment": "gqe_qsci_improved",
            "description": f"QSCI with tiled operators ({args.depth_multiplier}×n_qubits) + L-BFGS-B",
            "results": all_results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(partial, f, indent=2)

    # Final save
    final_result = {
        "experiment": "gqe_qsci_improved",
        "description": f"QSCI with tiled operators ({args.depth_multiplier}×n_qubits) + L-BFGS-B",
        "depth_multiplier": args.depth_multiplier,
        "n_shots": args.n_shots,
        "results": all_results,
    }
    with args.out.open("w") as f:
        json.dump(final_result, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Molecule':20s} {'Qubits':>6s} {'Ops':>5s} {'Bitstrings':>10s} {'QSCI E':>14s} {'HF E':>14s} {'Δ(mHa)':>10s}")
    print("-" * 80)
    for r in all_results:
        n_bs = max((s.get("n_unique_bitstrings", 0) for s in r.get("sweep_results", [])), default=0)
        best_e = min((s.get("qsci_energy", float("inf")) for s in r.get("sweep_results", []) if s.get("qsci_energy")), default=float("inf"))
        hf = r.get("hf_energy")
        delta = abs(best_e - hf) * 1000 if hf and best_e != float("inf") else None
        hf_str = f"{hf:14.6f}" if hf else f"{'N/A':>14s}"
        delta_str = f"{delta:10.3f}" if delta else f"{'N/A':>10s}"
        best_str = f"{best_e:14.6f}" if best_e != float("inf") else f"{'N/A':>14s}"
        print(f"{r['molecule']:20s} {r['n_qubits']:6d} {r.get('n_operators', 0):5d} {n_bs:10d} {best_str} {hf_str} {delta_str}")
    print(f"\nSaved QSCI results to {args.out}")
    print(f"Saved tiled operators to {args.operators_out}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
