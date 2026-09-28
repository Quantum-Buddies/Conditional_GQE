"""GQE-QSCI scaling experiment: sample determinants from H-cGQE state, diagonalize subspace.

This module implements the full QSCI pipeline:
1. Prepare a quantum state (HF + H-cGQE operators) on CUDA-Q MPS backend
2. Sample computational-basis bitstrings from the prepared state
3. Build the Hamiltonian matrix in the subspace spanned by those bitstrings
4. Diagonalize classically to get the refined ground-state energy

The key insight: QSCI only needs a "good enough" reference state. The classical
subspace diagonalization refines it. This allows scaling to 40+ qubits on a
single GPU via the MPS (matrix product state) backend.

Usage:
    python src/gqe/eval/qsci.py \
        --hamiltonians results/data/hamiltonians_40plus.json/hamiltonians.json \
        --molecules n2_1.1_full ethylene formaldehyde benzene_cas20 \
        --n-samples 1000 5000 10000 \
        --bond-dims 64 128 256 \
        --out results/phase3_final/qsci/qsci_scaling_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import cudaq
except ImportError:
    cudaq = None

try:
    from src.gqe.common.hamiltonian_utils import (
        load_hamiltonian_records,
        hamiltonian_to_spin_operator,
        get_active_electron_count,
        find_record_by_name,
        iter_terms,
    )
except ImportError:
    from gqe.common.hamiltonian_utils import (
        load_hamiltonian_records,
        hamiltonian_to_spin_operator,
        get_active_electron_count,
        find_record_by_name,
        iter_terms,
    )

try:
    from src.gqe.eval.qsci_postprocess import qsci_energy_from_bitstrings
except ImportError:
    from gqe.eval.qsci_postprocess import qsci_energy_from_bitstrings


def _make_hcgqe_kernel(operators: list[str], thetas: list[float], n_qubits: int, n_electrons: int):
    """Build a CUDA-Q kernel for HF state + H-cGQE operator sequence."""

    padded_ops = []
    for w in operators:
        if len(w) < n_qubits:
            w = w + "I" * (n_qubits - len(w))
        elif len(w) > n_qubits:
            w = w[:n_qubits]
        padded_ops.append(w)

    if not thetas:
        thetas = [0.01] * len(padded_ops)

    @cudaq.kernel
    def kernel(n_qubits: int, n_electrons: int, pauli_words: list[cudaq.pauli_word], thetas: list[float]):
        q = cudaq.qvector(n_qubits)
        for i in range(n_electrons):
            x(q[i])
        for i in range(len(pauli_words)):
            exp_pauli(thetas[i], q, pauli_words[i])

    pauli_words = [cudaq.pauli_word(w) for w in padded_ops]
    return kernel, pauli_words, thetas


def _make_hf_kernel():
    """Build a CUDA-Q kernel for just the Hartree-Fock reference state."""

    @cudaq.kernel
    def hf_kernel(n_qubits: int, n_electrons: int):
        q = cudaq.qvector(n_qubits)
        for i in range(n_electrons):
            x(q[i])

    return hf_kernel


def _generate_entangling_operators(n_qubits: int, n_electrons: int, max_ops: int = 30) -> list[str]:
    """Generate entangling Pauli operators for creating excitations.

    Generates single and double excitation operators between occupied and
    virtual orbitals, scaled to the molecule's qubit count.
    """
    n_virtual = n_qubits - n_electrons
    ops = []

    # Single excitations: X_i X_j and Y_i Y_j for occupied i, virtual j
    # Select a subset to stay within max_ops
    occ_indices = list(range(n_electrons))
    virt_indices = list(range(n_electrons, n_qubits))

    # Prioritize nearest-neighbor excitations (most chemically relevant)
    pairs = []
    for i in occ_indices:
        for j in virt_indices:
            pairs.append((i, j))
    # Sort by distance (nearest first)
    pairs.sort(key=lambda p: abs(p[0] - p[1]))

    for i, j in pairs:
        if len(ops) >= max_ops:
            break
        word = ["I"] * n_qubits
        word[i] = "X"
        word[j] = "X"
        ops.append("".join(word))
        if len(ops) >= max_ops:
            break
        word = ["I"] * n_qubits
        word[i] = "Y"
        word[j] = "Y"
        ops.append("".join(word))

    # Double excitations: X_i X_j X_k X_l for two occupied -> two virtual
    if n_electrons >= 2 and n_virtual >= 2 and len(ops) < max_ops:
        for i in range(n_electrons - 1):
            for j in range(i + 1, n_electrons):
                for k in range(n_electrons, n_qubits - 1):
                    for l in range(k + 1, n_qubits):
                        word = ["I"] * n_qubits
                        word[i] = "X"
                        word[j] = "X"
                        word[k] = "X"
                        word[l] = "X"
                        ops.append("".join(word))
                        if len(ops) >= max_ops:
                            break
                    if len(ops) >= max_ops:
                        break
                if len(ops) >= max_ops:
                    break
            if len(ops) >= max_ops:
                break

    return ops


def _make_entangled_hf_kernel():
    """Build a CUDA-Q kernel for HF state + entangling excitations.

    Uses exp_pauli with entangling operators (X/Y on occupied+virtual qubits)
    to create superposition between HF and excited determinants. This gives
    QSCI a rich set of determinants that includes the important configurations.
    """

    # Standard entangling operators for creating excitations.
    # These are generated per-molecule based on n_qubits and n_electrons.
    # See _generate_entangling_operators() for the generation logic.

    @cudaq.kernel
    def entangled_hf(n_qubits: int, n_electrons: int,
                     pauli_words: list[cudaq.pauli_word], thetas: list[float]):
        q = cudaq.qvector(n_qubits)
        for i in range(n_electrons):
            x(q[i])
        for i in range(len(pauli_words)):
            exp_pauli(thetas[i], q, pauli_words[i])

    return entangled_hf


def sample_bitstrings(
    kernel: Any,
    kernel_args: tuple,
    n_qubits: int,
    n_shots: int = 4096,
    seed: int | None = 42,
) -> list[str]:
    """Sample computational-basis bitstrings from a CUDA-Q kernel.

    Args:
        kernel: CUDA-Q kernel function.
        kernel_args: Arguments to pass to the kernel.
        n_qubits: Number of qubits (for validation).
        n_shots: Number of shots to sample.
        seed: Optional RNG seed.

    Returns:
        List of unique bitstrings, sorted by frequency (most frequent first).
    """
    if cudaq is None:
        raise ImportError("CUDA-Q is required for QSCI sampling")

    if seed is not None:
        cudaq.set_random_seed(seed)

    counts = cudaq.sample(kernel, *kernel_args, shots_count=n_shots)

    # Sort by frequency (most frequent first) and return unique bitstrings
    bitstring_counts = [(bs, int(count)) for bs, count in counts.items()]
    bitstring_counts.sort(key=lambda x: -x[1])
    return [bs for bs, _ in bitstring_counts]


def run_qsci_for_molecule(
    record: dict[str, Any],
    operators: list[str] | None = None,
    thetas: list[float] | None = None,
    n_samples_list: list[int] | None = None,
    bond_dims: list[int] | None = None,
    n_shots: int = 8192,
    backend: str = "tensornet-mps",
    seed: int = 42,
    excitation_angle: float = 0.3,
    max_entangling_ops: int = 30,
) -> dict[str, Any]:
    """Run QSCI for a single molecule across parameter sweeps.

    Args:
        record: Hamiltonian record.
        operators: H-cGQE operator sequence (if None, uses HF + entangling layer).
        thetas: Rotation parameters for operators.
        n_samples_list: List of subspace sizes to try.
        bond_dims: List of MPS bond dimensions to try.
        n_shots: Number of sampling shots.
        backend: CUDA-Q backend ("nvidia" for <=24q, "tensornet-mps" for >24q).
        seed: RNG seed for reproducibility.
        excitation_angle: Angle for entangling operators when no H-cGQE ops provided.
        max_entangling_ops: Maximum number of entangling operators to generate.

    Returns:
        Dictionary with QSCI results for each parameter combination.
    """
    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)

    if n_samples_list is None:
        n_samples_list = [100, 500, 1000, 5000]
    if bond_dims is None:
        bond_dims = [64, 128, 256]

    # Determine which kernel to use
    use_hcgqe = operators is not None and len(operators) > 0

    # Also compute HF energy as baseline (diagonal expectation)
    # Skip for >24q to avoid OOM with large Hamiltonians on MPS
    hf_energy = None
    if n_qubits <= 24:
        try:
            if cudaq is not None:
                cudaq.set_target("nvidia")
                spin_ham = hamiltonian_to_spin_operator(record)
                hf_kern = _make_hf_kernel()
                result = cudaq.observe(hf_kern, spin_ham, n_qubits, n_electrons)
                hf_energy = float(result.expectation())
        except Exception as e:
            print(f"    HF energy computation failed: {e}")
    else:
        print(f"    Skipping HF energy for {n_qubits}q (too large for statevector)")
        # Use HF energy from record if available
        hf_energy = record.get("hf_energy")

    results = {
        "molecule": record.get("name", "unknown"),
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "n_hamiltonian_terms": len(record.get("terms", [])),
        "hf_energy": hf_energy,
        "used_hcgqe_operators": use_hcgqe,
        "n_operators": len(operators) if use_hcgqe else 0,
        "backend": backend,
        "sweep_results": [],
    }

    # Sample bitstrings at each bond dimension
    for bond_dim in bond_dims:
        print(f"  Bond dimension D={bond_dim}...")

        # Set MPS bond dimension
        if backend == "tensornet-mps":
            os.environ["CUDAQ_MPS_MAX_BOND"] = str(bond_dim)

        try:
            cudaq.set_target(backend)
        except Exception as e:
            print(f"    Failed to set backend {backend}: {e}")
            continue

        # Build kernel
        if use_hcgqe:
            kernel, pauli_words, theta_vals = _make_hcgqe_kernel(
                operators, thetas, n_qubits, n_electrons
            )
            kernel_args = (n_qubits, n_electrons, pauli_words, theta_vals)
        else:
            kernel = _make_entangled_hf_kernel()
            # Generate entangling operators for this molecule
            ent_ops = _generate_entangling_operators(n_qubits, n_electrons, max_ops=max_entangling_ops)
            pauli_words = [cudaq.pauli_word(w) for w in ent_ops]
            # Use specified angle to create excitations
            theta_vals = [excitation_angle] * len(ent_ops)
            kernel_args = (n_qubits, n_electrons, pauli_words, theta_vals)

        # Sample bitstrings
        t0 = time.time()
        try:
            all_bitstrings = sample_bitstrings(
                kernel, kernel_args, n_qubits, n_shots=n_shots, seed=seed
            )
            sample_time = time.time() - t0

            # Always include the HF determinant in the subspace
            hf_bitstring = format((1 << n_electrons) - 1, f"0{n_qubits}b")
            if hf_bitstring not in all_bitstrings:
                all_bitstrings.insert(0, hf_bitstring)
                print(f"    Added HF determinant: {hf_bitstring}")

            print(f"    Sampled {len(all_bitstrings)} unique bitstrings in {sample_time:.1f}s")
        except Exception as e:
            print(f"    Sampling failed: {e}")
            sample_time = time.time() - t0
            results["sweep_results"].append({
                "bond_dim": bond_dim,
                "n_shots": n_shots,
                "n_unique_bitstrings": 0,
                "sample_time_seconds": sample_time,
                "error": str(e),
                "qsci_energies": {},
            })
            continue

        # Run QSCI at each subspace size
        for n_samples in n_samples_list:
            actual_samples = min(n_samples, len(all_bitstrings))
            if actual_samples == 0:
                continue

            selected_bitstrings = all_bitstrings[:actual_samples]

            t0 = time.time()
            try:
                qsci_energy = qsci_energy_from_bitstrings(record, selected_bitstrings)
                diag_time = time.time() - t0
                print(f"    QSCI N={actual_samples}: E={qsci_energy:.6f} Ha ({diag_time:.1f}s)")
            except Exception as e:
                print(f"    QSCI N={actual_samples} failed: {e}")
                qsci_energy = None
                diag_time = time.time() - t0

            results["sweep_results"].append({
                "bond_dim": bond_dim,
                "n_samples_requested": n_samples,
                "n_samples_used": actual_samples,
                "n_unique_bitstrings": len(all_bitstrings),
                "n_shots": n_shots,
                "sample_time_seconds": sample_time,
                "diag_time_seconds": diag_time,
                "qsci_energy": qsci_energy,
                "error_vs_hf": abs(qsci_energy - hf_energy) * 1000 if qsci_energy is not None and hf_energy is not None else None,
            })

    return results


def run_qsci_scaling(
    hamiltonian_path: Path,
    molecule_names: list[str],
    n_samples_list: list[int],
    bond_dims: list[int],
    n_shots: int = 8192,
    operators_map: dict[str, dict] | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Run QSCI scaling experiment across multiple molecules.

    Args:
        hamiltonian_path: Path to Hamiltonian JSON file.
        molecule_names: List of molecule names to run.
        n_samples_list: Subspace sizes to sweep.
        bond_dims: MPS bond dimensions to sweep.
        n_shots: Sampling shots per run.
        operators_map: Optional dict mapping molecule name -> {operators, thetas}.
        out_path: Optional path for incremental saving.

    Returns:
        Complete results dictionary.
    """
    records = load_hamiltonian_records(hamiltonian_path)
    record_map = {r["name"]: r for r in records}

    all_results = []
    total = len(molecule_names)

    for idx, name in enumerate(molecule_names):
        record = record_map.get(name)
        if record is None:
            print(f"\n[{idx+1}/{total}] {name}: NOT FOUND in Hamiltonian data")
            continue

        n_qubits = int(record["n_qubits"])
        print(f"\n[{idx+1}/{total}] {name} ({n_qubits} qubits, {len(record.get('terms',[]))} terms)")

        # Determine backend
        backend = "nvidia" if n_qubits <= 24 else "tensornet-mps"

        # Get operators if available
        ops_info = (operators_map or {}).get(name, {})
        operators = ops_info.get("operators")
        thetas = ops_info.get("thetas")

        mol_result = run_qsci_for_molecule(
            record,
            operators=operators,
            thetas=thetas,
            n_samples_list=n_samples_list,
            bond_dims=bond_dims,
            n_shots=n_shots,
            backend=backend,
        )

        all_results.append(mol_result)

        # Incremental save
        if out_path is not None:
            partial = {
                "experiment": "gqe_qsci_scaling",
                "description": "QSCI scaling: sample determinants from H-cGQE/HF state, diagonalize subspace",
                "molecules_run": len(all_results),
                "molecules_total": total,
                "n_samples_list": n_samples_list,
                "bond_dims": bond_dims,
                "n_shots": n_shots,
                "results": all_results,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                json.dump(partial, f, indent=2)

    return {
        "experiment": "gqe_qsci_scaling",
        "description": "QSCI scaling: sample determinants from H-cGQE/HF state, diagonalize subspace",
        "n_samples_list": n_samples_list,
        "bond_dims": bond_dims,
        "n_shots": n_shots,
        "results": all_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GQE-QSCI scaling experiment")
    parser.add_argument("--hamiltonians", type=Path, required=True, help="Path to Hamiltonian JSON")
    parser.add_argument("--molecules", type=str, nargs="+", required=True, help="Molecule names to run")
    parser.add_argument("--n-samples", type=int, nargs="+", default=[100, 500, 1000, 5000],
                        help="Subspace sizes to sweep")
    parser.add_argument("--bond-dims", type=int, nargs="+", default=[64, 128, 256],
                        help="MPS bond dimensions to sweep")
    parser.add_argument("--n-shots", type=int, default=8192, help="Number of sampling shots")
    parser.add_argument("--optimized", type=Path, default=None,
                        help="Path to optimized results JSON (for H-cGQE operator sequences)")
    parser.add_argument("--out", type=Path, required=True, help="Output path")
    args = parser.parse_args()

    # Load optimized operator sequences if provided
    operators_map: dict[str, dict] = {}
    if args.optimized and args.optimized.exists():
        with args.optimized.open("r") as f:
            opt_data = json.load(f)
        if isinstance(opt_data, list):
            for entry in opt_data:
                mol = entry.get("molecule")
                best = entry.get("best_sequence", {})
                ops = best.get("operators", [])
                thetas = best.get("thetas", [])
                if mol and ops:
                    operators_map[mol] = {"operators": ops, "thetas": thetas}
        elif isinstance(opt_data, dict):
            for entry in opt_data.get("results", []):
                mol = entry.get("molecule")
                best = entry.get("best_sequence", {})
                ops = best.get("operators", [])
                thetas = best.get("thetas", [])
                if mol and ops:
                    operators_map[mol] = {"operators": ops, "thetas": thetas}
        print(f"Loaded operator sequences for {len(operators_map)} molecules")

    result = run_qsci_scaling(
        hamiltonian_path=args.hamiltonians,
        molecule_names=args.molecules,
        n_samples_list=args.n_samples,
        bond_dims=args.bond_dims,
        n_shots=args.n_shots,
        operators_map=operators_map,
        out_path=args.out,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved QSCI scaling results to {args.out}")


if __name__ == "__main__":
    main()
