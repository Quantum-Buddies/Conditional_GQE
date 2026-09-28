#!/usr/bin/env python
"""Run local SQD control suite for H2/LiH pilot.

Executes multiple SQD control paths to validate the pipeline:
1. **Ideal**: Exact statevector sampling (CUDA-Q, no noise)
2. **Noiseless simulator**: Qiskit Aer statevector with finite shots
3. **Noisy simulator**: Qiskit Aer density matrix with device noise model
4. **Random control**: Uniform random bitstrings (negative control)
5. **Hardware counts**: Load pre-existing QPU counts from JSON (if available)

Each path produces:
- Measurement counts (bitstring -> frequency)
- SQD energy at multiple subspace sizes
- Symmetry-filtered SQD energy
- Nested subspace monotonicity check
- Comparison vs FCI and HF energy

Usage:
    python scripts/run_sqd_pilot.py \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --molecules h2 lih \
        --optimized results/eval/h_cgqe_optimized.json \
        --shots 4096 \
        --out results/eval/sqd_pilot/

    # Include noisy simulator control:
    python scripts/run_sqd_pilot.py --molecules h2 --noisy --shots 4096

    # Include hardware counts from prior QPU run:
    python scripts/run_sqd_pilot.py --molecules h2 --hardware-counts results/eval/qpu_rigetti_h2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
    get_active_electron_count,
)
from src.gqe.eval.sqd import (
    sqd_energy_from_bitstrings,
    sqd_energy_from_counts,
    filter_by_particle_number,
    filter_by_spin_parity,
    apply_symmetry_filters,
    select_subspace_by_counts,
    nested_subspace_energies,
    check_monotonicity,
    exact_diagonalize,
    run_sqd,
)


# ---------------------------------------------------------------------------
# Count generation strategies
# ---------------------------------------------------------------------------

def generate_ideal_counts(
    record: Dict[str, Any],
    operators: List[str],
    thetas: List[float],
    n_shots: int = 4096,
    seed: int = 42,
) -> Dict[str, int]:
    """Generate counts from exact CUDA-Q statevector sampling (no noise).

    Uses CUDA-Q's sample() on the nvidia backend for exact statevector.
    """
    import cudaq

    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)

    # Pad operators
    padded = []
    for w in operators:
        if len(w) < n_qubits:
            w = w + "I" * (n_qubits - len(w))
        elif len(w) > n_qubits:
            w = w[:n_qubits]
        padded.append(w)

    pauli_words = [cudaq.pauli_word(w) for w in padded]
    theta_vals = thetas if thetas else [0.01] * len(padded)

    @cudaq.kernel
    def kernel(n_q: int, n_e: int, pws: list[cudaq.pauli_word], ths: list[float]):
        q = cudaq.qvector(n_q)
        for i in range(n_e):
            x(q[i])
        for i in range(len(pws)):
            exp_pauli(ths[i], q, pws[i])

    cudaq.set_random_seed(seed)
    try:
        cudaq.set_target("nvidia")
    except Exception:
        cudaq.set_target("qpp-cpu")

    counts = cudaq.sample(kernel, n_qubits, n_electrons, pauli_words, theta_vals, shots_count=n_shots)
    return {bs: int(c) for bs, c in counts.items()}


def generate_noiseless_simulator_counts(
    record: Dict[str, Any],
    operators: List[str],
    thetas: List[float],
    n_shots: int = 4096,
    seed: int = 42,
) -> Dict[str, int]:
    """Generate counts using Qiskit Aer statevector simulator with finite shots."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)

    # Build circuit (same convention as qbraid_backend.py)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_electrons):
        qc.x(n_qubits - 1 - i)

    for op_idx, (pauli_word, theta) in enumerate(zip(operators, thetas or [0.01] * len(operators))):
        if len(pauli_word) < n_qubits:
            pauli_word = pauli_word + "I" * (n_qubits - len(pauli_word))
        elif len(pauli_word) > n_qubits:
            pauli_word = pauli_word[:n_qubits]

        non_identity = [(i, p) for i, p in enumerate(pauli_word) if p != "I"]
        if not non_identity:
            continue

        for i, p in non_identity:
            if p == "X":
                qc.h(i)
            elif p == "Y":
                qc.sdg(i)
                qc.h(i)

        for idx in range(len(non_identity) - 1):
            qc.cx(non_identity[idx][0], non_identity[idx + 1][0])

        last_q = non_identity[-1][0]
        qc.rz(2 * theta, last_q)

        for idx in range(len(non_identity) - 2, -1, -1):
            qc.cx(non_identity[idx][0], non_identity[idx + 1][0])

        for i, p in non_identity:
            if p == "X":
                qc.h(i)
            elif p == "Y":
                qc.h(i)
                qc.s(i)

    qc.measure_all()

    sim = AerSimulator(seed_simulator=seed)
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=n_shots).result()
    raw_counts = result.get_counts()

    # Qiskit uses big-endian bitstrings; convert to our convention (qubit 0 = LSB = rightmost)
    converted = {}
    for bs, count in raw_counts.items():
        # Qiskit: leftmost = qubit n-1, rightmost = qubit 0 -> already matches our convention
        converted[bs] = count
    return converted


def generate_noisy_simulator_counts(
    record: Dict[str, Any],
    operators: List[str],
    thetas: List[float],
    n_shots: int = 4096,
    seed: int = 42,
    noise_model: str = "depolarizing",
    error_rate: float = 0.001,
) -> Dict[str, int]:
    """Generate counts with a simple noise model (depolarizing or bit flip)."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)

    # Build circuit (same as noiseless)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_electrons):
        qc.x(n_qubits - 1 - i)

    for op_idx, (pauli_word, theta) in enumerate(zip(operators, thetas or [0.01] * len(operators))):
        if len(pauli_word) < n_qubits:
            pauli_word = pauli_word + "I" * (n_qubits - len(pauli_word))
        elif len(pauli_word) > n_qubits:
            pauli_word = pauli_word[:n_qubits]

        non_identity = [(i, p) for i, p in enumerate(pauli_word) if p != "I"]
        if not non_identity:
            continue

        for i, p in non_identity:
            if p == "X":
                qc.h(i)
            elif p == "Y":
                qc.sdg(i)
                qc.h(i)

        for idx in range(len(non_identity) - 1):
            qc.cx(non_identity[idx][0], non_identity[idx + 1][0])

        last_q = non_identity[-1][0]
        qc.rz(2 * theta, last_q)

        for idx in range(len(non_identity) - 2, -1, -1):
            qc.cx(non_identity[idx][0], non_identity[idx + 1][0])

        for i, p in non_identity:
            if p == "X":
                qc.h(i)
            elif p == "Y":
                qc.h(i)
                qc.s(i)

    qc.measure_all()

    # Build noise model
    noise = NoiseModel()
    if noise_model == "depolarizing":
        # 1-qubit gate error
        noise.add_all_qubit_quantum_error(
            depolarizing_error(error_rate, 1),
            ["x", "h", "s", "sdg", "rz"],
        )
        # 2-qubit gate error (higher rate)
        noise.add_all_qubit_quantum_error(
            depolarizing_error(error_rate * 3, 2),
            ["cx"],
        )
    elif noise_model == "bit_flip":
        noise.add_all_qubit_quantum_error(
            pauli_error([("X", error_rate), ("I", 1 - error_rate)], 1),
            ["x", "h", "s", "sdg", "rz"],
        )
        noise.add_all_qubit_quantum_error(
            pauli_error([("IX", error_rate), ("XI", error_rate), ("II", 1 - 2 * error_rate)], 2),
            ["cx"],
        )

    sim = AerSimulator(noise_model=noise, seed_simulator=seed)
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=n_shots).result()
    raw_counts = result.get_counts()
    return {bs: count for bs, count in raw_counts.items()}


def generate_random_counts(
    n_qubits: int,
    n_shots: int = 4096,
    seed: int = 42,
) -> Dict[str, int]:
    """Generate uniform random bitstring counts (negative control)."""
    rng = np.random.default_rng(seed)
    counts: Dict[str, int] = {}
    for _ in range(n_shots):
        bs = format(rng.integers(0, 2**n_qubits), f"0{n_qubits}b")
        counts[bs] = counts.get(bs, 0) + 1
    return counts


def load_hardware_counts(file_path: Path) -> Dict[str, int]:
    """Load hardware counts from a prior QPU job result JSON."""
    with file_path.open() as f:
        data = json.load(f)
    counts = data.get("counts", {})
    if isinstance(counts, dict):
        return {str(k): int(v) for k, v in counts.items()}
    raise ValueError(f"Could not extract counts from {file_path}")


# ---------------------------------------------------------------------------
# SQD analysis
# ---------------------------------------------------------------------------

def analyze_counts(
    record: Dict[str, Any],
    counts: Dict[str, int],
    fci_energy: float,
    hf_energy: Optional[float] = None,
    subspace_sizes: Optional[List[int]] = None,
    n_electrons: Optional[int] = None,
    apply_symmetry: bool = True,
) -> Dict[str, Any]:
    """Run full SQD analysis on a set of measurement counts.

    Returns a dict with:
    - sqd_energies: energy at each subspace size
    - symmetry_filtered_energies: energy with particle number + spin parity filtering
    - monotonicity_check: whether energies are monotonically decreasing
    - n_unique_bitstrings: number of unique bitstrings in counts
    - best_energy: lowest SQD energy achieved
    - error_vs_fci: |best_energy - fci| in mHa
    """
    n_qubits = int(record["n_qubits"])
    if n_electrons is None:
        n_electrons = get_active_electron_count(record)
    if subspace_sizes is None:
        n_unique = len(counts)
        subspace_sizes = sorted(set([min(s, n_unique) for s in [4, 8, 16, 32, 64, 128, 256, 512, n_unique]]))

    result: Dict[str, Any] = {
        "n_unique_bitstrings": len(counts),
        "n_total_shots": sum(counts.values()),
        "subspace_sizes": [],
        "sqd_energies": [],
        "symmetry_filtered_energies": [],
        "monotonicity_ok": True,
    }

    # Sort bitstrings by count
    sorted_bs = select_subspace_by_counts(counts)

    # Unfiltered SQD
    energies = []
    for k in subspace_sizes:
        if k > len(sorted_bs):
            break
        selected = sorted_bs[:k]
        e = sqd_energy_from_bitstrings(record, selected)
        energies.append(e)
        result["subspace_sizes"].append(k)
        result["sqd_energies"].append(e)

    # Symmetry-filtered SQD
    if apply_symmetry:
        filtered_bs = apply_symmetry_filters(
            sorted_bs, n_qubits, n_electrons,
            particle_number_tol=0,
            spin_parity=0,
        )
        sym_energies = []
        for k in subspace_sizes:
            if k > len(filtered_bs):
                break
            selected = filtered_bs[:k]
            if not selected:
                continue
            e = sqd_energy_from_bitstrings(record, selected)
            sym_energies.append(e)
        result["symmetry_filtered_energies"] = sym_energies
        result["n_symmetry_filtered_bitstrings"] = len(filtered_bs)

    # Monotonicity check
    if len(energies) > 1:
        result["monotonicity_ok"] = check_monotonicity(energies)

    # Best energy
    all_energies = energies + result.get("symmetry_filtered_energies", [])
    best_energy = min(all_energies) if all_energies else None
    result["best_energy"] = best_energy
    result["fci_energy"] = fci_energy
    result["hf_energy"] = hf_energy
    if best_energy is not None:
        result["error_vs_fci_mha"] = abs(best_energy - fci_energy) * 1000.0
    else:
        result["error_vs_fci_mha"] = None

    # Variational bound check
    if best_energy is not None:
        result["variational_bound_satisfied"] = best_energy >= fci_energy - 1e-10
    else:
        result["variational_bound_satisfied"] = None

    return result


# ---------------------------------------------------------------------------
# Multi-seed convergence study
# ---------------------------------------------------------------------------

def run_convergence_study(
    record: Dict[str, Any],
    operators: List[str],
    thetas: List[float],
    fci_energy: float,
    hf_energy: float,
    n_seeds: int = 20,
    shots: int = 4096,
    subspace_sizes: Optional[List[int]] = None,
    n_electrons: Optional[int] = None,
    use_noisy: bool = False,
    noise_model: str = "depolarizing",
    error_rate: float = 0.001,
    skip_ideal: bool = False,
) -> Dict[str, Any]:
    """Run multi-seed SQD convergence study with matched shot/R budgets.

    For each seed, generates counts, selects nested top-R subspaces, and
    computes SQD energy at each R. Aggregates across seeds to produce
    mean ± std convergence curves.

    Args:
        record: Hamiltonian record.
        operators: Circuit operators.
        thetas: Rotation angles.
        fci_energy: FCI energy for comparison.
        hf_energy: HF energy for comparison.
        n_seeds: Number of independent seeds (default 20).
        shots: Total shots per seed.
        subspace_sizes: List of R values for nested subspace sweep.
        n_electrons: Target electron count for symmetry filtering.
        use_noisy: Use noisy simulator instead of noiseless.
        skip_ideal: Skip CUDA-Q ideal sampling.

    Returns:
        Dict with per-seed energies, aggregate statistics, convergence data.
    """
    n_qubits = int(record["n_qubits"])
    if n_electrons is None:
        n_electrons = get_active_electron_count(record)
    if subspace_sizes is None:
        n_unique_max = min(2**n_qubits, shots)
        subspace_sizes = sorted(set([min(s, n_unique_max) for s in [4, 8, 16, 32, 64, 128, 256, 512]]))

    all_seed_energies: List[List[float]] = []
    all_seed_sym_energies: List[List[float]] = []

    for seed in tqdm(range(n_seeds), desc="Seeds"):
        # Generate counts
        if skip_ideal or not operators:
            if use_noisy:
                counts = generate_noisy_simulator_counts(
                    record, operators, thetas, shots, seed, noise_model, error_rate,
                )
            else:
                counts = generate_noiseless_simulator_counts(
                    record, operators, thetas, shots, seed,
                )
        else:
            try:
                counts = generate_ideal_counts(record, operators, thetas, shots, seed)
            except Exception:
                counts = generate_noiseless_simulator_counts(
                    record, operators, thetas, shots, seed,
                )

        # Nested top-R subspace energies
        sorted_bs = select_subspace_by_counts(counts)
        seed_energies = []
        seed_sym_energies = []

        for R in subspace_sizes:
            if R > len(sorted_bs):
                break
            selected = sorted_bs[:R]
            e = sqd_energy_from_bitstrings(record, selected)
            seed_energies.append(e)

            # Symmetry-filtered
            filtered = apply_symmetry_filters(
                sorted_bs[:R], n_qubits, n_electrons,
                particle_number_tol=0, spin_parity=0,
            )
            if filtered:
                e_sym = sqd_energy_from_bitstrings(record, filtered)
            else:
                e_sym = float("nan")
            seed_sym_energies.append(e_sym)

        all_seed_energies.append(seed_energies)
        all_seed_sym_energies.append(seed_sym_energies)

    # Pad shorter lists with nan for aggregation
    max_len = max(len(e) for e in all_seed_energies) if all_seed_energies else 0
    for energies in all_seed_energies:
        while len(energies) < max_len:
            energies.append(float("nan"))
    for energies in all_seed_sym_energies:
        while len(energies) < max_len:
            energies.append(float("nan"))

    energies_arr = np.array(all_seed_energies)
    sym_energies_arr = np.array(all_seed_sym_energies)

    # Aggregate
    mean_energies = np.nanmean(energies_arr, axis=0)
    std_energies = np.nanstd(energies_arr, axis=0)
    mean_sym_energies = np.nanmean(sym_energies_arr, axis=0)
    std_sym_energies = np.nanstd(sym_energies_arr, axis=0)

    actual_Rs = subspace_sizes[:max_len]

    return {
        "n_seeds": n_seeds,
        "shots": shots,
        "subspace_sizes": actual_Rs,
        "mean_energies": mean_energies.tolist(),
        "std_energies": std_energies.tolist(),
        "mean_sym_energies": mean_sym_energies.tolist(),
        "std_sym_energies": std_sym_energies.tolist(),
        "all_seed_energies": energies_arr.tolist(),
        "fci_energy": fci_energy,
        "hf_energy": hf_energy,
        "n_electrons": n_electrons,
    }


def plot_convergence(
    convergence_data: Dict[str, Any],
    molecule: str,
    out_path: Path,
) -> None:
    """Generate SQD convergence plot: energy vs R with error bands.

    Shows mean ± std SQD energy as a function of subspace size R,
    with FCI and HF reference lines. Both raw and symmetry-filtered curves.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  matplotlib not available, skipping convergence plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    Rs = convergence_data["subspace_sizes"]
    mean_e = np.array(convergence_data["mean_energies"])
    std_e = np.array(convergence_data["std_energies"])
    mean_sym = np.array(convergence_data["mean_sym_energies"])
    std_sym = np.array(convergence_data["std_sym_energies"])
    fci = convergence_data["fci_energy"]
    hf = convergence_data["hf_energy"]

    # Raw SQD
    ax.plot(Rs, mean_e, "b-o", label="SQD (raw)", markersize=4)
    ax.fill_between(Rs, mean_e - std_e, mean_e + std_e, alpha=0.2, color="blue")

    # Symmetry-filtered SQD
    valid_sym = ~np.isnan(mean_sym)
    if np.any(valid_sym):
        ax.plot(np.array(Rs)[valid_sym], mean_sym[valid_sym], "r-s", label="SQD (sym-filtered)", markersize=4)
        ax.fill_between(
            np.array(Rs)[valid_sym],
            mean_sym[valid_sym] - std_sym[valid_sym],
            mean_sym[valid_sym] + std_sym[valid_sym],
            alpha=0.2, color="red",
        )

    # Reference lines
    ax.axhline(y=fci, color="green", linestyle="--", linewidth=1.5, label=f"FCI = {fci:.4f} Ha")
    ax.axhline(y=hf, color="orange", linestyle=":", linewidth=1.5, label=f"HF = {hf:.4f} Ha")

    ax.set_xlabel("Subspace size R", fontsize=12, fontweight="bold")
    ax.set_ylabel("Energy (Ha)", fontsize=12, fontweight="bold")
    ax.set_title(f"SQD Convergence — {molecule} ({convergence_data['n_seeds']} seeds, {convergence_data['shots']} shots)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    if len(Rs) >= 2 and all(r > 0 for r in Rs):
        ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Convergence plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run local SQD control suite for H2/LiH pilot")
    parser.add_argument("--hamiltonians", type=Path, required=True,
                        help="Path to Hamiltonian JSON")
    parser.add_argument("--molecules", type=str, nargs="+", default=["h2"],
                        help="Molecule names to run")
    parser.add_argument("--optimized", type=Path, default=None,
                        help="Path to optimized results JSON (for operator sequences)")
    parser.add_argument("--shots", type=int, default=4096,
                        help="Number of sampling shots")
    parser.add_argument("--subspace-sizes", type=int, nargs="+", default=None,
                        help="Subspace sizes to sweep (default: auto)")
    parser.add_argument("--out", type=Path, default=Path("results/eval/sqd_pilot/"),
                        help="Output directory")
    parser.add_argument("--noisy", action="store_true",
                        help="Include noisy simulator control")
    parser.add_argument("--noise-model", type=str, default="depolarizing",
                        choices=["depolarizing", "bit_flip"],
                        help="Noise model for noisy control")
    parser.add_argument("--error-rate", type=float, default=0.001,
                        help="Error rate for noise model")
    parser.add_argument("--hardware-counts", type=Path, default=None,
                        help="JSON file with hardware counts from prior QPU run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip-ideal", action="store_true",
                        help="Skip CUDA-Q ideal sampling (use if no GPU available)")
    parser.add_argument("--convergence", action="store_true",
                        help="Run multi-seed convergence study (20 seeds, nested top-R subspaces)")
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Number of seeds for convergence study (default: 20)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load Hamiltonians
    records = load_hamiltonian_records(args.hamiltonians)
    record_map = {r["name"]: r for r in records}

    # Load optimized operator sequences
    operators_map: Dict[str, Dict] = {}
    if args.optimized and args.optimized.exists():
        with args.optimized.open() as f:
            opt_data = json.load(f)
        if isinstance(opt_data, list):
            for entry in opt_data:
                mol = entry.get("molecule")
                ops = entry.get("best_operators", [])
                thetas = entry.get("best_thetas", [])
                if mol and ops:
                    operators_map[mol] = {"operators": ops, "thetas": thetas}
        elif isinstance(opt_data, dict):
            for entry in opt_data.get("results", []):
                mol = entry.get("molecule")
                ops = entry.get("best_operators", [])
                thetas = entry.get("best_thetas", [])
                if mol and ops:
                    operators_map[mol] = {"operators": ops, "thetas": thetas}
        print(f"Loaded operator sequences for {len(operators_map)} molecules")

    all_results = []

    for mol_name in tqdm(args.molecules, desc="Molecules"):
        record = record_map.get(mol_name)
        if record is None:
            print(f"  {mol_name}: NOT FOUND in Hamiltonian data")
            continue

        n_qubits = int(record["n_qubits"])
        n_electrons = get_active_electron_count(record)
        print(f"\n{'='*60}")
        print(f"  {mol_name} ({n_qubits} qubits, {n_electrons} electrons)")
        print(f"{'='*60}")

        # Compute FCI
        fci_energy = exact_diagonalize(record)
        print(f"  FCI energy: {fci_energy:.6f} Ha")

        # Compute HF energy
        hf_bs = format((1 << n_electrons) - 1, f"0{n_qubits}b")
        hf_energy = sqd_energy_from_bitstrings(record, [hf_bs])
        print(f"  HF energy:  {hf_energy:.6f} Ha")
        print(f"  HF error:   {abs(hf_energy - fci_energy)*1000:.3f} mHa")

        # Get operators
        ops_info = operators_map.get(mol_name, {})
        operators = ops_info.get("operators", [])
        thetas = ops_info.get("thetas", [])

        if not operators:
            # Use HF-only if no operators
            operators = []
            thetas = []
            print(f"  No operator sequence found, using HF-only state")

        mol_result: Dict[str, Any] = {
            "molecule": mol_name,
            "n_qubits": n_qubits,
            "n_electrons": n_electrons,
            "fci_energy": fci_energy,
            "hf_energy": hf_energy,
            "n_operators": len(operators),
            "controls": {},
        }

        # --- Control 1: Ideal (CUDA-Q statevector) ---
        if not args.skip_ideal and operators:
            print(f"\n  [1/4] Ideal statevector (CUDA-Q)...")
            try:
                counts = generate_ideal_counts(record, operators, thetas, args.shots, args.seed)
                analysis = analyze_counts(record, counts, fci_energy, hf_energy,
                                         args.subspace_sizes, n_electrons)
                mol_result["controls"]["ideal"] = {
                    "counts": counts,
                    "analysis": analysis,
                    "source": "cudaq_statevector",
                }
                print(f"    Best SQD energy: {analysis['best_energy']:.6f} Ha")
                print(f"    Error vs FCI: {analysis['error_vs_fci_mha']:.3f} mHa")
                print(f"    Monotonicity: {'OK' if analysis['monotonicity_ok'] else 'FAIL'}")
            except Exception as e:
                print(f"    FAILED: {e}")
                mol_result["controls"]["ideal"] = {"error": str(e)}

        # --- Control 2: Noiseless simulator (Qiskit Aer) ---
        if operators:
            print(f"\n  [2/4] Noiseless simulator (Qiskit Aer)...")
            try:
                counts = generate_noiseless_simulator_counts(record, operators, thetas,
                                                             args.shots, args.seed)
                analysis = analyze_counts(record, counts, fci_energy, hf_energy,
                                         args.subspace_sizes, n_electrons)
                mol_result["controls"]["noiseless_simulator"] = {
                    "counts": counts,
                    "analysis": analysis,
                    "source": "qiskit_aer_statevector",
                }
                print(f"    Best SQD energy: {analysis['best_energy']:.6f} Ha")
                print(f"    Error vs FCI: {analysis['error_vs_fci_mha']:.3f} mHa")
            except Exception as e:
                print(f"    FAILED: {e}")
                mol_result["controls"]["noiseless_simulator"] = {"error": str(e)}

        # --- Control 3: Noisy simulator ---
        if args.noisy and operators:
            print(f"\n  [3/4] Noisy simulator ({args.noise_model}, rate={args.error_rate})...")
            try:
                counts = generate_noisy_simulator_counts(record, operators, thetas,
                                                         args.shots, args.seed,
                                                         args.noise_model, args.error_rate)
                analysis = analyze_counts(record, counts, fci_energy, hf_energy,
                                         args.subspace_sizes, n_electrons)
                mol_result["controls"]["noisy_simulator"] = {
                    "counts": counts,
                    "analysis": analysis,
                    "source": f"qiskit_aer_{args.noise_model}",
                    "noise_model": args.noise_model,
                    "error_rate": args.error_rate,
                }
                print(f"    Best SQD energy: {analysis['best_energy']:.6f} Ha")
                print(f"    Error vs FCI: {analysis['error_vs_fci_mha']:.3f} mHa")
            except Exception as e:
                print(f"    FAILED: {e}")
                mol_result["controls"]["noisy_simulator"] = {"error": str(e)}

        # --- Control 4: Random (negative control) ---
        print(f"\n  [4/4] Random uniform (negative control)...")
        counts = generate_random_counts(n_qubits, args.shots, args.seed)
        analysis = analyze_counts(record, counts, fci_energy, hf_energy,
                                 args.subspace_sizes, n_electrons)
        mol_result["controls"]["random"] = {
            "counts": counts,
            "analysis": analysis,
            "source": "uniform_random",
        }
        print(f"    Best SQD energy: {analysis['best_energy']:.6f} Ha")
        print(f"    Error vs FCI: {analysis['error_vs_fci_mha']:.3f} mHa")

        # --- Control 5: Hardware counts (if provided) ---
        if args.hardware_counts and args.hardware_counts.exists():
            print(f"\n  [5/5] Hardware counts from {args.hardware_counts}...")
            try:
                counts = load_hardware_counts(args.hardware_counts)
                analysis = analyze_counts(record, counts, fci_energy, hf_energy,
                                         args.subspace_sizes, n_electrons)
                mol_result["controls"]["hardware"] = {
                    "counts": counts,
                    "analysis": analysis,
                    "source": str(args.hardware_counts),
                }
                print(f"    Best SQD energy: {analysis['best_energy']:.6f} Ha")
                print(f"    Error vs FCI: {analysis['error_vs_fci_mha']:.3f} mHa")
            except Exception as e:
                print(f"    FAILED: {e}")
                mol_result["controls"]["hardware"] = {"error": str(e)}

        # Summary
        print(f"\n  Summary for {mol_name}:")
        print(f"    {'Control':<25s} {'Best E (Ha)':>14s} {'Err (mHa)':>10s} {'Mono':>5s}")
        print(f"    {'-'*55}")
        for ctrl_name, ctrl_data in mol_result["controls"].items():
            if "analysis" in ctrl_data:
                a = ctrl_data["analysis"]
                e = a.get("best_energy")
                err = a.get("error_vs_fci_mha")
                mono = "OK" if a.get("monotonicity_ok") else "FAIL"
                e_str = f"{e:.6f}" if e is not None else "N/A"
                err_str = f"{err:.3f}" if err is not None else "N/A"
                print(f"    {ctrl_name:<25s} {e_str:>14s} {err_str:>10s} {mono:>5s}")
            else:
                print(f"    {ctrl_name:<25s} {'ERROR':>14s}")

        # --- Convergence study ---
        if args.convergence and operators:
            print(f"\n  [Convergence] Multi-seed study ({args.n_seeds} seeds, {args.shots} shots/seed)...")
            try:
                conv_data = run_convergence_study(
                    record, operators, thetas, fci_energy, hf_energy,
                    n_seeds=args.n_seeds,
                    shots=args.shots,
                    subspace_sizes=args.subspace_sizes,
                    n_electrons=n_electrons,
                    use_noisy=args.noisy,
                    noise_model=args.noise_model,
                    error_rate=args.error_rate,
                    skip_ideal=args.skip_ideal,
                )
                mol_result["convergence"] = conv_data

                # Print convergence summary
                print(f"    R values: {conv_data['subspace_sizes']}")
                print(f"    Mean energies: {[f'{e:.6f}' for e in conv_data['mean_energies']]}")
                print(f"    Std energies:  {[f'{e:.6f}' for e in conv_data['std_energies']]}")

                # Generate convergence plot
                plot_path = args.out / f"sqd_convergence_{mol_name}.png"
                plot_convergence(conv_data, mol_name, plot_path)
            except Exception as e:
                print(f"    Convergence study FAILED: {e}")
                import traceback
                traceback.print_exc()
                mol_result["convergence"] = {"error": str(e)}

        all_results.append(mol_result)

        # Save per-molecule result
        mol_out = args.out / f"sqd_{mol_name}.json"
        with mol_out.open("w") as f:
            json.dump(mol_result, f, indent=2)
        print(f"\n  Saved: {mol_out}")

    # Save consolidated result
    consolidated = {
        "experiment": "sqd_pilot_control_suite",
        "description": "Local SQD control suite: ideal, noiseless, noisy, random, hardware",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shots": args.shots,
        "molecules": [r["molecule"] for r in all_results],
        "results": all_results,
    }
    consolidated_out = args.out / "sqd_pilot_consolidated.json"
    with consolidated_out.open("w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\nConsolidated results: {consolidated_out}")


if __name__ == "__main__":
    main()
