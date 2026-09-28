"""Benchmark acceleration modules: QWC grouping, parity computation, Pauli word caching.

Run on AIRE with cudaq-env:
  python scripts/benchmark_accel.py
"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.gqe.accel.fast_pauli import pauli_to_masks, pauli_list_to_masks, compute_parity_vectorized
from src.gqe.accel.fast_qwc import group_qwc_terms_vectorized
from src.gqe.accel.cudaq_tuning import ensure_applied


def generate_random_pauli_terms(n_terms: int, n_qubits: int, seed: int = 42) -> list[tuple[str, float]]:
    """Generate random Pauli terms for benchmarking."""
    rng = np.random.default_rng(seed)
    paulis = ["I", "X", "Y", "Z"]
    terms = []
    for _ in range(n_terms):
        word = "".join(rng.choice(paulis, size=n_qubits))
        coeff = rng.standard_normal()
        terms.append((word, float(coeff)))
    return terms


def benchmark_qwc_grouping(terms: list[tuple[str, float]], label: str) -> None:
    """Benchmark QWC grouping: original Python loop vs vectorized."""
    n = len(terms)
    print(f"\n{'='*60}")
    print(f"QWC Grouping: {label} ({n} terms)")
    print(f"{'='*60}")

    # Original Python loop
    t0 = time.perf_counter()
    groups_py = _group_qwc_terms_python(terms)
    t_py = time.perf_counter() - t0
    print(f"  Python loop:  {t_py*1000:.1f} ms ({len(groups_py)} groups)")

    # Vectorized CPU
    t0 = time.perf_counter()
    groups_vec = group_qwc_terms_vectorized(terms, use_gpu=False)
    t_vec = time.perf_counter() - t0
    print(f"  NumPy vector: {t_vec*1000:.1f} ms ({len(groups_vec)} groups)")

    speedup = t_py / t_vec if t_vec > 0 else float("inf")
    print(f"  CPU speedup:   {speedup:.1f}x")

    # Verify correctness: all terms in each group must be QWC
    from src.gqe.accel.fast_pauli import pauli_to_masks, are_qwc
    for grp in groups_vec:
        for i in grp:
            for j in grp:
                if i != j:
                    x1, z1 = pauli_to_masks(terms[i][0])
                    x2, z2 = pauli_to_masks(terms[j][0])
                    assert are_qwc(x1, z1, x2, z2), \
                        f"Invalid QWC group: {terms[i][0]} and {terms[j][0]}"
    # Verify all terms are assigned
    all_assigned = sorted(idx for grp in groups_vec for idx in grp)
    assert all_assigned == list(range(n)), f"Missing terms in vectorized grouping"
    print(f"  Correctness:   PASSED ({len(groups_vec)} groups, all QWC valid)")


def benchmark_parity(n_qubits: int, n_terms: int, n_shots: int) -> None:
    """Benchmark parity computation: Python loop vs NumPy vs C++."""
    print(f"\n{'='*60}")
    print(f"Parity Computation: {n_qubits}q, {n_terms} terms, {n_shots} shots")
    print(f"{'='*60}")

    # Generate fake counts
    rng = np.random.default_rng(42)
    paulis = ["I", "X", "Y", "Z"]
    terms = ["".join(rng.choice(paulis, size=n_qubits)) for _ in range(n_terms)]
    bitstrings = ["".join(rng.choice(["0", "1"], size=n_qubits)) for _ in range(n_shots)]
    counts = {bs: int(rng.integers(1, 100)) for bs in bitstrings}
    n_shots_actual = sum(counts.values())

    # Python loop
    t0 = time.perf_counter()
    for word in terms:
        exp = 0.0
        for bs, cnt in counts.items():
            parity = sum(int(bs[q]) for q, op in enumerate(word) if op != "I") % 2
            sign = -1 if parity == 1 else 1
            exp += sign * cnt / n_shots_actual
    t_py = time.perf_counter() - t0
    print(f"  Python loop:  {t_py*1000:.1f} ms")

    # NumPy vectorized
    from src.gqe.accel.fast_pauli import compute_grouped_expectations_vectorized
    term_infos = [{"term": w, "coeff": 1.0} for w in terms]
    t0 = time.perf_counter()
    energy, _ = compute_grouped_expectations_vectorized(counts, term_infos, n_qubits, n_shots_actual)
    t_np = time.perf_counter() - t0
    print(f"  NumPy vector: {t_np*1000:.1f} ms")
    print(f"  NumPy speedup: {t_py/t_np:.1f}x")

    # PyTorch GPU (if available)
    try:
        import torch
        if torch.cuda.is_available():
            from src.gqe.accel.gpu_parity import _parse_counts_torch
            t0 = time.perf_counter()
            _parse_counts_torch(counts, term_infos, n_qubits, n_shots_actual)
            t_gpu = time.perf_counter() - t0
            print(f"  PyTorch GPU:  {t_gpu*1000:.1f} ms")
            print(f"  GPU speedup:  {t_py/t_gpu:.1f}x")
    except Exception as e:
        print(f"  PyTorch GPU: unavailable ({e})")


def _group_qwc_terms_python(terms: list[tuple[str, float]]) -> list[list[int]]:
    """Original Python loop QWC grouping for benchmarking."""
    groups: list[list[int]] = []
    group_bases: list[str] = []
    for idx, (word, _) in enumerate(terms):
        placed = False
        for gi, base in enumerate(group_bases):
            compatible = True
            for q in range(len(word)):
                a, b = word[q], base[q]
                if a != "I" and b != "I" and a != b:
                    compatible = False
                    break
            if compatible:
                groups[gi].append(idx)
                new_base = list(base)
                for q in range(len(word)):
                    if word[q] != "I" and new_base[q] == "I":
                        new_base[q] = word[q]
                group_bases[gi] = "".join(new_base)
                placed = True
                break
        if not placed:
            groups.append([idx])
            group_bases.append(word)
    return groups


def main():
    print("GQE Pipeline Acceleration Benchmark")
    print("=" * 60)

    # Apply CUDA-Q env tuning
    env = ensure_applied()
    print(f"CUDA-Q env tuning applied: {env}")

    # Benchmark QWC grouping at different scales
    for n_terms, n_qubits, label in [
        (15, 4, "H2 (4q)"),
        (631, 12, "LiH (12q)"),
        (2951, 20, "N2 (20q)"),
    ]:
        terms = generate_random_pauli_terms(n_terms, n_qubits)
        benchmark_qwc_grouping(terms, label)

    # Benchmark parity computation
    for n_qubits, n_terms, n_shots in [
        (4, 15, 4096),
        (12, 631, 4096),
        (20, 2951, 4096),
    ]:
        benchmark_parity(n_qubits, n_terms, n_shots)

    print("\n" + "=" * 60)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
