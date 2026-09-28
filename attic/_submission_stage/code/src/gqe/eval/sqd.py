"""Hardware-agnostic Sample-based Quantum Diagonalization (SQD) core.

This module implements the SQD pipeline independent of any quantum backend:
1. Accept measurement counts (bitstring -> count) from any source
2. Optionally filter bitstrings by symmetry (particle number, spin parity)
3. Build the Hamiltonian matrix in the subspace spanned by selected bitstrings
4. Diagonalize classically to get the refined ground-state energy

Key design decisions:
- Bit order: qubit 0 = LSB = rightmost character in bitstring (consistent with int(bs, 2))
- Pauli phases: Z gives (-1)^{bit_i}, Y gives ±i based on bit_i
- Symmetry filtering: particle number (Hamming weight) and spin parity
- Variational guarantee: SQD energy >= FCI energy (subspace diagonalization is variational)
- Nested subspace monotonicity: larger subspace -> lower or equal energy

This module does NOT depend on CUDA-Q, Qiskit, or any quantum simulator.
It operates purely on classical data (counts, Hamiltonian terms).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse

from src.gqe.common.hamiltonian_utils import iter_terms


# ---------------------------------------------------------------------------
# Pauli matrix element computation
# ---------------------------------------------------------------------------

def pauli_matrix_element(
    ops: List[str],
    coeff: complex,
    b_int: int,
    b_flipped: int,
    n_qubits: int,
) -> complex:
    """Compute <b_int| (coeff * Pauli(ops)) |b_flipped>.

    The Pauli string acts on qubits 0..n_qubits-1 where qubit i corresponds
    to bit i of the integer (qubit 0 = LSB = rightmost char in bitstring).

    Args:
        ops: List of single-qubit Pauli operators, e.g. ["Y", "X", "X", "Y"].
        coeff: Complex coefficient of this Pauli term.
        b_int: Integer representation of the bra bitstring.
        b_flipped: Integer representation of the ket bitstring.
        n_qubits: Number of qubits (for validation).

    Returns:
        Complex matrix element. Returns 0 if the Pauli string does not
        connect b_int and b_flipped (i.e., b_flipped != b_int ^ xy_mask).
    """
    xy_mask = 0
    for i, op in enumerate(ops):
        if op in ("X", "Y"):
            xy_mask |= 1 << i

    if (b_int ^ b_flipped) != xy_mask:
        return 0.0 + 0.0j

    phase = 1.0 + 0.0j
    for i, op in enumerate(ops):
        bit = (b_int >> i) & 1
        if op == "Z" and bit:
            phase *= -1
        elif op == "Y":
            phase *= -1j if bit else 1j

    return coeff * phase


# ---------------------------------------------------------------------------
# Subspace Hamiltonian construction
# ---------------------------------------------------------------------------

def build_subspace_hamiltonian(
    record: Dict[str, Any],
    bitstrings: Sequence[str],
) -> scipy.sparse.csc_matrix:
    """Build the Hamiltonian matrix projected onto the subspace spanned by bitstrings.

    Args:
        record: Hamiltonian record with ``n_qubits`` and ``terms``.
        bitstrings: Computational-basis bitstrings (qubit 0 = LSB = rightmost char).

    Returns:
        Sparse CSC Hamiltonian matrix of shape (dim, dim).
    """
    if not bitstrings:
        raise ValueError("bitstrings must be non-empty")

    n_qubits = int(record["n_qubits"])
    basis = [int(b, 2) for b in bitstrings]
    bitset = {b: i for i, b in enumerate(basis)}
    dim = len(basis)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[complex] = []

    for ops, coeff in iter_terms(record):
        if abs(coeff) < 1e-14:
            continue

        xy_mask = 0
        for i, op in enumerate(ops):
            if op in ("X", "Y"):
                xy_mask |= 1 << i

        for idx, b_int in enumerate(basis):
            b_flipped = b_int ^ xy_mask
            jdx = bitset.get(b_flipped)
            if jdx is None:
                continue
            elem = pauli_matrix_element(ops, coeff, b_int, b_flipped, n_qubits)
            if abs(elem) < 1e-14:
                continue
            rows.append(idx)
            cols.append(jdx)
            vals.append(elem)

    if not vals:
        # Hamiltonian is diagonal in this subspace
        h_diag = np.zeros(dim, dtype=np.complex128)
        for ops, coeff in iter_terms(record):
            if all(op in ("I", "Z") for op in ops):
                for idx, b_int in enumerate(basis):
                    elem = pauli_matrix_element(ops, coeff, b_int, b_int, n_qubits)
                    h_diag[idx] += elem
        return scipy.sparse.csc_matrix(
            np.diag(h_diag), dtype=np.complex128
        )

    h_sub = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(dim, dim), dtype=np.complex128
    ).tocsc()
    # Ensure Hermitian
    h_sub = (h_sub + h_sub.getH()) / 2
    return h_sub


def sqd_energy_from_bitstrings(
    record: Dict[str, Any],
    bitstrings: Iterable[str],
) -> float:
    """Compute the SQD ground-state energy in the subspace spanned by bitstrings.

    This is the variational energy: the lowest eigenvalue of the Hamiltonian
    projected onto the subspace. It is guaranteed to be >= FCI energy.

    Args:
        record: Hamiltonian record with ``n_qubits`` and ``terms``.
        bitstrings: Computational-basis bitstrings (qubit 0 = LSB).

    Returns:
        Ground-state energy (real float) of the projected Hamiltonian.
    """
    bitstrings = list(bitstrings)
    if not bitstrings:
        raise ValueError("bitstrings must be non-empty")

    h_sub = build_subspace_hamiltonian(record, bitstrings)
    dim = h_sub.shape[0]

    if dim <= 2 or h_sub.nnz == 0:
        eigvals = np.linalg.eigvalsh(h_sub.toarray())
    else:
        eigvals = scipy.sparse.linalg.eigsh(h_sub, k=1, which="SA")[0]
    return float(np.real(eigvals[0]))


# ---------------------------------------------------------------------------
# Symmetry filtering
# ---------------------------------------------------------------------------

def filter_by_particle_number(
    bitstrings: Iterable[str],
    n_electrons: int,
    tol: int = 0,
) -> List[str]:
    """Filter bitstrings by particle number (Hamming weight).

    Args:
        bitstrings: Computational-basis bitstrings.
        n_electrons: Target number of electrons (particles).
        tol: Tolerance in electron count (0 = exact, 1 = ±1, etc.).

    Returns:
        Filtered list of bitstrings preserving order.
    """
    result = []
    for bs in bitstrings:
        n = bin(int(bs, 2)).count("1")
        if abs(n - n_electrons) <= tol:
            result.append(bs)
    return result


def filter_by_spin_parity(
    bitstrings: Iterable[str],
    n_qubits: int,
    target_parity: int = 0,
) -> List[str]:
    """Filter bitstrings by spin parity (alpha-beta electron count parity).

    Convention: even-indexed qubits are alpha spin-orbitals, odd-indexed are beta.
    Spin parity = (n_alpha - n_beta) % 2.
    For a singlet state (S=0), target_parity = 0.

    Args:
        bitstrings: Computational-basis bitstrings.
        n_qubits: Number of qubits.
        target_parity: Target (n_alpha - n_beta) % 2.

    Returns:
        Filtered list of bitstrings preserving order.
    """
    result = []
    for bs in bitstrings:
        val = int(bs, 2)
        n_alpha = sum((val >> i) & 1 for i in range(0, n_qubits, 2))
        n_beta = sum((val >> i) & 1 for i in range(1, n_qubits, 2))
        if (n_alpha - n_beta) % 2 == target_parity:
            result.append(bs)
    return result


def apply_symmetry_filters(
    bitstrings: Iterable[str],
    n_qubits: int,
    n_electrons: int,
    particle_number_tol: int = 0,
    spin_parity: Optional[int] = 0,
) -> List[str]:
    """Apply both particle number and spin parity filters.

    Args:
        bitstrings: Input bitstrings.
        n_qubits: Number of qubits.
        n_electrons: Target electron count.
        particle_number_tol: Tolerance for particle number filter.
        spin_parity: Target spin parity (None to skip).

    Returns:
        Filtered bitstrings.
    """
    filtered = filter_by_particle_number(bitstrings, n_electrons, tol=particle_number_tol)
    if spin_parity is not None:
        filtered = filter_by_spin_parity(filtered, n_qubits, target_parity=spin_parity)
    return filtered


# ---------------------------------------------------------------------------
# Counts-based SQD
# ---------------------------------------------------------------------------

def select_subspace_by_counts(
    counts: Dict[str, int],
    subspace_size: Optional[int] = None,
) -> List[str]:
    """Select bitstrings from counts, sorted by frequency (most frequent first).

    Args:
        counts: Dictionary mapping bitstring -> count.
        subspace_size: Maximum number of bitstrings to select. If None, selects all.

    Returns:
        List of selected bitstrings, most frequent first.
    """
    sorted_bs = sorted(counts.keys(), key=lambda bs: -counts[bs])
    if subspace_size is not None:
        sorted_bs = sorted_bs[:subspace_size]
    return sorted_bs


def sqd_energy_from_counts(
    record: Dict[str, Any],
    counts: Dict[str, int],
    subspace_size: Optional[int] = None,
    n_electrons: Optional[int] = None,
    particle_number_tol: int = 0,
    spin_parity: Optional[int] = None,
) -> float:
    """Compute SQD energy from measurement counts.

    Pipeline:
    1. Select bitstrings by frequency (most frequent first)
    2. Optionally filter by symmetry (particle number, spin parity)
    3. Build subspace Hamiltonian and diagonalize

    Args:
        record: Hamiltonian record.
        counts: Measurement counts (bitstring -> count).
        subspace_size: Maximum subspace size. If None, uses all unique bitstrings.
        n_electrons: If provided, filter by particle number.
        particle_number_tol: Tolerance for particle number filter.
        spin_parity: If not None, filter by spin parity.

    Returns:
        SQD ground-state energy (variational bound guaranteed).
    """
    n_qubits = int(record["n_qubits"])

    # Select bitstrings by count
    selected = select_subspace_by_counts(counts, subspace_size)

    # Apply symmetry filters
    if n_electrons is not None or spin_parity is not None:
        ne = n_electrons if n_electrons is not None else 0
        selected = apply_symmetry_filters(
            selected, n_qubits, ne,
            particle_number_tol=particle_number_tol,
            spin_parity=spin_parity,
        )

    if not selected:
        raise ValueError("No bitstrings remain after symmetry filtering")

    return sqd_energy_from_bitstrings(record, selected)


# ---------------------------------------------------------------------------
# Nested subspace analysis
# ---------------------------------------------------------------------------

def nested_subspace_energies(
    record: Dict[str, Any],
    bitstrings: Sequence[str],
) -> List[float]:
    """Compute SQD energies for nested subspaces (monotonicity check).

    Builds subspaces S_1 ⊂ S_2 ⊂ ... ⊂ S_n where S_k = {bs_0, ..., bs_{k-1}}
    and returns the SQD energy for each. Energies should be monotonically
    non-increasing.

    Args:
        record: Hamiltonian record.
        bitstrings: Ordered bitstrings (most important first).

    Returns:
        List of SQD energies, one per nested subspace size.
    """
    energies = []
    accumulated: List[str] = []
    for bs in bitstrings:
        accumulated.append(bs)
        e = sqd_energy_from_bitstrings(record, accumulated)
        energies.append(e)
    return energies


def exact_diagonalize(record: Dict[str, Any]) -> float:
    """Compute the exact ground-state energy (FCI) without requiring qiskit.

    Builds the full 2^n_qubits Hamiltonian matrix directly from Pauli terms
    using numpy. Limited to <= 14 qubits for memory safety.

    Args:
        record: Hamiltonian record with ``n_qubits`` and ``terms``.

    Returns:
        Ground-state energy (real float).
    """
    n_qubits = int(record["n_qubits"])
    if n_qubits > 14:
        raise ValueError(f"Exact diagonalization limited to <= 14 qubits, got {n_qubits}")

    dim = 2 ** n_qubits
    h_full = np.zeros((dim, dim), dtype=np.complex128)

    for ops, coeff in iter_terms(record):
        if abs(coeff) < 1e-14:
            continue
        xy_mask = 0
        for i, op in enumerate(ops):
            if op in ("X", "Y"):
                xy_mask |= 1 << i

        for b_int in range(dim):
            b_flipped = b_int ^ xy_mask
            elem = pauli_matrix_element(ops, coeff, b_int, b_flipped, n_qubits)
            if abs(elem) > 1e-14:
                h_full[b_int, b_flipped] += elem

    # Ensure Hermitian
    h_full = (h_full + h_full.conj().T) / 2
    eigvals = np.linalg.eigvalsh(h_full)
    return float(eigvals[0])


def check_monotonicity(energies: Sequence[float], tol: float = 1e-10) -> bool:
    """Check that energies are monotonically non-increasing.

    Args:
        energies: List of energies from nested subspaces.
        tol: Numerical tolerance.

    Returns:
        True if monotonic, False otherwise.
    """
    for i in range(1, len(energies)):
        if energies[i] > energies[i - 1] + tol:
            return False
    return True


# ---------------------------------------------------------------------------
# Occupancy-guided configuration recovery
# ---------------------------------------------------------------------------

def occupancy_guided_recovery(
    counts: Dict[str, int],
    n_qubits: int,
    n_electrons: int,
    n_recovered: int = 50,
    n_top: int = 100,
) -> List[str]:
    """Generate additional bitstrings using occupancy-guided recovery.

    Computes per-orbital occupancy probabilities from the sampled counts,
    then generates new bitstrings by promoting electrons from over-occupied
    to under-occupied orbitals. This is inspired by the SQD configuration
    recovery protocol (Robledo-Moreno et al., Nature 2024).

    The procedure:
    1. Compute occupancy probability p_i for each qubit i from the top-N bitstrings
    2. Identify occupied (p_i > 0.5) and virtual (p_i <= 0.5) orbitals
    3. Generate new determinants by single/double excitations from occupied -> virtual
    4. Filter by correct particle number

    Args:
        counts: Measurement counts (bitstring -> count).
        n_qubits: Number of qubits.
        n_electrons: Target electron count.
        n_recovered: Number of new bitstrings to generate.
        n_top: Number of top bitstrings to use for occupancy statistics.

    Returns:
        List of recovered bitstrings (not present in the original counts).
    """
    # Select top bitstrings by frequency
    sorted_bs = sorted(counts.keys(), key=lambda bs: -counts[bs])[:n_top]
    if not sorted_bs:
        return []

    # Compute per-orbital occupancy
    occupancy = np.zeros(n_qubits)
    total_weight = 0
    for bs in sorted_bs:
        val = int(bs, 2)
        w = counts[bs]
        total_weight += w
        for i in range(n_qubits):
            if (val >> i) & 1:
                occupancy[i] += w
    if total_weight > 0:
        occupancy /= total_weight

    # Classify orbitals
    occupied = [i for i in range(n_qubits) if occupancy[i] > 0.5]
    virtual = [i for i in range(n_qubits) if occupancy[i] <= 0.5]

    # Ensure we have the right number of occupied orbitals
    # (occupancy may not perfectly separate due to noise)
    if len(occupied) < n_electrons:
        # Promote highest-occupancy virtuals
        virtual.sort(key=lambda i: -occupancy[i])
        occupied.extend(virtual[:n_electrons - len(occupied)])
        occupied.sort()
        virtual = [i for i in range(n_qubits) if i not in occupied]
    elif len(occupied) > n_electrons:
        # Demote lowest-occupancy occupied
        occupied.sort(key=lambda i: occupancy[i])
        to_demote = occupied[:len(occupied) - n_electrons]
        occupied = occupied[len(occupied) - n_electrons:]
        virtual = sorted(virtual + to_demote)
        occupied.sort()

    # Build HF-like reference determinant
    ref_int = 0
    for i in occupied:
        ref_int |= 1 << i

    # Generate excitations
    existing = set(int(bs, 2) for bs in counts.keys())
    recovered: List[str] = []
    rng = np.random.default_rng(42)

    # Single excitations: i -> a
    singles = []
    for i in occupied:
        for a in virtual:
            new_int = ref_int & ~(1 << i) | (1 << a)
            if new_int not in existing:
                singles.append(new_int)

    # Double excitations: i,j -> a,b
    doubles = []
    for idx, i in enumerate(occupied):
        for j in occupied[idx + 1:]:
            for idx2, a in enumerate(virtual):
                for b in virtual[idx2 + 1:]:
                    new_int = ref_int & ~(1 << i) & ~(1 << j) | (1 << a) | (1 << b)
                    if new_int not in existing:
                        doubles.append(new_int)

    # Shuffle and select
    pool = singles + doubles
    rng.shuffle(pool)

    for val in pool[:n_recovered]:
        bs = format(val, f"0{n_qubits}b")
        recovered.append(bs)

    return recovered


def sqd_energy_with_recovery(
    record: Dict[str, Any],
    counts: Dict[str, int],
    subspace_size: Optional[int] = None,
    n_electrons: Optional[int] = None,
    particle_number_tol: int = 0,
    spin_parity: Optional[int] = None,
    n_recovered: int = 50,
) -> Dict[str, Any]:
    """Compute SQD energy with occupancy-guided configuration recovery.

    Runs standard SQD on the raw counts, then augments the subspace with
    recovered bitstrings and re-diagonalizes. Reports both energies and
    the improvement from recovery.

    Args:
        record: Hamiltonian record.
        counts: Measurement counts.
        subspace_size: Maximum subspace size for raw counts.
        n_electrons: Target electron count for symmetry filtering.
        particle_number_tol: Tolerance for particle number filter.
        spin_parity: Target spin parity (None to skip).
        n_recovered: Number of bitstrings to recover via occupancy guidance.

    Returns:
        Dict with 'sqd_raw', 'sqd_recovered', 'recovered_bitstrings', and 'improvement_mha'.
    """
    n_qubits = int(record["n_qubits"])
    ne = n_electrons if n_electrons is not None else 0

    # Raw SQD energy
    raw_energy = sqd_energy_from_counts(
        record, counts, subspace_size, n_electrons, particle_number_tol, spin_parity
    )

    # Recover additional bitstrings
    recovered_bs = occupancy_guided_recovery(
        counts, n_qubits, ne, n_recovered=n_recovered
    )

    # Combine raw selected + recovered
    selected = select_subspace_by_counts(counts, subspace_size)
    if n_electrons is not None or spin_parity is not None:
        selected = apply_symmetry_filters(
            selected, n_qubits, ne,
            particle_number_tol=particle_number_tol,
            spin_parity=spin_parity,
        )

    # Filter recovered bitstrings too
    if n_electrons is not None or spin_parity is not None:
        recovered_bs = apply_symmetry_filters(
            recovered_bs, n_qubits, ne,
            particle_number_tol=particle_number_tol,
            spin_parity=spin_parity,
        )

    combined = list(set(selected + recovered_bs))
    recovered_energy = sqd_energy_from_bitstrings(record, combined) if combined else raw_energy

    improvement_mha = (raw_energy - recovered_energy) * 1000.0  # positive = improvement

    return {
        "sqd_raw": raw_energy,
        "sqd_recovered": recovered_energy,
        "n_raw_bitstrings": len(selected),
        "n_recovered_bitstrings": len(recovered_bs),
        "n_combined": len(combined),
        "improvement_mha": improvement_mha,
    }


# ---------------------------------------------------------------------------
# Masterplan API: canonicalize, target_spin, filter_configurations,
# apply_pauli, project_pauli_hamiltonian, solve_subspace, run_sqd
# ---------------------------------------------------------------------------

def canonicalize_counts(
    counts: Dict[str, int],
    n_qubits: int,
) -> Dict[str, int]:
    """Canonicalize raw measurement counts to fixed-width bitstrings.

    Normalizes bitstring keys to:
    - Zero-padded to n_qubits characters
    - Consistent bit order (qubit 0 = LSB = rightmost char)
    - Merged duplicate keys after padding

    Args:
        counts: Raw counts dict from any quantum backend.
        n_qubits: Number of qubits for zero-padding.

    Returns:
        Canonicalized counts dict with fixed-width keys.
    """
    canonical: Dict[str, int] = {}
    for bs, count in counts.items():
        # Strip whitespace and ensure binary string
        bs = bs.strip().replace(" ", "")
        # Handle hex or decimal input (unlikely but safe)
        if not all(c in "01" for c in bs):
            val = int(bs, 2) if bs else 0
        else:
            val = int(bs, 2)
        canonical_bs = format(val, f"0{n_qubits}b")
        canonical[canonical_bs] = canonical.get(canonical_bs, 0) + int(count)
    return canonical


def target_spin_counts(
    counts: Dict[str, int],
    n_qubits: int,
    n_electrons: int,
    spin_squared: int = 0,
) -> Dict[str, int]:
    """Filter counts to target spin sector.

    For a given S² value, keeps bitstrings where the spin quantum number
    matches. For singlet (S=0, spin_squared=0), requires n_alpha == n_beta.

    Args:
        counts: Measurement counts.
        n_qubits: Number of qubits.
        n_electrons: Target electron count.
        spin_squared: Target S(S+1) value (0 for singlet, 2 for triplet, etc.).

    Returns:
        Filtered counts dict.
    """
    s = {0: 0, 2: 1, 6: 2}.get(spin_squared, 0)
    n_alpha_target = (n_electrons + 2 * s) // 2
    n_beta_target = (n_electrons - 2 * s) // 2

    filtered: Dict[str, int] = {}
    for bs, count in counts.items():
        val = int(bs, 2)
        n_alpha = sum((val >> i) & 1 for i in range(0, n_qubits, 2))
        n_beta = sum((val >> i) & 1 for i in range(1, n_qubits, 2))
        if n_alpha == n_alpha_target and n_beta == n_beta_target:
            filtered[bs] = count
    return filtered


def filter_configurations(
    bitstrings: Sequence[str],
    n_qubits: int,
    n_electrons: Optional[int] = None,
    particle_number_tol: int = 0,
    spin_parity: Optional[int] = None,
    spin_squared: Optional[int] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Filter bitstrings by symmetry with invalid-reason accounting.

    Returns both the filtered list and a dict of rejection reasons with counts.

    Args:
        bitstrings: Input bitstrings.
        n_qubits: Number of qubits.
        n_electrons: Target electron count (None to skip).
        particle_number_tol: Tolerance for particle number.
        spin_parity: Target (n_alpha - n_beta) % 2 (None to skip).
        spin_squared: Target S(S+1) (None to skip, overrides spin_parity if set).

    Returns:
        (valid_bitstrings, rejection_counts) where rejection_counts has keys:
        'wrong_particle_number', 'wrong_spin_parity', 'wrong_spin_squared', 'total_valid', 'total_invalid'
    """
    rejection = {
        "wrong_particle_number": 0,
        "wrong_spin_parity": 0,
        "wrong_spin_squared": 0,
        "total_valid": 0,
        "total_invalid": 0,
    }

    valid: List[str] = []
    for bs in bitstrings:
        val = int(bs, 2)
        rejected = False

        # Particle number check
        if n_electrons is not None:
            n = bin(val).count("1")
            if abs(n - n_electrons) > particle_number_tol:
                rejection["wrong_particle_number"] += 1
                rejected = True
                rejection["total_invalid"] += 1
                continue

        # Spin squared check (more specific than parity)
        if spin_squared is not None:
            n_alpha = sum((val >> i) & 1 for i in range(0, n_qubits, 2))
            n_beta = sum((val >> i) & 1 for i in range(1, n_qubits, 2))
            s = (n_alpha - n_beta) / 2
            actual_s2 = s * (s + 1)
            if abs(actual_s2 - spin_squared) > 0.01:
                rejection["wrong_spin_squared"] += 1
                rejected = True
                rejection["total_invalid"] += 1
                continue
        elif spin_parity is not None:
            n_alpha = sum((val >> i) & 1 for i in range(0, n_qubits, 2))
            n_beta = sum((val >> i) & 1 for i in range(1, n_qubits, 2))
            if (n_alpha - n_beta) % 2 != spin_parity:
                rejection["wrong_spin_parity"] += 1
                rejected = True
                rejection["total_invalid"] += 1
                continue

        if not rejected:
            valid.append(bs)
            rejection["total_valid"] += 1

    return valid, rejection


def apply_pauli_to_bitstring(
    ops: List[str],
    b_int: int,
    n_qubits: int,
) -> Tuple[int, complex]:
    """Apply a Pauli string to a computational basis state.

    Computes P|b> where P = ops[0] ⊗ ops[1] ⊗ ... ⊗ ops[n-1].

    Args:
        ops: Per-qubit Pauli operators.
        b_int: Integer representation of the bitstring.
        n_qubits: Number of qubits.

    Returns:
        (result_int, phase) where result_int is the resulting basis state
        and phase is the complex phase factor.
    """
    result = b_int
    phase = 1.0 + 0.0j
    for i, op in enumerate(ops):
        bit = (b_int >> i) & 1
        if op == "X":
            result ^= (1 << i)
        elif op == "Y":
            result ^= (1 << i)
            phase *= -1j if bit else 1j
        elif op == "Z":
            phase *= -1 if bit else 1
        # I: no change
    return result, phase


def project_pauli_hamiltonian(
    record: Dict[str, Any],
    bitstrings: Sequence[str],
) -> np.ndarray:
    """Project the full Hamiltonian onto a subspace and return dense matrix.

    This is the same as build_subspace_hamiltonian but returns a dense numpy
    array instead of a sparse matrix, for use with dense diagonalization.

    Args:
        record: Hamiltonian record.
        bitstrings: Computational basis bitstrings defining the subspace.

    Returns:
        Dense (dim, dim) complex numpy array.
    """
    h_sparse = build_subspace_hamiltonian(record, bitstrings)
    return h_sparse.toarray()


def solve_subspace(
    record: Dict[str, Any],
    bitstrings: Sequence[str],
    return_eigvec: bool = False,
) -> Any:
    """Diagonalize the Hamiltonian in the subspace and return ground state info.

    Args:
        record: Hamiltonian record.
        bitstrings: Subspace-defining bitstrings.
        return_eigvec: If True, also return the ground-state eigenvector.

    Returns:
        If return_eigvec is False: (ground_energy, dim)
        If return_eigvec is True: (ground_energy, eigenvector, dim)
    """
    h_dense = project_pauli_hamiltonian(record, bitstrings)
    dim = h_dense.shape[0]

    if dim <= 64:
        eigvals, eigvecs = np.linalg.eigh(h_dense)
    else:
        from scipy.sparse.linalg import eigsh
        h_sparse = scipy.sparse.csc_matrix(h_dense)
        eigvals, eigvecs = eigsh(h_sparse, k=1, which="SA")

    ground_energy = float(np.real(eigvals[0]))
    if return_eigvec:
        return ground_energy, eigvecs[:, 0], dim
    return ground_energy, dim


def run_sqd(
    record: Dict[str, Any],
    counts: Dict[str, int],
    n_electrons: Optional[int] = None,
    subspace_size: Optional[int] = None,
    particle_number_tol: int = 0,
    spin_parity: Optional[int] = 0,
    spin_squared: Optional[int] = None,
    n_recovered: int = 0,
    return_details: bool = True,
) -> Dict[str, Any]:
    """Full SQD pipeline from raw counts to refined energy.

    This is the main entry point for the SQD workflow:
    1. Canonicalize counts
    2. Select subspace by frequency
    3. Filter by symmetry (particle number, spin)
    4. Optionally recover additional configurations
    5. Build and diagonalize subspace Hamiltonian
    6. Report energy with variational bound check

    Args:
        record: Hamiltonian record with n_qubits and terms.
        counts: Raw measurement counts (bitstring -> count).
        n_electrons: Target electron count for symmetry filtering.
        subspace_size: Max subspace size (None = all unique bitstrings).
        particle_number_tol: Tolerance for particle number filter.
        spin_parity: Target spin parity (None to skip).
        spin_squared: Target S(S+1) (None to skip, overrides spin_parity).
        n_recovered: Number of bitstrings to recover via occupancy guidance (0 = skip).
        return_details: If True, include rejection counts, monotonicity, etc.

    Returns:
        Dict with keys:
        - 'energy': SQD ground-state energy
        - 'fci_energy': Exact FCI energy (if n_qubits <= 14)
        - 'n_bitstrings': Number of bitstrings in subspace
        - 'n_unique_raw': Number of unique bitstrings in raw counts
        - 'rejection': Rejection accounting dict (if return_details)
        - 'monotonicity_ok': Whether nested subspace energies are monotonic (if return_details)
        - 'recovered_energy': Energy with recovered bitstrings (if n_recovered > 0)
        - 'variational_bound_satisfied': bool (if FCI available)
    """
    n_qubits = int(record["n_qubits"])

    # Step 1: Canonicalize
    counts = canonicalize_counts(counts, n_qubits)

    # Step 2: Select by frequency
    selected = select_subspace_by_counts(counts, subspace_size)

    # Step 3: Filter by symmetry
    if n_electrons is not None or spin_parity is not None or spin_squared is not None:
        ne = n_electrons if n_electrons is not None else 0
        selected, rejection = filter_configurations(
            selected, n_qubits, n_electrons=ne,
            particle_number_tol=particle_number_tol,
            spin_parity=spin_parity,
            spin_squared=spin_squared,
        )
    else:
        rejection = {"total_valid": len(selected), "total_invalid": 0,
                      "wrong_particle_number": 0, "wrong_spin_parity": 0,
                      "wrong_spin_squared": 0}

    if not selected:
        raise ValueError("No bitstrings remain after symmetry filtering")

    # Step 4: Optional recovery
    recovered_info = {}
    if n_recovered > 0 and n_electrons is not None:
        rec_result = sqd_energy_with_recovery(
            record, counts, subspace_size, n_electrons,
            particle_number_tol, spin_parity, n_recovered,
        )
        recovered_info = {
            "recovered_energy": rec_result["sqd_recovered"],
            "n_recovered_bitstrings": rec_result["n_recovered_bitstrings"],
            "improvement_mha": rec_result["improvement_mha"],
        }

    # Step 5: Diagonalize
    energy, dim = solve_subspace(record, selected)

    # Step 6: Assemble result
    result: Dict[str, Any] = {
        "energy": energy,
        "n_bitstrings": dim,
        "n_unique_raw": len(counts),
    }

    # FCI comparison (if small enough)
    try:
        fci = exact_diagonalize(record)
        result["fci_energy"] = fci
        result["variational_bound_satisfied"] = energy >= fci - 1e-10
        result["error_vs_fci_mha"] = abs(energy - fci) * 1000.0
    except (ValueError, Exception):
        pass

    if return_details:
        result["rejection"] = rejection
        # Monotonicity check (only for small subspaces to avoid expensive computation)
        if dim <= 32:
            energies = nested_subspace_energies(record, selected)
            result["monotonicity_ok"] = check_monotonicity(energies)
            result["nested_energies"] = energies

    if recovered_info:
        result.update(recovered_info)

    return result
