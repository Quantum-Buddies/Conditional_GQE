"""Quantum-Selected Configuration Interaction (QSCI) post-processing.

Another concrete idea from the eQoSystem proposal that is solid: after the
H-cGQE/L-BFGS-B pipeline produces a candidate circuit, sample bitstrings from
it, build a truncated Hamiltonian in the subspace spanned by those bitstrings,
and diagonalize classically. This often gives a lower (better) energy than the
raw expectation value of the optimized circuit.

The implementation avoids full statevector expansion and works directly with the
Pauli terms stored in the Hamiltonian record.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse

from src.gqe.common.hamiltonian_utils import iter_terms


def _pauli_matrix_element(
    ops: List[str], coeff: complex, b_int: int, b_flipped: int, n_qubits: int
) -> complex:
    """Compute <b_int| (coeff * Pauli(ops)) |b_flipped>.

    This assumes b_flipped differs from b_int exactly on the qubits where ops
    contains X or Y. Returns 0 if the Pauli string does not connect the two
    bitstrings.
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


def qsci_energy_from_bitstrings(
    record: Dict[str, Any],
    bitstrings: Iterable[str],
) -> float:
    """Return the lowest QSCI energy in the subspace spanned by ``bitstrings``.

    Args:
        record: Hamiltonian record with ``n_qubits`` and ``terms``.
        bitstrings: Computational-basis bitstrings (e.g., ``"0101"``). The
            rightmost character is qubit 0 (LSB), consistent with
            ``int(bitstring, 2)``.

    Returns:
        Ground-state energy of the Hamiltonian projected onto the bitstring
        subspace. If the subspace is empty or has one element, falls back to
        the expectation value of that single determinant.
    """
    bitstrings = list(bitstrings)
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
            elem = _pauli_matrix_element(ops, coeff, b_int, b_flipped, n_qubits)
            if abs(elem) < 1e-14:
                continue
            rows.append(idx)
            cols.append(jdx)
            vals.append(elem)

    if not vals:
        # Hamiltonian is diagonal in this subspace; fall back to diagonal terms.
        h_diag = np.zeros(dim, dtype=np.complex128)
        for ops, coeff in iter_terms(record):
            if all(op in ("I", "Z") for op in ops):
                for idx, b_int in enumerate(basis):
                    elem = _pauli_matrix_element(ops, coeff, b_int, b_int, n_qubits)
                    h_diag[idx] += elem
        return float(np.min(h_diag.real))

    h_sub = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(dim, dim), dtype=np.complex128
    ).tocsc()
    # Ensure Hermitian.
    h_sub = (h_sub + h_sub.getH()) / 2

    if dim <= 2 or h_sub.nnz == 0:
        eigvals = np.linalg.eigvalsh(h_sub.toarray())
    else:
        eigvals = scipy.sparse.linalg.eigsh(h_sub, k=1, which="SA")[0]
    return float(np.real(eigvals[0]))


def collect_bitstrings_from_circuit(
    circuit: Any,
    n_qubits: int,
    n_shots: int = 1024,
    seed: int | None = None,
) -> List[str]:
    """Sample computational-basis bitstrings from a CUDA-Q kernel.

    Args:
        circuit: A CUDA-Q kernel function that takes no arguments and prepares
            the optimized state (or a callable that returns one).
        n_qubits: Number of qubits in the circuit.
        n_shots: Number of shots to sample.
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of bitstrings, e.g., ``["0101", "0011", ...]``.
    """
    import cudaq

    if seed is not None:
        cudaq.set_random_seed(seed)
    counts = cudaq.sample(circuit, shots_count=n_shots)
    # counts is a dict-like {bitstring: count}
    return list(counts.keys())
