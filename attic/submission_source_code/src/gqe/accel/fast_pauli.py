"""Vectorized Pauli word operations using NumPy bit manipulation.

Encodes Pauli words as integer bitmasks for O(1) operations:
  - Each Pauli word → 2 integers (x_mask, z_mask) where:
    I = (0,0), X = (1,0), Y = (1,1), Z = (0,1)
  - QWC compatibility: two terms are QWC iff (x1 & z2) | (z1 & x2) == 0
    (no position where one has X and other has Z, or vice versa)
  - Parity computation: popcount(bitstring & mask) % 2

This replaces character-by-character Python loops with vectorized NumPy ops.
"""
from __future__ import annotations

import numpy as np


# Pauli encoding: (x_bit, z_bit)
_PAULI_ENCODING = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}


def pauli_to_masks(word: str) -> tuple[int, int]:
    """Convert a Pauli word to (x_mask, z_mask) integers.

    Each character maps to 2 bits. The qubit position q maps to bit q.
    """
    x_mask = 0
    z_mask = 0
    for q, ch in enumerate(word):
        xb, zb = _PAULI_ENCODING.get(ch, (0, 0))
        x_mask |= xb << q
        z_mask |= zb << q
    return x_mask, z_mask


def pauli_list_to_masks(words: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized: convert list of Pauli words to (x_masks, z_masks) arrays.

    Returns:
        x_masks: np.ndarray shape (n_terms,) dtype int64
        z_masks: np.ndarray shape (n_terms,) dtype int64
    """
    n = len(words)
    x_masks = np.zeros(n, dtype=np.int64)
    z_masks = np.zeros(n, dtype=np.int64)
    for i, word in enumerate(words):
        x, z = pauli_to_masks(word)
        x_masks[i] = x
        z_masks[i] = z
    return x_masks, z_masks


def pad_pauli_word(word: str, n_qubits: int) -> str:
    """Pad or truncate a Pauli word to n_qubits (fast, no copy if already correct)."""
    if len(word) == n_qubits:
        return word
    if len(word) < n_qubits:
        return word + "I" * (n_qubits - len(word))
    return word[:n_qubits]


def pad_pauli_batch(words: list[str], n_qubits: int) -> list[str]:
    """Vectorized padding for a batch of Pauli words."""
    return [pad_pauli_word(w, n_qubits) for w in words]


def are_qwc(x1: int, z1: int, x2: int, z2: int) -> bool:
    """Check if two Pauli words (as masks) are qubit-wise commuting.

    Two terms are QWC iff no position has conflicting Pauli operators.
    Conflict: one has X (x=1,z=0) and other has Z (x=0,z=1) at same position.
    This is: (x1 & z2) | (z1 & x2) == 0
    """
    return ((x1 & z2) | (z1 & x2)) == 0


def qwc_compatibility_matrix(x_masks: np.ndarray, z_masks: np.ndarray) -> np.ndarray:
    """Compute QWC compatibility matrix for all pairs.

    Args:
        x_masks: shape (n,) int64
        z_masks: shape (n,) int64

    Returns:
        bool matrix shape (n, n) where True = QWC compatible
    """
    n = len(x_masks)
    # Broadcast: (n, 1) & (1, n) -> (n, n)
    x1 = x_masks[:, None]
    z1 = z_masks[:, None]
    x2 = x_masks[None, :]
    z2 = z_masks[None, :]

    # QWC iff (x1 & z2) | (z1 & x2) == 0 at all positions
    conflict = (x1 & z2) | (z1 & x2)
    return conflict == 0


def compute_parity_vectorized(
    bitstrings: np.ndarray,
    mask: int,
) -> np.ndarray:
    """Compute parity of bitstrings under a Pauli mask.

    Args:
        bitstrings: shape (n_shots, n_qubits) uint8, each row is a bitstring
        mask: integer bitmask where bit q = 1 if Pauli is non-I at position q

    Returns:
        shape (n_shots,) int8: 0 for even parity, 1 for odd
    """
    # Convert mask to per-qubit boolean array
    n_qubits = bitstrings.shape[1]
    mask_bits = np.array([(mask >> q) & 1 for q in range(n_qubits)], dtype=np.uint8)

    # Mask the bitstrings and compute parity via XOR reduction
    masked = bitstrings & mask_bits[None, :]
    parity = np.bitwise_xor.reduce(masked, axis=1).astype(np.int8)
    return parity


def compute_expectation_from_counts(
    counts: dict[str, int],
    x_mask: int,
    z_mask: int,
    n_qubits: int,
    n_shots: int,
) -> float:
    """Compute expectation value for a single Pauli term from counts.

    Uses vectorized NumPy operations instead of Python loops.

    Args:
        counts: {bitstring: count} dict
        x_mask: X bitmask for the Pauli term
        z_mask: Z bitmask for the Pauli term
        n_qubits: number of qubits
        n_shots: total shots
    """
    non_identity_mask = x_mask | z_mask

    if non_identity_mask == 0:
        return 1.0  # Identity term

    # Convert counts to arrays
    bitstrings_list = list(counts.keys())
    counts_arr = np.array(list(counts.values()), dtype=np.float64)

    # Convert bitstrings to bit array
    bs_array = np.zeros((len(bitstrings_list), n_qubits), dtype=np.uint8)
    for i, bs in enumerate(bitstrings_list):
        for q in range(n_qubits):
            bs_array[i, q] = int(bs[q])

    # Compute parity under the non-identity mask
    parity = compute_parity_vectorized(bs_array, non_identity_mask)

    # Expectation = sum((-1)^parity * count) / n_shots
    signs = 1.0 - 2.0 * parity.astype(np.float64)
    exp = np.dot(signs, counts_arr) / n_shots
    return float(exp)


def compute_grouped_expectations_vectorized(
    counts: dict[str, int],
    group_terms: list[dict[str, Any]],
    n_qubits: int,
    n_shots: int,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Compute all term expectations for a QWC group from shared counts.

    Vectorized over terms and bitstrings — replaces nested Python loops.

    Args:
        counts: {bitstring: count} dict from the measurement circuit
        group_terms: list of {term_idx, term, coeff} dicts for this group
        n_qubits: number of qubits
        n_shots: total shots

    Returns:
        (energy_contribution, {term: {coeff, expectation}})
    """
    import numpy as np
    from typing import Any

    n_bs = len(counts)
    n_terms = len(group_terms)

    if n_bs == 0 or n_terms == 0:
        return 0.0, {}

    # Convert bitstrings to bit array once
    bitstrings_list = list(counts.keys())
    counts_arr = np.array(list(counts.values()), dtype=np.float64)
    bs_array = np.zeros((n_bs, n_qubits), dtype=np.uint8)
    for i, bs in enumerate(bitstrings_list):
        for q in range(n_qubits):
            bs_array[i, q] = int(bs[q])

    # Precompute masks for all terms
    masks = np.zeros(n_terms, dtype=np.int64)
    coeffs = np.zeros(n_terms, dtype=np.float64)
    terms = []
    for ti, term_info in enumerate(group_terms):
        word = term_info["term"]
        padded = pad_pauli_word(word, n_qubits)
        x_mask, z_mask = pauli_to_masks(padded)
        masks[ti] = x_mask | z_mask
        coeffs[ti] = term_info["coeff"]
        terms.append(word)

    # Compute parity for all terms × all bitstrings
    # parity[ti, bi] = popcount(bs_array[bi] & mask_bits[ti]) % 2
    # Vectorized: convert masks to bit arrays
    mask_bits = np.zeros((n_terms, n_qubits), dtype=np.uint8)
    for ti in range(n_terms):
        m = int(masks[ti])
        for q in range(n_qubits):
            mask_bits[ti, q] = (m >> q) & 1

    # masked[ti, bi, q] = bs_array[bi, q] & mask_bits[ti, q]
    masked = bs_array[None, :, :] & mask_bits[:, None, :]
    parity = np.bitwise_xor.reduce(masked, axis=2).astype(np.float64)  # (n_terms, n_bs)

    # signs[ti, bi] = (-1)^parity
    signs = 1.0 - 2.0 * parity

    # expectations[ti] = sum(signs * counts) / n_shots
    expectations = signs @ counts_arr / n_shots

    # Energy contribution
    energy = float(np.dot(coeffs, expectations))

    # Build term_expectations dict
    term_exp: dict[str, dict[str, float]] = {}
    for ti, word in enumerate(terms):
        term_exp[word] = {
            "coeff_real": float(coeffs[ti]),
            "coeff_imag": 0.0,
            "expectation": float(expectations[ti]),
        }

    return energy, term_exp
