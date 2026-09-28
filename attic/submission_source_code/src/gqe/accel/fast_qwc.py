"""Vectorized QWC grouping using bit manipulation.

Replaces the O(n²) Python loop in _group_qwc_terms with:
  1. Convert Pauli words to (x_mask, z_mask) integer arrays
  2. Compute QWC compatibility matrix via vectorized bitwise ops
  3. Greedy grouping using the compatibility matrix

For LiH (631 terms): ~400K comparisons done in one NumPy operation.
For N2 (2951 terms): ~8.7M comparisons — still fast with vectorization.

Also provides a GPU-accelerated path via PyTorch when available.
"""
from __future__ import annotations

import numpy as np

from .fast_pauli import pauli_to_masks, pauli_list_to_masks, pad_pauli_word


def group_qwc_terms_vectorized(
    terms: list[tuple[str, float]],
    use_gpu: bool = True,
) -> list[list[int]]:
    """Group Pauli terms by qubit-wise commutativity using vectorized bit ops.

    Args:
        terms: list of (pauli_word, coefficient) tuples
        use_gpu: If True and PyTorch+CUDA available, use GPU for compatibility matrix

    Returns:
        List of groups, each group being a list of indices into terms.
    """
    n = len(terms)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    words = [t[0] for t in terms]
    x_masks, z_masks = pauli_list_to_masks(words)

    # GPU path not beneficial: greedy grouping loop is the bottleneck,
    # not the compatibility matrix computation. Use CPU path.
    # CPU vectorized path
    compat = _qwc_compat_matrix_cpu(x_masks, z_masks)

    return _greedy_grouping(compat, n)


def _qwc_compat_matrix_cpu(
    x_masks: np.ndarray, z_masks: np.ndarray,
) -> np.ndarray:
    """Compute QWC compatibility matrix on CPU using NumPy."""
    n = len(x_masks)
    # Broadcast to (n, n) matrices
    x1 = x_masks[:, None]
    z1 = z_masks[:, None]
    x2 = x_masks[None, :]
    z2 = z_masks[None, :]

    # QWC iff (x1 & z2) | (z1 & x2) == 0
    conflict = (x1 & z2) | (z1 & x2)
    return conflict == 0


def _qwc_compat_matrix_gpu(
    x_masks: np.ndarray, z_masks: np.ndarray,
) -> np.ndarray:
    """Compute QWC compatibility matrix on GPU using PyTorch.

    For very large term counts (N2: 2951 terms → 8.7M pairs), GPU is faster.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return _qwc_compat_matrix_cpu(x_masks, z_masks)

    x = torch.from_numpy(x_masks).to(device)
    z = torch.from_numpy(z_masks).to(device)

    # (n, 1) & (1, n) -> (n, n)
    x1 = x.unsqueeze(1)
    z1 = z.unsqueeze(1)
    x2 = x.unsqueeze(0)
    z2 = z.unsqueeze(0)

    conflict = (x1 & z2) | (z1 & x2)
    compat = (conflict == 0).cpu().numpy()
    return compat


def _group_qwc_gpu(
    x_masks: np.ndarray, z_masks: np.ndarray, n: int,
) -> list[list[int]] | None:
    """GPU-accelerated QWC grouping using PyTorch."""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    compat = _qwc_compat_matrix_gpu(x_masks, z_masks)
    return _greedy_grouping(compat, n)


def _greedy_grouping(compat: np.ndarray, n: int) -> list[list[int]]:
    """Greedy grouping from a compatibility matrix.

    For each term, find first group whose intersection compat still includes it.
    group_compat[gi] tracks which unassigned terms are compatible with ALL members.

    Optimized: uses a 2D array for group_compat instead of list of arrays,
    enabling vectorized batch checking when n is large.
    """
    if n > 1000:
        return _greedy_grouping_large(compat, n)

    groups: list[list[int]] = []
    assigned = np.zeros(n, dtype=bool)
    group_compat: list[np.ndarray] = []

    for idx in range(n):
        if assigned[idx]:
            continue

        placed = False
        for gi, gc in enumerate(group_compat):
            if gc[idx]:
                groups[gi].append(idx)
                group_compat[gi] = group_compat[gi] & compat[idx]
                assigned[idx] = True
                placed = True
                break

        if not placed:
            groups.append([idx])
            group_compat.append(compat[idx].copy())
            assigned[idx] = True

    return groups


def _greedy_grouping_large(compat: np.ndarray, n: int) -> list[list[int]]:
    """Optimized greedy grouping for large n (>1000 terms).

    Uses a preallocated 2D boolean array for group_compat to enable vectorized
    column indexing. Avoids O(n²) vstack copying.
    """
    max_groups = n  # worst case: each term in its own group
    group_compat_2d = np.zeros((max_groups, n), dtype=bool)
    n_groups = 0

    groups: list[list[int]] = []
    assigned = np.zeros(n, dtype=bool)

    for idx in range(n):
        if assigned[idx]:
            continue

        if n_groups > 0:
            compatible = np.nonzero(group_compat_2d[:n_groups, idx])[0]
            if len(compatible) > 0:
                gi = int(compatible[0])
                groups[gi].append(idx)
                group_compat_2d[gi] = group_compat_2d[gi] & compat[idx]
                assigned[idx] = True
                continue

        # Create new group
        groups.append([idx])
        group_compat_2d[n_groups] = compat[idx]
        n_groups += 1
        assigned[idx] = True

    return groups


def group_qwc_terms_fast(
    terms: list[tuple[str, float]],
    n_qubits: int,
) -> tuple[list[list[int]], list[str]]:
    """Fast QWC grouping with measurement basis computation.

    Args:
        terms: list of (pauli_word, coefficient) tuples
        n_qubits: number of qubits

    Returns:
        (groups, measurement_bases) where:
          groups: list of list of indices
          measurement_bases: list of basis strings (one per group)
    """
    groups = group_qwc_terms_vectorized(terms)

    # Compute measurement basis for each group
    bases = []
    for group_indices in groups:
        base = ["I"] * n_qubits
        for ti in group_indices:
            word = terms[ti][0]
            padded = pad_pauli_word(word, n_qubits)
            for q in range(n_qubits):
                if padded[q] != "I" and base[q] == "I":
                    base[q] = padded[q]
        bases.append("".join(base))

    return groups, bases
