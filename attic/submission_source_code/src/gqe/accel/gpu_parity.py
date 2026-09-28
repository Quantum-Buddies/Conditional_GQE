"""GPU-accelerated parity computation for QWC result parsing.

Replaces the Python loop:
    for bitstring, count in counts.items():
        parity = sum(int(bitstring[q]) for q, op in enumerate(term) if op != "I") % 2
        sign = -1 if parity == 1 else 1
        exp += sign * count / n_shots

With either:
  1. A Triton kernel (if GPU available) — processes all terms × bitstrings in one launch
  2. A NumPy vectorized path (CPU fallback)
  3. A C extension (optional, compiled at import time)

The Triton kernel computes parity via popcount(bitstring & mask) % 2
for all (term, bitstring) pairs in parallel.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from .fast_pauli import pad_pauli_word, pauli_to_masks


def parse_grouped_results_gpu(
    results: list[Any],
    group_mapping: list[list[dict[str, Any]]],
    n_qubits: int,
    shots: int,
    use_triton: bool = True,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Parse QWC grouped results with GPU acceleration.

    Args:
        results: list of result objects (one per QWC group), each with .get_counts()
        group_mapping: per-group list of {term_idx, term, coeff} dicts
        n_qubits: number of qubits
        shots: total shots per circuit
        use_triton: if True and GPU available, use Triton kernel

    Returns:
        (total_energy, {term: {coeff, expectation}})
    """
    total_energy = 0.0
    all_term_exp: dict[str, dict[str, float]] = {}

    for gi, (result, term_infos) in enumerate(zip(results, group_mapping)):
        counts = _get_counts(result)
        if not counts:
            continue

        energy, term_exp = _parse_group_counts_gpu(
            counts, term_infos, n_qubits, shots, use_triton,
        )
        total_energy += energy
        all_term_exp.update(term_exp)

    return total_energy, all_term_exp


def _get_counts(result: Any) -> dict[str, int]:
    """Extract counts from various result object types."""
    if hasattr(result, "get_counts"):
        return result.get_counts()
    if isinstance(result, dict):
        return result
    if hasattr(result, "counts"):
        return result.counts
    return {}


def _parse_group_counts_gpu(
    counts: dict[str, int],
    term_infos: list[dict[str, Any]],
    n_qubits: int,
    n_shots: int,
    use_triton: bool = True,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Parse counts for one QWC group using GPU/Triton if available."""
    n_bs = len(counts)
    n_terms = len(term_infos)

    if n_bs == 0 or n_terms == 0:
        return 0.0, {}

    # Try Triton GPU path
    if use_triton and n_bs * n_terms > 10000:
        try:
            return _parse_counts_triton(counts, term_infos, n_qubits, n_shots)
        except Exception:
            pass

    # Try PyTorch GPU path
    if n_bs * n_terms > 5000:
        try:
            return _parse_counts_torch(counts, term_infos, n_qubits, n_shots)
        except Exception:
            pass

    # NumPy vectorized fallback
    return _parse_counts_numpy(counts, term_infos, n_qubits, n_shots)


def _parse_counts_torch(
    counts: dict[str, int],
    term_infos: list[dict[str, Any]],
    n_qubits: int,
    n_shots: int,
) -> tuple[float, dict[str, dict[str, float]]]:
    """PyTorch GPU path: compute all parities on GPU."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_bs = len(counts)
    n_terms = len(term_infos)

    # Convert bitstrings to GPU tensor
    bs_list = list(counts.keys())
    counts_arr = np.array(list(counts.values()), dtype=np.float64)

    # Bit array: (n_bs, n_qubits)
    bs_bits = torch.zeros((n_bs, n_qubits), dtype=torch.uint8, device=device)
    for i, bs in enumerate(bs_list):
        for q in range(n_qubits):
            bs_bits[i, q] = int(bs[q])

    counts_t = torch.from_numpy(counts_arr).to(device)

    # Masks for each term
    masks = torch.zeros(n_terms, dtype=torch.int64, device=device)
    coeffs = torch.zeros(n_terms, dtype=torch.float64, device=device)
    terms = []
    for ti, info in enumerate(term_infos):
        word = info["term"]
        padded = pad_pauli_word(word, n_qubits)
        x, z = pauli_to_masks(padded)
        masks[ti] = x | z
        coeffs[ti] = info["coeff"]
        terms.append(word)

    # Convert masks to bit arrays: (n_terms, n_qubits)
    mask_bits = torch.zeros((n_terms, n_qubits), dtype=torch.uint8, device=device)
    for ti in range(n_terms):
        m = int(masks[ti].item())
        for q in range(n_qubits):
            mask_bits[ti, q] = (m >> q) & 1

    # Compute parity: (n_terms, n_bs)
    # masked[ti, bi, q] = bs_bits[bi, q] & mask_bits[ti, q]
    masked = bs_bits.unsqueeze(0) & mask_bits.unsqueeze(1)  # (n_terms, n_bs, n_qubits)
    parity = masked.sum(dim=2) % 2  # (n_terms, n_bs)

    # signs[ti, bi] = (-1)^parity
    signs = 1.0 - 2.0 * parity.double()

    # expectations[ti] = sum(signs * counts) / n_shots
    expectations = signs @ counts_t / n_shots

    # Energy
    energy = float((coeffs * expectations).sum().item())

    # Build result dict (matching original _parse_grouped_results structure)
    exp_cpu = expectations.cpu().numpy()
    term_exp: dict[str, dict[str, float]] = {}
    for ti, word in enumerate(terms):
        term_exp[word] = {
            "coeff_real": float(coeffs[ti].item()),
            "coeff_imag": 0.0,
            "expectation": float(exp_cpu[ti]),
        }

    return energy, term_exp


def _parse_counts_triton(
    counts: dict[str, int],
    term_infos: list[dict[str, Any]],
    n_qubits: int,
    n_shots: int,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Triton kernel path for maximum GPU throughput.

    Uses a custom Triton kernel that computes parity for all (term, bitstring)
    pairs in a single kernel launch.
    """
    try:
        import triton
        import triton.language as tl
        import torch
    except ImportError:
        return _parse_counts_torch(counts, term_infos, n_qubits, n_shots)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return _parse_counts_numpy(counts, term_infos, n_qubits, n_shots)

    n_bs = len(counts)
    n_terms = len(term_infos)

    # Prepare inputs
    bs_list = list(counts.keys())
    counts_arr = np.array(list(counts.values()), dtype=np.float32)

    # Pack bitstrings as int64 (up to 64 qubits)
    bs_packed = torch.zeros(n_bs, dtype=torch.int64, device=device)
    for i, bs in enumerate(bs_list):
        val = 0
        for q in range(n_qubits):
            if int(bs[q]):
                val |= 1 << q
        bs_packed[i] = val

    counts_t = torch.from_numpy(counts_arr).to(device)

    # Pack masks as int64
    masks = torch.zeros(n_terms, dtype=torch.int64, device=device)
    coeffs = torch.zeros(n_terms, dtype=torch.float32, device=device)
    terms = []
    for ti, info in enumerate(term_infos):
        word = info["term"]
        padded = pad_pauli_word(word, n_qubits)
        x, z = pauli_to_masks(padded)
        masks[ti] = x | z
        coeffs[ti] = info["coeff"]
        terms.append(word)

    # Triton kernel: compute expectations
    @triton.jit
    def _parity_expectation_kernel(
        bs_ptr, masks_ptr, counts_ptr, out_ptr,
        n_bs, n_terms, n_qubits,
        BLOCK_BS: tl.constexpr,
    ):
        ti = tl.program_id(0)  # term index
        mask = tl.load(masks_ptr + ti)

        # Process bitstrings in blocks
        acc = 0.0
        for bs_start in range(0, n_bs, BLOCK_BS):
            offsets = bs_start + tl.arange(0, BLOCK_BS)
            mask_valid = offsets < n_bs

            # Load bitstrings
            bs_vals = tl.load(bs_ptr + offsets, mask=mask_valid, other=0)
            cnts = tl.load(counts_ptr + offsets, mask=mask_valid, other=0.0)

            # Compute parity: popcount(bs & mask) % 2
            masked = bs_vals & mask
            # popcount via bit manipulation
            parity = tl.zeros_like(masked)
            for q in range(64):
                parity = parity + ((masked >> q) & 1)
            parity = parity % 2

            # sign = (-1)^parity = 1 - 2*parity
            signs = 1.0 - 2.0 * parity.to(tl.float32)
            acc += tl.sum(signs * cnts, mask=mask_valid)

        # Store expectation
        tl.store(out_ptr + ti, acc / n_shots)

    expectations = torch.zeros(n_terms, dtype=torch.float32, device=device)
    BLOCK_BS = 256
    _parity_expectation_kernel[(n_terms,)](
        bs_packed, masks, counts_t, expectations,
        n_bs, n_terms, n_qubits,
        BLOCK_BS=BLOCK_BS,
    )

    # Energy
    energy = float((coeffs * expectations).sum().item())

    # Build result dict (matching original structure)
    exp_cpu = expectations.cpu().numpy()
    term_exp: dict[str, dict[str, float]] = {}
    for ti, word in enumerate(terms):
        term_exp[word] = {
            "coeff_real": float(coeffs[ti].item()),
            "coeff_imag": 0.0,
            "expectation": float(exp_cpu[ti]),
        }

    return energy, term_exp


def _parse_counts_numpy(
    counts: dict[str, int],
    term_infos: list[dict[str, Any]],
    n_qubits: int,
    n_shots: int,
) -> tuple[float, dict[str, dict[str, float]]]:
    """NumPy vectorized fallback (CPU)."""
    from .fast_pauli import compute_grouped_expectations_vectorized
    return compute_grouped_expectations_vectorized(counts, term_infos, n_qubits, n_shots)
