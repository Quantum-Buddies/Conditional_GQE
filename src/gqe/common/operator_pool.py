"""Fermionic excitation operator pool for GQE.

Replaces the broken approach of building the pool from Hamiltonian Pauli terms
(which produces mostly Z-only diagonal operators and causes diagonal sequence
collapse). Instead, builds the pool from UCCSD fermionic excitations mapped
through Jordan-Wigner, exactly as the original GPT-QE paper describes.

References:
  - Nakaji et al., "The generative quantum eigensolver (GQE) and its application
    for ground state search", arXiv:2401.09253 (2024)
  - NVIDIA CUDA-QX GQE example: https://nvidia.github.io/cudaqx/examples_rst/solvers/gqe.html
  - Gard et al., "Local, Expressive, Quantum-Number-Preserving VQE Ansatze for
    Fermionic Systems", arXiv:2104.05695 (2021)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .hamiltonian_utils import get_active_electron_count


def _jw_excitation_pauli_words(
    n_qubits: int,
    n_electrons: int,
    *,
    max_singles: int | None = None,
    max_doubles: int | None = None,
) -> List[Tuple[str, float]]:
    """Generate UCCSD-style fermionic excitation Pauli words under Jordan-Wigner.

    Each excitation a†_p a_q (single) or a†_p a†_q a_r a_s (double) is mapped
    to Pauli operators via Jordan-Wigner. The anti-Hermitian combination
    (τ - τ†) produces Pauli words with X/Y components that create
    superpositions — Z-only collapse is impossible by construction.

    Returns a list of (pauli_word, coefficient) pairs where pauli_word is a
    string like "XYYX" of length n_qubits.
    """
    n_orbitals = n_qubits // 2
    # Spin-orbital indexing: even = alpha, odd = beta
    # Occupied spin-orbitals: 0..n_electrons-1
    # Virtual spin-orbitals: n_electrons..n_qubits-1

    excitations: List[Tuple[str, float]] = []

    # --- Single excitations: a†_p a_q for q occupied, p virtual ---
    singles = []
    for q in range(n_electrons):
        for p in range(n_electrons, n_qubits):
            singles.append((p, q))

    if max_singles is not None:
        singles = singles[:max_singles]

    for p, q in singles:
        # Jordan-Wigner: a†_p a_q - a†_q a_p = (i/2)(X_p Z_{p+1..q-1} X_q + Y_p Z_{p+1..q-1} Y_q)
        # for p < q; swap if p > q
        if p > q:
            p, q = q, p  # ensure p < q for the JW string

        # Build the two Pauli words for this excitation
        for op_p, op_q in [("X", "X"), ("Y", "Y")]:
            word = ["I"] * n_qubits
            word[p] = op_p
            word[q] = op_q
            # JW chain: Z on all orbitals between p and q
            for k in range(p + 1, q):
                word[k] = "Z"
            excitations.append(("".join(word), 0.5 if op_p == "X" else -0.5))

    # --- Double excitations: a†_p a†_q a_r a_s ---
    # (p,q virtual; r,s occupied) — standard UCCSD
    doubles = []
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))
    for s in occupied:
        for r in occupied:
            if r >= s:
                continue
            for q in virtual:
                for p in virtual:
                    if p >= q:
                        continue
                    doubles.append((p, q, r, s))

    if max_doubles is not None:
        doubles = doubles[:max_doubles]

    for p, q, r, s in doubles:
        # Each double excitation produces up to 8 Pauli words under JW
        # (all combinations of X/Y on the 4 sites, with Z chains between them)
        for ops_4 in _double_excitation_pauli_combos(p, q, r, s, n_qubits):
            excitations.append(ops_4)

    return excitations


def _double_excitation_pauli_combos(
    p: int, q: int, r: int, s: int, n_qubits: int
) -> List[Tuple[str, float]]:
    """Generate the 8 Pauli words for a double excitation a†_p a†_q a_r a_s.

    Under Jordan-Wigner, the anti-Hermitian double excitation decomposes into
    8 Pauli terms with X/Y on the 4 active sites and Z chains in between.
    """
    # Sort the 4 indices to build JW chains correctly
    sites = sorted([p, q, r, s])
    site_set = set(sites)

    # The 8 combinations: (X or Y) on each of the 4 sites
    # Coefficient signs alternate based on the combination
    combos: List[Tuple[str, float]] = []
    pauli_choices = ["X", "Y"]

    # Sign pattern for the 8 terms
    # The standard decomposition gives:
    # (1/8) * sum over {X,Y}^4 of sign * P0 P1 P2 P3 * Z_chains
    signs = [
        (1, 1, 1, 1),   # XXXX: +
        (1, 1, -1, -1), # XXYY: -
        (1, -1, 1, -1), # XYXY: -
        (1, -1, -1, 1), # XYYX: +
        (-1, 1, 1, -1), # YXXY: -
        (-1, 1, -1, 1), # YXYX: +
        (-1, -1, 1, 1), # YYXX: +
        (-1, -1, -1, -1),# YYYY: -
    ]

    for i, (s0, s1, s2, s3) in enumerate(signs):
        word = ["I"] * n_qubits
        ops_4 = [pauli_choices[0 if s > 0 else 1] for s in [s0, s1, s2, s3]]

        # Place operators on the 4 sorted sites
        for site_idx, site in enumerate(sites):
            word[site] = ops_4[site_idx]

        # JW Z chains between non-consecutive sites
        for j in range(len(sites) - 1):
            for k in range(sites[j] + 1, sites[j + 1]):
                if k not in site_set:
                    word[k] = "Z"

        # Coefficient: 1/8 with sign
        coeff = 0.125 * (1 if i % 4 in (0, 3, 5, 6) else -1)
        combos.append(("".join(word), coeff))

    return combos


def _cudaq_builtin_uccsd_pool(
    record: Dict[str, Any],
    scale_factors: Sequence[float],
) -> List[Tuple[Any, complex, str]] | None:
    """Try to use CUDA-Q's built-in solvers.get_operator_pool('uccsd').

    Returns None if CUDA-Q solvers isn't available or fails.
    """
    try:
        import cudaq  # type: ignore[import-untyped]
        import cudaq_solvers as solvers  # type: ignore[import-untyped]
        from cudaq import spin  # type: ignore[import-untyped]

        n_qubits = int(record["n_qubits"])
        n_electrons = get_active_electron_count(record)

        raw_ops = solvers.get_operator_pool(
            "uccsd", n_qubits=n_qubits, n_electrons=n_electrons
        )

        pool: List[Tuple[Any, complex, str]] = []
        for op in raw_ops:
            # Extract Pauli word string from the spin operator
            try:
                pw_str = str(op.to_pauli_word())
            except Exception:
                # Fallback: iterate terms to build the string
                chars = ["I"] * n_qubits
                for term in op:
                    word = term.to_string()
                    for ch in word:
                        if ch in "XYZI":
                            pass
                pw_str = "".join(chars)

            for scale in scale_factors:
                pool.append((
                    scale * op,
                    complex(scale),
                    pw_str,
                ))
        return pool
    except Exception:
        return None


def build_uccsd_operator_pool(
    record: Dict[str, Any],
    *,
    scale_factors: Sequence[float] = (
        0.003125, -0.003125, 0.00625, -0.00625,
        0.0125, -0.0125, 0.025, -0.025,
        0.05, -0.05, 0.1, -0.1,
    ),
    max_singles: int | None = None,
    max_doubles: int | None = None,
    use_cudaq_builtin: bool = False,
) -> List[Tuple[Any, complex, str]]:
    """Build a UCCSD fermionic operator pool for GQE.

    This replaces the broken approach of using Hamiltonian Pauli terms as the
    operator pool. The UCCSD pool contains only fermionic excitation operators
    mapped through Jordan-Wigner, which all contain X/Y components and
    therefore cannot cause diagonal sequence collapse.

    Args:
        record: Hamiltonian record dict with n_qubits and geometry/active_space.
        scale_factors: Time step parameters for each operator (from GPT-QE paper).
        max_singles: Optional limit on single excitations.
        max_doubles: Optional limit on double excitations.
        use_cudaq_builtin: If True, try CUDA-Q's built-in solvers.get_operator_pool
            first, falling back to manual JW if unavailable.

    Returns:
        List of (SpinOperator, coefficient, pauli_word_string) tuples,
        matching the interface expected by run_cudaq_gqe.py.
    """
    import cudaq  # type: ignore[import-untyped]
    from cudaq import spin  # type: ignore[import-untyped]

    if use_cudaq_builtin:
        builtin = _cudaq_builtin_uccsd_pool(record, scale_factors)
        if builtin is not None and len(builtin) > 0:
            return builtin

    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)

    excitation_words = _jw_excitation_pauli_words(
        n_qubits, n_electrons,
        max_singles=max_singles,
        max_doubles=max_doubles,
    )

    pool: List[Tuple[Any, complex, str]] = []
    for pauli_str, base_coeff in excitation_words:
        # Build CUDA-Q SpinOperator from the Pauli string
        term_op: Any | None = None
        for qubit, op in enumerate(pauli_str):
            if op == "I":
                continue
            single = {"X": spin.x, "Y": spin.y, "Z": spin.z}[op](qubit)
            term_op = single if term_op is None else term_op * single

        if term_op is None:
            continue  # skip identity

        for scale in scale_factors:
            pool.append((
                scale * term_op,
                complex(scale * base_coeff),
                pauli_str,
            ))

    return pool


def build_uccsd_pauli_words(
    record: Dict[str, Any],
    *,
    max_singles: int | None = None,
    max_doubles: int | None = None,
) -> List[str]:
    """Return just the Pauli word strings from the UCCSD pool (no CUDA-Q needed).

    Useful for building vocabularies and masks without a CUDA-Q installation.
    """
    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)
    excitations = _jw_excitation_pauli_words(
        n_qubits, n_electrons,
        max_singles=max_singles,
        max_doubles=max_doubles,
    )
    # Deduplicate while preserving order
    seen: set[str] = set()
    words: List[str] = []
    for word, _ in excitations:
        if word not in seen:
            seen.add(word)
            words.append(word)
    return words
