"""Bitstring / QSCI mapping that matches MatGen-Q Demo A.

CUDA-Q ``sample`` and MatGen-Q NumpySampler emit little-endian strings where
character ``q`` is the outcome of qubit ``q`` (leftmost = qubit 0). Hartree-Fock
for ``n_electrons`` occupies qubits ``0 .. n_electrons-1``, so the HF string is
``'1' * n_electrons + '0' * (n_qubits - n_electrons)`` (e.g. SnO 14q:
``11000000000000``).

Pauli-subspace QSCI in ``qsci_postprocess.py`` uses the opposite integer
convention: ``int(bitstring, 2)`` with the **rightmost** character as qubit 0.
Convert with :func:`cudaq_bits_to_lsb` before calling that scorer.

Job 7431320 scored CUDA-Q strings with the Pauli convention unchanged and
inserted the LSB-right HF string ``00000000000011``. That mixed two endianness
conventions and could not reproduce Demo A (+2.869 mHa).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MATGENQ_CODE = Path("/scratch/kcwp264/baselines/gqe-qsci-euv-photoresists/code")


def hf_bitstring_cudaq(n_qubits: int, n_electrons: int) -> str:
    """HF computational-basis string in CUDA-Q / MatGen-Q order."""
    if not 0 <= n_electrons <= n_qubits:
        raise ValueError(f"n_electrons={n_electrons} not in [0, {n_qubits}]")
    return "1" * n_electrons + "0" * (n_qubits - n_electrons)


def hf_bitstring_lsb_right(n_qubits: int, n_electrons: int) -> str:
    """HF string for Pauli QSCI (``int(bits, 2)`` has qubit 0 as LSB)."""
    return format((1 << n_electrons) - 1, f"0{n_qubits}b")


def cudaq_bits_to_lsb(bits: str) -> str:
    """CUDA-Q (char q = qubit q) → LSB-right (rightmost = qubit 0)."""
    return bits[::-1]


def lsb_bits_to_cudaq(bits: str) -> str:
    """LSB-right → CUDA-Q."""
    return bits[::-1]


def cudaq_counts_to_lsb(counts: dict[str, int]) -> dict[str, int]:
    converted: dict[str, int] = {}
    for bits, n in counts.items():
        key = cudaq_bits_to_lsb(str(bits))
        converted[key] = converted.get(key, 0) + int(n)
    return converted


def unique_bitstrings_from_counts(counts: dict[str, int]) -> list[str]:
    """Unique bitstrings, most frequent first."""
    return [bs for bs, _ in sorted(counts.items(), key=lambda kv: -int(kv[1]))]


def import_matgenq_qsci(matgenq_code: Path | None = None):
    """Import MatGen-Q ``demo.qsci`` without importing cudaq."""
    code = Path(matgenq_code) if matgenq_code is not None else DEFAULT_MATGENQ_CODE
    code = code.resolve()
    if not (code / "demo" / "qsci.py").is_file():
        raise FileNotFoundError(f"MatGen-Q demo/qsci.py not under {code}")
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from demo.qsci import (  # type: ignore[import-not-found]
        SubspaceEngine,
        circuit_qsci_energy,
        determinant_from_bitstring,
        post_select,
        symmetry_complete,
    )

    return {
        "SubspaceEngine": SubspaceEngine,
        "circuit_qsci_energy": circuit_qsci_energy,
        "determinant_from_bitstring": determinant_from_bitstring,
        "post_select": post_select,
        "symmetry_complete": symmetry_complete,
    }


def load_sno_subspace_engine(matgenq_code: Path | None = None, verbose: bool = False):
    """Build SnO (2e,7o) molecular QSCI engine (PySCF integrals, no cudaq)."""
    code = Path(matgenq_code) if matgenq_code is not None else DEFAULT_MATGENQ_CODE
    code = code.resolve()
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from demo.molecule import SnOMolecule  # type: ignore[import-not-found]
    from demo.qsci import SubspaceEngine  # type: ignore[import-not-found]

    mol = SnOMolecule(verbose=verbose)
    engine = SubspaceEngine(mol.h1, mol.h2, mol.norb, mol.nelec, mol.e_core)
    return mol, engine


def score_cudaq_counts_matgenq(
    engine: Any,
    counts: dict[str, int],
    nelec: tuple[int, int],
    casci: float,
    *,
    max_dim: int = 2000,
    full_ci_dets: int = 49,
) -> dict[str, Any]:
    """Score CUDA-Q-ordered counts with MatGen-Q molecular QSCI."""
    qsci = import_matgenq_qsci()
    energy, dets, n_valid = qsci["circuit_qsci_energy"](
        engine, counts, nelec, max_dim=max_dim
    )
    n_dets = len(dets)
    modal = None
    if counts:
        modal = max(counts, key=lambda k: int(counts[k]))
    err = None if energy is None else abs(float(energy) - float(casci)) * 1000.0
    return {
        "n_valid_particle_number": int(n_valid),
        "n_dets": int(n_dets),
        "qsci_energy": None if energy is None else float(energy),
        "error_vs_casci_mha": err,
        "filled_full_ci": n_dets >= full_ci_dets,
        "modal_bitstring": modal,
        "n_unique_raw": len(counts),
        "qsci_backend": "matgenq_subspace",
    }


def merge_counts(*count_dicts: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for counts in count_dicts:
        for bits, n in counts.items():
            merged[str(bits)] = merged.get(str(bits), 0) + int(n)
    return merged


def sample_counts_numpy(
    operators: list[str] | None,
    thetas: list[float] | None,
    n_qubits: int,
    n_electrons: int,
    n_shots: int,
    seed: int,
) -> dict[str, int]:
    """MatGen-Q NumpySampler kernel: HF on qubits 0..n_e-1, then exp(i θ P).

    Bitstrings use character q = qubit q. No cudaq import.
    """
    dim = 1 << n_qubits
    indices = np.arange(dim, dtype=np.int64)
    psi = np.zeros(dim, dtype=np.complex128)
    psi[(1 << n_electrons) - 1] = 1.0
    words = list(operators or [])
    angles = list(thetas or [])
    if words and not angles:
        angles = [0.01] * len(words)
    for theta, word in zip(angles, words):
        if len(word) < n_qubits:
            word = word + "I" * (n_qubits - len(word))
        elif len(word) > n_qubits:
            word = word[:n_qubits]
        flip = 0
        ymask = 0
        zmask = 0
        for q, p in enumerate(word):
            bit = 1 << q
            if p == "X":
                flip |= bit
            elif p == "Y":
                flip |= bit
                ymask |= bit
            elif p == "Z":
                zmask |= bit
        if flip == 0 and ymask == 0 and zmask == 0:
            psi = psi * np.exp(1j * float(theta))
            continue
        n_y = bin(ymask).count("1")
        signs = np.array(
            [(-1) ** bin(int(s) & (ymask | zmask)).count("1") for s in indices]
        )
        p_psi = np.zeros_like(psi)
        p_psi[indices ^ flip] = (1j ** n_y) * signs * psi
        psi = np.cos(float(theta)) * psi + 1j * np.sin(float(theta)) * p_psi
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed)
    shot_counts = rng.multinomial(n_shots, probs)
    out: dict[str, int] = {}
    for s in np.flatnonzero(shot_counts):
        bits = "".join(str((int(s) >> q) & 1) for q in range(n_qubits))
        out[bits] = int(shot_counts[s])
    hf = hf_bitstring_cudaq(n_qubits, n_electrons)
    if hf not in out:
        out[hf] = 1
    return out

