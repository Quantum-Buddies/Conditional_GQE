"""Helpers for moving between stored Pauli terms and solver-native objects."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PAULI_PATTERN = re.compile(r"([IXYZ])(\d+)?")

# Atomic number lookup for common elements used in this challenge
_ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}


def get_active_electron_count(record: Dict[str, Any]) -> int:
    """Return the number of active electrons for a Hamiltonian record.

    Uses the stored active_space count if available; otherwise computes it
    from the molecular geometry and charge.
    """
    active_space = record.get("active_space", {})
    n_active_electrons = active_space.get("n_active_electrons")
    if n_active_electrons is not None:
        return int(n_active_electrons)

    charge = int(record.get("charge", 0))
    geometry = record.get("geometry", [])
    total_protons = 0
    for atom in geometry:
        symbol = atom[0] if isinstance(atom, (list, tuple)) else atom.get("symbol")
        if symbol is None:
            continue
        # Handle symbols that may include numeric labels (e.g., H1, C1)
        symbol = "".join(ch for ch in str(symbol) if ch.isalpha())
        total_protons += _ATOMIC_NUMBERS.get(symbol, 0)
    return max(total_protons - charge, 0)


def load_hamiltonian_records(hamiltonian_json_path: Path) -> List[Dict[str, Any]]:
    """Load molecular Hamiltonian records from the generated JSON dataset."""
    with hamiltonian_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("records", [])


def _parse_pauli_label(label: str, n_qubits: int) -> List[str]:
    """Return explicit single-qubit operators for the compact Pauli label."""

    ops = ["I"] * n_qubits
    if label == "I":
        return ops

    # If n_qubits == 1, a bare operator like 'X' or 'Z' is qubit 0.
    if n_qubits == 1 and len(label) == 1 and label in "XYZ":
        ops[0] = label
        return ops

    for match in PAULI_PATTERN.finditer(label):
        op, index = match.groups()
        if index is None:
            # Bare 'I' implies the global identity, nothing to do
            continue
        ops[int(index)] = op
    return ops


def iter_terms(record: Dict[str, Any]) -> Iterable[Tuple[List[str], complex]]:
    """Yield (per-qubit ops, coefficient) pairs from a record."""

    n_qubits = int(record["n_qubits"])
    for term in record.get("terms", []):
        coeff = complex(float(term.get("real", 0.0)), float(term.get("imag", 0.0)))
        if abs(coeff) < 1e-14:
            continue
        yield _parse_pauli_label(term["term"], n_qubits), coeff


def hamiltonian_to_sparse_pauli_op(record: Dict[str, Any]) -> Any:
    """Convert a record to Qiskit's SparsePauliOp (requires qiskit)."""

    from qiskit.quantum_info import SparsePauliOp  # type: ignore[import-untyped]

    pauli_list = []
    coeffs = []
    for ops, coeff in iter_terms(record):
        pauli_list.append("".join(ops))
        coeffs.append(coeff)

    if not pauli_list:
        n_qubits = int(record["n_qubits"])
        pauli_list = ["I" * n_qubits]
        coeffs = [0.0]

    return SparsePauliOp.from_list(list(zip(pauli_list, coeffs)))


def pauli_ops_to_spin_term(ops: List[str]) -> Any | None:
    """Return a CUDA-Q SpinOperator term for the provided Pauli list."""

    import cudaq  # type: ignore[import-untyped]
    from cudaq import spin  # type: ignore[import-untyped]

    term_op: Any | None = None
    for qubit, op in enumerate(ops):
        if op == "I":
            continue
        single = {
            "X": spin.x,
            "Y": spin.y,
            "Z": spin.z,
        }[op](qubit)
        term_op = single if term_op is None else term_op * single
    return term_op


def hamiltonian_to_spin_operator(record: Dict[str, Any]) -> Any:
    """Convert a record to CUDA-Q's SpinOperator (requires cudaq)."""

    import cudaq  # type: ignore[import-untyped]

    spin_op = cudaq.SpinOperator()
    for ops, coeff in iter_terms(record):
        term_op = pauli_ops_to_spin_term(ops)
        if term_op is None:
            spin_op += coeff
        else:
            spin_op += coeff * term_op

    return spin_op


def exact_diagonalize_hamiltonian(record: Dict[str, Any]) -> Tuple[float, float]:
    """Exact diagonalization of a small Hamiltonian using dense matrix methods.

    Returns (ground_state_energy, energy_gap_to_first_excited).
    """
    n_qubits = int(record["n_qubits"])
    if n_qubits > 14:
        raise ValueError(
            f"Exact diagonalization is limited to <= 14 qubits, got {n_qubits}."
        )

    sparse_op = hamiltonian_to_sparse_pauli_op(record)
    # Convert to dense numpy array
    dense = sparse_op.to_matrix()

    # Eigenvalues only for Hermitian matrix
    eigenvalues = np.linalg.eigvalsh(dense)
    eigenvalues = np.sort(eigenvalues)

    ground_energy = float(eigenvalues[0])
    gap = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0

    return ground_energy, gap


def find_record_by_name(records: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """Find a Hamiltonian record by its molecule name."""
    for rec in records:
        if rec.get("name") == name:
            return rec
    available = [r.get("name", "<unnamed>") for r in records]
    raise ValueError(f"Molecule {name!r} not found. Available: {available}")
