"""Z2 qubit tapering for molecular Hamiltonian records.

This is one of the concrete ideas from the eQoSystem GQEx proposal that actually
makes sense: exploit molecular Z2 symmetries to reduce the qubit count before
running the GQE pipeline. The resulting tapered Hamiltonian can be passed to
the existing H-cGQE / L-BFGS-B pipeline without changing its core logic.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse
from qiskit.quantum_info import Pauli, SparsePauliOp, Z2Symmetries

from .hamiltonian_utils import hamiltonian_to_sparse_pauli_op, iter_terms


def _hf_bitstring(n_qubits: int, n_electrons: int) -> str:
    """Return a canonical Hartree-Fock bitstring: first n_electrons qubits occupied."""
    return "1" * n_electrons + "0" * (n_qubits - n_electrons)


def _symmetry_eigenvalues(symmetries: SparsePauliOp, bitstring: str) -> List[int]:
    """Eigenvalues (+/-1) of each Z2 symmetry operator on the given bitstring."""
    values = []
    for sym_pauli in symmetries:
        label = sym_pauli.to_label()  # label[i] is qubit i
        val = 1
        for i, op in enumerate(label):
            if op == "Z" and bitstring[i] == "1":
                val *= -1
        values.append(val)
    return values


def _exact_sector_ground_state(sparse_op: SparsePauliOp) -> float:
    """Return the lowest eigenvalue of a SparsePauliOp via sparse diagonalization."""
    if sparse_op.num_qubits == 0:
        return float(np.real(sparse_op.coeffs[0]))
    mat = sparse_op.to_matrix(sparse=True)
    if mat.shape[0] <= 2:
        eigvals = np.linalg.eigvalsh(mat.toarray())
    else:
        eigvals = scipy.sparse.linalg.eigsh(mat, k=1, which="SA")[0]
    return float(np.real(eigvals[0]))


def taper_hamiltonian_record(
    record: Dict[str, Any],
    n_electrons: int | None = None,
    exact_sector_search: bool = False,
    max_exact_qubits: int = 16,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Z2-taper a Hamiltonian record and return the tapered record + context.

    Args:
        record: Hamiltonian record with ``n_qubits`` and ``terms``.
        n_electrons: Number of active electrons. If None, read from the record's
            active_space or infer from geometry.
        exact_sector_search: If True, try all 2^k symmetry sectors and pick the
            one with the lowest ground-state energy. Fast enough only for small
            tapered Hamiltonians (<= ``max_exact_qubits``). Default is False, which
            uses the Hartree-Fock sector (correct for molecular ground states).
        max_exact_qubits: Maximum tapered qubits for which exact sector search is
            attempted.

    Returns:
        (tapered_record, context). ``context`` contains the symmetry operators,
        chosen tapering values, and an ``untaper_fn`` that can map operators back.
    """
    n_qubits = int(record["n_qubits"])
    if n_electrons is None:
        from .hamiltonian_utils import get_active_electron_count

        n_electrons = get_active_electron_count(record)
    n_electrons = max(0, min(n_electrons, n_qubits))

    sparse_op = hamiltonian_to_sparse_pauli_op(record)
    z2 = Z2Symmetries.find_z2_symmetries(sparse_op)

    symmetries = z2.symmetries
    n_sym = len(symmetries) if symmetries is not None else 0
    if n_sym == 0:
        # No tapering possible; return a copy with identity context.
        return deepcopy(record), {
            "n_tapered": n_qubits,
            "symmetries": [],
            "tapering_values": [],
            "sq_list": list(range(n_qubits)),
        }

    tapered_qubits = n_qubits - n_sym
    if exact_sector_search and tapered_qubits <= max_exact_qubits:
        # Try every sector and keep the one with the lowest ground-state energy.
        sectors = np.array(np.meshgrid(*[[-1, 1]] * n_sym)).T.reshape(-1, n_sym)
        best_energy = float("inf")
        best_values = sectors[0].tolist()
        for values in sectors:
            z2_sector = Z2Symmetries(
                symmetries=z2.symmetries,
                sq_paulis=z2.sq_paulis,
                tapering_values=values.tolist(),
                sq_list=z2.sq_list,
            )
            tapered = z2_sector.taper(sparse_op)
            energy = _exact_sector_ground_state(tapered)
            if energy < best_energy:
                best_energy = energy
                best_values = values.tolist()
        tapering_values = best_values
    else:
        # Use the Hartree-Fock sector (standard for molecular ground states).
        hf_bits = _hf_bitstring(n_qubits, n_electrons)
        tapering_values = _symmetry_eigenvalues(symmetries, hf_bits)

    z2_use = Z2Symmetries(
        symmetries=z2.symmetries,
        sq_paulis=z2.sq_paulis,
        tapering_values=tapering_values,
        sq_list=z2.sq_list,
    )
    tapered_sparse = z2_use.taper(sparse_op)

    def _qiskit_label_to_compact(label: str) -> str:
        """Convert Qiskit Pauli label (e.g., 'ZIIZ' or 'X') to compact 'X0 Z3'."""
        if len(label) == 1 and label != "I":
            return f"{label}0"
        parts = []
        for i, op in enumerate(label):
            if op != "I":
                parts.append(f"{op}{i}")
        return " ".join(parts) if parts else "I"

    # Build tapered record in the same format as the original.
    tapered_terms = []
    for pauli, coeff in zip(tapered_sparse.paulis, tapered_sparse.coeffs):
        label = _qiskit_label_to_compact(pauli.to_label())
        tapered_terms.append(
            {
                "term": label,
                "real": float(np.real(coeff)),
                "imag": float(np.imag(coeff)),
            }
        )

    tapered_record = deepcopy(record)
    tapered_record["n_qubits"] = tapered_sparse.num_qubits
    tapered_record["terms"] = tapered_terms
    tapered_record["tapered"] = True
    tapered_record["original_n_qubits"] = n_qubits

    context = {
        "n_tapered": tapered_sparse.num_qubits,
        "symmetries": [p.to_label() for p in symmetries],
        "tapering_values": tapering_values,
        "sq_list": list(z2.sq_list),
        "untaper_fn": z2_use,  # can be used to map operators back
    }
    return tapered_record, context
