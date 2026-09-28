#!/usr/bin/env python3
"""Validate the prototype active-space MBE2 in fixed particle-number sectors.

Stored Pauli indices follow OpenFermion's Jordan-Wigner convention.  Internally,
integer basis bit q is the occupation of spin orbital/qubit q.  Qiskit labels
are reversed when used for the optional ordering cross-check because Qiskit
prints qubit n-1 at the left of a Pauli label.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "results" / "data" / "fragments"
DEFAULT_OUTPUT = (
    ROOT / "results" / "phase3_final" / "fmo"
    / "fmo2_particle_number_validation.json"
)
METHOD = "non_embedded_active_space_mbe2_prototype"


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is absent: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a 'records' list")
    return records


def active_electrons(record: dict[str, Any]) -> int:
    value = record.get("active_space", {}).get("n_active_electrons")
    if value is None:
        raise ValueError(
            f"{record.get('name', '<unnamed>')} has no declared "
            "active_space.n_active_electrons"
        )
    return int(value)


def parse_term(label: str, n_qubits: int) -> list[str]:
    ops = ["I"] * n_qubits
    if label.strip() == "I":
        return ops
    for token in label.split():
        if len(token) < 2 or token[0] not in "XYZ":
            raise ValueError(f"Invalid Pauli token {token!r} in {label!r}")
        qubit = int(token[1:])
        if not 0 <= qubit < n_qubits:
            raise ValueError(f"Qubit {qubit} outside {n_qubits}-qubit record")
        ops[qubit] = token[0]
    return ops


def sector_basis(n_qubits: int, n_electrons: int) -> np.ndarray:
    if not 0 <= n_electrons <= n_qubits:
        raise ValueError(f"Invalid N={n_electrons} for {n_qubits} qubits")
    return np.fromiter(
        (
            sum(1 << qubit for qubit in occupied)
            for occupied in itertools.combinations(range(n_qubits), n_electrons)
        ),
        dtype=np.int64,
        count=math.comb(n_qubits, n_electrons),
    )


def sparse_hamiltonian(record: dict[str, Any]) -> sparse.csr_matrix:
    """Construct H with integer basis bit q representing occupation qubit q."""
    n_qubits = int(record["n_qubits"])
    dimension = 1 << n_qubits
    columns = np.arange(dimension, dtype=np.int64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for term in record.get("terms", []):
        coefficient = complex(
            float(term.get("real", 0.0)), float(term.get("imag", 0.0))
        )
        if abs(coefficient) < 1e-14:
            continue
        targets = columns.copy()
        phases = np.full(dimension, coefficient, dtype=np.complex128)
        for qubit, op in enumerate(parse_term(term["term"], n_qubits)):
            bit = (columns >> qubit) & 1
            if op == "X":
                targets ^= 1 << qubit
            elif op == "Y":
                targets ^= 1 << qubit
                phases *= 1j * (1 - 2 * bit)
            elif op == "Z":
                phases *= 1 - 2 * bit
        rows.append(targets)
        cols.append(columns)
        values.append(phases)

    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dimension, dimension),
        dtype=np.complex128,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def qiskit_hf_crosscheck(
    record: dict[str, Any], hf_index: int, expected: float
) -> dict[str, Any]:
    try:
        from qiskit.quantum_info import SparsePauliOp
    except ImportError:
        return {"available": False, "reason": "qiskit is not installed"}

    n_qubits = int(record["n_qubits"])
    terms = []
    for term in record.get("terms", []):
        ops = parse_term(term["term"], n_qubits)
        coefficient = complex(
            float(term.get("real", 0.0)), float(term.get("imag", 0.0))
        )
        terms.append(("".join(reversed(ops)), coefficient))
    matrix = SparsePauliOp.from_list(terms).to_matrix(sparse=True).tocsr()
    qiskit_value = float(np.real(matrix[hf_index, hf_index]))
    difference = abs(qiskit_value - expected)
    if difference > 1e-9:
        raise RuntimeError(
            f"Qiskit/OpenFermion ordering check failed: delta={difference:.3e}"
        )
    return {
        "available": True,
        "qiskit_label_rule": "reverse ops[q] so qubit n-1 is leftmost",
        "qiskit_hf_expectation": qiskit_value,
        "absolute_difference": difference,
        "passed": True,
    }


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    name = record.get("name", "<unnamed>")
    n_qubits = int(record["n_qubits"])
    n_electrons = active_electrons(record)
    basis = sector_basis(n_qubits, n_electrons)
    matrix = sparse_hamiltonian(record)
    sector = matrix[basis][:, basis].tocsr()

    outside = np.setdiff1d(np.arange(1 << n_qubits), basis, assume_unique=True)
    leakage = matrix[outside][:, basis]
    max_leakage = float(np.max(np.abs(leakage.data))) if leakage.nnz else 0.0
    if max_leakage > 1e-10:
        raise RuntimeError(
            f"{name} does not conserve particle number: leakage={max_leakage:.3e}"
        )

    if sector.shape[0] <= 128:
        eigenvalue = float(np.linalg.eigvalsh(sector.toarray())[0])
        solver = "numpy.linalg.eigvalsh"
    else:
        eigenvalue = float(
            eigsh(sector, k=1, which="SA", return_eigenvectors=False, tol=1e-11)[0]
        )
        solver = "scipy.sparse.linalg.eigsh"
    unconstrained = float(
        eigsh(matrix, k=1, which="SA", return_eigenvectors=False, tol=1e-11)[0]
    )

    hf_index = (1 << n_electrons) - 1
    if hf_index not in set(basis.tolist()):
        raise RuntimeError(f"HF determinant for {name} is outside fixed-N sector")
    hf_expectation = float(np.real(matrix[hf_index, hf_index]))
    ordering_check = qiskit_hf_crosscheck(record, hf_index, hf_expectation)

    return {
        "name": name,
        "n_qubits": n_qubits,
        "full_hilbert_dimension": 1 << n_qubits,
        "declared_active_electrons": n_electrons,
        "fixed_n_sector_dimension": int(sector.shape[0]),
        "expected_sector_dimension": math.comb(n_qubits, n_electrons),
        "fixed_n_energy_hartree": eigenvalue,
        "unconstrained_energy_hartree": unconstrained,
        "unconstrained_minus_fixed_n_hartree": unconstrained - eigenvalue,
        "solver": solver,
        "particle_number_leakage_max_abs": max_leakage,
        "hf_determinant": {
            "occupied_qubits": list(range(n_electrons)),
            "basis_index": hf_index,
            "expectation_hartree": hf_expectation,
            "is_in_sector": True,
        },
        "bit_ordering_crosscheck": ordering_check,
    }


def validate_inventory(
    monomers: list[dict[str, Any]],
    dimers: list[dict[str, Any]],
    parents: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(parents) != 1:
        raise ValueError(f"Expected exactly one parent record, found {len(parents)}")
    if not monomers:
        raise ValueError("No monomer records found")
    expected_pairs = set(itertools.combinations(range(len(monomers)), 2))
    actual_pairs: set[tuple[int, int]] = set()
    monomer_electrons = [active_electrons(record) for record in monomers]
    for dimer in dimers:
        pair = tuple(sorted((int(dimer["frag_i"]), int(dimer["frag_j"]))))
        if pair in actual_pairs:
            raise ValueError(f"Duplicate dimer pair {pair}")
        actual_pairs.add(pair)
        expected_electrons = sum(monomer_electrons[index] for index in pair)
        if active_electrons(dimer) != expected_electrons:
            raise ValueError(
                f"{dimer.get('name')} declares {active_electrons(dimer)} active "
                f"electrons; constituent monomers require {expected_electrons}"
            )
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"Missing/unexpected dimers: missing={sorted(expected_pairs - actual_pairs)}, "
            f"unexpected={sorted(actual_pairs - expected_pairs)}"
        )
    parent_electrons = active_electrons(parents[0])
    if parent_electrons != sum(monomer_electrons):
        raise ValueError(
            f"Parent declares {parent_electrons} active electrons; monomer sum is "
            f"{sum(monomer_electrons)}"
        )
    return {
        "monomer_count": len(monomers),
        "dimer_count": len(dimers),
        "parent_count": 1,
        "complete_pair_inventory": True,
        "active_electron_counts_consistent": True,
    }


def mbe_summary(
    monomers: list[dict[str, Any]],
    dimers: list[dict[str, Any]],
    parent: dict[str, Any],
    energy_key: str,
) -> dict[str, float]:
    monomer_sum = sum(float(item[energy_key]) for item in monomers)
    dimer_sum = sum(float(item[energy_key]) for item in dimers)
    mbe2 = dimer_sum - monomer_sum
    parent_energy = float(parent[energy_key])
    signed_error = mbe2 - parent_energy
    return {
        "monomer_sum_hartree": monomer_sum,
        "dimer_sum_hartree": dimer_sum,
        "mbe2_energy_hartree": mbe2,
        "parent_energy_hartree": parent_energy,
        "signed_error_hartree": signed_error,
        "absolute_error_hartree": abs(signed_error),
    }


def reference_summaries(
    path: Path,
    monomer_names: list[str],
    dimer_names: list[str],
) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        refs = json.load(handle)
    if "parent" not in refs:
        raise ValueError("ccsd_refs.json has no parent reference")
    required_names = monomer_names + dimer_names
    missing = [name for name in required_names if name not in refs]
    if missing:
        raise ValueError(f"ccsd_refs.json is missing records: {missing}")

    output: dict[str, Any] = {}
    for method, key in (
        ("hf", "hf_energy"),
        ("ccsd", "ccsd_energy"),
        ("ccsd_t", "ccsd_t_total"),
    ):
        all_refs = [refs[name] for name in required_names] + [refs["parent"]]
        if any(key not in item for item in all_refs):
            raise ValueError(f"Reference field {key!r} is incomplete")
        output[method] = mbe_summary(
            [refs[name] for name in monomer_names],
            [refs[name] for name in dimer_names],
            refs["parent"],
            key,
        )
        output[method]["source_field"] = key
    return output


def self_test() -> None:
    record = {
        "name": "two_qubit_number_conserving_test",
        "n_qubits": 2,
        "active_space": {"n_active_electrons": 1},
        "terms": [
            {"term": "I", "real": 1.0, "imag": 0.0},
            {"term": "Z0", "real": 0.25, "imag": 0.0},
            {"term": "Z1", "real": -0.5, "imag": 0.0},
            {"term": "X0 X1", "real": 0.1, "imag": 0.0},
            {"term": "Y0 Y1", "real": 0.1, "imag": 0.0},
        ],
    }
    result = evaluate_record(record)
    expected = float(np.linalg.eigvalsh(np.array([[0.25, 0.2], [0.2, 1.75]]))[0])
    if not np.isclose(result["fixed_n_energy_hartree"], expected, atol=1e-12):
        raise AssertionError("Fixed-N toy Hamiltonian eigenvalue is incorrect")
    if result["fixed_n_sector_dimension"] != 2:
        raise AssertionError("Fixed-N toy sector dimension is incorrect")
    print("Self-test passed: sector construction, Pauli phases, and ordering")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    monomer_records = load_records(args.input_dir / "monomers.json")
    dimer_records = load_records(args.input_dir / "dimers.json")
    parent_records = load_records(args.input_dir / "parent.json")
    inventory = validate_inventory(monomer_records, dimer_records, parent_records)

    monomers = [evaluate_record(record) for record in monomer_records]
    dimers = [evaluate_record(record) for record in dimer_records]
    parent = evaluate_record(parent_records[0])
    fixed_n = mbe_summary(
        monomers, dimers, parent, "fixed_n_energy_hartree"
    )
    unconstrained = mbe_summary(
        monomers, dimers, parent, "unconstrained_energy_hartree"
    )
    references = reference_summaries(
        args.input_dir / "ccsd_refs.json",
        [item["name"] for item in monomers],
        [item["name"] for item in dimers],
    )

    artifact = {
        "schema_version": 1,
        "method": METHOD,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "input_files": {
                name: str((args.input_dir / name).resolve().relative_to(ROOT))
                for name in (
                    "monomers.json",
                    "dimers.json",
                    "parent.json",
                    "ccsd_refs.json",
                )
            },
            "python": sys.version,
            "basis_convention": (
                "integer bit q is occupation of OpenFermion/Jordan-Wigner "
                "spin-orbital q; Qiskit Pauli labels are reversed"
            ),
        },
        "validation": inventory,
        "components": {
            "monomers": monomers,
            "dimers": dimers,
            "parent": parent,
        },
        "mbe2": {
            "formula": "sum(dimer energies) - sum(monomer energies)",
            "fixed_particle_number": fixed_n,
            "unconstrained_diagnostic_only": unconstrained,
            "classical_references": references,
        },
        "limitations": [
            "This is a non-embedded active-space MBE2 prototype, not FMO2.",
            "Independently optimized fragment active spaces are not a common "
            "orbital partition of the parent Hamiltonian.",
            "No electrostatic embedding, projection operator, or environmental "
            "polarization is included.",
            "Unconstrained ground energies are reported only to diagnose "
            "particle-number-sector collapse.",
            "HF/CCSD/CCSD(T) references use full molecular calculations and are "
            "not directly equivalent to the truncated active-space Hamiltonians.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing validation artifact: {args.output}"
        ) from exc
    print(json.dumps(fixed_n, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
