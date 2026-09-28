#!/usr/bin/env python3
"""Export MatGen-Q's published SnO (2e,7o) Hamiltonian into H-cGQE JSON.

Their GATE 1 instance is def2-SVP + ECP28MDF on Sn at 1.8325 Å, CASCI
-288.10912104 Ha. openfermionpyscf.generate_molecular_hamiltonian does not
pass that ECP, so rebuilding from YAML would be a different molecule.
This exporter uses their SnOMolecule class and writes our Pauli JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

MATGENQ_ENV = "GQE_MATGENQ_DIR"
MATGENQ_HINT = (
    "Set GQE_MATGENQ_DIR to your clone of the "
    "gqe-qsci-euv-photoresists baselines repo"
)


def _to_serializable_terms(qubit_ham: Any) -> list[dict[str, float | str]]:
    terms: list[dict[str, float | str]] = []
    items = list(qubit_ham.terms.items())
    for pauli_term, coeff in tqdm(
        items,
        desc="Serializing Pauli terms",
        unit="term",
        dynamic_ncols=True,
        disable=None,
    ):
        label = " ".join([f"{p}{i}" for i, p in pauli_term]) if pauli_term else "I"
        terms.append({"term": label, "real": float(coeff.real), "imag": float(coeff.imag)})
    return terms


def export_sno_14q(
    *,
    matgenq_code: Path,
    bond_length: float,
    skip_jw_check: bool,
) -> dict[str, Any]:
    code_dir = matgenq_code.resolve()
    if not (code_dir / "demo" / "molecule.py").is_file():
        raise FileNotFoundError(f"MatGen-Q molecule.py not found under {code_dir}")
    sys.path.insert(0, str(code_dir))
    from demo.molecule import CASCI_REFERENCE_HA, SnOMolecule

    mol = SnOMolecule(bond_length=bond_length, verbose=True)
    qubit_ham = mol.qubit_hamiltonian()
    if not skip_jw_check:
        mol.validate_jw(verbose=True)
    terms = _to_serializable_terms(qubit_ham)
    return {
        "name": "sno_14q",
        "split": "test",
        "smiles": "[Sn]=O",
        "geometry": [["Sn", [0.0, 0.0, 0.0]], ["O", [0.0, 0.0, float(bond_length)]]],
        "basis": "def2-svp",
        "ecp": {"Sn": "def2-svp"},
        "charge": 0,
        "multiplicity": 1,
        "active_space": {
            "mode": "explicit",
            "n_active_electrons": 2,
            "n_active_orbitals": 7,
            "n_core_orbitals": int(mol.ncore),
            "occupied_indices": [],
            "active_indices": list(mol.active_indices),
        },
        "n_qubits": int(mol.n_qubits),
        "n_pauli_terms": len(terms),
        "hf_energy": float(mol.e_hf),
        "casci_energy": float(mol.e_casci),
        "casci_reference_ha": float(CASCI_REFERENCE_HA),
        "source": "KarimElgammal/gqe-qsci-euv-photoresists SnOMolecule",
        "arxiv": "2607.23988",
        "terms": terms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="hamiltonians.json path, or a directory that will contain it",
    )
    parser.add_argument(
        "--matgenq-code",
        type=Path,
        default=None,
        help=f"Path to the baselines 'code' dir (defaults to ${MATGENQ_ENV})",
    )
    parser.add_argument("--bond-length", type=float, default=1.8325)
    parser.add_argument(
        "--skip-jw-check",
        action="store_true",
        help="Skip 2^14 sparse diagonalisation (GATE 1 CASCI is still asserted)",
    )
    args = parser.parse_args()

    raw = args.matgenq_code or os.environ.get(MATGENQ_ENV, "")
    if not str(raw).strip():
        raise SystemExit(MATGENQ_HINT)
    matgenq_code = Path(str(raw)).expanduser()
    if not matgenq_code.is_dir():
        raise SystemExit(MATGENQ_HINT)
    record = export_sno_14q(
        matgenq_code=matgenq_code,
        bond_length=args.bond_length,
        skip_jw_check=args.skip_jw_check,
    )
    out_path = args.out
    if out_path.suffix != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / "hamiltonians.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "MatGen-Q published SnO (2e,7o) Hamiltonian for H-cGQE A/B",
        "records": [record],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"Wrote {out_path}  n_qubits={record['n_qubits']}  "
        f"terms={record['n_pauli_terms']}  CASCI={record['casci_energy']:.8f} Ha"
    )


if __name__ == "__main__":
    main()
