#!/usr/bin/env python3
"""Add HF and FCI reference energies to Hamiltonian records using PySCF.

Computes Hartree-Fock and FCI (for <=20q) or CASCI energies and injects
them into the 'hf_energy' and 'fci_energy' fields of each record.

Usage:
    python scripts/add_reference_energies.py \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --max-fci-qubits 20
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from pyscf import gto, scf, fci, mcscf
except ImportError:
    print("ERROR: PySCF not installed. Run: pip install pyscf")
    sys.exit(1)


def _parse_geometry(geom_list):
    """Convert [['H', [0,0,0]], ...] to PySCF format string."""
    lines = []
    for atom, coords in geom_list:
        x, y, z = coords
        lines.append(f"{atom} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines)


def _get_active_space(record):
    """Extract active space info from record."""
    aspace = record.get("active_space", {})
    n_active_e = aspace.get("n_active_electrons")
    n_active_o = aspace.get("n_active_orbitals")
    n_core = aspace.get("n_core_orbitals")
    occ_idx = aspace.get("occupied_indices", [])
    act_idx = aspace.get("active_indices", [])
    return n_active_e, n_active_o, n_core, occ_idx, act_idx


def compute_reference_energies(record, max_fci_qubits=20):
    """Compute HF and FCI/CASCI energy for a molecule record."""
    name = record.get("name", "unknown")
    geom = _parse_geometry(record["geometry"])
    basis = record.get("basis", "sto-3g")
    charge = record.get("charge", 0)
    multiplicity = record.get("multiplicity", 1)
    spin = multiplicity - 1  # PySCF spin = 2S = n_alpha - n_beta
    n_qubits = record.get("n_qubits", 0)

    mol = gto.M(atom=geom, basis=basis, charge=charge, spin=spin, verbose=0)

    # Hartree-Fock
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    hf_energy = mf.kernel()

    # FCI or CASCI
    n_active_e, n_active_o, n_core, occ_idx, act_idx = _get_active_space(record)

    if n_qubits <= max_fci_qubits and n_active_e is None:
        # Full FCI in the given basis
        try:
            if spin == 0:
                cisolver = fci.FCI(mol, mf.mo_coeff)
            else:
                cisolver = fci.FCI(mol, mf.mo_coeff, singlet=False)
            fci_energy = cisolver.kernel()[0]
        except Exception as e:
            print(f"  {name}: FCI failed ({e}), using CASCI fallback")
            fci_energy = None
    elif n_active_e is not None and n_active_o is not None:
        # CASCI in active space
        try:
            mc = mcscf.CASCI(mf, n_active_o, n_active_e)
            if n_core is not None:
                mc.ncore = n_core
            fci_energy = mc.kernel()[0]
        except Exception as e:
            print(f"  {name}: CASCI failed ({e})")
            fci_energy = None
    else:
        # No active space info and too large for full FCI
        fci_energy = None

    return hf_energy, fci_energy


def main():
    parser = argparse.ArgumentParser(description="Add HF/FCI reference energies to Hamiltonian records")
    parser.add_argument("--hamiltonians", type=str, required=True, help="Path to hamiltonians.json")
    parser.add_argument("--max-fci-qubits", type=int, default=20, help="Max qubits for full FCI")
    parser.add_argument("--dry-run", action="store_true", help="Print energies without saving")
    args = parser.parse_args()

    ham_path = Path(args.hamiltonians)
    with ham_path.open() as f:
        data = json.load(f)

    records = data.get("records", data if isinstance(data, list) else [])
    print(f"Processing {len(records)} molecules from {ham_path}")

    updated = 0
    for record in tqdm(records, desc="Computing reference energies"):
        name = record.get("name", "unknown")
        if "hf_energy" in record and record["hf_energy"] is not None:
            print(f"  {name}: already has hf_energy={record['hf_energy']:.6f}, skipping")
            continue

        try:
            hf_e, fci_e = compute_reference_energies(record, args.max_fci_qubits)
            record["hf_energy"] = hf_e
            if fci_e is not None:
                record["fci_energy"] = fci_e
            print(f"  {name}: HF={hf_e:.6f}, FCI={fci_e if fci_e is not None else 'N/A'}")
            updated += 1
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    if not args.dry_run and updated > 0:
        backup = ham_path.with_suffix(".json.bak")
        backup.write_text(ham_path.read_text())
        with ham_path.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"\nUpdated {updated}/{len(records)} records. Backup saved to {backup}")
    else:
        print(f"\n{'Dry run' if args.dry_run else 'No updates needed'}. {updated} energies computed.")


if __name__ == "__main__":
    main()
