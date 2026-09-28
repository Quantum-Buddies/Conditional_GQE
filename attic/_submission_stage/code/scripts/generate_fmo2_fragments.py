#!/usr/bin/env python3
"""Generate monomer, dimer, and parent Hamiltonians for FMO2 scaling demonstration.

Splits iodobenzene_cas12 (12q parent) into 3 spatial fragments so that
FMO2 many-body expansion recovers the parent energy from max 8q dimer circuits.

Output:
  results/data/fragments/monomers.json   — 3 monomer Hamiltonian records
  results/data/fragments/dimers.json     — 3 dimer Hamiltonian records
  results/data/fragments/parent.json     — parent Hamiltonian record
  results/data/fragments/ccsd_refs.json  — CCSD/CCSD(T) reference energies

Usage:
    python scripts/generate_fmo2_fragments.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openfermion import count_qubits, get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermionpyscf import generate_molecular_hamiltonian

from src.gqe.data.fragmentation import (
    ActiveSpaceSpec,
    build_active_space_spec,
    build_fragment_records,
    build_dimer_records,
    fragment_geometry,
)
from openfermion.ops import QubitOperator


def _to_serializable_terms(qubit_ham: QubitOperator) -> list[dict]:
    terms = []
    for pauli_term, coeff in qubit_ham.terms.items():
        label = " ".join([f"{p}{i}" for i, p in pauli_term]) if pauli_term else "I"
        terms.append({"term": label, "real": float(coeff.real), "imag": float(coeff.imag)})
    return terms

# --- Iodobenzene geometry (from iodobenzene_cas12 record) ---
PARENT_GEOMETRY = [
    ["C", [0.0, 1.4, 0.0]],      # 0
    ["C", [1.212, 0.7, 0.0]],    # 1
    ["C", [1.212, -0.7, 0.0]],   # 2
    ["C", [0.0, -1.4, 0.0]],     # 3
    ["C", [-1.212, -0.7, 0.0]],  # 4
    ["C", [-1.212, 0.7, 0.0]],   # 5
    ["I", [-2.42, 1.4, 0.0]],    # 6
    ["H", [2.147, 1.24, 0.0]],   # 7
    ["H", [2.147, -1.24, 0.0]],  # 8
    ["H", [0.0, -2.48, 0.0]],    # 9
    ["H", [-2.147, -1.24, 0.0]], # 10
    ["H", [-2.147, 1.24, 0.0]],  # 11
]

# 3-fragment plan: I-aryl, ortho, meta-para
# Charges adjusted so each fragment is closed-shell (even electron count)
# I(53e) + C(6e) + H(1e) + H(1e) = 61e → charge=+1 → 60e (even)
# C(6e) + C(6e) + H(1e) = 13e → charge=-1 → 14e (even)
# C(6e) + C(6e) + C(6e) + H(1e) + H(1e) = 20e → charge=0 → 20e (even)
# Total charge: +1 + (-1) + 0 = 0 ✓
FRAGMENT_PLAN = [
    {"name": "frag_iodo", "atom_indices": [6, 5, 10, 11], "charge": 1},
    {"name": "frag_ortho", "atom_indices": [0, 1, 7], "charge": -1},
    {"name": "frag_meta_para", "atom_indices": [2, 3, 4, 8, 9], "charge": 0},
]

BASIS = "sto-3g"
CHARGE = 0
MULTIPLICITY = 1


def _generate_hamiltonian(
    geometry: list,
    basis: str,
    charge: int,
    multiplicity: int,
    active_space: ActiveSpaceSpec | None = None,
) -> dict[str, Any]:
    """Generate a molecular Hamiltonian record from geometry."""
    if active_space is None:
        active_space = build_active_space_spec(
            geometry=geometry, basis=basis, charge=charge, multiplicity=multiplicity,
        )

    import inspect
    raw_kwargs = {
        "geometry": geometry,
        "basis": basis,
        "multiplicity": multiplicity,
        "charge": charge,
    }
    raw_kwargs.update(active_space.as_kwargs())
    if active_space.n_core_orbitals is not None:
        raw_kwargs["n_core_orbitals"] = int(active_space.n_core_orbitals)

    sig = inspect.signature(generate_molecular_hamiltonian)
    call_kwargs = {}
    alias_map = {
        "occupied_indices": ("occupied_indices", "docc_mo_indices"),
        "active_indices": ("active_indices", "active_mo_indices", "active_orbital_indices"),
        "n_active_electrons": ("n_active_electrons",),
        "n_active_orbitals": ("n_active_orbitals",),
        "n_core_orbitals": ("n_core_orbitals", "n_core"),
    }
    for key in ("geometry", "basis", "multiplicity", "charge"):
        if key in sig.parameters:
            call_kwargs[key] = raw_kwargs[key]
    for canonical_key, aliases in alias_map.items():
        value = raw_kwargs.get(canonical_key)
        if value is None:
            continue
        for alias in aliases:
            if alias in sig.parameters:
                call_kwargs[alias] = value
                break

    mol_ham = generate_molecular_hamiltonian(**call_kwargs)
    fermion_ham = get_fermion_operator(mol_ham)
    qubit_ham = jordan_wigner(fermion_ham)
    terms = _to_serializable_terms(qubit_ham)

    return {
        "geometry": geometry,
        "basis": basis,
        "charge": charge,
        "multiplicity": multiplicity,
        "active_space": active_space.as_dict(),
        "n_qubits": int(count_qubits(fermion_ham)),
        "n_pauli_terms": len(terms),
        "terms": terms,
    }


def _compute_ccsd(geometry: list, basis: str, charge: int, multiplicity: int) -> dict[str, float]:
    """Compute HF, MP2, CCSD, and CCSD(T) energies via PySCF."""
    from pyscf import gto, scf, cc

    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.build(parse_arg=False)

    if mol.spin == 0:
        mf = scf.RHF(mol).run()
        mycc = cc.RCCSD(mf).run()
        try:
            from pyscf.cc.ccsd_t import RCCSD as _RCCSD_T
            ccsd_t = mycc.ccsd_t()
        except Exception:
            try:
                ccsd_t = mycc.ccsd_t()
            except Exception:
                ccsd_t = None
    else:
        mf = scf.UHF(mol).run()
        mycc = cc.UCCSD(mf).run()
        ccsd_t = None

    result = {
        "hf_energy": float(mf.e_tot),
        "ccsd_energy": float(mycc.e_tot),
    }
    if ccsd_t is not None:
        result["ccsd_t_energy"] = float(ccsd_t)
        result["ccsd_t_total"] = float(mycc.e_tot + ccsd_t)
    return result


def main() -> None:
    out_dir = ROOT / "results/data/fragments"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # --- Build fragment records ---
    print("=== FMO2 Fragment Generation ===")
    print(f"Parent: iodobenzene_cas12, 12 atoms, {BASIS}")

    fragment_records = build_fragment_records(
        parent_name="iodobenzene_cas12",
        geometry=PARENT_GEOMETRY,
        fragments=FRAGMENT_PLAN,
        charge=CHARGE,
        multiplicity=MULTIPLICITY,
        basis=BASIS,
    )

    print(f"\nFragment plan: {len(fragment_records)} fragments")
    for fr in fragment_records:
        print(f"  {fr['name']}: atoms={fr['atom_indices']}, "
              f"n_atoms={len(fr['geometry'])}")

    # --- Generate monomer Hamiltonians ---
    # Explicit CAS(2e,2o)=4q per monomer — keeps dimers at CAS(4e,4o)=8q < parent 12q
    MONOMER_CAS = {"n_active_electrons": 2, "n_active_orbitals": 2}
    print("\n--- Generating monomer Hamiltonians ---")
    monomer_hams = []
    for fr in fragment_records:
        print(f"  {fr['name']}...")
        frag_charge = int(fr.get("charge", CHARGE))
        as_spec = build_active_space_spec(
            mode="explicit",
            n_active_electrons=MONOMER_CAS["n_active_electrons"],
            n_active_orbitals=MONOMER_CAS["n_active_orbitals"],
            geometry=fr["geometry"],
            basis=BASIS,
            charge=frag_charge,
            multiplicity=MULTIPLICITY,
        )
        ham = _generate_hamiltonian(
            geometry=fr["geometry"],
            basis=BASIS,
            charge=frag_charge,
            multiplicity=MULTIPLICITY,
            active_space=as_spec,
        )
        ham["name"] = fr["name"]
        ham["parent_name"] = "iodobenzene_cas12"
        ham["atom_indices"] = fr["atom_indices"]
        print(f"    {ham['n_qubits']}q, {ham['n_pauli_terms']} terms")
        monomer_hams.append(ham)

    # --- Build dimer records ---
    dimer_records = build_dimer_records(
        parent_name="iodobenzene_cas12",
        geometry=PARENT_GEOMETRY,
        fragment_records=fragment_records,
        charge=CHARGE,
        multiplicity=MULTIPLICITY,
        basis=BASIS,
    )

    # Explicit CAS(4e,4o)=8q per dimer — strictly < parent 12q
    DIMER_CAS = {"n_active_electrons": 4, "n_active_orbitals": 4}
    print(f"\n--- Generating dimer Hamiltonians ({len(dimer_records)} dimers) ---")
    dimer_hams = []
    for dr in dimer_records:
        print(f"  {dr['name']} (frag {dr['frag_i']}-{dr['frag_j']})...")
        dimer_charge = int(dr.get("charge", CHARGE))
        as_spec = build_active_space_spec(
            mode="explicit",
            n_active_electrons=DIMER_CAS["n_active_electrons"],
            n_active_orbitals=DIMER_CAS["n_active_orbitals"],
            geometry=dr["geometry"],
            basis=BASIS,
            charge=dimer_charge,
            multiplicity=MULTIPLICITY,
        )
        ham = _generate_hamiltonian(
            geometry=dr["geometry"],
            basis=BASIS,
            charge=dimer_charge,
            multiplicity=MULTIPLICITY,
            active_space=as_spec,
        )
        ham["name"] = dr["name"]
        ham["parent_name"] = "iodobenzene_cas12"
        ham["frag_i"] = dr["frag_i"]
        ham["frag_j"] = dr["frag_j"]
        ham["atom_indices"] = dr["atom_indices"]
        print(f"    {ham['n_qubits']}q, {ham['n_pauli_terms']} terms")
        dimer_hams.append(ham)

    # --- Generate parent Hamiltonian ---
    print("\n--- Generating parent Hamiltonian ---")
    parent_as = build_active_space_spec(
        mode="explicit",
        n_active_electrons=6,
        n_active_orbitals=6,
        geometry=PARENT_GEOMETRY,
        basis=BASIS,
        charge=CHARGE,
        multiplicity=MULTIPLICITY,
    )
    parent_ham = _generate_hamiltonian(
        geometry=PARENT_GEOMETRY,
        basis=BASIS,
        charge=CHARGE,
        multiplicity=MULTIPLICITY,
        active_space=parent_as,
    )
    parent_ham["name"] = "iodobenzene_cas12"
    print(f"  {parent_ham['n_qubits']}q, {parent_ham['n_pauli_terms']} terms")

    # --- Compute CCSD/CCSD(T) references ---
    print("\n--- Computing CCSD/CCSD(T) references ---")
    ccsd_refs = {}

    print("  Parent...")
    try:
        ccsd_refs["parent"] = _compute_ccsd(PARENT_GEOMETRY, BASIS, CHARGE, MULTIPLICITY)
        print(f"    HF={ccsd_refs['parent']['hf_energy']:.6f}, "
              f"CCSD={ccsd_refs['parent']['ccsd_energy']:.6f}")
    except Exception as e:
        print(f"    FAILED: {e}")
        ccsd_refs["parent"] = {"error": str(e)}

    for fr in fragment_records:
        print(f"  {fr['name']}...")
        try:
            ccsd_refs[fr["name"]] = _compute_ccsd(
                fr["geometry"], BASIS, int(fr.get("charge", CHARGE)), MULTIPLICITY,
            )
            ref = ccsd_refs[fr["name"]]
            print(f"    HF={ref['hf_energy']:.6f}, CCSD={ref['ccsd_energy']:.6f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            ccsd_refs[fr["name"]] = {"error": str(e)}

    for dr in dimer_records:
        print(f"  {dr['name']}...")
        try:
            ccsd_refs[dr["name"]] = _compute_ccsd(
                dr["geometry"], BASIS, int(dr.get("charge", CHARGE)), MULTIPLICITY,
            )
            ref = ccsd_refs[dr["name"]]
            print(f"    HF={ref['hf_energy']:.6f}, CCSD={ref['ccsd_energy']:.6f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            ccsd_refs[dr["name"]] = {"error": str(e)}

    # --- Save outputs ---
    monomers_file = out_dir / "monomers.json"
    with open(monomers_file, "w") as f:
        json.dump({"records": monomer_hams}, f, indent=2)
    print(f"\nMonomers -> {monomers_file}")

    dimers_file = out_dir / "dimers.json"
    with open(dimers_file, "w") as f:
        json.dump({"records": dimer_hams}, f, indent=2)
    print(f"Dimers -> {dimers_file}")

    parent_file = out_dir / "parent.json"
    with open(parent_file, "w") as f:
        json.dump({"records": [parent_ham]}, f, indent=2)
    print(f"Parent -> {parent_file}")

    ccsd_file = out_dir / "ccsd_refs.json"
    with open(ccsd_file, "w") as f:
        json.dump(ccsd_refs, f, indent=2)
    print(f"CCSD refs -> {ccsd_file}")

    elapsed = time.time() - t_start
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"\nScaling summary:")
    print(f"  Monomers: {[h['n_qubits'] for h in monomer_hams]}q")
    print(f"  Dimers:   {[h['n_qubits'] for h in dimer_hams]}q")
    print(f"  Parent:   {parent_ham['n_qubits']}q")
    max_dimer = max(h["n_qubits"] for h in dimer_hams)
    print(f"  Max circuit: {max_dimer}q < Parent: {parent_ham['n_qubits']}q")
    print(f"  Genuine scaling: {max_dimer < parent_ham['n_qubits']}")


if __name__ == "__main__":
    main()
