#!/usr/bin/env python3
"""Compute HF, CCSD, FCI classical baseline energies for all GPU benchmark molecules.

Reads consolidated_results_gic2026.json to get molecule geometries and H-cGQE/GQE
energies, then computes Hartree-Fock, CCSD, and FCI (where feasible) using PySCF.
Merges with existing VQE baseline data and writes updated classical_baseline_comparison.json.

Usage:
    python scripts/compute_classical_baselines.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from pyscf import gto, scf, fci, cc, mcscf
except ImportError:
    print("ERROR: PySCF not installed. Run: pip install pyscf")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
CONSOLIDATED = ROOT / "results/phase3_final/consolidated_results_gic2026.json"
VQE_FILE = ROOT / "results/baselines/cudaq_vqe.json"
ADAPT_FILE = ROOT / "results/baselines/adapt_vqe_h2.json"
EXACT_DIAG = ROOT / "results/baselines/exact_diagonalization.json"
OUTPUT = ROOT / "results/phase3_final/classical_baseline_comparison.json"


def _geom_to_str(geom_list):
    lines = []
    for atom, coords in geom_list:
        x, y, z = coords
        lines.append(f"{atom} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines)


def compute_classical_energies(geometry, basis="sto-3g", charge=0, multiplicity=1, n_qubits=4, active_space=None):
    """Compute HF, CCSD, FCI/CASCI for a molecule."""
    spin = multiplicity - 1
    mol = gto.M(atom=_geom_to_str(geometry), basis=basis, charge=charge, spin=spin, verbose=0)

    # Hartree-Fock
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    mf.kernel()
    hf_energy = float(mf.e_tot)

    # CCSD
    ccsd_energy = None
    ccsd_t_total = None
    try:
        if spin == 0:
            mycc = cc.RCCSD(mf)
        else:
            mycc = cc.UCCSD(mf)
        mycc.kernel()
        ccsd_energy = float(mycc.e_tot)
        try:
            ccsd_t = mycc.ccsd_t()
            ccsd_t_total = float(mycc.e_tot + ccsd_t)
        except Exception:
            pass
    except Exception as e:
        print(f"    CCSD failed: {e}")

    # FCI or CASCI
    fci_energy = None
    n_active_e = None
    n_active_o = None
    if active_space:
        n_active_e = active_space.get("n_active_electrons")
        n_active_o = active_space.get("n_active_orbitals")

    if n_active_e is not None and n_active_o is not None:
        try:
            mc = mcscf.CASCI(mf, n_active_o, n_active_e)
            mc.kernel()
            fci_energy = float(mc.e_tot)
        except Exception as e:
            print(f"    CASCI failed: {e}")
    elif n_qubits <= 20:
        try:
            if spin == 0:
                cisolver = fci.FCI(mol, mf.mo_coeff)
            else:
                cisolver = fci.FCI(mol, mf.mo_coeff, singlet=False)
            fci_energy = float(cisolver.kernel()[0])
        except Exception as e:
            print(f"    FCI failed: {e}")

    return {
        "hf_energy": hf_energy,
        "ccsd_energy": ccsd_energy,
        "ccsd_t_total": ccsd_t_total,
        "fci_energy": fci_energy,
    }


def load_vqe_data():
    """Load existing VQE baseline results."""
    vqe_map = {}
    if VQE_FILE.exists():
        with VQE_FILE.open() as f:
            data = json.load(f)
        for r in data.get("results", []):
            name = r.get("system", "")
            vqe_map[name] = {
                "hea_vqe_energy": r.get("baseline_energy"),
                "hea_vqe_error_mha": abs(r.get("delta_energy", 0)) * 1000 if r.get("delta_energy") else None,
            }
    return vqe_map


def load_adapt_data():
    """Load ADAPT-VQE results."""
    adapt_map = {}
    if ADAPT_FILE.exists():
        with ADAPT_FILE.open() as f:
            data = json.load(f)
        for r in data.get("results", []):
            name = r.get("system", r.get("molecule", ""))
            adapt_map[name] = {
                "adapt_vqe_energy": r.get("baseline_energy"),
                "adapt_vqe_error_mha": abs(r.get("delta_energy", 0)) * 1000 if r.get("delta_energy") else None,
            }
    return adapt_map


def load_exact_diag():
    """Load exact diagonalization FCI references."""
    ed_map = {}
    if EXACT_DIAG.exists():
        with EXACT_DIAG.open() as f:
            data = json.load(f)
        if isinstance(data, list):
            for r in data:
                name = r.get("molecule", r.get("system", ""))
                ed_map[name] = r.get("fci_energy") or r.get("exact_energy")
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    ed_map[k] = v.get("fci_energy") or v.get("exact_energy")
    return ed_map


def main():
    # Load consolidated GPU results
    with CONSOLIDATED.open() as f:
        consolidated = json.load(f)

    gpu_results = consolidated.get("gpu_benchmark", [])
    print(f"Loaded {len(gpu_results)} GPU benchmark molecules")

    # Load existing baselines
    vqe_data = load_vqe_data()
    adapt_data = load_adapt_data()
    ed_data = load_exact_diag()
    print(f"VQE data: {list(vqe_data.keys())}")
    print(f"ADAPT data: {list(adapt_data.keys())}")
    print(f"Exact diag: {list(ed_data.keys())}")

    # Load config for active spaces
    import yaml
    config_path = ROOT / "configs/gic2026_molecules.yaml"
    config_molecules = {}
    if config_path.exists():
        with config_path.open() as f:
            cfg = yaml.safe_load(f)
        for m in cfg.get("dataset", {}).get("molecules", []):
            config_molecules[m["name"]] = m

    # Compute classical energies for each GPU molecule
    molecules_out = []
    for r in tqdm(gpu_results, desc="Computing classical baselines"):
        name = r["molecule"]
        n_qubits = r["n_qubits"]
        h_cgqe_energy = r.get("h_cgqe_optimized_energy")
        gqe_energy = r.get("cudaq_gqe_energy")
        ref_energy = r.get("reference_energy")

        # Get geometry from config
        cfg_mol = config_molecules.get(name, {})
        geometry = cfg_mol.get("geometry")
        basis = cfg_mol.get("basis", "sto-3g")
        charge = cfg_mol.get("charge", 0)
        multiplicity = cfg_mol.get("multiplicity", 1)
        active_space = cfg_mol.get("active_space")

        if geometry is None:
            print(f"  WARNING: No geometry for {name}, using reference energy only")
            entry = {
                "molecule": name,
                "n_qubits": n_qubits,
                "fci_energy": ref_energy,
                "hcgqe_energy": h_cgqe_energy,
                "hcgqe_error_mha": r.get("error_vs_reference_mha", 0),
                "chemical_accuracy": r.get("error_vs_reference_mha", 999) < 1.6,
                "cudaq_gqe_energy": gqe_energy,
                "cudaq_gqe_error_mha": abs(gqe_energy - ref_energy) * 1000 if gqe_energy and ref_energy else None,
            }
            molecules_out.append(entry)
            continue

        print(f"\n  {name} ({n_qubits}q)...")
        classical = compute_classical_energies(
            geometry, basis, charge, multiplicity, n_qubits, active_space
        )

        # Use FCI from computation or from exact diag or from reference
        fci_e = classical["fci_energy"]
        if fci_e is None and name in ed_data:
            fci_e = ed_data[name]
        if fci_e is None:
            fci_e = ref_energy  # Fall back to consolidated reference

        hf_e = classical["hf_energy"]
        ccsd_e = classical["ccsd_energy"]
        ccsd_t = classical["ccsd_t_total"]

        # Compute errors vs FCI
        hf_error = abs(hf_e - fci_e) * 1000 if fci_e else None
        ccsd_error = abs(ccsd_e - fci_e) * 1000 if ccsd_e and fci_e else None
        hcgqe_error = abs(h_cgqe_energy - fci_e) * 1000 if h_cgqe_energy and fci_e else None
        gqe_error = abs(gqe_energy - fci_e) * 1000 if gqe_energy and fci_e else None

        # Merge VQE data
        vqe = vqe_data.get(name, {})
        adapt = adapt_data.get(name, {})

        entry = {
            "molecule": name,
            "n_qubits": n_qubits,
            "hf_energy": hf_e,
            "hf_error_mha": hf_error,
            "ccsd_energy": ccsd_e,
            "ccsd_error_mha": ccsd_error,
            "ccsd_t_energy": ccsd_t,
            "fci_energy": fci_e,
            "hcgqe_energy": h_cgqe_energy,
            "hcgqe_error_mha": hcgqe_error,
            "chemical_accuracy": hcgqe_error is not None and hcgqe_error < 1.6,
            "cudaq_gqe_energy": gqe_energy,
            "cudaq_gqe_error_mha": gqe_error,
            "hea_vqe_energy": vqe.get("hea_vqe_energy"),
            "hea_vqe_error_mha": vqe.get("hea_vqe_error_mha"),
            "adapt_vqe_energy": adapt.get("adapt_vqe_energy"),
            "adapt_vqe_error_mha": adapt.get("adapt_vqe_error_mha"),
        }
        molecules_out.append(entry)

        ccsd_str = f"{ccsd_e:.6f}" if ccsd_e is not None else "N/A"
        fci_str = f"{fci_e:.6f}" if fci_e is not None else "N/A"
        print(f"    HF={hf_e:.6f}  CCSD={ccsd_str}  FCI={fci_str}")
        hf_err_str = f"{hf_error:.2f}" if hf_error is not None else "N/A"
        hcgqe_err_str = f"{hcgqe_error:.2f}" if hcgqe_error is not None else "N/A"
        print(f"    HF err={hf_err_str} mHa  H-cGQE err={hcgqe_err_str} mHa")

    # Build output
    output = {
        "description": "Classical and quantum baseline comparison for GIC 2026 Phase 3",
        "methods_compared": ["HF", "CCSD", "CCSD(T)", "HEA-VQE", "ADAPT-VQE", "CUDA-Q GQE", "H-cGQE (SFT+RL+opt)"],
        "n_molecules": len(molecules_out),
        "molecules": molecules_out,
        "ccsd_baselines": {},
        "ch3i_headline": {
            "hcgqe_gpu_error_mha": 0.63,
            "qpu_sqd_error_mha": 13.95,
            "note": "GPU = noiseless RL-optimized circuit; QPU = same circuit on Rigetti with SQD post-processing"
        },
    }

    # Preserve CCSD FMO2 data from existing file
    if OUTPUT.exists():
        with OUTPUT.open() as f:
            old = json.load(f)
        if "ccsd_baselines" in old:
            output["ccsd_baselines"] = old["ccsd_baselines"]

    with OUTPUT.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(molecules_out)} molecules to {OUTPUT}")

    # Summary table
    print(f"\n{'Molecule':25s} {'Q':>3s} {'HF(mHa)':>10s} {'CCSD(mHa)':>10s} {'GQE(mHa)':>10s} {'H-cGQE':>10s} {'CHEM':>5s}")
    print("-" * 75)
    for m in molecules_out:
        name = m["molecule"]
        q = m["n_qubits"]
        hf = f"{m.get('hf_error_mha',0):.2f}" if m.get("hf_error_mha") else "—"
        ccsd = f"{m.get('ccsd_error_mha',0):.2f}" if m.get("ccsd_error_mha") else "—"
        gqe = f"{m.get('cudaq_gqe_error_mha',0):.2f}" if m.get("cudaq_gqe_error_mha") else "—"
        hcgqe = f"{m.get('hcgqe_error_mha',0):.2f}" if m.get("hcgqe_error_mha") else "—"
        chem = "YES" if m.get("chemical_accuracy") else ""
        print(f"{name:25s} {q:>3d} {hf:>10s} {ccsd:>10s} {gqe:>10s} {hcgqe:>10s} {chem:>5s}")


if __name__ == "__main__":
    main()
