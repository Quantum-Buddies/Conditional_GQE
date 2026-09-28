"""FMO2 error decomposition: solver error vs fragmentation error.

ΔE_solver = E_hcgqe_fmo2 - E_exact_fragment_fmo2
ΔE_fragmentation = E_exact_fragment_fmo2 - E_parent_reference
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="FMO2 error decomposition")
    parser.add_argument("--exact", type=Path, required=True, help="Exact-fragment FMO2 result JSON")
    parser.add_argument("--hcgqe", type=Path, required=True, help="H-cGQE FMO2 result JSON")
    parser.add_argument("--parent-reference", type=float, default=None, help="Parent molecule exact reference energy")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(args.exact) as f:
        exact = json.load(f)
    with open(args.hcgqe) as f:
        hcgqe = json.load(f)

    e_exact_fmo2 = exact["fmo2_energy"]
    e_hcgqe_fmo2 = hcgqe["fmo2_energy"]

    # Parent reference
    if args.parent_reference is not None:
        e_parent = args.parent_reference
    else:
        # Try to load from hamiltonians
        from src.gqe.common.hamiltonian_utils import load_hamiltonian_records, find_record_by_name
        records = load_hamiltonian_records(Path("results/data/hamiltonians_phase3.json/hamiltonians.json"))
        parent = find_record_by_name(records, "imeph")
        if parent:
            from src.gqe.eval.run_fmo2 import exact_energy_from_hamiltonian
            e_parent = exact_energy_from_hamiltonian(parent)
        else:
            e_parent = e_exact_fmo2  # fallback

    dE_solver = e_hcgqe_fmo2 - e_exact_fmo2
    dE_fragmentation = e_exact_fmo2 - e_parent
    dE_total = e_hcgqe_fmo2 - e_parent

    result = {
        "parent_reference_energy": e_parent,
        "exact_fmo2_energy": e_exact_fmo2,
        "hcgqe_fmo2_energy": e_hcgqe_fmo2,
        "dE_solver_mha": dE_solver * 1000,
        "dE_fragmentation_mha": dE_fragmentation * 1000,
        "dE_total_mha": dE_total * 1000,
        "interpretation": {
            "solver_error": f"H-cGQE vs exact within fragments: {dE_solver*1000:.3f} mHa",
            "fragmentation_error": f"FMO2 decomposition vs parent: {dE_fragmentation*1000:.3f} mHa",
            "total_error": f"H-cGQE FMO2 vs parent reference: {dE_total*1000:.3f} mHa",
        }
    }

    print(f"Parent reference:    {e_parent:.6f} Ha")
    print(f"Exact FMO2:          {e_exact_fmo2:.6f} Ha")
    print(f"H-cGQE FMO2:         {e_hcgqe_fmo2:.6f} Ha")
    print(f"")
    print(f"ΔE_solver:           {dE_solver*1000:.3f} mHa")
    print(f"ΔE_fragmentation:    {dE_fragmentation*1000:.3f} mHa")
    print(f"ΔE_total:            {dE_total*1000:.3f} mHa")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
