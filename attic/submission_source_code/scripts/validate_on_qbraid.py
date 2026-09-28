#!/usr/bin/env python3
"""Validation script for judges to verify ground state energies on qBraid simulator.

Loads H-cGQE optimized operator sequences and coefficients, runs them on qBraid's
free QIR simulator (up to 30 qubits, zero credits), and outputs a validation report.

Usage:
    python scripts/validate_on_qbraid.py \
        --hamiltonians results/data/hamiltonians_scaling.json/hamiltonians.json \
        --optimized results/eval/h_cgqe_uccsd_optimized.json \
        --out results/eval/qbraid_validation_report.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
)
from src.gqe.eval.qbraid_backend import evaluate_energy_qbraid_batched


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate H-cGQE energies on qBraid simulator")
    parser.add_argument(
        "--hamiltonians",
        type=Path,
        default=PROJECT_ROOT / "results" / "data" / "hamiltonians_scaling.json" / "hamiltonians.json",
        help="Path to Hamiltonians JSON record file"
    )
    parser.add_argument(
        "--optimized",
        type=Path,
        default=PROJECT_ROOT / "results" / "eval" / "h_cgqe_uccsd_optimized.json",
        help="Path to optimized results JSON file"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "eval" / "qbraid_validation_report.json",
        help="Path to output validation report JSON"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=2000,
        help="Number of simulator shots per term (default: 2000)"
    )
    parser.add_argument(
        "--molecule",
        type=str,
        default=None,
        help="Specific molecule to validate (default: None, validate all)"
    )
    args = parser.parse_args()

    # Verify input paths
    args.hamiltonians = args.hamiltonians.resolve()
    args.optimized = args.optimized.resolve()
    args.out = args.out.resolve()
    if not args.hamiltonians.exists():
        sys.exit(f"Error: Hamiltonians file not found at {args.hamiltonians}")
    if not args.optimized.exists():
        sys.exit(f"Error: Optimized result file not found at {args.optimized}")

    print("================================================================================")
    print("H-cGQE qBraid Reproducibility Validation Suite")
    print("================================================================================")
    print(f"Hamiltonians: {args.hamiltonians.relative_to(PROJECT_ROOT)}")
    print(f"Optimized:    {args.optimized.relative_to(PROJECT_ROOT)}")
    print(f"Simulator:    qbraid:qbraid:sim:qir-sv (Free Target)")
    print(f"Shots:        {args.shots}")
    print("--------------------------------------------------------------------------------")

    # Load data
    ham_records = load_hamiltonian_records(args.hamiltonians)
    with args.optimized.open("r", encoding="utf-8") as f:
        optimized_data = json.load(f)

    # If it is a list of results (optimized format)
    if isinstance(optimized_data, list):
        entries = optimized_data
    elif isinstance(optimized_data, dict) and "results" in optimized_data:
        entries = optimized_data["results"]
    else:
        sys.exit("Error: Invalid optimized data format (expected list of results).")

    if args.molecule:
        entries = [e for e in entries if e.get("molecule") == args.molecule]
        if not entries:
            sys.exit(f"Error: Molecule '{args.molecule}' not found in optimized results.")

    print(f"Loaded {len(entries)} molecules for validation.")

    report_results = []
    
    for idx, entry in enumerate(entries):
        molecule = entry.get("molecule")
        if not molecule:
            continue

        print(f"\n[{idx+1}/{len(entries)}] Validating {molecule}...")
        
        try:
            mol_record = find_record_by_name(ham_records, molecule)
        except ValueError:
            print(f"  Warning: No Hamiltonian record found for {molecule}, skipping.")
            continue

        best_seq = entry.get("best_sequence")
        if not best_seq:
            # Fallback if optimized format varies
            best_seq = entry
            
        operators = best_seq.get("best_operators", best_seq.get("operators", []))
        thetas = best_seq.get("best_thetas", best_seq.get("thetas", []))

        if not operators:
            print("  Warning: No operator sequence found, skipping.")
            continue

        print(f"  Circuit structure: {len(operators)} operators, {mol_record['n_qubits']} qubits.")

        try:
            # Run on free state-vector simulator
            res = evaluate_energy_qbraid_batched(
                molecule_record=mol_record,
                operators=operators,
                theta_values=np.asarray(thetas),
                device="qbraid:qbraid:sim:qir-sv",
                shots=args.shots,
                submit_only=False
            )
            
            assert isinstance(res, dict)
            sim_energy = res["energy"]
            opt_energy = best_seq.get("best_energy", best_seq.get("energy", 0.0))
            diff = abs(sim_energy - opt_energy)

            print(f"  --> Local Optimized Energy: {opt_energy:.6f} Ha")
            print(f"  --> qBraid Sim Energy:     {sim_energy:.6f} Ha")
            print(f"  --> Difference:            {diff:.6f} Ha")

            report_results.append({
                "molecule": molecule,
                "n_qubits": int(mol_record["n_qubits"]),
                "optimized_energy": opt_energy,
                "qbraid_sim_energy": sim_energy,
                "difference": diff,
                "status": "SUCCESS"
            })
            
        except Exception as e:
            print(f"  Validation FAILED: {e}")
            report_results.append({
                "molecule": molecule,
                "status": "FAILED",
                "error": str(e)
            })

    # Save validation report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "simulator": "qbraid:qbraid:sim:qir-sv",
        "shots": args.shots,
        "results": report_results
    }
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Molecule':25s} {'Qubits':>6s} {'Opt Energy':>14s} {'Sim Energy':>14s} {'Status':>10s}")
    print("-" * 80)
    for r in report_results:
        status = r["status"]
        if status == "SUCCESS":
            print(f"{r['molecule']:25s} {r['n_qubits']:6d} {r['optimized_energy']:14.6f} {r['qbraid_sim_energy']:14.6f} {status:>10s}")
        else:
            print(f"{r['molecule']:25s} {'-':6s} {'-':14s} {'-':14s} {status:>10s}")
    print("-" * 80)
    print(f"Saved validation report to: {args.out}")


if __name__ == "__main__":
    main()
