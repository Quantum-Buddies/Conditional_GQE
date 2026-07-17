#!/usr/bin/env python3
"""Test corrected Hamiltonian energy evaluation on AWS Braket SV1 simulator.

Free for first minute per task. Validates that the QWC-grouped term-by-term
Pauli measurement approach produces energies consistent with GPU-optimized references.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
)
from src.gqe.eval.qbraid_backend import evaluate_energy_qbraid_batched


def main() -> None:
    ham_path = ROOT / "results/data/hamiltonians_merged.json"
    opt_path = ROOT / "results/eval/h_cgqe_uccsd_optimized.json"

    print("=== Free qBraid Simulator Energy Validation ===")
    print(f"  Hamiltonians: {ham_path}")
    print(f"  Optimized:    {opt_path}")
    print()

    records = load_hamiltonian_records(ham_path)
    with open(opt_path) as f:
        optimized = json.load(f)

    # Test molecules: H2 (4q) and LiH (12q) — both fit in 29q IonQ free sim
    test_molecules = ["h2_0.74", "lih_1.6_full"]
    device = "aws:aws:sim:sv1"  # AWS Braket SV1 (free for first min/task, 34q)
    shots = 4096

    results = []
    for mol_name in test_molecules:
        print(f"\n--- {mol_name} ---")
        try:
            record = find_record_by_name(records, mol_name)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        # Find optimized entry
        mol_opt = None
        for entry in optimized:
            if entry.get("molecule") == mol_name:
                mol_opt = entry
                break
        if mol_opt is None:
            print(f"  SKIP: No optimized data found")
            continue

        operators = mol_opt.get("best_operators", [])
        thetas = mol_opt.get("best_thetas", [])
        gpu_energy = mol_opt.get("best_energy")
        ref_energy = mol_opt.get("reference_energy_hartree", mol_opt.get("reference_energy"))

        n_qubits = int(record["n_qubits"])
        n_terms = len(record.get("terms", []))
        print(f"  Qubits: {n_qubits}, Hamiltonian terms: {n_terms}")
        print(f"  Operators: {operators}")
        print(f"  Thetas: {thetas}")
        print(f"  GPU energy: {gpu_energy:.6f} Ha")
        if ref_energy is not None:
            print(f"  Reference:  {ref_energy:.6f} Ha")

        # Run on free qBraid simulator
        print(f"  Submitting to {device} with {shots} shots/term...")
        try:
            result = evaluate_energy_qbraid_batched(
                record,
                operators,
                theta_values=np.asarray(thetas),
                device=device,
                shots=shots,
                submit_only=False,
            )

            sim_energy = result["energy"]
            print(f"  Simulator energy: {sim_energy:.6f} Ha")

            if gpu_energy is not None:
                diff_mha = abs(sim_energy - gpu_energy) * 1000
                print(f"  |sim - GPU|: {diff_mha:.3f} mHa")

            if ref_energy is not None:
                sim_err = abs(sim_energy - ref_energy) * 1000
                gpu_err = abs(gpu_energy - ref_energy) * 1000
                print(f"  sim error: {sim_err:.3f} mHa | GPU error: {gpu_err:.3f} mHa")

            # Print per-term breakdown for first few terms
            term_exps = result.get("term_expectations", {})
            print(f"  Term expectations ({len(term_exps)} terms):")
            for word, info in list(term_exps.items())[:5]:
                coeff = info["coeff_real"]
                exp = info["expectation"]
                contrib = coeff * exp
                print(f"    {word:20s}  coeff={coeff:+.6f}  <P>={exp:+.6f}  contrib={contrib:+.6f}")
            if len(term_exps) > 5:
                print(f"    ... ({len(term_exps) - 5} more terms)")

            results.append({
                "molecule": mol_name,
                "n_qubits": n_qubits,
                "n_terms": n_terms,
                "operators": operators,
                "thetas": thetas,
                "gpu_energy": gpu_energy,
                "sim_energy": sim_energy,
                "reference_energy": ref_energy,
                "diff_mha": abs(sim_energy - gpu_energy) * 1000 if gpu_energy else None,
                "device": result.get("device"),
                "shots": result.get("shots"),
                "runtime_seconds": result.get("runtime_seconds"),
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"molecule": mol_name, "error": str(e)})

    # Summary
    print("\n\n=== Summary ===")
    print(f"{'Molecule':20s} {'Qubits':>6s} {'GPU Energy':>12s} {'Sim Energy':>12s} {'|diff| mHa':>10s}")
    print("-" * 64)
    for r in results:
        if "error" in r:
            print(f"{r['molecule']:20s}  ERROR: {r['error'][:40]}")
            continue
        print(
            f"{r['molecule']:20s} {r['n_qubits']:>6d} "
            f"{r['gpu_energy']:>12.6f} {r['sim_energy']:>12.6f} "
            f"{r['diff_mha']:>10.3f}"
        )

    # Save results
    out_path = ROOT / "results/eval/simulator_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
