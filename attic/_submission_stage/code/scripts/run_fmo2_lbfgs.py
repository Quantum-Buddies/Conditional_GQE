#!/usr/bin/env python3
"""Run L-BFGS-B coefficient optimization on FMO2 fragments using RL operator sequences.

This closes the gap between H-cGQE (HF-level with zero thetas) and exact FMO2.
The RL model already found good operator sequences; L-BFGS-B optimizes the rotation angles.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_fmo2_lbfgs.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cudaq
    cudaq.set_target("nvidia")

    from src.gqe.eval.optimize_h_cgqe_coefficients import (
        _build_kernel_for_sequence,
        _evaluate_energy,
        _pad_pauli_word,
    )
    from src.gqe.common.hamiltonian_utils import (
        get_active_electron_count,
        hamiltonian_to_spin_operator,
    )
    from scipy.optimize import minimize

    # Load fragments
    with open(ROOT / "results/data/fragments/monomers.json") as f:
        monos = json.load(f)["records"]
    with open(ROOT / "results/data/fragments/dimers.json") as f:
        dimers = json.load(f)["records"]

    # Load RL circuits — use h2 operators for 4q, lih operators for 8q (truncate/pad)
    with open(ROOT / "results/train/h_cgqe_model_qbraid_rl_best_circuits.json") as f:
        rl_data = json.load(f)
    bc = rl_data.get("best_circuits", rl_data)

    # Also load saved thetas from h_cgqe_rl_optimized.json
    with open(ROOT / "results/eval/h_cgqe_rl_optimized.json") as f:
        rl_opt = json.load(f)
    saved_thetas = {}
    for e in rl_opt:
        saved_thetas[e["molecule"]] = {
            "thetas": e.get("best_thetas", []),
            "operators": e.get("best_operators", []),
            "n_qubits": e["n_qubits"],
        }

    print("=== FMO2 L-BFGS-B Coefficient Optimization ===\n")

    def optimize_fragment(record, operators, init_thetas=None, label=""):
        n_qubits = int(record["n_qubits"])
        n_electrons = get_active_electron_count(record)
        spin_ham = hamiltonian_to_spin_operator(record)

        padded = [_pad_pauli_word(w, n_qubits) for w in operators]
        pauli_words = [cudaq.pauli_word(w) for w in padded]

        kernel, theta_symbols = _build_kernel_for_sequence(n_qubits, n_electrons, operators)

        def objective(thetas):
            return _evaluate_energy(thetas, kernel, spin_ham, n_qubits, n_electrons, pauli_words)

        # Initial guess
        if init_thetas is not None and len(init_thetas) == len(operators):
            x0 = np.array(init_thetas, dtype=float)
        else:
            x0 = np.zeros(len(operators))

        # Run L-BFGS-B (single start, fast)
        t0 = time.time()
        result = minimize(
            objective, x0, method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-8},
        )
        best_e = float(result.fun)
        best_thetas = result.x.copy()

        elapsed = time.time() - t0
        hf_e = objective(np.zeros(len(operators)))
        print(f"  {label}: HF={hf_e:.6f}, L-BFGS={best_e:.6f}, delta={abs(hf_e-best_e)*1000:.1f}mHa ({elapsed:.1f}s)")
        return best_e, best_thetas.tolist()

    # --- Monomers (4q) — use h2 operators + saved thetas ---
    print("--- Monomers (4q, h2 RL operators + saved thetas) ---")
    h2_ops = saved_thetas.get("h2", {}).get("operators", bc["h2"]["operators"])
    h2_thetas = saved_thetas.get("h2", {}).get("thetas", None)

    mono_results = []
    for r in monos:
        # Use h2 operators (4q) — should match
        ops = h2_ops if h2_ops else bc["h2"]["operators"]
        # If the number of operators doesn't match, fall back to RL best circuits
        if len(ops) != len(h2_thetas or []):
            ops = bc["h2"]["operators"]
            h2_thetas = None
        E, thetas = optimize_fragment(r, ops, h2_thetas, label=r["name"])
        mono_results.append({"name": r["name"], "energy": E, "thetas": thetas, "operators": ops})

    # --- Dimers (8q) — use lih operators (12q, truncate) or n2_1.1_631g_cas8 (8q) ---
    print("\n--- Dimers (8q) ---")
    # Find 8q RL circuits
    rl_8q = [(mol, info) for mol, info in bc.items()
             if isinstance(info, dict) and info.get("n_qubits") == 8]
    if rl_8q:
        rl_8q.sort(key=lambda x: abs(x[1].get("energy", 0) - x[1].get("hf_energy", 0)))
        best_8q_mol, best_8q_info = rl_8q[0]
        print(f"  Using 8q RL circuit from {best_8q_mol}")
        dimer_ops = best_8q_info["operators"]
    else:
        # Use h2 operators — they work on any qubit count (padded)
        print("  No 8q RL circuit found, using h2 operators (padded)")
        dimer_ops = bc["h2"]["operators"]

    dim_results = []
    for r in dimers:
        E, thetas = optimize_fragment(r, dimer_ops, None, label=r["name"])
        dim_results.append({
            "name": r["name"],
            "frag_i": r.get("frag_i", 0),
            "frag_j": r.get("frag_j", 1),
            "energy": E,
            "thetas": thetas,
            "operators": dimer_ops,
        })

    # --- FMO2 Reassembly ---
    mono_E = [r["energy"] for r in mono_results]
    dim_E = {}
    for r in dim_results:
        pk = f"{r['frag_i']}_{r['frag_j']}"
        dim_E[pk] = r["energy"]

    e_mono = sum(mono_E)
    e_pair = sum(E_ij - mono_E[i] - mono_E[j]
                 for pk, E_ij in dim_E.items()
                 for i, j in [map(int, pk.split("_"))])
    e_fmo2 = e_mono + e_pair

    # Load parent exact for comparison
    with open(ROOT / "results/phase3_final/fmo/fmo2_exact_3frag.json") as f:
        exact = json.load(f)
    parent_E = exact["parent_energy"]
    fmo2_exact = exact["fmo2_energy"]

    err_vs_parent = abs(e_fmo2 - parent_E) * 1000
    err_vs_exact = abs(e_fmo2 - fmo2_exact) * 1000

    print(f"\n=== FMO2 L-BFGS-B Results ===")
    print(f"  FMO2 L-BFGS:      {e_fmo2:.6f}")
    print(f"  FMO2 exact:       {fmo2_exact:.6f}")
    print(f"  Parent exact:     {parent_E:.6f}")
    print(f"  vs parent:        {err_vs_parent:.3f} mHa")
    print(f"  vs exact FMO2:    {err_vs_exact:.3f} mHa")

    # Save
    out = {
        "method": "hcgqe_rl_lbfgsb",
        "molecule": "iodobenzene_cas12",
        "n_fragments": 3,
        "fmo2_energy": e_fmo2,
        "monomer_energies": mono_E,
        "dimer_energies": dim_E,
        "parent_energy": parent_E,
        "fmo2_exact": fmo2_exact,
        "error_vs_parent_mha": err_vs_parent,
        "error_vs_exact_fmo2_mha": err_vs_exact,
        "max_dimer_qubits": 8,
        "parent_qubits": 12,
        "genuine_scaling": True,
        "fragment_details": {
            "monomers": [{"name": r["name"], "energy": r["energy"], "n_thetas": len(r["thetas"])} for r in mono_results],
            "dimers": [{"name": r["name"], "energy": r["energy"], "n_thetas": len(r["thetas"])} for r in dim_results],
        },
    }
    out_path = ROOT / "results/phase3_final/fmo/fmo2_hcgqe_lbfgs_3frag.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
