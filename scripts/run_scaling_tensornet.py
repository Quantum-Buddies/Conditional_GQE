#!/usr/bin/env python3
"""Run scaling experiments with CUDA-Q tensornet backend on single L40S GPU.

This script:
1. Generates Hamiltonians with larger active spaces
2. Runs GQE baseline with tensornet backend
3. Evaluates H-cGQE generated circuits with tensornet
4. Reports qubit counts, timings, and energy errors

Usage (on AIRE GPU node):
    ssh gpu013
    cd /scratch/kcwp264/Conditional-GQE_materials
    /mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python scripts/run_scaling_tensornet.py
"""

import json
import time
import subprocess
import sys
from pathlib import Path

ROOT = Path("/scratch/kcwp264/Conditional-GQE_materials")
PYTHON = "/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"
CONFIG = ROOT / "configs" / "experiment_scaling.yaml"
HAM_DIR = ROOT / "results" / "data"
HAM_PATH = HAM_DIR / "hamiltonians_scaling.json"
GQE_OUT = ROOT / "results" / "baselines" / "cudaq_gqe_scaling.json"
EVAL_OUT = ROOT / "results" / "eval" / "h_cgqe_evaluation_scaling.json"
OPT_OUT = ROOT / "results" / "eval" / "h_cgqe_optimized_scaling.json"
INFER_OUT = ROOT / "results" / "inference" / "h_cgqe_generated_scaling.json"
RLQF_CKPT = ROOT / "results" / "train" / "h_cgqe_model_rlqf_phase3.pt"

# Molecules to run through the full pipeline
SCALING_MOLECULES = [
    "lih_1.6_full",      # 12 qubits (full active space)
    "lih_1.6_cas8",      # 16 qubits (larger active space)
    "n2_1.1_cas12",      # 24 qubits (large active space)
    "beh2_1.3_full",     # 14 qubits (full)
    "iodobenzene_cas16", # 16 qubits (2x previous)
    "methyl_iodide_cas16", # 16 qubits
    "imeph_cas16",       # 16 qubits
    "phenol_cas16",      # 16 qubits
]


def run_step(name, cmd, cwd=ROOT):
    """Run a pipeline step and report timing."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"CMD: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s")
    if result.returncode != 0:
        print(f"STDERR (last 800 chars): {result.stderr[-800:]}")
        print(f"STDOUT (last 800 chars): {result.stdout[-800:]}")
        return False, elapsed
    print(f"STDOUT (last 500 chars): {result.stdout[-500:]}")
    return True, elapsed


def main():
    # Step 1: Generate Hamiltonians with larger active spaces
    ok, t1 = run_step("Generate Hamiltonians (scaling config)", [
        PYTHON, "src/gqe/data/generate_hamiltonians.py",
        "--config", str(CONFIG),
        "--out", str(HAM_PATH),
    ])
    if not ok:
        print("Hamiltonian generation failed. Check errors above.")
        sys.exit(1)

    # Find the generated Hamiltonians file
    ham_path = HAM_PATH
    if not ham_path.exists():
        print(f"Cannot find generated Hamiltonians at {ham_path}")
        print(f"Searching {HAM_DIR}...")
        for f in HAM_DIR.iterdir():
            print(f"  {f}")
        sys.exit(1)

    with open(ham_path) as f:
        ham_data = json.load(f)

    # Handle both dict and list formats
    if isinstance(ham_data, list):
        ham_dict = {item["name"]: item for item in ham_data}
    else:
        ham_dict = ham_data

    print(f"\nGenerated {len(ham_dict)} Hamiltonians:")
    for name, h in ham_dict.items():
        n_qubits = h.get("n_qubits", "?")
        terms = h.get("terms", h.get("pauli_terms", []))
        n_terms = len(terms) if isinstance(terms, list) else "?"
        print(f"  {name:30s}  qubits={n_qubits:3}  terms={n_terms}")

    # Filter to molecules that exist in the generated data
    available_molecules = list(ham_dict.keys())
    molecules_to_run = [m for m in SCALING_MOLECULES if m in ham_dict]
    if not molecules_to_run:
        print(f"Warning: None of {SCALING_MOLECULES} found in generated data.")
        print(f"Available: {available_molecules}")
        molecules_to_run = available_molecules

    # Determine max qubits for GQE baseline
    max_qubits = max(int(ham_dict[m].get("n_qubits", 0)) for m in molecules_to_run)
    print(f"\nMax qubits in dataset: {max_qubits}")

    # Step 2: Run GQE baseline with tensornet backend
    ok, t2 = run_step("GQE baseline (tensornet)", [
        PYTHON, "src/gqe/baselines/run_cudaq_gqe.py",
        "--ham", str(ham_path),
        "--out", str(GQE_OUT),
        "--target", "tensornet",
        "--max-qubits", str(max_qubits + 1),
    ])
    if not ok:
        print("GQE baseline with tensornet failed. Trying nvidia backend...")
        ok, t2 = run_step("GQE baseline (nvidia fallback)", [
            PYTHON, "src/gqe/baselines/run_cudaq_gqe.py",
            "--ham", str(ham_path),
            "--out", str(GQE_OUT),
            "--max-qubits", str(max_qubits + 1),
        ])

    # Step 3: Inference with existing RLQF model
    infer_cmd = [
        PYTHON, "src/gqe/models/infer_h_cgqe.py",
        "--checkpoint", str(RLQF_CKPT),
        "--hamiltonians", str(ham_path),
        "--out", str(INFER_OUT),
        "--n-samples", "50",
        "--sample",
        "--use-cuda",
        "--max-pauli-len", str(max_qubits),
        "--max-seq-len", "64",
    ] + ["--molecules"] + molecules_to_run

    ok, t3 = run_step("H-cGQE inference (RLQF model)", infer_cmd)
    if not ok:
        print("Inference failed. Check errors above.")
        sys.exit(1)

    # Step 4: Optimize coefficients with tensornet
    ok, t4 = run_step("Optimize coefficients (tensornet)", [
        PYTHON, "src/gqe/eval/optimize_h_cgqe_coefficients.py",
        "--generated", str(INFER_OUT),
        "--hamiltonians", str(ham_path),
        "--out", str(OPT_OUT),
        "--n-sequences", "5",
        "--target", "tensornet",
    ])

    # Step 5: Evaluate with tensornet
    ok, t5 = run_step("Evaluate H-cGQE (tensornet)", [
        PYTHON, "src/gqe/eval/evaluate_h_cgqe.py",
        "--generated", str(INFER_OUT),
        "--hamiltonians", str(ham_path),
        "--out", str(EVAL_OUT),
        "--target", "tensornet",
    ])

    # Summary
    print(f"\n{'='*60}")
    print("SCALING EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Hamiltonian generation:  {t1:.1f}s")
    print(f"GQE baseline:            {t2:.1f}s")
    print(f"H-cGQE inference:        {t3:.1f}s")
    print(f"Coefficient optimization: {t4:.1f}s")
    print(f"Final evaluation:        {t5:.1f}s")
    print(f"Total:                   {t1+t2+t3+t4+t5:.1f}s")

    # Print results if available
    if EVAL_OUT.exists():
        with open(EVAL_OUT) as f:
            eval_data = json.load(f)
        if isinstance(eval_data, list):
            eval_list = eval_data
        else:
            eval_list = eval_data.get("results", eval_data.get("molecules", []))
        print(f"\nEvaluation Results:")
        print(f"{'Molecule':30s} {'Nq':>3s} {'Ref(E)':>12s} {'Best(E)':>12s} {'Err(mHa)':>10s}")
        for item in eval_list:
            name = item.get("molecule", item.get("name", "?"))
            nq = item.get("n_qubits", item.get("n_spin_orbitals", "?"))
            ref = item.get("reference_energy", 0)
            best = item.get("best_generated_energy", 0)
            err = abs(ref - best) * 1000
            print(f"{name:30s} {str(nq):>3s} {ref:12.4f} {best:12.4f} {err:10.2f}")

    if OPT_OUT.exists():
        with open(OPT_OUT) as f:
            opt_data = json.load(f)
        if isinstance(opt_data, list):
            opt_list = opt_data
        else:
            opt_list = opt_data.get("results", opt_data.get("molecules", []))
        print(f"\nOptimized Results:")
        print(f"{'Molecule':30s} {'Nq':>3s} {'Opt(E)':>12s} {'Err(mHa)':>10s} {'Best Ops':>25s}")
        for item in opt_list:
            name = item.get("molecule", item.get("name", "?"))
            nq = item.get("n_qubits", "?")
            best = item.get("best_energy", 0)
            ref = item.get("reference_energy", 0)
            err = abs(ref - best) * 1000
            ops = item.get("best_operators", [])
            print(f"{name:30s} {str(nq):>3s} {best:12.4f} {err:10.2f} {str(ops)[:25]}")

    print(f"\nOutput files:")
    print(f"  Hamiltonians: {ham_path}")
    print(f"  GQE baseline: {GQE_OUT}")
    print(f"  Inference:    {INFER_OUT}")
    print(f"  Optimized:    {OPT_OUT}")
    print(f"  Evaluation:   {EVAL_OUT}")


if __name__ == "__main__":
    main()
