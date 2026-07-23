#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]

# Load Hamiltonians
ham_path = repo_root / "results/data/hamiltonians.json"
with open(ham_path) as f:
    hamiltonians = json.load(f)

ham_dict = {item["name"]: item for item in hamiltonians}

# Load exact diagonalization references
exact_path = repo_root / "results/baselines/exact_diagonalization.json"
exact_data = {}
if exact_path.exists():
    with open(exact_path) as f:
        exact_json = json.load(f)
        for r in exact_json.get("results", []):
            exact_data[r["system"]] = r.get("exact_energy")

# Load CUDA-Q GQE baseline results
cudaq_gqe_path = repo_root / "results/baselines/cudaq_gqe.json"
cudaq_gqe_data = {}
if cudaq_gqe_path.exists():
    with open(cudaq_gqe_path) as f:
        cudaq_json = json.load(f)
        for r in cudaq_json.get("results", []):
            cudaq_gqe_data[r["system"]] = r

# Load H-cGQE optimized results
h_cgqe_opt_path = repo_root / "results/eval/h_cgqe_optimized.json"
h_cgqe_opt_data = {}
if h_cgqe_opt_path.exists():
    with open(h_cgqe_opt_path) as f:
        h_cgqe_opt_json = json.load(f)
        for r in h_cgqe_opt_json:
            h_cgqe_opt_data[r["molecule"]] = r

# Load H-cGQE evaluation results
h_cgqe_eval_path = repo_root / "results/eval/h_cgqe_evaluation.json"
h_cgqe_eval_data = {}
if h_cgqe_eval_path.exists():
    with open(h_cgqe_eval_path) as f:
        h_cgqe_eval_json = json.load(f)
        for r in h_cgqe_eval_json:
            h_cgqe_eval_data[r["molecule"]] = r

# Load GQE comparison results
gqe_comp_path = repo_root / "results/baselines/gqe_comparison.json"
gqe_comp_data = {}
if gqe_comp_path.exists():
    with open(gqe_comp_path) as f:
        gqe_comp_json = json.load(f)
        for r in gqe_comp_json.get("comparison", []):
            gqe_comp_data[r["system"]] = r

# Diversity function
def compute_diversity(file_path):
    if not file_path.exists():
        return {}
    with open(file_path) as f:
        data = json.load(f)
    results = {}
    for entry in data:
        mol = entry["molecule"]
        seqs = entry["generated_sequences"]
        op_lists = [s["operators"] for s in seqs]
        unique = len(set(tuple(ops) for ops in op_lists))
        div = unique / len(op_lists) if op_lists else 0.0
        results[mol] = {
            "n_seqs": len(op_lists),
            "n_unique": unique,
            "diversity": div,
            "mean_len": float(np.mean([len(ops) for ops in op_lists])) if op_lists else 0
        }
    return results

sup_div = compute_diversity(repo_root / "results/inference/supervised_generated.json")
chm_div = compute_diversity(repo_root / "results/inference/chemeleon2_generated.json")

print("=== BENCHMARK EVALUATION METRICS SUMMARY ===")
print(f"{'Molecule':<12} | {'Qubits':<6} | {'FCI Ref (Ha)':<14} | {'CUDA-Q GQE (Ha)':<16} | {'H-cGQE Opt (Ha)':<16} | {'CUDA-Q Err (mHa)':<16} | {'H-cGQE Err (mHa)':<16} | {'Diversity':<10}")
print("-" * 115)

for name in sorted(ham_dict.keys()):
    rec = ham_dict[name]
    n_qubits = rec["n_qubits"]
    ref_e = exact_data.get(name)
    if ref_e is None and name in gqe_comp_data:
        ref_e = gqe_comp_data[name].get("reference_energy")
    
    cq_res = cudaq_gqe_data.get(name, {})
    cq_e = cq_res.get("baseline_energy")
    if cq_e is None and name in gqe_comp_data:
        cq_e = gqe_comp_data[name].get("baseline_energy")

    hcgqe_res = h_cgqe_opt_data.get(name, {})
    hcgqe_e = hcgqe_res.get("best_energy")

    cq_err = abs(cq_e - ref_e) * 1000 if (cq_e is not None and ref_e is not None) else None
    hcgqe_err = abs(hcgqe_e - ref_e) * 1000 if (hcgqe_e is not None and ref_e is not None) else None

    div_val = chm_div.get(name, {}).get("diversity", 0.0)

    ref_str = f"{ref_e:.6f}" if ref_e is not None else "N/A"
    cq_str = f"{cq_e:.6f}" if cq_e is not None else "N/A"
    hc_str = f"{hcgqe_e:.6f}" if hcgqe_e is not None else "N/A"
    cq_err_str = f"{cq_err:.3f}" if cq_err is not None else "N/A"
    hc_err_str = f"{hcgqe_err:.3f}" if hcgqe_err is not None else "N/A"

    print(f"{name:<12} | {n_qubits:<6} | {ref_str:<14} | {cq_str:<16} | {hc_str:<16} | {cq_err_str:<16} | {hc_err_str:<16} | {div_val:<10.3f}")
