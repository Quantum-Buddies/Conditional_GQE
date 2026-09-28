#!/usr/bin/env python3
"""Evaluate clever and transparent surrogates for the RL reward."""

import json
import numpy as np
import scipy.stats as stats
from pathlib import Path
from multiprocessing import Pool
from functools import partial

# Import CUDA-Q evaluation logic
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from gqe.eval.optimize_h_cgqe_coefficients import _optimize_coefficients, _evaluate_fixed_theta_energy
from scipy.optimize import minimize
import cudaq

RESULTS_PATH = Path("results/eval/verify_rl_proxy_iodobenzene.json")
MOL_PATH = Path("results/data/hamiltonians.json")

# Helper to compute gradient via parameter shift rule
def _compute_gradient_norm(molecule_record, operators, base_theta=0.0):
    """Computes the L1 norm of the gradient at a base theta using parameter shift."""
    try:
        from gqe.eval.optimize_h_cgqe_coefficients import _build_kernel_for_sequence, _evaluate_energy
        from gqe.eval.evaluate_h_cgqe import get_active_electron_count, hamiltonian_to_spin_operator
        
        n_qubits = molecule_record["n_qubits"]
        n_electrons = get_active_electron_count(molecule_record)
        
        # Limit to 24 qubits for safety
        if n_qubits > 24:
            return 0.0
            
        spin_ham = hamiltonian_to_spin_operator(molecule_record)
        kernel, pauli_words = _build_kernel_for_sequence(n_qubits, n_electrons, operators)
        
        grad_norm = 0.0
        delta = 1e-4
        
        for i in range(len(operators)):
            # + shift
            thetas_plus = np.full(len(operators), base_theta)
            thetas_plus[i] += delta
            e_plus = _evaluate_energy(thetas_plus, kernel, spin_ham, n_qubits, n_electrons, pauli_words)
            
            # - shift
            thetas_minus = np.full(len(operators), base_theta)
            thetas_minus[i] -= delta
            e_minus = _evaluate_energy(thetas_minus, kernel, spin_ham, n_qubits, n_electrons, pauli_words)
            
            # Gradient for this parameter (finite difference)
            grad_i = (e_plus - e_minus) / (2.0 * delta)
            grad_norm += abs(grad_i)
            
        return grad_norm
    except Exception as e:
        print(f"Gradient failed: {e}")
        return 0.0

def test_circuit(circuit_idx, item, mol_record):
    operators = item["operators"]
    final_energy = item["final_energy"]
    proxy_energy = item["proxy_energy"]
    
    # Use 0.01 as initial to avoid saddle points at 0.0
    init_thetas = np.full(len(operators), 0.01)
    e_trunc_3, _ = _optimize_coefficients(mol_record, operators, initial_thetas=init_thetas, max_iter=3)
    
    # 2. Truncated L-BFGS-B (10 steps)
    e_trunc_10, _ = _optimize_coefficients(mol_record, operators, initial_thetas=init_thetas, max_iter=10)
    
    # 3. Gradient Norm at theta=0.01 (same base as proxy)
    grad_norm = _compute_gradient_norm(mol_record, operators, base_theta=0.01)
    
    return {
        "idx": circuit_idx,
        "final": final_energy,
        "proxy": proxy_energy,
        "trunc_3": e_trunc_3,
        "trunc_10": e_trunc_10,
        "grad_norm": grad_norm
    }

def main():
    print("Loading data...")
    with RESULTS_PATH.open() as f:
        data = json.load(f)["data"]
        
    with MOL_PATH.open() as f:
        mols = json.load(f)["records"]
        
    # Find iodobenzene
    mol_record = next(m for m in mols if m["name"] == "iodobenzene")
    
    print(f"Testing {len(data)} circuits with clever surrogates...")
    
    results = []
    for i, item in enumerate(data):
        print(f"  Circuit {i+1}/{len(data)}...")
        res = test_circuit(i, item, mol_record)
        results.append(res)
        
    # Extract arrays
    finals = np.array([r["final"] for r in results])
    proxies = np.array([r["proxy"] for r in results])
    truncs_3 = np.array([r["trunc_3"] for r in results])
    truncs_10 = np.array([r["trunc_10"] for r in results])
    grad_norms = np.array([r["grad_norm"] for r in results])
    
    # Compute Spearmans
    rho_proxy, p_proxy = stats.spearmanr(proxies, finals)
    rho_t3, p_t3 = stats.spearmanr(truncs_3, finals)
    rho_t10, p_t10 = stats.spearmanr(truncs_10, finals)
    # For gradient norm, higher is better (more trainability), so we expect negative correlation with energy
    rho_grad, p_grad = stats.spearmanr(grad_norms, finals)
    
    print("\n" + "="*60)
    print("SURROGATE CORRELATION RESULTS")
    print("="*60)
    print(f"1. Fixed Theta=0.01 (Current):  rho = {rho_proxy:.3f} (p={p_proxy:.3f})")
    print(f"2. Truncated L-BFGS-B (3 iter): rho = {rho_t3:.3f} (p={p_t3:.3f})")
    print(f"3. Truncated L-BFGS-B (10 iter):rho = {rho_t10:.3f} (p={p_t10:.3f})")
    print(f"4. Gradient Norm (at theta=0):  rho = {rho_grad:.3f} (p={p_grad:.3f})")
    print("="*60)
    
    # Save results
    out_file = Path("results/eval/surrogate_validation.json")
    with out_file.open("w") as f:
        json.dump({
            "metrics": {
                "proxy": {"rho": rho_proxy, "p": p_proxy},
                "trunc_3": {"rho": rho_t3, "p": p_t3},
                "trunc_10": {"rho": rho_t10, "p": p_t10},
                "grad_norm": {"rho": rho_grad, "p": p_grad}
            },
            "data": results
        }, f, indent=2)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
