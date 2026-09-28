#!/usr/bin/env python3
"""
Verify the RL proxy energy (fixed theta) against full multi-start L-BFGS-B optimization.
This ensures the structures the policy ranks highly are actually good when converged.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

try:
    import cudaq
except ImportError:
    cudaq = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
)
from src.gqe.eval.optimize_h_cgqe_coefficients import (
    _evaluate_fixed_theta_energy,
    _optimize_coefficients,
    _ensure_cuda_context,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def optimize_multistart(mol_record, operators, n_restarts=5, max_iter=100):
    """Run L-BFGS-B multiple times with different random initializations and return the best."""
    best_energy = float("inf")
    best_thetas = None
    
    for _ in range(n_restarts):
        # Sample initial thetas uniformly from [-pi, pi] as is standard for VQE angles,
        # or [-0.1, 0.1] to stay near HF. We'll use a mix or just [-0.1, 0.1] to match original
        initial_thetas = np.random.uniform(-0.1, 0.1, size=len(operators))
        try:
            e, t = _optimize_coefficients(mol_record, operators, initial_thetas, max_iter=max_iter)
            if e < best_energy:
                best_energy = e
                best_thetas = t
        except Exception as ex:
            logging.debug(f"Optimization failed: {ex}")
            
    return best_energy, best_thetas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--hamiltonians", type=Path, default=Path("results/data/hamiltonians.json"))
    parser.add_argument("--molecule", type=str, required=True, help="e.g. 'h2_0.5' or 'lih'")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of sequences to evaluate")
    parser.add_argument("--n-restarts", type=int, default=10, help="Number of multi-start L-BFGS-B runs")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--plot", type=Path, help="Save scatter plot to this file")
    parser.add_argument("--out", type=Path, help="Save JSON results")
    args = parser.parse_args()

    if cudaq:
        cudaq.set_target("nvidia")

    # Load data
    with args.generated.open() as f:
        generated_data = json.load(f)
    ham_records = load_hamiltonian_records(args.hamiltonians)

    mol_record = find_record_by_name(ham_records, args.molecule)
    
    # Find generated sequences for the molecule
    mol_gen = next((m for m in generated_data if m["molecule"] == args.molecule), None)
    if not mol_gen:
        logging.error(f"Molecule {args.molecule} not found in {args.generated}")
        return

    sequences = mol_gen["generated_sequences"]
    if len(sequences) > args.n_samples:
        # Sample uniformly or just take first N? Let's take the first N (often ranked or sequential)
        # Actually random choice ensures we get diverse circuits
        idx = np.random.choice(len(sequences), args.n_samples, replace=False)
        sequences = [sequences[i] for i in idx]

    logging.info(f"Evaluating {len(sequences)} sequences for {args.molecule}")
    logging.info(f"Multi-start restarts: {args.n_restarts}")

    proxy_energies = []
    final_energies = []
    valid_sequences = []

    for i, seq in enumerate(sequences):
        ops = seq["operators"]
        if not ops:
            continue
            
        logging.info(f"[{i+1}/{len(sequences)}] Circuit depth: {len(ops)}")
        
        # 1. RL Proxy Energy
        try:
            e_proxy = _evaluate_fixed_theta_energy(mol_record, ops, theta=0.01)
        except Exception as e:
            logging.error(f"Proxy failed: {e}")
            continue

        # 2. Final Optimized Energy
        e_final, _ = optimize_multistart(mol_record, ops, n_restarts=args.n_restarts, max_iter=args.max_iter)
        
        if e_final != float("inf"):
            proxy_energies.append(e_proxy)
            final_energies.append(e_final)
            valid_sequences.append(ops)
            logging.info(f"  Proxy: {e_proxy:.6f} Ha | Final: {e_final:.6f} Ha")

    if len(proxy_energies) < 3:
        logging.error("Not enough valid sequences to compute correlation.")
        return

    # Compute Rank Correlation
    rho, p_val = spearmanr(proxy_energies, final_energies)
    logging.info("=" * 50)
    logging.info(f"Spearman Rank Correlation (rho): {rho:.4f}")
    logging.info(f"p-value: {p_val:.4e}")
    logging.info("=" * 50)

    # Save results
    if args.out:
        results = {
            "molecule": args.molecule,
            "n_samples": len(proxy_energies),
            "n_restarts": args.n_restarts,
            "spearman_rho": rho,
            "p_value": p_val,
            "data": [
                {"operators": ops, "proxy_energy": float(p), "final_energy": float(f)}
                for ops, p, f in zip(valid_sequences, proxy_energies, final_energies)
            ]
        }
        with args.out.open("w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved JSON results to {args.out}")

    # Plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.scatter(proxy_energies, final_energies, alpha=0.7)
            plt.xlabel(r"RL Proxy Energy ($E_{\theta=0.01}$)")
            plt.ylabel(r"Converged Multi-start Energy ($E^\star$)")
            plt.title(f"Ansatz Verification: {args.molecule}\nSpearman $\\rho={rho:.3f}$")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.plot)
            logging.info(f"Saved plot to {args.plot}")
        except ImportError:
            logging.warning("matplotlib not installed, skipping plot.")

if __name__ == "__main__":
    main()
