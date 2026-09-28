#!/usr/bin/env python3
"""Analyze and compare generated H-cGQE circuits."""

import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np


def is_entangling(word: str) -> bool:
    """True if operator contains X or Y (entangling)."""
    return any(c in word for c in "XY")


def is_z_only(word: str) -> bool:
    """True if operator is Z-only or identity."""
    non_identity = [i for i, ch in enumerate(word) if ch != "I"]
    if not non_identity:
        return True
    return all(word[i] == "Z" for i in non_identity)


def analyze_file(path: str, label: str) -> dict:
    """Analyze generated circuits from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    results = {}
    print(f"\n{'='*60}")
    print(f"{label} ({path})")
    print(f"{'='*60}")

    for mol_entry in data:
        molecule = mol_entry["molecule"]
        sequences = mol_entry["generated_sequences"]

        operator_lists = [s["operators"] for s in sequences]
        lengths = [len(ops) for ops in operator_lists]

        # Diversity
        unique_seqs = set(tuple(ops) for ops in operator_lists)
        n_unique = len(unique_seqs)
        diversity = n_unique / len(operator_lists) if operator_lists else 0.0

        # Entanglement content
        total_ops = sum(len(ops) for ops in operator_lists)
        entangling_ops = sum(
            1 for ops in operator_lists for op in ops if is_entangling(op)
        )
        z_only_ops = sum(
            1 for ops in operator_lists for op in ops if is_z_only(op)
        )
        entangling_frac = entangling_ops / total_ops if total_ops > 0 else 0.0
        z_only_frac = z_only_ops / total_ops if total_ops > 0 else 0.0

        # Operator frequency
        all_ops = [op for ops in operator_lists for op in ops]
        op_counts = Counter(all_ops)
        most_common = op_counts.most_common(5)

        mol_results = {
            "n_sequences": len(operator_lists),
            "n_unique": n_unique,
            "diversity": diversity,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "std_length": float(np.std(lengths)) if lengths else 0.0,
            "min_length": int(np.min(lengths)) if lengths else 0,
            "max_length": int(np.max(lengths)) if lengths else 0,
            "entangling_frac": entangling_frac,
            "z_only_frac": z_only_frac,
            "top_5_operators": most_common,
        }
        results[molecule] = mol_results

        print(f"\n{molecule.upper()}:")
        print(f"  Sequences: {mol_results['n_sequences']}")
        print(f"  Unique: {mol_results['n_unique']} (diversity={diversity:.3f})")
        print(f"  Length: {mol_results['mean_length']:.1f} ± {mol_results['std_length']:.1f} "
              f"[{mol_results['min_length']}, {mol_results['max_length']}]")
        print(f"  Entangling ops: {entangling_frac:.3f}")
        print(f"  Z-only ops: {z_only_frac:.3f}")
        print(f"  Top operators: {most_common}")

    return results


def compare(supervised: dict, chemeleon2: dict):
    """Compare supervised vs Chemeleon2 results."""
    print(f"\n{'='*60}")
    print("COMPARISON: Supervised vs Chemeleon2 RL")
    print(f"{'='*60}")

    molecules = set(supervised.keys()) | set(chemeleon2.keys())
    for molecule in sorted(molecules):
        sup = supervised.get(molecule, {})
        chm = chemeleon2.get(molecule, {})

        print(f"\n{molecule.upper()}:")
        print(f"  Diversity:       {sup.get('diversity', 0):.3f} → {chm.get('diversity', 0):.3f} "
              f"({(chm.get('diversity', 0) - sup.get('diversity', 0)):+.3f})")
        print(f"  Entangling frac: {sup.get('entangling_frac', 0):.3f} → {chm.get('entangling_frac', 0):.3f} "
              f"({(chm.get('entangling_frac', 0) - sup.get('entangling_frac', 0)):+.3f})")
        print(f"  Z-only frac:     {sup.get('z_only_frac', 0):.3f} → {chm.get('z_only_frac', 0):.3f} "
              f"({(chm.get('z_only_frac', 0) - sup.get('z_only_frac', 0)):+.3f})")
        print(f"  Mean length:     {sup.get('mean_length', 0):.1f} → {chm.get('mean_length', 0):.1f} "
              f"({(chm.get('mean_length', 0) - sup.get('mean_length', 0)):+.1f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervised", type=str,
                        default="results/inference/supervised_generated.json")
    parser.add_argument("--chemeleon2", type=str,
                        default="results/inference/chemeleon2_generated.json")
    args = parser.parse_args()

    sup_results = analyze_file(args.supervised, "SUPERVISED")
    chm_results = analyze_file(args.chemeleon2, "CHEMELEON2 RL")
    compare(sup_results, chm_results)


if __name__ == "__main__":
    main()
