#!/usr/bin/env python3
"""Recover (operators, energy) pairs from existing PersistentEnergyCache.

The B200 precompute stored circuit→energy mappings in SQLite using MD5 hashes
of the operator sequences. The operator sequences themselves were not saved.

This script replays the deterministic circuit generation from
precompute_rl_energy_cache.py (same seed, molecule order, vocab, params),
computes the cache keys, and looks them up in the existing SQLite cache.
Matched pairs are exported as a pretrain JSON file for RL bootstrapping.

Usage:
    python3 src/gqe/data/cache_to_pretrain.py \
        --hamiltonians results/data/hamiltonians_rl_b200/hamiltonians.json \
        --cache results/train/rl_energy_cache.sqlite \
        --out results/train/rl_pretrain_from_cache.json \
        --n-per-mol 512 --max-qubits 28 --theta 0.01 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
    get_active_electron_count,
)
from src.gqe.common.operator_pool import _jw_excitation_pauli_words
from src.gqe.models.h_cgqe_transformer import build_operator_vocab, SPECIAL_TOKENS
from src.gqe.rl.energy_cache import circuit_energy_cache_key

# Reuse the exact same circuit sampling logic from precompute_rl_energy_cache.py
from src.gqe.data.precompute_rl_energy_cache import (
    _build_vocab,
    _sample_circuits_for_molecule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover (operators, energy) pairs from cache")
    parser.add_argument("--hamiltonians", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-per-mol", type=int, default=512)
    parser.add_argument("--n-per-mol-large", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--max-qubits", type=int, default=28)
    parser.add_argument("--theta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mps-threshold", type=int, default=33)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    records = load_hamiltonian_records(args.hamiltonians)
    mol_names = [
        r["name"]
        for r in sorted(records, key=lambda x: (x.get("n_qubits", 99), x.get("name", "")))
        if int(r.get("n_qubits", 99)) <= args.max_qubits
    ]

    print(f"Recovering circuits for {len(mol_names)} molecules from cache")
    print(f"  Hamiltonians: {args.hamiltonians}")
    print(f"  Cache: {args.cache}")
    print(f"  Output: {args.out}")
    print(f"  n_per_mol={args.n_per_mol}  max_qubits={args.max_qubits}  theta={args.theta}  seed={args.seed}")

    vocab, inv_vocab, op_tokens = _build_vocab(args.hamiltonians, mol_names)
    print(f"  Vocab operators: {len(op_tokens)}")

    # Open cache for lookup
    conn = sqlite3.connect(str(args.cache))

    # Pre-load all (key, energy) pairs for fast lookup
    print("  Loading cache entries...")
    cache_lookup: dict[str, float] = {}
    for row in conn.execute("SELECT key, energy FROM energies"):
        cache_lookup[row[0]] = float(row[1])
    print(f"  Loaded {len(cache_lookup)} cache entries")

    results: list[dict] = []
    total_matched = 0
    total_unmatched = 0

    for mol_name in tqdm(mol_names, desc="Molecules", unit="mol"):
        record = find_record_by_name(records, mol_name)
        if record is None:
            continue
        n_qubits = int(record["n_qubits"])
        n_electrons = get_active_electron_count(record)
        n_circuits = args.n_per_mol_large if n_qubits > args.mps_threshold else args.n_per_mol

        circuits = _sample_circuits_for_molecule(
            op_tokens, n_qubits, n_circuits, args.max_seq_len, rng,
        )

        matched = 0
        for ops in circuits:
            key = circuit_energy_cache_key(ops, mol_name, n_qubits, n_electrons, args.theta)
            energy = cache_lookup.get(key)
            if energy is not None:
                results.append({
                    "molecule": mol_name,
                    "gqe_selected_operators": [{"pauli_word": op} for op in ops],
                    "best_energy": energy,
                    "n_qubits": n_qubits,
                    "n_electrons": n_electrons,
                    "n_ops": len(ops),
                })
                matched += 1
            else:
                total_unmatched += 1

        total_matched += matched
        tqdm.write(f"  {mol_name:25s}  q={n_qubits:2d}  circuits={len(circuits):3d}  matched={matched:3d}")

    conn.close()

    print(f"\n=== Recovery complete ===")
    print(f"  Matched: {total_matched}/{total_matched + total_unmatched} circuits")
    print(f"  Unmatched: {total_unmatched}")

    # Save as pretrain JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"  Output: {args.out} ({len(results)} samples)")

    # Per-molecule summary
    per_mol: dict[str, int] = {}
    for r in results:
        per_mol[r["molecule"]] = per_mol.get(r["molecule"], 0) + 1
    print(f"  Per-molecule:")
    for mol, count in sorted(per_mol.items()):
        print(f"    {mol:25s}  {count:4d}")


if __name__ == "__main__":
    main()
