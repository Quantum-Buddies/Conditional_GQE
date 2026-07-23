#!/usr/bin/env python3
"""Precompute a persistent circuit→energy cache for fast DAPO RL.

Generates diverse operator sequences per molecule (random vocab, entangler-
biased, UCCSD-pool shuffles), evaluates them with CUDA-Q (chunked async),
and stores results in SQLite. Training then hits this cache instead of
re-running observe() for every repeated circuit.

Usage:
    # Source Blackwell env first (or use launch_b200_training.sh cache)
    source scripts/env_b200_blackwell.sh
    python3 src/gqe/data/precompute_rl_energy_cache.py \\
        --hamiltonians results/data/hamiltonians_rl_b200/hamiltonians.json \\
        --out results/train/rl_energy_cache.sqlite \\
        --n-per-mol 512 --max-qubits 40
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import cudaq
except ImportError:
    cudaq = None

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
    get_active_electron_count,
)
from src.gqe.common.operator_pool import _jw_excitation_pauli_words
from src.gqe.models.h_cgqe_transformer import (
    build_operator_vocab,
    SPECIAL_TOKENS,
)
from src.gqe.models.train_rl_dapo import (
    evaluate_energies_batch,
    _get_cached_spin_ham,
    _set_cudaq_target_cached,
    _warmup_cudaq_observe,
)
from src.gqe.rl.energy_cache import PersistentEnergyCache


def _build_vocab(ham_path: Path, molecules: list[str]) -> tuple[dict[str, int], dict[int, str], list[str]]:
    """Build operator vocab from UCCSD pool (same recipe as train_rl_dapo --from-scratch)."""
    MAX_SINGLES, MAX_DOUBLES = 10, 10
    all_pauli_words: list[str] = []
    ham_records = load_hamiltonian_records(ham_path)
    for mol_name in molecules:
        record = find_record_by_name(ham_records, mol_name)
        if record is None:
            continue
        n_qubits_mol = int(record.get("n_qubits", 0))
        n_electrons = get_active_electron_count(record)
        try:
            excitation_words = _jw_excitation_pauli_words(
                n_qubits_mol, n_electrons,
                max_singles=MAX_SINGLES, max_doubles=MAX_DOUBLES,
            )
            # _jw_excitation_pauli_words returns (pauli_word, coeff) pairs
            for word, _coeff in excitation_words:
                if isinstance(word, str):
                    all_pauli_words.append(word)
        except Exception:
            continue
    vocab = build_operator_vocab(all_pauli_words)
    inv_vocab = {i: t for t, i in vocab.items()}
    # Operator tokens only (exclude specials); keep strings only
    special_ids = set(SPECIAL_TOKENS.values())
    op_tokens = [
        t for t, i in vocab.items()
        if isinstance(t, str) and i not in special_ids and t not in SPECIAL_TOKENS
    ]
    return vocab, inv_vocab, op_tokens


def _is_entangling(word: str) -> bool:
    return sum(1 for c in word if c in ("X", "Y")) >= 2


def _sample_circuits_for_molecule(
    op_tokens: list[str],
    n_qubits: int,
    n_circuits: int,
    max_seq_len: int,
    rng: random.Random,
) -> list[list[str]]:
    """Generate diverse operator sequences without a trained policy."""
    if not op_tokens:
        return []

    length_ok = [w for w in op_tokens if len(w) == n_qubits or len(w.replace("I", "")) <= n_qubits]
    # Prefer exact-length words when available
    exact = [w for w in op_tokens if len(w) == n_qubits]
    pool = exact if exact else (length_ok if length_ok else op_tokens)
    entanglers = [w for w in pool if _is_entangling(w)]
    diagonal = [w for w in pool if not _is_entangling(w)]

    circuits: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def _add(ops: list[str]) -> None:
        key = tuple(ops)
        if key in seen or not ops:
            return
        seen.add(key)
        circuits.append(ops)

    # 40% random lengths
    n_random = max(1, n_circuits * 2 // 5)
    for _ in range(n_random * 3):  # oversample then unique-cap
        if len(circuits) >= n_random:
            break
        length = rng.randint(2, max(2, min(max_seq_len, 16)))
        ops = [rng.choice(pool) for _ in range(length)]
        _add(ops)

    # 40% force-entangler biased (first op entangling when possible)
    n_ent = max(1, n_circuits * 2 // 5)
    target_ent = len(circuits) + n_ent
    for _ in range(n_ent * 3):
        if len(circuits) >= target_ent:
            break
        length = rng.randint(3, max(3, min(max_seq_len, 12)))
        ops = []
        if entanglers:
            ops.append(rng.choice(entanglers))
        while len(ops) < length:
            # Prefer entanglers ~70% of the time
            if entanglers and rng.random() < 0.7:
                ops.append(rng.choice(entanglers))
            else:
                ops.append(rng.choice(pool))
        _add(ops)

    # 20% UCCSD-pool shuffles (short)
    n_pool = n_circuits - len(circuits)
    for _ in range(max(n_pool, 1) * 3):
        if len(circuits) >= n_circuits:
            break
        length = rng.randint(2, max(2, min(8, len(pool), max_seq_len)))
        ops = rng.sample(pool, k=min(length, len(pool)))
        rng.shuffle(ops)
        _add(ops)

    # Fill remaining with random if still short
    while len(circuits) < n_circuits:
        length = rng.randint(2, max(2, min(max_seq_len, 10)))
        ops = [rng.choice(pool) for _ in range(length)]
        before = len(circuits)
        _add(ops)
        if len(circuits) == before:
            # forced unique via pad noise
            ops = ops + [rng.choice(pool)]
            _add(ops[:max_seq_len])
            if len(circuits) == before:
                break

    return circuits[:n_circuits]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute RL energy cache (CUDA-Q → SQLite)")
    parser.add_argument("--hamiltonians", type=Path, required=True)
    parser.add_argument("--molecules", nargs="*", default=None,
                        help="Molecule names (default: all with n_qubits <= --max-qubits)")
    parser.add_argument("--out", type=Path, default=Path("results/train/rl_energy_cache.sqlite"))
    parser.add_argument("--n-per-mol", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--max-qubits", type=int, default=40)
    parser.add_argument("--theta", type=float, default=0.01)
    parser.add_argument("--eval-async-chunk", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=str, default="nvidia")
    parser.add_argument("--target-option", type=str, default="fp32")
    parser.add_argument("--mps-threshold", type=int, default=28)
    args = parser.parse_args()

    if cudaq is None:
        raise SystemExit("cudaq is required for energy precompute")

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    records = load_hamiltonian_records(args.hamiltonians)
    if args.molecules:
        mol_names = list(args.molecules)
    else:
        mol_names = [
            r["name"] for r in sorted(records, key=lambda x: (x.get("n_qubits", 99), x.get("name", "")))
            if int(r.get("n_qubits", 99)) <= args.max_qubits
        ]

    print(f"Precomputing energy cache for {len(mol_names)} molecules → {args.out}")
    print(f"  n_per_mol={args.n_per_mol}  theta={args.theta}  chunk={args.eval_async_chunk}")

    vocab, inv_vocab, op_tokens = _build_vocab(args.hamiltonians, mol_names)
    print(f"  Vocab operators: {len(op_tokens)}")

    # CUDA-Q Blackwell / fp32 target
    opt = args.target_option or ("fp32" if args.target == "nvidia" else "")
    try:
        _set_cudaq_target_cached(args.target, opt)
        print(f"  CUDA-Q target: {args.target} ({opt})")
        _warmup_cudaq_observe()
    except Exception as e:
        print(f"  WARNING: CUDA-Q setup: {e}")

    cache = PersistentEnergyCache(args.out)
    total_new = 0
    total_skipped = 0

    mol_bar = tqdm(mol_names, desc="Molecules", unit="mol")
    for mol_name in mol_bar:
        record = find_record_by_name(records, mol_name)
        if record is None:
            continue
        n_qubits = int(record["n_qubits"])
        n_electrons = get_active_electron_count(record)
        mol_bar.set_postfix(mol=mol_name, q=n_qubits)

        circuits = _sample_circuits_for_molecule(
            op_tokens, n_qubits, args.n_per_mol, args.max_seq_len, rng,
        )
        # Filter already cached
        to_eval: list[list[str]] = []
        for ops in circuits:
            if cache.has_key(ops, mol_name, n_qubits, n_electrons, args.theta):
                total_skipped += 1
            else:
                to_eval.append(ops)

        if not to_eval:
            continue

        # MPS switch for large molecules
        use_mps = n_qubits > args.mps_threshold
        if use_mps:
            try:
                _set_cudaq_target_cached("tensornet-mps")
            except Exception as e:
                print(f"  WARNING: MPS switch failed for {mol_name}: {e}")
        else:
            try:
                _set_cudaq_target_cached(args.target, opt)
            except Exception:
                pass

        spin_ham = _get_cached_spin_ham(record, cache_key=mol_name)
        # Evaluate in chunks matching async depth
        chunk = max(args.eval_async_chunk, 1)
        for start in tqdm(
            range(0, len(to_eval), chunk),
            desc=f"  {mol_name} eval",
            leave=False,
            unit="batch",
        ):
            batch = to_eval[start:start + chunk]
            energies = evaluate_energies_batch(
                batch,
                record,
                theta=args.theta,
                eval_async=True,
                spin_ham=spin_ham,
                async_chunk=chunk,
                show_progress=False,
                mol_name=mol_name,
            )
            cache.put_many(batch, energies, mol_name, n_qubits, n_electrons, args.theta)
            total_new += len(batch)

        mol_bar.set_postfix(mol=mol_name, q=n_qubits, new=total_new, skip=total_skipped)

    stats = cache.stats()
    cache.close()
    print("\n=== Precompute complete ===")
    print(f"  Cache path : {args.out}")
    print(f"  New entries: {total_new}")
    print(f"  Skipped    : {total_skipped} (already cached)")
    print(f"  Total size : {stats['n_entries']} circuits")
    top = sorted(stats["per_molecule"].items(), key=lambda x: -x[1])[:8]
    print(f"  Per-mol (top): {top}")


if __name__ == "__main__":
    main()
