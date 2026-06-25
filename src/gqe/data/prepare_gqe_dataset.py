"""Prepare a supervised dataset from GQE JSON outputs for H-cGQE training.

Reads:
- results/data/hamiltonians.json (input Hamiltonians)
- results/baselines/cudaq_gqe_*.json (target operator sequences)

Writes:
- results/train/gqe_operator_vocab.json
- results/train/gqe_supervised_dataset.pt (torch dataset)
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Import the tokenizer utilities from the transformer module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.gqe.models.h_cgqe_transformer import (
    build_operator_vocab,
    tokenize_hamiltonian,
    tokenize_operator_sequence,
    PAULI_CHAR_VOCAB,
    SPECIAL_TOKENS,
)


def _load_hamiltonians(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["name"]: r for r in data.get("records", [])}


def _extract_operator_sequences(gqe_jsons: list[Path]) -> dict[str, list[list[str]]]:
    """Map molecule name -> list of operator sequences (one per GQE run)."""
    sequences: dict[str, list[list[str]]] = {}
    for path in gqe_jsons:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for result in data.get("results", []):
            name = result.get("system")
            if not name:
                continue
            ops = result.get("gqe_selected_operators", [])
            words = [op["pauli_word"] for op in ops if "pauli_word" in op]
            if words:
                if name not in sequences:
                    sequences[name] = []
                sequences[name].append(words)
    return sequences


def _augment_terms(
    terms: list[tuple[str, float]],
    *,
    coeff_noise: float = 0.05,
    shuffle_terms: bool = True,
    subsample_ratio: float = 1.0,
    seed: int = 42,
) -> list[tuple[str, float]]:
    """Apply data augmentation to Hamiltonian terms.

    Args:
        coeff_noise: Standard deviation of Gaussian noise for coefficients (e.g., 0.05 = ±5%)
        shuffle_terms: Randomize term order
        subsample_ratio: Fraction of terms to keep (e.g., 0.9 = keep 90%)
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    augmented: list[tuple[str, float]] = []

    # Coefficient perturbation
    for word, coeff in terms:
        noise = rng.normal(0, coeff_noise * abs(coeff))
        new_coeff = coeff + noise
        augmented.append((word, new_coeff))

    # Subsample
    if subsample_ratio < 1.0:
        n_keep = max(1, int(len(augmented) * subsample_ratio))
        indices = rng.choice(len(augmented), size=n_keep, replace=False)
        augmented = [augmented[i] for i in indices]

    # Shuffle (re-sort by |coeff| with noise)
    if shuffle_terms:
        augmented.sort(key=lambda x: abs(x[1]), reverse=True)

    return augmented


def _hamiltonian_to_terms(record: dict[str, Any]) -> list[tuple[str, float]]:
    """Convert a Hamiltonian record to a list of (pauli_word, coefficient)."""
    terms = record.get("terms", [])
    out: list[tuple[str, float]] = []
    for t in terms:
        if isinstance(t, dict):
            word = t.get("term", "")
            coeff = float(t.get("real", 0.0))
            out.append((word, coeff))
        elif isinstance(t, (list, tuple)) and len(t) == 2:
            out.append((str(t[0]), float(t[1])))
    # Sort by absolute coefficient descending
    out.sort(key=lambda x: abs(x[1]), reverse=True)
    return out


def prepare_dataset(
    hamiltonian_path: Path,
    gqe_jsons: list[Path],
    out_dir: Path,
    max_terms: int = 128,
    max_pauli_len: int = 24,
    max_seq_len: int = 64,
    augment_multiplier: int = 1,
    coeff_noise: float = 0.05,
    subsample_ratio: float = 1.0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    hamiltonians = _load_hamiltonians(hamiltonian_path)
    sequences = _extract_operator_sequences(gqe_jsons)

    # Find overlap
    common_names = sorted(set(hamiltonians.keys()) & set(sequences.keys()))
    if not common_names:
        raise RuntimeError("No matching molecules between Hamiltonians and GQE results.")

    total_sequences = sum(len(seqs) for seqs in sequences.values())
    print(f"Found {len(common_names)} molecules with both Hamiltonian and GQE data:")
    for name in common_names:
        n_terms = len(_hamiltonian_to_terms(hamiltonians[name]))
        n_seqs = len(sequences[name])
        print(f"  {name:20s}  {n_terms:4d} terms  {n_seqs:3d} sequences")

    # Build vocabulary from all selected operators
    all_words: list[str] = []
    for seq_list in sequences.values():
        for words in seq_list:
            all_words.extend(words)
    vocab = build_operator_vocab(all_words)
    vocab_path = out_dir / "gqe_operator_vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"\nBuilt vocabulary: {len(vocab)} tokens (including 4 special tokens)")
    print(f"  Pauli words: {len(vocab) - 4}")
    print(f"  Saved vocab to: {vocab_path}")

    # Build dataset samples: one sample per GQE sequence, plus augmented copies
    samples: list[dict[str, Any]] = []
    aug_seed = 0
    for name in common_names:
        ham = hamiltonians[name]
        base_terms = _hamiltonian_to_terms(ham)

        for target_words in sequences[name]:
            # Original (unaugmented) sample
            ham_tokens = tokenize_hamiltonian(base_terms, vocab, max_terms, max_pauli_len)
            tgt_tokens = tokenize_operator_sequence(target_words, vocab, max_seq_len)
            samples.append({
                "name": name,
                "pauli_ids": ham_tokens["pauli_ids"],
                "coeffs": ham_tokens["coeffs"],
                "term_mask": ham_tokens["term_mask"],
                "tgt_tokens": tgt_tokens,
                "n_terms": len(base_terms),
                "n_ops": len(target_words),
                "augmented": False,
            })

            # Augmented copies
            for i in range(augment_multiplier):
                aug_seed += 1
                aug_terms = _augment_terms(
                    base_terms,
                    coeff_noise=coeff_noise,
                    shuffle_terms=True,
                    subsample_ratio=subsample_ratio,
                    seed=aug_seed,
                )
                ham_tokens = tokenize_hamiltonian(aug_terms, vocab, max_terms, max_pauli_len)
                tgt_tokens = tokenize_operator_sequence(target_words, vocab, max_seq_len)
                samples.append({
                    "name": name,
                    "pauli_ids": ham_tokens["pauli_ids"],
                    "coeffs": ham_tokens["coeffs"],
                    "term_mask": ham_tokens["term_mask"],
                    "tgt_tokens": tgt_tokens,
                    "n_terms": len(aug_terms),
                    "n_ops": len(target_words),
                    "augmented": True,
                })

    # Stack into tensors
    dataset = {
        "vocab": vocab,
        "inv_vocab": {v: k for k, v in vocab.items()},
        "samples": samples,
        "pauli_ids": torch.stack([s["pauli_ids"] for s in samples]),
        "coeffs": torch.stack([s["coeffs"] for s in samples]),
        "term_mask": torch.stack([s["term_mask"] for s in samples]),
        "tgt_tokens": torch.stack([s["tgt_tokens"] for s in samples]),
        "names": [s["name"] for s in samples],
        "metadata": {
            "max_terms": max_terms,
            "max_pauli_len": max_pauli_len,
            "max_seq_len": max_seq_len,
            "n_samples": len(samples),
            "vocab_size": len(vocab),
            "n_pauli_char_vocab": len(PAULI_CHAR_VOCAB),
        },
    }

    dataset_path = out_dir / "gqe_supervised_dataset.pt"
    torch.save(dataset, dataset_path)
    print(f"Saved dataset to: {dataset_path}")
    print(f"  Samples: {len(samples)}")
    print(f"  Input shape:  pauli_ids {dataset['pauli_ids'].shape}")
    print(f"  Target shape:   tgt_tokens {dataset['tgt_tokens'].shape}")

    # Also save a readable JSON summary
    n_augmented = sum(1 for s in samples if s.get("augmented", False))
    summary = {
        "molecules": common_names,
        "vocab_size": len(vocab),
        "n_samples": len(samples),
        "n_original": len(samples) - n_augmented,
        "n_augmented": n_augmented,
        "augmentation": {
            "multiplier": augment_multiplier,
            "coeff_noise": coeff_noise,
            "subsample_ratio": subsample_ratio,
        },
        "max_terms": max_terms,
        "max_pauli_len": max_pauli_len,
        "max_seq_len": max_seq_len,
        "sample_shapes": {
            "pauli_ids": list(dataset["pauli_ids"].shape),
            "coeffs": list(dataset["coeffs"].shape),
            "term_mask": list(dataset["term_mask"].shape),
            "tgt_tokens": list(dataset["tgt_tokens"].shape),
        },
    }
    summary_path = out_dir / "gqe_dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare supervised GQE dataset.")
    parser.add_argument("--ham", type=Path, default=Path("results/data/hamiltonians.json"))
    parser.add_argument("--gqe-results", type=Path, nargs="+", required=True, help="GQE JSON result files")
    parser.add_argument("--out-dir", type=Path, default=Path("results/train"))
    parser.add_argument("--max-terms", type=int, default=128)
    parser.add_argument("--max-pauli-len", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--augment-multiplier", type=int, default=5, help="Number of augmented copies per sample")
    parser.add_argument("--coeff-noise", type=float, default=0.05, help="Coefficient perturbation std (e.g., 0.05 = ±5%)")
    parser.add_argument("--subsample-ratio", type=float, default=0.9, help="Fraction of terms to keep (1.0 = no subsampling)")
    args = parser.parse_args()

    prepare_dataset(
        hamiltonian_path=args.ham,
        gqe_jsons=args.gqe_results,
        out_dir=args.out_dir,
        max_terms=args.max_terms,
        max_pauli_len=args.max_pauli_len,
        max_seq_len=args.max_seq_len,
        augment_multiplier=args.augment_multiplier,
        coeff_noise=args.coeff_noise,
        subsample_ratio=args.subsample_ratio,
    )


if __name__ == "__main__":
    main()
