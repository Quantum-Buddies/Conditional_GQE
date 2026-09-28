#!/usr/bin/env python3
"""Build a 14-char H-cGQE SFT dataset from MatGen-Q SnO pool + Demo A teacher.

Vocab is the unique 14-character Pauli words in pool_sno14q.json (not the mixed
GIC UCCSD checkpoint). Teacher sequences: Demo A smoke-best tokens plus
synthetic pool walks. Hamiltonian conditioning is the exported SnO 14q JW H.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from src.gqe.models.h_cgqe_transformer import (
    PAULI_CHAR_VOCAB,
    SPECIAL_TOKENS,
    build_operator_vocab,
    tokenize_hamiltonian,
    tokenize_operator_sequence,
)


def _ham_terms(record: dict[str, Any]) -> list[tuple[str, float]]:
    terms: list[tuple[str, float]] = []
    for t in record.get("terms", []):
        terms.append((str(t["term"]), float(t.get("real", 0.0))))
    terms.sort(key=lambda x: abs(x[1]), reverse=True)
    return terms


def _tokens_to_words(pool: dict[str, Any], token_ids: list[int]) -> list[str]:
    by_id = {int(t["token_id"]): t for t in pool["tokens"]}
    words: list[str] = []
    for tid in token_ids:
        tok = by_id[int(tid)]
        word = str(tok["word"])
        if word == "I" * len(word):
            continue
        words.append(word)
    return words


def _augment_terms(
    terms: list[tuple[str, float]],
    rng: np.random.Generator,
    *,
    coeff_noise: float = 0.05,
    subsample_ratio: float = 0.95,
) -> list[tuple[str, float]]:
    keep = max(1, int(len(terms) * subsample_ratio))
    idx = rng.choice(len(terms), size=keep, replace=False)
    idx.sort()
    out: list[tuple[str, float]] = []
    for i in idx:
        word, coeff = terms[int(i)]
        noise = rng.normal(0.0, coeff_noise * abs(coeff))
        out.append((word, coeff + noise))
    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--hamiltonians", type=Path, required=True)
    parser.add_argument("--matgenq-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--molecule", type=str, default="sno_14q")
    parser.add_argument("--n-synthetic", type=int, default=2048)
    parser.add_argument("--max-terms", type=int, default=128)
    parser.add_argument("--max-pauli-len", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    pool = json.loads(args.pool.read_text())
    ham_data = json.loads(args.hamiltonians.read_text())
    record = next(r for r in ham_data["records"] if r["name"] == args.molecule)
    matgenq = json.loads(args.matgenq_results.read_text())

    unique_words = [w for w in pool["unique_words"] if w != "I" * len(w)]
    if not unique_words:
        raise SystemExit("pool has no non-identity words")
    if any(len(w) != 14 for w in unique_words):
        raise SystemExit("pool words are not all length 14")

    vocab = build_operator_vocab(pool["unique_words"])
    base_terms = _ham_terms(record)
    sequences: list[list[str]] = []

    teacher = _tokens_to_words(
        pool, list(matgenq.get("best_circuit", {}).get("tokens") or [])
    )
    if teacher:
        sequences.append(teacher)
        print(f"Teacher (Demo A smoke-best): {len(teacher)} words")

    for w in unique_words:
        sequences.append([w])

    seq_lens = (4, 6, 8, 10)
    for _ in tqdm(range(args.n_synthetic), desc="Synthetic 14-char sequences"):
        length = int(py_rng.choice(seq_lens))
        sequences.append([py_rng.choice(unique_words) for _ in range(length)])

    samples: list[dict[str, Any]] = []
    for i, words in enumerate(tqdm(sequences, desc="Tokenizing SFT samples")):
        terms = base_terms if i == 0 else _augment_terms(base_terms, rng)
        ham_tokens = tokenize_hamiltonian(terms, vocab, args.max_terms, args.max_pauli_len)
        tgt = tokenize_operator_sequence(words, vocab, args.max_seq_len)
        samples.append({
            "name": args.molecule,
            "pauli_ids": ham_tokens["pauli_ids"],
            "coeffs": ham_tokens["coeffs"],
            "term_mask": ham_tokens["term_mask"],
            "tgt_tokens": tgt,
            "n_terms": len(terms),
            "n_ops": len(words),
            "augmented": i > 0,
        })

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
            "max_terms": args.max_terms,
            "max_pauli_len": args.max_pauli_len,
            "max_seq_len": args.max_seq_len,
            "n_samples": len(samples),
            "vocab_size": len(vocab),
            "n_pauli_char_vocab": len(PAULI_CHAR_VOCAB),
            "dialect": "matgenq_sno14q_14char",
            "n_unique_words": len(unique_words),
            "special_tokens": list(SPECIAL_TOKENS),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds_path = args.out_dir / "gqe_supervised_dataset_sno14.pt"
    torch.save(dataset, ds_path)
    word_lens = {}
    for w in vocab:
        if w in SPECIAL_TOKENS:
            continue
        word_lens[str(len(w))] = word_lens.get(str(len(w)), 0) + 1
    summary = {
        "dataset": str(ds_path),
        "vocab_size": len(vocab),
        "n_samples": len(samples),
        "word_length_counts": word_lens,
        "all_operator_words_length_14": all(
            len(w) == 14 for w in vocab if w not in SPECIAL_TOKENS
        ),
        "max_seq_len": args.max_seq_len,
        "teacher_n_ops": len(teacher),
    }
    (args.out_dir / "sft_pool14_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {ds_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
