"""Inference script for H-cGQE Transformer.

Loads a trained H-cGQE model and generates operator sequences for given Hamiltonians.
Outputs the generated sequences as JSON for downstream evaluation.

Usage:
    python src/gqe/models/infer_h_cgqe.py \
        --checkpoint results/train/h_cgqe_model_augmented.pt \
        --hamiltonians results/data/hamiltonians.json \
        --molecules h2 lih beh2 \
        --out results/inference/h_cgqe_generated.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.gqe.models.h_cgqe_transformer import (
    HcGQEModel,
    tokenize_hamiltonian,
    PAULI_CHAR_VOCAB,
    SPECIAL_TOKENS,
)


def load_hamiltonian_record(ham_path: Path, molecule: str) -> dict[str, Any]:
    """Extract the Hamiltonian record for a specific molecule."""
    with ham_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for record in data.get("records", []):
        if record["name"] == molecule:
            return record
    raise ValueError(f"Molecule {molecule} not found in {ham_path}")


def load_hamiltonian_terms(ham_path: Path, molecule: str) -> list[tuple[str, float]]:
    """Extract Hamiltonian terms for a specific molecule."""
    record = load_hamiltonian_record(ham_path, molecule)
    terms = record.get("terms", [])
    out: list[tuple[str, float]] = []
    for t in terms:
        if isinstance(t, dict):
            word = t.get("term", "")
            coeff = float(t.get("real", 0.0))
            out.append((word, coeff))
    # Sort by absolute coefficient
    out.sort(key=lambda x: abs(x[1]), reverse=True)
    return out


def _is_trailing_noise(word: str) -> bool:
    """True for operators that are all-identity or single-qubit Z-only.

    Such terms add no entanglement and are frequently emitted as padding by the
    autoregressive model; trimming them from the end of a sequence makes the
    downstream coefficient optimization much cheaper without changing the
    expressive power of the ansatz.
    """
    non_identity_positions = [i for i, ch in enumerate(word) if ch != "I"]
    if not non_identity_positions:
        return True
    if len(non_identity_positions) == 1 and word[non_identity_positions[0]] == "Z":
        return True
    return False


def decode_operator_sequence(
    token_ids: torch.Tensor,
    inv_vocab: dict[int, str],
    trim_trailing: bool = True,
) -> list[str]:
    """Convert token IDs back to Pauli words.

    Args:
        token_ids: Tensor of token IDs including BOS/EOS/PAD.
        inv_vocab: Inverse vocabulary mapping token IDs to Pauli words.
        trim_trailing: If True, remove trailing identity or single-qubit Z-only
            operators that do not contribute entanglement.
    """
    words = []
    for tid in token_ids:
        if tid == SPECIAL_TOKENS["<EOS>"]:
            break
        if tid == SPECIAL_TOKENS["<PAD>"]:
            continue
        word = inv_vocab.get(tid.item(), "<UNK>")
        if word not in ["<BOS>", "<UNK>"]:
            words.append(word)

    if trim_trailing:
        while words and _is_trailing_noise(words[-1]):
            words.pop()

    return words


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with H-cGQE Transformer")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hamiltonians", type=Path, default=Path("results/data/hamiltonians.json"))
    parser.add_argument("--molecules", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=100, help="Number of circuits to sample per molecule")
    parser.add_argument("--max-terms", type=int, default=128)
    parser.add_argument("--max-pauli-len", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--force-entanglement", action="store_true",
                        help="Prevent Z-only sequences by forcing at least one X/Y operator")
    parser.add_argument("--sample", action="store_true",
                        help="Use stochastic sampling instead of greedy decoding")
    parser.add_argument("--max-repeat", type=int, default=4,
                        help="Stop generation if the same token repeats this many times")
    parser.add_argument("--no-trim", action="store_true",
                        help="Disable trimming of trailing identity / single-qubit Z-only operators")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    inv_vocab = ckpt["inv_vocab"]
    config = ckpt["config"]

    # Load model
    model = HcGQEModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_pauli_len=config["max_pauli_len"],
        max_seq_len=config["max_seq_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Generate sequences for each molecule
    results: list[dict[str, Any]] = []
    for molecule in args.molecules:
        print(f"\nGenerating for {molecule}...")
        record = load_hamiltonian_record(args.hamiltonians, molecule)
        terms = load_hamiltonian_terms(args.hamiltonians, molecule)
        n_qubits = int(record.get("n_qubits", 0))
        print(f"  Hamiltonian: {len(terms)} terms, {n_qubits} qubits")

        ham_tokens = tokenize_hamiltonian(
            terms, vocab, args.max_terms, args.max_pauli_len
        )

        pauli_ids = ham_tokens["pauli_ids"].unsqueeze(0).to(device)
        coeffs = ham_tokens["coeffs"].unsqueeze(0).to(device)
        term_mask = ham_tokens["term_mask"].unsqueeze(0).to(device)

        # Generate multiple samples
        molecule_results = []
        for i in range(args.n_samples):
            generated = model.generate(
                pauli_ids,
                coeffs,
                term_mask,
                bos_id=SPECIAL_TOKENS["<BOS>"],
                eos_id=SPECIAL_TOKENS["<EOS>"],
                max_len=args.max_seq_len,
                temperature=args.temperature,
                vocab=vocab,
                force_entanglement=args.force_entanglement,
                max_repeat=args.max_repeat,
                sample=args.sample,
                n_qubits=n_qubits,
            )
            words = decode_operator_sequence(generated[0], inv_vocab, trim_trailing=not args.no_trim)
            molecule_results.append({"sample_id": i, "operators": words})

        results.append({
            "molecule": molecule,
            "n_terms": len(terms),
            "generated_sequences": molecule_results,
        })
        print(f"  Generated {len(molecule_results)} sequences")

    # Save results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
