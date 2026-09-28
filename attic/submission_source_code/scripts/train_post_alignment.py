#!/usr/bin/env python3
"""Post-Training Alignment via Rejection Sampling Fine-Tuning (RAFT) for H-cGQE.

Generates candidate sequences from a trained model checkpoint, optimizes their 
parameters classically (Stage 2 L-BFGS-B), filters the top-performing sequences
(rejection sampling), and fine-tunes the model prior on this high-fidelity dataset.

Usage:
    python scripts/train_post_alignment.py \
        --checkpoint results/train/h_cgqe_rl_warmstart.pt \
        --hamiltonians results/data/hamiltonians_scaling.json/hamiltonians.json \
        --out results/train/h_cgqe_raft_aligned.pt \
        --epochs 50 --batch-size 4 --lr 5e-5 \
        --n-samples 50 --top-k 5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cudaq
except ImportError:
    cudaq = None

from src.gqe.models.h_cgqe_transformer import (
    HcGQEModel,
    SPECIAL_TOKENS,
    tokenize_hamiltonian,
    tokenize_operator_sequence,
)
from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
)
from src.gqe.eval.optimize_h_cgqe_coefficients import _optimize_coefficients
from src.gqe.models.train_h_cgqe import train_epoch


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_candidates(
    model: nn.Module,
    pauli_ids: torch.Tensor,
    coeffs: torch.Tensor,
    term_mask: torch.Tensor,
    n_samples: int,
    max_seq_len: int,
    temperature: float,
    vocab: dict[str, int],
    inv_vocab: dict[int, str],
    device: torch.device,
    top_p: float = 0.9,
) -> list[list[str]]:
    """Sample multiple candidate sequences for a molecule from the model policy."""
    model.eval()
    bos_id = SPECIAL_TOKENS["<BOS>"]
    eos_id = SPECIAL_TOKENS["<EOS>"]
    pad_id = SPECIAL_TOKENS["<PAD>"]

    # Expand inputs to batch dimension for fast sampling
    pauli_ids_batch = pauli_ids.expand(n_samples, -1, -1).to(device)
    coeffs_batch = coeffs.expand(n_samples, -1).to(device)
    term_mask_batch = term_mask.expand(n_samples, -1).to(device)

    # Initialize decoder input with BOS tokens
    decoder_input = torch.full((n_samples, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)

    for _ in range(max_seq_len - 1):
        if finished.all():
            break

        # Forward pass
        with torch.no_grad():
            logits = model(
                pauli_ids_batch,
                coeffs_batch,
                decoder_input,
                term_mask=term_mask_batch,
            )
            # Focus on logits for the last generated token
            next_token_logits = logits[:, -1, :] / temperature

        # Apply top-p (nucleus) filtering
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F_softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter mask back to original logits shape
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        next_token_logits[indices_to_remove] = -float("Inf")

        # Sample next tokens
        probs = F_softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Handle finished sequences
        next_tokens = torch.where(finished, torch.tensor(pad_id, device=device), next_tokens)
        finished = finished | (next_tokens == eos_id)

        # Append to decoder input
        decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(-1)], dim=-1)

    # Convert generated token IDs back to operator sequences
    sampled_ops_list = []
    for b in range(n_samples):
        seq = decoder_input[b].tolist()
        ops = []
        for tid in seq[1:]:  # skip BOS
            if tid == eos_id or tid == pad_id:
                break
            op = inv_vocab.get(tid)
            if op and op not in SPECIAL_TOKENS:
                ops.append(op)
        sampled_ops_list.append(ops)

    return sampled_ops_list


def F_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax utility helper."""
    return torch.softmax(x, dim=dim)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAFT (Rejection Sampling Fine-Tuning) for H-cGQE")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Input model checkpoint path")
    parser.add_argument("--hamiltonians", type=Path, required=True, help="Path to Hamiltonians JSON record file")
    parser.add_argument("--out", type=Path, required=True, help="Output model checkpoint path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of alignment fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Fine-tuning batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Fine-tuning learning rate")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of candidate circuits to sample per molecule")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top-performing sequences to select per molecule")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling cutoff")
    parser.add_argument("--max-seq-len", type=int, default=64, help="Max sequence length of generated circuits")
    parser.add_argument("--max-terms", type=int, default=128, help="Max terms to keep in input tokenization")
    parser.add_argument("--max-pauli-len", type=int, default=24, help="Max Pauli word length in tokenization")
    parser.add_argument("--max-qubits", type=int, default=48, help="Skip molecules with more qubits")
    parser.add_argument("--target", type=str, default="nvidia", help="CUDA-Q target device for optimization")
    parser.add_argument("--target-option", type=str, default="mqpu", help="CUDA-Q target option")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA for training")
    parser.add_argument("--adaptive-n-samples", action="store_true",
                        help="Scale n_samples by qubit count (test-time compute scaling). "
                             "Allocates more samples to harder molecules: N = n_samples * max(1, n_qubits/4). "
                             "Following Snell et al. 2024 (Google DeepMind), compute-optimal allocation "
                             "is 4x more efficient than uniform Best-of-N.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    _seed_everything(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set CUDA-Q target if available
    if cudaq and args.target:
        try:
            if args.target == "nvidia" and args.target_option:
                cudaq.set_target(args.target, option=args.target_option)
            else:
                cudaq.set_target(args.target)
            print(f"CUDA-Q target set to: {args.target}")
        except Exception as e:
            print(f"Warning: Could not set CUDA-Q target {args.target}, error: {e}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    vocab = checkpoint["vocab"]
    inv_vocab = {v: k for k, v in vocab.items()}
    model_state = checkpoint["model_state_dict"]

    # Reconstruct model architecture from checkpoint metadata
    hparams = checkpoint.get("hyperparameters", {})
    hidden_size = hparams.get("hidden_size", 256)
    num_layers = hparams.get("num_layers", 6)
    vocab_size = len(vocab)

    print(f"Reconstructed H-cGQE Model: hidden_size={hidden_size}, layers={num_layers}, vocab_size={vocab_size}")
    model = HcGQEModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    model.load_state_dict(model_state)
    model.to(device)

    # Load Hamiltonians
    ham_records = load_hamiltonian_records(args.hamiltonians)
    print(f"Loaded {len(ham_records)} Hamiltonian records.")

    # Step 1 & 2: Generate and Classically Optimize Candidates (Rejection Sampling)
    print("\n" + "=" * 80)
    print("STAGE 1: Rejection Sampling & Candidate Filtration")
    print("=" * 80)

    aligned_samples = []

    for record in ham_records:
        molecule = record["name"]
        n_qubits = int(record["n_qubits"])
        if n_qubits > args.max_qubits:
            print(f"\nSkipping {molecule} ({n_qubits} qubits > max {args.max_qubits})")
            continue
        print(f"\nProcessing molecule: {molecule} ({n_qubits} qubits)...")

        # Adaptive test-time compute: scale samples by molecule difficulty
        if args.adaptive_n_samples:
            effective_n_samples = max(args.n_samples, args.n_samples * max(1, n_qubits // 4))
            print(f"  Adaptive N: {effective_n_samples} (base={args.n_samples}, qubits={n_qubits})")
        else:
            effective_n_samples = args.n_samples

        # Extract terms
        terms = []
        for t in record.get("terms", []):
            terms.append((t["term"], float(t.get("real", 0.0))))
        terms.sort(key=lambda x: abs(x[1]), reverse=True)

        # Tokenize Hamiltonian input
        ham_tokens = tokenize_hamiltonian(terms, vocab, args.max_terms, args.max_pauli_len)
        pauli_ids = ham_tokens["pauli_ids"].unsqueeze(0)
        coeffs = ham_tokens["coeffs"].unsqueeze(0)
        term_mask = ham_tokens["term_mask"].unsqueeze(0)

        # Sample sequences from policy
        print(f"  Sampling {effective_n_samples} candidate sequences (T={args.temperature}, top_p={args.top_p})...")
        candidates = sample_candidates(
            model=model,
            pauli_ids=pauli_ids,
            coeffs=coeffs,
            term_mask=term_mask,
            n_samples=effective_n_samples,
            max_seq_len=args.max_seq_len,
            temperature=args.temperature,
            vocab=vocab,
            inv_vocab=inv_vocab,
            device=device,
            top_p=args.top_p,
        )

        # De-duplicate candidate sequences
        unique_candidates = []
        seen = set()
        for c in candidates:
            c_tuple = tuple(c)
            if c_tuple and c_tuple not in seen:
                seen.add(c_tuple)
                unique_candidates.append(c)

        print(f"  Found {len(unique_candidates)} unique non-empty sequences.")

        # Classically optimize all candidate sequences
        print(f"  Optimizing rotation coefficients classically via L-BFGS-B...")
        results = []
        for i, ops in enumerate(unique_candidates):
            try:
                energy, thetas = _optimize_coefficients(
                    molecule_record=record,
                    operators=ops,
                    max_iter=100
                )
                results.append({"operators": ops, "energy": energy, "thetas": thetas})
            except Exception as e:
                pass

        if not results:
            print(f"  Warning: No successful optimizations for {molecule}, skipping.")
            continue

        # Sort candidate sequences by lowest optimized energy
        results.sort(key=lambda x: x["energy"])

        # Filter the top-k winning sequences
        top_winners = results[:args.top_k]
        print(f"  Selected top-{len(top_winners)} winning sequences:")
        for rank, w in enumerate(top_winners):
            print(f"    Rank {rank+1}: E = {w['energy']:.6f} Ha ({len(w['operators'])} ops)")

        # Prepare tokens for SFT fine-tuning dataset
        for w in top_winners:
            tgt_tokens = tokenize_operator_sequence(w["operators"], vocab, args.max_seq_len)
            aligned_samples.append({
                "pauli_ids": ham_tokens["pauli_ids"],
                "coeffs": ham_tokens["coeffs"],
                "term_mask": ham_tokens["term_mask"],
                "tgt_tokens": tgt_tokens,
            })

    if not aligned_samples:
        sys.exit("Error: No winning sequences found across all molecules. Alignment failed.")

    # Stack into tensors for dataset
    pauli_ids_t = torch.stack([s["pauli_ids"] for s in aligned_samples])
    coeffs_t = torch.stack([s["coeffs"] for s in aligned_samples])
    term_mask_t = torch.stack([s["term_mask"] for s in aligned_samples])
    tgt_tokens_t = torch.stack([s["tgt_tokens"] for s in aligned_samples])

    print(f"\nFinal RAFT dataset built with {len(aligned_samples)} total samples.")
    dataset = TensorDataset(pauli_ids_t, coeffs_t, term_mask_t, tgt_tokens_t)
    
    # Custom loader collate matching SFT formats
    def collate_fn(batch):
        return {
            "pauli_ids": torch.stack([item[0] for item in batch]),
            "coeffs": torch.stack([item[1] for item in batch]),
            "term_mask": torch.stack([item[2] for item in batch]),
            "tgt_tokens": torch.stack([item[3] for item in batch]),
        }

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Step 3: Fine-Tuning
    print("\n" + "=" * 80)
    print("STAGE 2: Supervised Alignment Fine-Tuning (SFT)")
    print("=" * 80)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["<PAD>"])

    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            pad_id=SPECIAL_TOKENS["<PAD>"],
            scaler=None,
            grad_accum_steps=1,
            inv_vocab=inv_vocab,
            commutator_weight=0.0,
            epoch=epoch,
        )
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{args.epochs:2d} | Loss: {metrics['loss']:.4f} | Accuracy: {metrics['acc']*100:.2f}% | Perplexity: {metrics['perplexity']:.4f}")

    # Save aligned checkpoint
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["model_state_dict"] = model.state_dict()
    # Add RAFT metadata
    checkpoint["metadata"]["raft_samples"] = len(aligned_samples)
    checkpoint["metadata"]["raft_alignment_complete"] = True
    
    torch.save(checkpoint, args.out)
    print(f"\nSuccessfully aligned model saved to: {args.out}")


if __name__ == "__main__":
    main()
