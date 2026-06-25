"""Train the H-cGQE Transformer on real GQE operator sequences.

Loads a prepared dataset (from prepare_gqe_dataset.py) and trains the
Hamiltonian-conditioned Transformer to predict operator sequences.

Usage:
    python src/gqe/models/train_h_cgqe.py \
        --dataset results/train/gqe_supervised_dataset.pt \
        --out results/train/h_cgqe_model.pt \
        --epochs 500 --batch-size 4 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.gqe.models.h_cgqe_transformer import HcGQEModel, SPECIAL_TOKENS


def _pauli_word_from_token_id(token_id: int, inv_vocab: dict[int, str]) -> str | None:
    """Return the Pauli word for a token ID, or None for special tokens."""
    word = inv_vocab.get(token_id)
    if word is None or word in SPECIAL_TOKENS:
        return None
    return word


def _words_commute(word_a: str, word_b: str) -> bool:
    """Return True if two Pauli words commute.

    Two Pauli words commute iff the number of qubits where they differ and
    neither is identity is even.
    """
    # Pad to same length
    length = max(len(word_a), len(word_b))
    n_anticomm = 0
    for i in range(length):
        a = word_a[i] if i < len(word_a) else "I"
        b = word_b[i] if i < len(word_b) else "I"
        if a == "I" or b == "I" or a == b:
            continue
        n_anticomm += 1
    return n_anticomm % 2 == 0


def _commutator_penalty(
    token_ids: torch.Tensor,
    inv_vocab: dict[int, str],
    pad_id: int = 0,
) -> torch.Tensor:
    """Compute a penalty for sequences that contain mostly commuting operators.

    Returns a scalar tensor: mean fraction of commuting operator pairs per
    non-padding sequence. Higher values mean the sequence is more Z-like /
    diagonal and should be penalized.
    """
    bsz, seq_len = token_ids.shape
    device = token_ids.device
    total_penalty = 0.0
    total_seqs = 0

    for b in range(bsz):
        seq = token_ids[b].tolist()
        words: list[str] = []
        for tid in seq:
            if tid == pad_id:
                break
            word = _pauli_word_from_token_id(tid, inv_vocab)
            if word is not None:
                words.append(word)

        n = len(words)
        if n < 2:
            continue
        n_pairs = n * (n - 1) // 2
        n_commute = 0
        for i in range(n):
            for j in range(i + 1, n):
                if _words_commute(words[i], words[j]):
                    n_commute += 1
        total_penalty += n_commute / n_pairs
        total_seqs += 1

    if total_seqs == 0:
        return torch.tensor(0.0, device=device)
    return torch.tensor(total_penalty / total_seqs, device=device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_random_baseline(vocab_size: int, seq_len: int) -> float:
    """Cross-entropy of uniform random guessing."""
    return -np.log(1.0 / vocab_size) * seq_len


def _collate_fn(batch: list[tuple[Any, ...]]) -> dict[str, torch.Tensor]:
    """Collate a batch of dataset samples."""
    pauli_ids = torch.stack([item[0] for item in batch])
    coeffs = torch.stack([item[1] for item in batch])
    term_mask = torch.stack([item[2] for item in batch])
    tgt_tokens = torch.stack([item[3] for item in batch])
    return {
        "pauli_ids": pauli_ids,
        "coeffs": coeffs,
        "term_mask": term_mask,
        "tgt_tokens": tgt_tokens,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    pad_id: int = 0,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_accum_steps: int = 1,
    inv_vocab: dict[int, str] | None = None,
    commutator_weight: float = 0.0,
    epoch: int = 0,
    commutator_ramp_epochs: int = 0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_comm_loss = 0.0
    total_acc = 0.0
    total_tokens = 0
    n_batches = 0

    optimizer.zero_grad()

    # Curriculum ramp: linearly increase commutator weight over ramp_epochs
    effective_comm_weight = commutator_weight
    if commutator_ramp_epochs > 0:
        effective_comm_weight = min(1.0, epoch / max(commutator_ramp_epochs, 1)) * commutator_weight

    for i, batch in enumerate(loader):
        pauli_ids = batch["pauli_ids"].to(device)
        coeffs = batch["coeffs"].to(device)
        term_mask = batch["term_mask"].to(device)
        tgt_tokens = batch["tgt_tokens"].to(device)

        # Target: shift right for teacher forcing
        # Input to decoder: all but last token
        # Target labels: all but first token (after BOS)
        tgt_input = tgt_tokens[:, :-1]
        tgt_labels = tgt_tokens[:, 1:]

        # Padding mask for target
        tgt_key_padding_mask = tgt_input == pad_id

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(
                    pauli_ids,
                    coeffs,
                    tgt_input,
                    term_mask=term_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                ce_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
                comm_penalty = _commutator_penalty(tgt_tokens, inv_vocab, pad_id) if (inv_vocab and effective_comm_weight > 0) else torch.tensor(0.0, device=device)
                loss = ce_loss + effective_comm_weight * comm_penalty
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(
                pauli_ids,
                coeffs,
                tgt_input,
                term_mask=term_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            ce_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
            comm_penalty = _commutator_penalty(tgt_tokens, inv_vocab, pad_id) if (inv_vocab and effective_comm_weight > 0) else torch.tensor(0.0, device=device)
            loss = ce_loss + effective_comm_weight * comm_penalty
            loss = loss / grad_accum_steps
            loss.backward()

        # Gradient accumulation: step optimizer every grad_accum_steps
        if (i + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        # Metrics (ignore padding)
        mask = tgt_labels != pad_id
        preds = logits.argmax(dim=-1)
        correct = ((preds == tgt_labels) & mask).sum().item()
        n_tokens = mask.sum().item()

        total_loss += loss.item() * grad_accum_steps  # unscale for reporting
        total_ce_loss += ce_loss.item()
        total_comm_loss += comm_penalty.item()
        total_acc += correct
        total_tokens += n_tokens
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "ce_loss": total_ce_loss / max(n_batches, 1),
        "comm_loss": total_comm_loss / max(n_batches, 1),
        "acc": total_acc / max(total_tokens, 1),
        "perplexity": np.exp(total_loss / max(n_batches, 1)),
        "effective_comm_weight": effective_comm_weight,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_id: int = 0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in loader:
        pauli_ids = batch["pauli_ids"].to(device)
        coeffs = batch["coeffs"].to(device)
        term_mask = batch["term_mask"].to(device)
        tgt_tokens = batch["tgt_tokens"].to(device)

        tgt_input = tgt_tokens[:, :-1]
        tgt_labels = tgt_tokens[:, 1:]
        tgt_key_padding_mask = tgt_input == pad_id

        logits = model(
            pauli_ids,
            coeffs,
            tgt_input,
            term_mask=term_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))

        mask = tgt_labels != pad_id
        preds = logits.argmax(dim=-1)
        correct = ((preds == tgt_labels) & mask).sum().item()
        n_tokens = mask.sum().item()

        total_loss += loss.item()
        total_acc += correct
        total_tokens += n_tokens
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": total_acc / max(total_tokens, 1),
        "perplexity": np.exp(total_loss / max(n_batches, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train H-cGQE Transformer on real GQE data.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to .pt dataset")
    parser.add_argument("--out", type=Path, required=True, help="Path to save model checkpoint")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--use-fp16", action="store_true", help="Use mixed precision (FP16) training")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--commutator-weight", type=float, default=0.0,
                        help="Weight for the commutator penalty term (set >0 to enable curriculum entanglement loss)")
    parser.add_argument("--commutator-ramp-epochs", type=int, default=100,
                        help="Number of epochs over which to linearly ramp the commutator penalty to full weight")
    args = parser.parse_args()

    _seed_everything(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    vocab = dataset["vocab"]
    vocab_size = len(vocab)
    metadata = dataset["metadata"]
    print(f"  Vocab size: {vocab_size}")
    print(f"  Samples: {metadata['n_samples']}")
    print(f"  Max terms: {metadata['max_terms']}")
    print(f"  Max seq len: {metadata['max_seq_len']}")

    # Build TensorDataset
    full_ds = TensorDataset(
        dataset["pauli_ids"],
        dataset["coeffs"],
        dataset["term_mask"],
        dataset["tgt_tokens"],
    )

    n = len(full_ds)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * args.train_split)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_fn
    )
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Model
    model = HcGQEModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        encoder_layers=args.enc_layers,
        decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        max_pauli_len=metadata["max_pauli_len"],
        max_seq_len=metadata["max_seq_len"],
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None
    if args.use_fp16:
        print(f"Mixed precision (FP16) enabled")
    if args.grad_accum > 1:
        print(f"Gradient accumulation: {args.grad_accum} steps (effective batch size = {args.batch_size * args.grad_accum})")

    # Random baseline
    random_baseline = _compute_random_baseline(vocab_size, metadata["max_seq_len"] - 1)
    print(f"Random-guess CE baseline (per token): {np.log(vocab_size):.4f}")
    print(f"Random-guess total CE baseline: {random_baseline:.4f}")

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    train_ce_losses = []
    train_comm_losses = []
    val_losses = []
    val_accs = []
    comm_weights = []

    pbar = tqdm(range(args.epochs), desc="Epoch", unit="epoch")
    for epoch in pbar:
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, grad_accum_steps=args.grad_accum,
            inv_vocab=dataset.get("inv_vocab"),
            commutator_weight=args.commutator_weight,
            epoch=epoch,
            commutator_ramp_epochs=args.commutator_ramp_epochs,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_metrics["loss"])
        train_ce_losses.append(train_metrics["ce_loss"])
        train_comm_losses.append(train_metrics["comm_loss"])
        val_losses.append(val_metrics["loss"])
        val_accs.append(val_metrics["acc"])
        comm_weights.append(train_metrics["effective_comm_weight"])

        pbar.set_postfix_str(
            f"train_loss={train_metrics['loss']:.4f} "
            f"ce_loss={train_metrics['ce_loss']:.4f} "
            f"comm_loss={train_metrics['comm_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "inv_vocab": dataset["inv_vocab"],
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "encoder_layers": args.enc_layers,
                    "decoder_layers": args.dec_layers,
                    "dim_feedforward": args.dim_ff,
                    "dropout": args.dropout,
                    "max_pauli_len": metadata["max_pauli_len"],
                    "max_seq_len": metadata["max_seq_len"],
                },
                "metrics": {
                    "best_val_loss": best_val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_accs": val_accs,
                },
            }, args.out)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Final val accuracy: {val_accs[-1]:.4f}")
    print(f"Model saved to: {args.out}")

    # Write metrics JSON
    metrics_path = args.out.parent / f"{args.out.stem}_metrics.json"
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": config_dict,
            "model_params": n_params,
            "vocab_size": vocab_size,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_val_acc": val_accs[-1],
            "train_losses": train_losses,
            "train_ce_losses": train_ce_losses,
            "train_comm_losses": train_comm_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "comm_weights": comm_weights,
            "commutator_weight": args.commutator_weight,
            "commutator_ramp_epochs": args.commutator_ramp_epochs,
            "random_baseline_ce": random_baseline,
        }, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
