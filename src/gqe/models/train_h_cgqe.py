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
    length = max(len(word_a), len(word_b))
    n_anticomm = 0
    for i in range(length):
        a = word_a[i] if i < len(word_a) else "I"
        b = word_b[i] if i < len(word_b) else "I"
        if a == "I" or b == "I" or a == b:
            continue
        n_anticomm += 1
    return n_anticomm % 2 == 0


def _normalize_inv_vocab(inv_vocab: dict[Any, str] | None) -> dict[int, str]:
    """Ensure inv_vocab keys are ints (torch/json may store string keys)."""
    if not inv_vocab:
        return {}
    return {int(k): v for k, v in inv_vocab.items()}


def build_commute_table(inv_vocab: dict[int, str], vocab_size: int) -> np.ndarray:
    """Precompute V×V pairwise commute matrix for O(1) pair lookups.

    commute[i, j] = 1 if token i and j commute (or either is special/unknown).
    """
    words: list[str | None] = [None] * vocab_size
    for tid, word in inv_vocab.items():
        if 0 <= tid < vocab_size and word not in SPECIAL_TOKENS:
            words[tid] = word

    table = np.ones((vocab_size, vocab_size), dtype=np.float32)
    for i in range(vocab_size):
        wi = words[i]
        if wi is None:
            continue
        for j in range(i + 1, vocab_size):
            wj = words[j]
            if wj is None:
                continue
            c = 1.0 if _words_commute(wi, wj) else 0.0
            table[i, j] = c
            table[j, i] = c
    return table


def _commutator_penalty(
    token_ids: torch.Tensor,
    inv_vocab: dict[int, str] | None = None,
    pad_id: int = 0,
    commute_table: np.ndarray | None = None,
) -> torch.Tensor:
    """Mean fraction of commuting operator pairs (higher = more diagonal).

    Uses a precomputed commute table when available (fast path). Falls back
    to the original Python loop only if no table is provided.
    """
    device = token_ids.device
    bsz = token_ids.shape[0]
    total_penalty = 0.0
    total_seqs = 0

    if commute_table is not None:
        # Fast path: numpy index lookups instead of string Pauli comparisons
        ids_cpu = token_ids.detach().cpu().numpy()
        for b in range(bsz):
            seq = ids_cpu[b]
            ops = [int(t) for t in seq if t != pad_id and 0 <= int(t) < commute_table.shape[0]]
            # Drop specials (PAD/BOS/EOS/UNK live in 0..3)
            ops = [t for t in ops if t >= 4]
            n = len(ops)
            if n < 2:
                continue
            n_pairs = n * (n - 1) // 2
            # Cap pair checks for very long sequences
            if n_pairs > 200:
                # Sample pairs uniformly
                rng = np.random.default_rng(abs(hash((b, n))) % (2**32))
                i_idx = rng.integers(0, n, size=200)
                j_idx = rng.integers(0, n, size=200)
                mask = i_idx != j_idx
                i_idx, j_idx = i_idx[mask], j_idx[mask]
                if len(i_idx) == 0:
                    continue
                ops_arr = np.asarray(ops, dtype=np.int64)
                n_commute = float(commute_table[ops_arr[i_idx], ops_arr[j_idx]].sum())
                total_penalty += n_commute / len(i_idx)
            else:
                ops_arr = np.asarray(ops, dtype=np.int64)
                # Upper triangle via broadcasting
                ii, jj = np.triu_indices(n, k=1)
                n_commute = float(commute_table[ops_arr[ii], ops_arr[jj]].sum())
                total_penalty += n_commute / n_pairs
            total_seqs += 1
    else:
        inv = inv_vocab or {}
        for b in range(bsz):
            seq = token_ids[b].tolist()
            words: list[str] = []
            for tid in seq:
                if tid == pad_id:
                    break
                word = _pauli_word_from_token_id(int(tid), inv)
                if word is not None:
                    words.append(word)
            n = len(words)
            if n < 2:
                continue
            n_pairs = n * (n - 1) // 2
            n_commute = sum(
                1 for i in range(n) for j in range(i + 1, n) if _words_commute(words[i], words[j])
            )
            total_penalty += n_commute / n_pairs
            total_seqs += 1

    if total_seqs == 0:
        return torch.tensor(0.0, device=device)
    return torch.tensor(total_penalty / total_seqs, device=device)


class EarlyStopping:
    """Simple early stopping based on validation loss."""
    def __init__(self, patience: int = 30, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_random_baseline(vocab_size: int, seq_len: int) -> float:
    """Cross-entropy of uniform random guessing."""
    return -np.log(1.0 / vocab_size) * seq_len


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap torch.compile and/or DataParallel for checkpoint saving."""
    m = getattr(model, "_orig_mod", model)
    if isinstance(m, nn.DataParallel):
        m = m.module
    return m


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
    scaler: Any | None = None,
    amp_dtype: torch.dtype | None = None,
    grad_accum_steps: int = 1,
    inv_vocab: dict[int, str] | None = None,
    commute_table: np.ndarray | None = None,
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

    optimizer.zero_grad(set_to_none=True)

    # Curriculum ramp: linearly increase commutator weight over ramp_epochs
    effective_comm_weight = commutator_weight
    if commutator_ramp_epochs > 0:
        effective_comm_weight = min(1.0, epoch / max(commutator_ramp_epochs, 1)) * commutator_weight

    use_amp = amp_dtype is not None and device.type == "cuda"

    for i, batch in enumerate(loader):
        pauli_ids = batch["pauli_ids"].to(device, non_blocking=True)
        coeffs = batch["coeffs"].to(device, non_blocking=True)
        term_mask = batch["term_mask"].to(device, non_blocking=True)
        tgt_tokens = batch["tgt_tokens"].to(device, non_blocking=True)

        # Target: shift right for teacher forcing
        tgt_input = tgt_tokens[:, :-1]
        tgt_labels = tgt_tokens[:, 1:]
        tgt_key_padding_mask = tgt_input == pad_id

        def _forward_loss() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits_ = model(
                pauli_ids,
                coeffs,
                tgt_input,
                term_mask=term_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            ce_ = criterion(logits_.reshape(-1, logits_.size(-1)), tgt_labels.reshape(-1))
            if effective_comm_weight > 0 and (commute_table is not None or inv_vocab):
                comm_ = _commutator_penalty(
                    tgt_tokens, inv_vocab=inv_vocab, pad_id=pad_id, commute_table=commute_table,
                )
            else:
                comm_ = torch.zeros((), device=device)
            return logits_, ce_, comm_, ce_ + effective_comm_weight * comm_

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, ce_loss, comm_penalty, loss = _forward_loss()
            loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            logits, ce_loss, comm_penalty, loss = _forward_loss()
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
            optimizer.zero_grad(set_to_none=True)

        # Metrics (ignore padding)
        with torch.no_grad():
            mask = tgt_labels != pad_id
            preds = logits.argmax(dim=-1)
            correct = ((preds == tgt_labels) & mask).sum().item()
            n_tokens = mask.sum().item()

        total_loss += loss.item() * grad_accum_steps  # unscale for reporting
        total_ce_loss += ce_loss.item()
        total_comm_loss += float(comm_penalty.detach().item())
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
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0
    n_batches = 0
    use_amp = amp_dtype is not None and device.type == "cuda"

    for batch in loader:
        pauli_ids = batch["pauli_ids"].to(device, non_blocking=True)
        coeffs = batch["coeffs"].to(device, non_blocking=True)
        term_mask = batch["term_mask"].to(device, non_blocking=True)
        tgt_tokens = batch["tgt_tokens"].to(device, non_blocking=True)

        tgt_input = tgt_tokens[:, :-1]
        tgt_labels = tgt_tokens[:, 1:]
        tgt_key_padding_mask = tgt_input == pad_id

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(
                    pauli_ids,
                    coeffs,
                    tgt_input,
                    term_mask=term_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
        else:
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
    parser.add_argument("--multi-gpu", action="store_true", help="Use nn.DataParallel for multi-GPU training")
    parser.add_argument("--use-fp16", action="store_true", help="Use mixed precision (FP16) training")
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        help="Use mixed precision (BF16) training (preferred on Blackwell; no GradScaler)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile to the model after moving to device (slow warmup; usually skip for small SFT)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default 0: avoid CUDA fork hangs; pin_memory still helps)",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Run validation / early-stopping / checkpoint every N epochs (last epoch always validated)",
    )
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--commutator-weight", type=float, default=0.0,
                        help="Weight for the commutator penalty term (set >0 to enable curriculum entanglement loss)")
    parser.add_argument("--commutator-ramp-epochs", type=int, default=100,
                        help="Number of epochs over which to linearly ramp the commutator penalty to full weight")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0.1 recommended for small datasets)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum val loss improvement to reset patience counter")
    args = parser.parse_args()

    _seed_everything(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print("TF32 enabled for matmul/cudnn")

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

    pin_memory = device.type == "cuda"
    num_workers = args.num_workers
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "collate_fn": _collate_fn,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"  DataLoader: num_workers={num_workers}, pin_memory={pin_memory}")

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

    if args.compile:
        print("NOTE: torch.compile warmup can take minutes on first epoch for this model size.")
        model = torch.compile(model)
        print("torch.compile enabled")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if args.multi_gpu and torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        model = nn.DataParallel(model)
        print(f"Using nn.DataParallel with {n_gpus} GPUs (effective batch size = {args.batch_size * n_gpus})")

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)  # ignore PAD
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    inv = _normalize_inv_vocab(dataset.get("inv_vocab"))
    commute_table = build_commute_table(inv, vocab_size) if args.commutator_weight > 0 else None
    if commute_table is not None:
        print(f"Precomputed commute table: {commute_table.shape}")

    # Mixed precision setup (--use-bf16 preferred over --use-fp16)
    if args.use_bf16 and device.type == "cuda":
        amp_dtype = torch.bfloat16
        scaler = None
        print("Mixed precision (BF16) enabled")
    elif args.use_fp16 and device.type == "cuda":
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
        print("Mixed precision (FP16) enabled")
    else:
        amp_dtype = None
        scaler = None
        if args.use_bf16 or args.use_fp16:
            print("AMP requested but CUDA unavailable; running in FP32")

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

    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    val_every = max(1, int(args.val_every))
    if val_every > 1:
        print(f"Validation every {val_every} epochs (last epoch always validated)")

    pbar = tqdm(range(args.epochs), desc="Epoch", unit="epoch")
    last_val_metrics: dict[str, float] = {"loss": float("inf"), "acc": 0.0}
    for epoch in pbar:
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            amp_dtype=amp_dtype,
            grad_accum_steps=args.grad_accum,
            inv_vocab=inv,
            commute_table=commute_table,
            commutator_weight=args.commutator_weight,
            epoch=epoch,
            commutator_ramp_epochs=args.commutator_ramp_epochs,
        )
        do_val = ((epoch + 1) % val_every == 0) or (epoch + 1 == args.epochs)
        if do_val:
            last_val_metrics = evaluate(model, val_loader, criterion, device, amp_dtype=amp_dtype)
        val_metrics = last_val_metrics
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

        if do_val and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": _unwrap_model(model).state_dict(),
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

        # Early stopping only when we actually validated
        if do_val and early_stopping(val_metrics["loss"]):
            print(f"\nEarly stopping triggered at epoch {epoch+1} (patience={args.patience})")
            break

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
            "early_stopped": early_stopping.early_stop,
            "epochs_run": len(train_losses),
        }, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
