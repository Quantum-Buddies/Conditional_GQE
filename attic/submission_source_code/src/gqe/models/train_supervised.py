import argparse
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import yaml
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=r"CUDA initialization: The NVIDIA driver on your system is too old.*",
    category=UserWarning,
)


def _coerce_optional_int(value: str | None, default: int) -> int:
    return default if value is None else int(value)


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_synthetic_dataset(torch, n: int, seq_len: int, vocab_size: int = 64):
    x = torch.randint(0, vocab_size, (n, seq_len), dtype=torch.long)
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict supervised CE training scaffold.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seq-samples", type=int, default=None, help="Override synthetic sample count.")
    parser.add_argument("--model-hidden", type=int, default=None, help="Override GRU hidden width.")
    parser.add_argument("--model-layers", type=int, default=None, help="Override GRU layer count.")
    parser.add_argument("--model-vocab", type=int, default=None, help="Override token vocab size.")
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Opt into CUDA if available.",
    )
    args = parser.parse_args()

    if not args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    import torch.nn as nn
    import torch.optim as optim

    class TinySeqModel(nn.Module):
        def __init__(
            self,
            vocab_size: int = 64,
            hidden: int = 64,
            num_layers: int = 1,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden)
            self.rnn = nn.GRU(
                hidden,
                hidden,
                batch_first=True,
                num_layers=max(1, num_layers),
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.proj = nn.Linear(hidden, vocab_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            out, _ = self.rnn(x)
            return self.proj(out)

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    _seed_everything(seed, torch)

    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])
    seq_len = int(cfg["training"]["max_seq_len"])
    seq_samples = int(cfg["training"].get("synthetic_samples", 256))
    if args.seq_samples is not None:
        seq_samples = args.seq_samples

    model_cfg = cfg.get("model", {})
    vocab_size = _coerce_optional_int(args.model_vocab, int(model_cfg.get("vocab_size", 64)))
    hidden_size = _coerce_optional_int(args.model_hidden, int(model_cfg.get("hidden_size", 64)))
    num_layers = _coerce_optional_int(args.model_layers, int(model_cfg.get("num_layers", 1)))
    dropout = float(model_cfg.get("dropout", 0.0))

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    if args.use_cuda and device.type != "cuda":
        print("CUDA requested but unavailable. Falling back to CPU.")

    model = TinySeqModel(
        vocab_size=vocab_size,
        hidden=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.to(device)
    x, y = _build_synthetic_dataset(torch, n=seq_samples, seq_len=seq_len, vocab_size=vocab_size)
    x = x.to(device)
    y = y.to(device)
    model.train()
    batch_losses = []
    epoch_losses = []
    n_batch_iters = (x.size(0) + batch_size - 1) // batch_size
    epoch_iter = tqdm(range(epochs), desc="Epoch", unit="epoch", dynamic_ncols=True, disable=None)
    for epoch in epoch_iter:
        running_loss = 0.0
        n_batches = 0
        for i in tqdm(
            range(0, x.size(0), batch_size),
            desc="Batch",
            unit="batch",
            total=n_batch_iters,
            leave=False,
            dynamic_ncols=True,
            disable=None,
        ):
            xb = x[i : i + batch_size]
            yb = y[i : i + batch_size]
            logits = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = float(loss.item())
            batch_losses.append(batch_loss)
            running_loss += batch_loss
            n_batches += 1
        epoch_t = running_loss / max(n_batches, 1)
        epoch_losses.append(epoch_t)
        epoch_iter.set_postfix_str(f"loss={epoch_t:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out.parent / "supervised_model.pt")
    with (args.out.parent / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "strict_supervised_ce",
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "model": {
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                },
                "synthetic_samples": seq_samples,
                "batch_losses": batch_losses,
                "epoch_losses": epoch_losses,
                "final_loss": epoch_losses[-1] if epoch_losses else None,
                "mean_batch_loss": float(np.mean(batch_losses)) if batch_losses else None,
            },
            f,
            indent=2,
        )
    print(f"Wrote model and metrics under: {args.out.parent}")


if __name__ == "__main__":
    main()

