#!/usr/bin/env python3
"""qBraid Skill wrapper: train the H-cGQE Transformer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.models.train_h_cgqe import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(description="Train H-cGQE Transformer (qBraid Skill)")
    parser.add_argument("--dataset", type=Path, default=ROOT / "results" / "train" / "gqe_supervised_dataset.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "train" / "h_cgqe_model.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--commutator-weight", type=float, default=0.5)
    parser.add_argument("--commutator-ramp-epochs", type=int, default=100)
    parser.add_argument("--use-cuda", action="store_true", default=True)
    parser.add_argument("--use-fp16", action="store_true", default=True)
    args = parser.parse_args()

    sys.argv = [
        "train_h_cgqe.py",
        "--dataset", str(args.dataset),
        "--out", str(args.out),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--commutator-weight", str(args.commutator_weight),
        "--commutator-ramp-epochs", str(args.commutator_ramp_epochs),
        "--use-cuda" if args.use_cuda else "",
        "--use-fp16" if args.use_fp16 else "",
    ]
    sys.argv = [x for x in sys.argv if x]
    _main()


if __name__ == "__main__":
    main()
