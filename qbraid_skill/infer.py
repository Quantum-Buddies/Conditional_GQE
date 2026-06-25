#!/usr/bin/env python3
"""qBraid Skill wrapper: run H-cGQE inference with constrained decoding."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.models.infer_h_cgqe import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(description="H-cGQE inference (qBraid Skill)")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results" / "train" / "h_cgqe_model.pt")
    parser.add_argument("--hamiltonians", type=Path, default=ROOT / "results" / "data" / "hamiltonians.json")
    parser.add_argument("--molecules", nargs="+", default=["h2", "lih", "beh2", "n2"])
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "inference" / "h_cgqe_generated.json")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--force-entanglement", action="store_true", default=True)
    parser.add_argument("--sample", action="store_true", default=True)
    parser.add_argument("--max-repeat", type=int, default=4)
    parser.add_argument("--use-cuda", action="store_true", default=True)
    args = parser.parse_args()

    sys.argv = [
        "infer_h_cgqe.py",
        "--checkpoint", str(args.checkpoint),
        "--hamiltonians", str(args.hamiltonians),
        "--molecules", *args.molecules,
        "--out", str(args.out),
        "--n-samples", str(args.n_samples),
        "--force-entanglement" if args.force_entanglement else "",
        "--sample" if args.sample else "",
        "--max-repeat", str(args.max_repeat),
        "--use-cuda" if args.use_cuda else "",
    ]
    sys.argv = [x for x in sys.argv if x]
    _main()


if __name__ == "__main__":
    main()
