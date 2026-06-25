#!/usr/bin/env python3
"""qBraid Skill wrapper: optimize H-cGQE operator coefficients on CUDA-Q."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.eval.optimize_h_cgqe_coefficients import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize H-cGQE coefficients (qBraid Skill)")
    parser.add_argument("--generated", type=Path, default=ROOT / "results" / "inference" / "h_cgqe_generated.json")
    parser.add_argument("--hamiltonians", type=Path, default=ROOT / "results" / "data" / "hamiltonians.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "eval" / "h_cgqe_optimized.json")
    parser.add_argument("--target", type=str, default="nvidia")
    parser.add_argument("--target-option", type=str, default="mqpu")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--parallel-gpus", type=int, default=None)
    args = parser.parse_args()

    sys.argv = [
        "optimize_h_cgqe_coefficients.py",
        "--generated", str(args.generated),
        "--hamiltonians", str(args.hamiltonians),
        "--out", str(args.out),
        "--target", args.target,
    ]
    if args.target_option:
        sys.argv += ["--target-option", args.target_option]
    if args.parallel_gpus is not None:
        sys.argv += ["--parallel-gpus", str(args.parallel_gpus)]
    sys.argv += ["--max-iter", str(args.max_iter), "--top-k", str(args.top_k)]
    _main()


if __name__ == "__main__":
    main()
