#!/usr/bin/env python3
"""qBraid Skill wrapper: evaluate H-cGQE circuits against the GQE baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.eval.evaluate_h_cgqe import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate H-cGQE circuits (qBraid Skill)")
    parser.add_argument("--generated", type=Path, default=ROOT / "results" / "inference" / "h_cgqe_generated.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "results" / "baselines" / "cudaq_gqe.json")
    parser.add_argument("--hamiltonians", type=Path, default=ROOT / "results" / "data" / "hamiltonians.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "eval" / "h_cgqe_evaluation.json")
    parser.add_argument("--target", type=str, default="nvidia")
    parser.add_argument("--target-option", type=str, default="mqpu")
    args = parser.parse_args()

    sys.argv = [
        "evaluate_h_cgqe.py",
        "--generated", str(args.generated),
        "--baseline", str(args.baseline),
        "--hamiltonians", str(args.hamiltonians),
        "--out", str(args.out),
        "--target", args.target,
    ]
    if args.target_option:
        sys.argv += ["--target-option", args.target_option]
    _main()


if __name__ == "__main__":
    main()
