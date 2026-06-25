#!/usr/bin/env python3
"""qBraid Skill wrapper: generate H-cGQE benchmark plots."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.eval.plot_h_cgqe_scaling import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot H-cGQE scaling (qBraid Skill)")
    parser.add_argument("--eval", type=Path, default=ROOT / "results" / "eval" / "h_cgqe_evaluation.json")
    parser.add_argument("--optimized", type=Path, default=ROOT / "results" / "eval" / "h_cgqe_optimized.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "plots")
    args = parser.parse_args()

    sys.argv = [
        "plot_h_cgqe_scaling.py",
        "--eval", str(args.eval),
        "--optimized", str(args.optimized),
        "--out-dir", str(args.out_dir),
    ]
    _main()


if __name__ == "__main__":
    main()
