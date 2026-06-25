#!/usr/bin/env python3
"""qBraid Skill wrapper: evaluate H-cGQE circuits on a qBraid device."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.eval.qbraid_backend import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate H-cGQE on qBraid device (qBraid Skill)")
    parser.add_argument("--hamiltonians", type=Path, default=ROOT / "results" / "data" / "hamiltonians.json")
    parser.add_argument("--generated", type=Path, default=ROOT / "results" / "inference" / "h_cgqe_generated.json")
    parser.add_argument("--optimized", type=Path, default=ROOT / "results" / "eval" / "h_cgqe_optimized.json")
    parser.add_argument("--molecule", type=str, default="h2")
    parser.add_argument("--device", type=str, default="qbraid_qir_simulator")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "eval" / "qbraid_energy.json")
    args = parser.parse_args()

    sys.argv = [
        "qbraid_backend.py",
        "--hamiltonians", str(args.hamiltonians),
        "--generated", str(args.generated),
        "--optimized", str(args.optimized),
        "--molecule", args.molecule,
        "--device", args.device,
        "--shots", str(args.shots),
        "--out", str(args.out),
    ]
    _main()


if __name__ == "__main__":
    main()
