#!/usr/bin/env python3
"""qBraid Skill wrapper: prepare the supervised GQE dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.data.prepare_gqe_dataset import main as _main


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare supervised GQE dataset (qBraid Skill)")
    parser.add_argument("--ham", type=Path, default=ROOT / "results" / "data" / "hamiltonians.json")
    parser.add_argument("--gqe-results", type=Path, nargs="+", default=[ROOT / "results" / "baselines" / "cudaq_gqe.json"])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "train")
    args = parser.parse_args()

    sys.argv = [
        "prepare_gqe_dataset.py",
        "--ham", str(args.ham),
        "--gqe-results", *[str(p) for p in args.gqe_results],
        "--out-dir", str(args.out_dir),
    ]
    _main()


if __name__ == "__main__":
    main()
