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
    
    # Forward the new async options
    parser.add_argument("--submit-only", action="store_true", help="Submit batch job asynchronously and exit")
    parser.add_argument("--retrieve", type=Path, default=None, help="Retrieve asynchronously submitted job from metadata file")
    
    args = parser.parse_args()

    # Reconstruct sys.argv for the underlying qbraid_backend.py script
    sys.argv = ["qbraid_backend.py"]
    
    if args.retrieve:
        sys.argv += ["--retrieve", str(args.retrieve)]
        if args.out:
            sys.argv += ["--out", str(args.out)]
    else:
        sys.argv += [
            "--hamiltonians", str(args.hamiltonians),
            "--generated", str(args.generated),
            "--optimized", str(args.optimized),
            "--molecule", args.molecule,
            "--device", args.device,
            "--shots", str(args.shots),
            "--out", str(args.out),
        ]
        if args.submit_only:
            sys.argv += ["--submit-only"]
            
    _main()


if __name__ == "__main__":
    main()
