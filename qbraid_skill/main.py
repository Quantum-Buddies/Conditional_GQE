#!/usr/bin/env python3
"""Main entry point for the H-cGQE qBraid Skill.

Dispatches to the appropriate sub-command wrapper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import submodules directly to avoid circular imports from eager __init__.py loading.
import qbraid_skill.install
import qbraid_skill.prepare_dataset
import qbraid_skill.train
import qbraid_skill.train_rlqf
import qbraid_skill.infer
import qbraid_skill.optimize
import qbraid_skill.evaluate
import qbraid_skill.qbraid_eval
import qbraid_skill.plot


COMMANDS = {
    "install": qbraid_skill.install.main,
    "prepare-dataset": qbraid_skill.prepare_dataset.main,
    "train": qbraid_skill.train.main,
    "train-rlqf": qbraid_skill.train_rlqf.main,
    "infer": qbraid_skill.infer.main,
    "optimize": qbraid_skill.optimize.main,
    "evaluate": qbraid_skill.evaluate.main,
    "qbraid-eval": qbraid_skill.qbraid_eval.main,
    "plot": qbraid_skill.plot.main,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="H-cGQE qBraid Skill")
    parser.add_argument("command", choices=list(COMMANDS.keys()) + ["list"], help="Command to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the command")
    args = parser.parse_args()

    if args.command == "list":
        print("Available commands:")
        for cmd in COMMANDS:
            print(f"  {cmd}")
        return

    # Replace sys.argv with the command-specific arguments so the wrapper scripts
    # can parse them as if they were called directly.
    sys.argv = [args.command] + args.args
    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
