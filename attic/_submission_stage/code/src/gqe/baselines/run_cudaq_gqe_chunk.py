"""Run CUDA-Q GQE baseline for a chunk of molecules.

This helper is used for embarrassingly-parallel multi-GPU/multi-node scaling of
the GQE baseline step. Each MPI/Slurm task runs this script with a different
--task-id and processes the molecules whose index satisfies
`index % num_tasks == task_id`. Results are merged by the caller.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CUDA-Q GQE baseline for a chunk of molecules")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML config with dataset.molecules")
    parser.add_argument("--task-id", type=int, required=True, help="This task's chunk id")
    parser.add_argument("--num-tasks", type=int, required=True, help="Total number of chunks")
    parser.add_argument("--ham", type=Path, required=True, help="Path to hamiltonians.json")
    parser.add_argument("--out", type=Path, required=True, help="Chunk output JSON")
    parser.add_argument("--target", type=str, default="nvidia")
    parser.add_argument("--target-option", type=str, default="mqpu")
    parser.add_argument("--max-qubits", type=int, default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    molecules = [m["name"] for m in cfg["dataset"]["molecules"]]

    chunk = [m for i, m in enumerate(molecules) if i % args.num_tasks == args.task_id]
    if not chunk:
        print(f"Task {args.task_id}: no molecules assigned")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump({"results": []}, f, indent=2)
        return

    print(f"Task {args.task_id}: processing {chunk}")
    results: list[dict] = []
    for mol in chunk:
        tmp = args.out.parent / f"chunk_{args.task_id}_{mol}.json"
        cmd = [
            sys.executable,
            "src/gqe/baselines/run_cudaq_gqe.py",
            "--ham",
            str(args.ham),
            "--out",
            str(tmp),
            "--target",
            args.target,
            "--target-option",
            args.target_option,
            "--molecule",
            mol,
        ]
        if args.max_qubits is not None:
            cmd.extend(["--max-qubits", str(args.max_qubits)])
        subprocess.run(cmd, check=True)
        with tmp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        results.extend(data.get("results", []))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Task {args.task_id}: wrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
