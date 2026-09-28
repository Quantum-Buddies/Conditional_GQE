#!/usr/bin/env python3
"""qBraid QIR-SV smoke: Bell pair, then optional GQE kernel.

Run with CUDA-Q 0.15+ and QBRAID_API_KEY in the environment. Never logs the key.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cudaq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@cudaq.kernel
def bell_pair():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--gqe-json", type=Path, default=None)
    args = parser.parse_args()
    if not os.environ.get("QBRAID_API_KEY"):
        raise SystemExit("QBRAID_API_KEY unset")
    from src.gqe.eval.cudaq_qbraid_target import QIR_SV_MACHINE, set_qbraid_machine

    machine = set_qbraid_machine(QIR_SV_MACHINE)
    print(f"target=qbraid machine={machine}")
    counts = cudaq.sample(bell_pair, shots_count=int(args.shots))
    print("BELL_OK", {str(k): int(v) for k, v in counts.items()})
    if args.gqe_json is None:
        return
    from src.gqe.eval.gqe_qsci_sample import load_gqe_operators
    from src.gqe.eval.qsci import _make_hcgqe_kernel

    words, thetas, _ = load_gqe_operators(args.gqe_json, "sno_14q")
    kernel, pauli_words, thetas_used = _make_hcgqe_kernel(words, thetas, 14, 2)
    future = cudaq.sample_async(
        kernel, 14, 2, pauli_words, thetas_used, shots_count=int(args.shots)
    )
    gqe_counts = future.get()
    print("GQE_OK", len(dict(gqe_counts)), "unique")


if __name__ == "__main__":
    main()
