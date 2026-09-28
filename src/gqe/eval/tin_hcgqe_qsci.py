#!/usr/bin/env python3
"""Frozen 14-char H-cGQE infer → MatGen-Q QSCI with union cap N_dets≤32."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

try:
    import cudaq
except ImportError:
    cudaq = None

from src.gqe.eval.qsci import _load_operators_map, _make_hcgqe_kernel, sample_counts_cudaq
from src.gqe.eval.qsci_map import (
    DEFAULT_MATGENQ_CODE,
    hf_bitstring_cudaq,
    load_sno_subspace_engine,
    merge_counts,
    score_cudaq_counts_matgenq,
)


CHEMICAL_ACCURACY_MHA = 1.6
FULL_CI_DETS = 49


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matgenq-code", type=Path, default=DEFAULT_MATGENQ_CODE)
    parser.add_argument("--molecule", type=str, default="sno_14q")
    parser.add_argument("--n-qubits", type=int, default=14)
    parser.add_argument("--n-electrons", type=int, default=2)
    parser.add_argument("--n-shots", type=int, default=8192)
    parser.add_argument("--max-union-dets", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--casci", type=float, default=-288.10912104)
    args = parser.parse_args()

    if cudaq is None:
        raise SystemExit("CUDA-Q is required")
    cudaq.set_target("nvidia")

    mol, engine = load_sno_subspace_engine(args.matgenq_code, verbose=True)
    nelec = (int(mol.nelec[0]), int(mol.nelec[1]))
    op_map = _load_operators_map(args.infer)
    ops_info = op_map.get(args.molecule, {})
    infer_data = json.loads(args.infer.read_text())
    entries = infer_data if isinstance(infer_data, list) else infer_data.get("results", [infer_data])
    sequences: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("molecule") != args.molecule:
            continue
        for seq in entry.get("generated_sequences") or []:
            if seq.get("operators"):
                sequences.append(seq)
    if not sequences and ops_info.get("operators"):
        sequences.append({"operators": ops_info["operators"], "thetas": ops_info.get("thetas") or []})

    ranked: list[tuple[float, dict[str, int], int]] = []
    circuit_rows: list[dict[str, Any]] = []
    hf = hf_bitstring_cudaq(args.n_qubits, args.n_electrons)

    for i, seq in enumerate(tqdm(sequences, desc="14-char QSCI", unit="circ")):
        ops = list(seq.get("operators") or [])
        thetas = list(seq.get("thetas") or [])
        if any(len(w) != args.n_qubits for w in ops):
            row = {
                "sample_id": seq.get("sample_id", i),
                "n_operators": len(ops),
                "word_lengths": sorted({len(w) for w in ops}),
                "error": "non_14char_word",
            }
            circuit_rows.append(row)
            continue
        kernel, words, th = _make_hcgqe_kernel(ops, thetas, args.n_qubits, args.n_electrons)
        counts = sample_counts_cudaq(
            kernel,
            (args.n_qubits, args.n_electrons, words, th),
            n_shots=args.n_shots,
            seed=args.seed + i,
        )
        if hf not in counts:
            counts[hf] = 1
        scored = score_cudaq_counts_matgenq(
            engine, counts, nelec, args.casci, full_ci_dets=FULL_CI_DETS
        )
        row = {
            "sample_id": seq.get("sample_id", i),
            "n_operators": len(ops),
            "operators_preview": ops[:8],
            **scored,
        }
        circuit_rows.append(row)
        err = scored.get("error_vs_casci_mha")
        if err is not None:
            ranked.append((float(err), counts, i))

    valid = [r for r in circuit_rows if r.get("error_vs_casci_mha") is not None]
    best = min(valid, key=lambda r: r["error_vs_casci_mha"]) if valid else None

    union = {}
    union_sources: list[int] = []
    union_row = None
    for err, counts, idx in sorted(ranked, key=lambda x: x[0]):
        del err
        trial = merge_counts(union, counts)
        scored = score_cudaq_counts_matgenq(
            engine, trial, nelec, args.casci, full_ci_dets=FULL_CI_DETS
        )
        n_dets = int(scored.get("n_dets") or 0)
        if n_dets >= FULL_CI_DETS or n_dets > args.max_union_dets:
            continue
        union = trial
        union_sources.append(idx)
        union_row = {"column": "union_capped", "source_sample_ids": union_sources, **scored}

    def _pass(row: dict[str, Any] | None) -> bool:
        if row is None or row.get("error_vs_casci_mha") is None:
            return False
        return row["error_vs_casci_mha"] < CHEMICAL_ACCURACY_MHA and int(row["n_dets"]) < FULL_CI_DETS

    summary = {
        "experiment": "tin_hcgqe_14char_qsci",
        "infer": str(args.infer),
        "n_sequences": len(sequences),
        "n_scored": len(valid),
        "best_single": best,
        "union_capped": union_row,
        "max_union_dets": args.max_union_dets,
        "s4_pass": _pass(best) or _pass(union_row),
        "circuits": circuit_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {args.out}")
    print(f"S4 pass: {summary['s4_pass']}")
    if best:
        print(
            f"Best single: sample {best.get('sample_id')} "
            f"ΔE={best['error_vs_casci_mha']:.3f} mHa N_dets={best['n_dets']}"
        )


if __name__ == "__main__":
    main()
