#!/usr/bin/env python3
"""Gate 1 QSCI controls on identical SnO 14q Hamiltonian.

Four columns on the same H / CASCI, scored with MatGen-Q molecular QSCI
(interleaved α/β, particle-number post-select, spin completion):
  1. HF only
  2. MatGen-Q smoke best circuit (token sequence -> pool words+angles)
  3. Frozen 12-char H-cGQE (padded to 14)
  4. 14q-pool native sequences (random + short greedy singles)

Honest metric: best single-circuit ΔE vs CASCI (mHa) and N_dets after
symmetry completion. If a union fills 49/49, report best single-circuit.
Pass: any pool-native circuit or capped union < 1.6 mHa without 49/49 fill.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from src.gqe.common.hamiltonian_utils import (
    find_record_by_name,
    get_active_electron_count,
    load_hamiltonian_records,
)
from src.gqe.eval.qsci_map import (
    DEFAULT_MATGENQ_CODE,
    cudaq_bits_to_lsb,
    hf_bitstring_cudaq,
    load_sno_subspace_engine,
    merge_counts,
    sample_counts_numpy,
    score_cudaq_counts_matgenq,
    unique_bitstrings_from_counts,
)
from src.gqe.eval.qsci_postprocess import qsci_energy_from_bitstrings


CASCI_REF = -288.10912104
CHEMICAL_ACCURACY_MHA = 1.6
FULL_CI_DETS = 49


def _load_operators_map(optimized_path: Path) -> dict[str, dict]:
    with optimized_path.open("r", encoding="utf-8") as f:
        opt_data = json.load(f)
    if isinstance(opt_data, list):
        entries = [e for e in opt_data if isinstance(e, dict)]
    elif isinstance(opt_data, dict):
        entries = [e for e in opt_data.get("results", []) if isinstance(e, dict)]
        if not entries and opt_data.get("molecule"):
            entries = [opt_data]
    else:
        entries = []
    operators_map: dict[str, dict] = {}
    for entry in entries:
        mol = entry.get("molecule")
        best = entry.get("best_sequence") or {}
        ops = list(best.get("operators") or [])
        thetas = list(best.get("thetas") or [])
        if not ops:
            sequences = entry.get("generated_sequences") or []
            nonempty = [seq for seq in sequences if seq.get("operators")]
            if nonempty:
                chosen = max(nonempty, key=lambda seq: len(seq.get("operators") or []))
                ops = list(chosen.get("operators") or [])
                thetas = list(chosen.get("thetas") or [])
        if mol and ops:
            operators_map[str(mol)] = {"operators": ops, "thetas": thetas}
    return operators_map


def _sample_counts(
    operators: list[str] | None,
    thetas: list[float] | None,
    n_qubits: int,
    n_electrons: int,
    n_shots: int,
    seed: int,
    backend: str,
) -> dict[str, int]:
    if backend == "numpy":
        return sample_counts_numpy(
            operators, thetas, n_qubits, n_electrons, n_shots, seed
        )
    import cudaq
    from src.gqe.eval.qsci import (
        _make_hcgqe_kernel,
        _make_hf_kernel,
        sample_counts_cudaq,
    )

    cudaq.set_target("nvidia")
    if operators:
        kernel, pauli_words, theta_vals = _make_hcgqe_kernel(
            operators, thetas or [], n_qubits, n_electrons
        )
        args = (n_qubits, n_electrons, pauli_words, theta_vals)
    else:
        kernel = _make_hf_kernel()
        args = (n_qubits, n_electrons)
    counts = sample_counts_cudaq(kernel, args, n_shots=n_shots, seed=seed)
    hf = hf_bitstring_cudaq(n_qubits, n_electrons)
    if hf not in counts:
        counts[hf] = 1
    return counts


def _tokens_to_ops(pool_tokens: list[dict[str, Any]], token_ids: list[int]) -> tuple[list[str], list[float]]:
    by_id = {int(t["token_id"]): t for t in pool_tokens}
    ops: list[str] = []
    thetas: list[float] = []
    for tid in token_ids:
        tok = by_id[int(tid)]
        word = tok["word"]
        angle = float(tok["angle"])
        if word == "I" * len(word) or abs(angle) < 1e-15:
            continue
        ops.append(word)
        thetas.append(angle)
    return ops, thetas


def _build_pool_circuits(
    pool: dict[str, Any],
    *,
    n_random: int = 16,
    seq_len: int = 8,
    seed: int = 42,
    preferred_angles: tuple[float, ...] = (0.1, -0.1),
) -> list[dict[str, Any]]:
    """Random multi-word sequences + short greedy singles (preferred angles)."""
    rng = random.Random(seed)
    tokens = pool["tokens"]
    excit = [t for t in tokens if t["word"] != "I" * len(t["word"])]
    preferred = {round(a, 6) for a in preferred_angles}
    circuits: list[dict[str, Any]] = []

    for t in excit:
        if round(float(t["angle"]), 6) not in preferred:
            continue
        circuits.append({
            "name": f"single_{t['token_id']}",
            "kind": "single_excitation",
            "operators": [t["word"]],
            "thetas": [float(t["angle"])],
        })

    for i in range(n_random):
        picks = [rng.choice(excit) for _ in range(seq_len)]
        circuits.append({
            "name": f"random_{i}",
            "kind": "random",
            "operators": [p["word"] for p in picks],
            "thetas": [float(p["angle"]) for p in picks],
        })
    return circuits


def _eval_column(
    *,
    engine: Any,
    nelec: tuple[int, int],
    n_qubits: int,
    n_electrons: int,
    name: str,
    operators: list[str] | None,
    thetas: list[float] | None,
    n_shots: int,
    seed: int,
    casci: float,
    backend: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    print(f"\n=== Column: {name} ===")
    counts = _sample_counts(
        operators, thetas, n_qubits, n_electrons, n_shots, seed, backend
    )
    scored = score_cudaq_counts_matgenq(
        engine, counts, nelec, casci, full_ci_dets=FULL_CI_DETS
    )
    out = {
        "column": name,
        "n_operators": len(operators or []),
        "operators_preview": (operators or [])[:8],
        "thetas_preview": (thetas or [])[:8],
        "n_shots": n_shots,
        "hf_cudaq": hf_bitstring_cudaq(n_qubits, n_electrons),
        **scored,
    }
    err = scored.get("error_vs_casci_mha")
    print(
        f"  ΔE={err:.3f} mHa  N_dets={scored['n_dets']}  "
        f"N_valid={scored['n_valid_particle_number']}  "
        f"unique_raw={scored['n_unique_raw']}  modal={scored.get('modal_bitstring')}"
        if err is not None
        else f"  failed / empty  unique={len(counts)}"
    )
    return out, counts


def _greedy_union(
    engine: Any,
    nelec: tuple[int, int],
    casci: float,
    ranked: list[tuple[float, dict[str, int], str]],
    *,
    max_union_dets: int,
) -> dict[str, Any] | None:
    """Add circuits by increasing ΔE while keeping N_dets < full CI and ≤ cap."""
    if not ranked:
        return None
    ranked = sorted(ranked, key=lambda x: x[0])
    union: dict[str, int] = {}
    sources: list[str] = []
    best: dict[str, Any] | None = None
    for err, counts, name in ranked:
        del err
        trial = merge_counts(union, counts)
        scored = score_cudaq_counts_matgenq(
            engine, trial, nelec, casci, full_ci_dets=FULL_CI_DETS
        )
        n_dets = int(scored.get("n_dets") or 0)
        if n_dets >= FULL_CI_DETS or n_dets > max_union_dets:
            continue
        union = trial
        sources.append(name)
        best = {
            "column": "pool14q_union_greedy",
            "source_circuits": list(sources),
            **scored,
            "honest_note": (
                "If N_dets==49, treat best single-circuit as the honest metric"
            ),
        }
        if (
            scored.get("error_vs_casci_mha") is not None
            and scored["error_vs_casci_mha"] < CHEMICAL_ACCURACY_MHA
        ):
            break
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Tin Gate 1 QSCI controls")
    parser.add_argument("--hamiltonians", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--matgenq-results", type=Path, required=True)
    parser.add_argument("--frozen-infer", type=Path, required=True)
    parser.add_argument("--matgenq-code", type=Path, default=DEFAULT_MATGENQ_CODE)
    parser.add_argument("--molecules", type=str, default="sno_14q")
    parser.add_argument("--n-shots", type=int, default=8192)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=49,
        help="Ignored; subspace size is spin-completed N_dets (kept for sbatch compat)",
    )
    parser.add_argument("--n-random", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-union-dets", type=int, default=32)
    parser.add_argument(
        "--backend",
        choices=("cudaq", "numpy"),
        default="cudaq",
        help="cudaq: GPU job. numpy: MatGen-Q statevector fallback (no cudaq import)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.backend == "cudaq":
        try:
            import cudaq  # noqa: F401
        except ImportError:
            raise SystemExit("CUDA-Q is required for --backend cudaq")

    records = load_hamiltonian_records(args.hamiltonians)
    record = find_record_by_name(records, args.molecules)
    if record is None:
        raise SystemExit(f"Molecule {args.molecules} not found")

    casci = float(record.get("casci_energy") or record.get("fci_energy") or CASCI_REF)
    pool = json.loads(args.pool.read_text())
    matgenq = json.loads(args.matgenq_results.read_text())
    frozen_map = _load_operators_map(args.frozen_infer)

    print("Building MatGen-Q SnO SubspaceEngine (PySCF integrals)...")
    mol, engine = load_sno_subspace_engine(args.matgenq_code, verbose=True)
    nelec = (int(mol.nelec[0]), int(mol.nelec[1]))
    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)
    print(
        f"  Mapping: HF CUDA-Q = {hf_bitstring_cudaq(n_qubits, n_electrons)} "
        f"(qubits 0..{n_electrons - 1}); QSCI = interleaved α/β + spin complete"
    )

    columns: list[dict[str, Any]] = []

    hf_col, _ = _eval_column(
        engine=engine,
        nelec=nelec,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        name="hf",
        operators=None,
        thetas=None,
        n_shots=args.n_shots,
        seed=args.seed,
        casci=casci,
        backend=args.backend,
    )
    columns.append(hf_col)

    token_ids = list(matgenq.get("best_circuit", {}).get("tokens") or [])
    mq_ops, mq_thetas = _tokens_to_ops(pool["tokens"], token_ids)
    mq_col, mq_counts = _eval_column(
        engine=engine,
        nelec=nelec,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        name="matgenq_smoke_best",
        operators=mq_ops,
        thetas=mq_thetas,
        n_shots=args.n_shots,
        seed=args.seed,
        casci=casci,
        backend=args.backend,
    )
    mq_col["source_tokens"] = token_ids
    mq_col["matgenq_reported_best_mha"] = matgenq.get("errors_mha_vs_casci", {}).get(
        "best_single_circuit"
    )
    mq_col["matgenq_reported_union_mha"] = matgenq.get("errors_mha_vs_casci", {}).get(
        "union_refined"
    )
    top_bits = unique_bitstrings_from_counts(mq_counts)[:20]
    mq_col["bitstrings_preview"] = [
        {"bits": b, "count": int(mq_counts[b])} for b in top_bits
    ]
    # Pauli LSB-right cross-check (same sampled strings, reversed).
    lsb_bits = [cudaq_bits_to_lsb(b) for b in unique_bitstrings_from_counts(mq_counts)]
    try:
        e_pauli = float(qsci_energy_from_bitstrings(record, lsb_bits))
        mq_col["pauli_lsb_crosscheck_mha"] = abs(e_pauli - casci) * 1000.0
    except Exception as exc:
        mq_col["pauli_lsb_crosscheck_error"] = str(exc)
    columns.append(mq_col)

    frozen = frozen_map.get(args.molecules, {})
    frozen_col, _ = _eval_column(
        engine=engine,
        nelec=nelec,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        name="frozen_hcgqe_12char_padded",
        operators=list(frozen.get("operators") or []),
        thetas=list(frozen.get("thetas") or []),
        n_shots=args.n_shots,
        seed=args.seed,
        casci=casci,
        backend=args.backend,
    )
    columns.append(frozen_col)

    pool_circuits = _build_pool_circuits(
        pool, n_random=args.n_random, seq_len=args.seq_len, seed=args.seed
    )
    pool_results: list[dict[str, Any]] = []
    singles = [c for c in pool_circuits if c["kind"] == "single_excitation"]
    randoms = [c for c in pool_circuits if c["kind"] == "random"]
    singles.sort(key=lambda c: -abs(c["thetas"][0]))

    ranked: list[tuple[float, dict[str, int], str]] = []

    for circ in tqdm(singles + randoms, desc="Pool-native circuits", unit="circ"):
        seed_i = args.seed + (abs(hash(circ["name"])) % 10_000)
        counts = _sample_counts(
            circ["operators"],
            circ["thetas"],
            n_qubits,
            n_electrons,
            args.n_shots,
            seed_i,
            args.backend,
        )
        scored = score_cudaq_counts_matgenq(
            engine, counts, nelec, casci, full_ci_dets=FULL_CI_DETS
        )
        col = {
            "column": f"pool14q_{circ['name']}",
            "kind": circ["kind"],
            "n_operators": len(circ["operators"]),
            "operators_preview": circ["operators"][:8],
            "thetas_preview": circ["thetas"][:8],
            "n_shots": args.n_shots,
            **scored,
        }
        err = scored.get("error_vs_casci_mha")
        if err is not None:
            print(
                f"\n=== Column: {col['column']} ===\n"
                f"  ΔE={err:.3f} mHa  N_dets={scored['n_dets']}  "
                f"N_valid={scored['n_valid_particle_number']}"
            )
            ranked.append((err, counts, col["column"]))
        else:
            print(f"\n=== Column: {col['column']} ===\n  failed")
        pool_results.append(col)

    valid = [r for r in pool_results if r.get("error_vs_casci_mha") is not None]
    best = min(valid, key=lambda r: r["error_vs_casci_mha"]) if valid else None

    union_result = _greedy_union(
        engine, nelec, casci, ranked, max_union_dets=args.max_union_dets
    )
    if union_result is not None:
        print(
            f"\n=== Column: pool14q_union_greedy ===\n"
            f"  ΔE={union_result['error_vs_casci_mha']:.3f} mHa  "
            f"N_dets={union_result['n_dets']}  "
            f"n_circuits={len(union_result.get('source_circuits') or [])}"
        )

    def _passes(row: dict[str, Any] | None) -> bool:
        if row is None or row.get("error_vs_casci_mha") is None:
            return False
        return (
            row["error_vs_casci_mha"] < CHEMICAL_ACCURACY_MHA
            and int(row.get("n_dets", FULL_CI_DETS)) < FULL_CI_DETS
        )

    pool_pass = _passes(best) or _passes(union_result)
    if union_result and int(union_result.get("n_dets", 0)) >= FULL_CI_DETS:
        pool_pass = _passes(best)

    smoke_err = mq_col.get("error_vs_casci_mha")
    smoke_ok = smoke_err is not None and abs(float(smoke_err) - 2.869) < 1.5
    summary = {
        "experiment": "tin_qsci_controls_gate1",
        "qsci_backend": "matgenq_subspace",
        "sampler_backend": args.backend,
        "bitstring_convention": "cudaq_char_q_equals_qubit_q",
        "molecule": args.molecules,
        "casci_energy": casci,
        "chemical_accuracy_mha": CHEMICAL_ACCURACY_MHA,
        "full_ci_dets": FULL_CI_DETS,
        "n_shots": args.n_shots,
        "max_union_dets": args.max_union_dets,
        "columns": columns,
        "pool14q_circuits": pool_results,
        "pool14q_union_greedy": union_result,
        "best_pool14q": best,
        "gate1_pool_native_pass": pool_pass,
        "smoke_best_near_demo_a": smoke_ok,
        "pass_criterion": (
            "pool-native circuit(s) < 1.6 mHa vs CASCI without filling all 49 dets"
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved {args.out}")
    print(f"Gate 1 pool-native pass: {pool_pass}")
    print(f"Smoke-best near Demo A +2.87 mHa: {smoke_ok} (got {smoke_err})")
    if best:
        print(
            f"Best pool14q: {best['column']}  "
            f"ΔE={best['error_vs_casci_mha']:.3f} mHa  N_dets={best['n_dets']}"
        )


if __name__ == "__main__":
    main()
