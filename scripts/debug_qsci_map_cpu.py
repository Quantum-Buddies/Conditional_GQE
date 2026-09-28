#!/usr/bin/env python3
"""CPU-only QSCI mapping debug (no cudaq). Replay smoke-best via NumpySampler."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/kcwp264/Conditional-GQE_materials")
BASE = Path("/scratch/kcwp264/baselines/gqe-qsci-euv-photoresists/code")
# Login node: never import cudaq (AIRE A2). NumpySampler is sufficient here.
sys.modules.setdefault("cudaq", None)
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT))

from demo.molecule import SnOMolecule  # noqa: E402
from demo.pool import build_pool  # noqa: E402
from demo.qsci import SubspaceEngine, circuit_qsci_energy, determinant_from_bitstring  # noqa: E402
from demo.sampler import NumpySampler  # noqa: E402

from src.gqe.common.hamiltonian_utils import (  # noqa: E402
    find_record_by_name,
    load_hamiltonian_records,
)
from src.gqe.eval.qsci_postprocess import qsci_energy_from_bitstrings  # noqa: E402

CASCI = -288.10912104
SMOKE_TOKENS = [0, 189, 73, 153, 107, 110, 272, 161, 225, 76, 235]


def mha(e: float) -> float:
    return abs(e - CASCI) * 1e3


def main() -> None:
    print("Building SnOMolecule (CPU PySCF, no cudaq)...", flush=True)
    mol = SnOMolecule(verbose=True)
    pool = build_pool(mol, verbose=True)
    engine = SubspaceEngine(mol.h1, mol.h2, mol.norb, mol.nelec, mol.e_core)

    ham_path = ROOT / "results/tin_ab/hamiltonians_tin/hamiltonians.json"
    record = find_record_by_name(load_hamiltonian_records(ham_path), "sno_14q")
    print(f"Exported H terms={len(record['terms'])} nq={record['n_qubits']}", flush=True)

    for shots in (1000, 8192):
        sampler = NumpySampler(pool, mol.n_qubits, sum(mol.nelec), shots, seed=42)
        counts = sampler.sample(SMOKE_TOKENS)
        n_unique = len(counts)
        e_mq, dets, n_valid = circuit_qsci_energy(engine, counts, mol.nelec, max_dim=2000)
        print(
            f"\n=== NumpySampler shots={shots} unique={n_unique} "
            f"N-valid={n_valid} spin-complete={len(dets)} ==="
        )
        print(f"  MatGen-Q QSCI: {e_mq:.10f} Ha  ΔE={mha(e_mq):.3f} mHa")

        modal = max(counts, key=counts.get)
        hf_cudaq = "1" * sum(mol.nelec) + "0" * (mol.n_qubits - sum(mol.nelec))
        hf_qiskit = format((1 << sum(mol.nelec)) - 1, f"0{mol.n_qubits}b")
        print(f"  modal={modal} count={counts[modal]}")
        print(f"  hf_cudaq={hf_cudaq} in_counts={hf_cudaq in counts}")
        print(f"  hf_qiskit={hf_qiskit} in_counts={hf_qiskit in counts}")
        print(f"  det(modal)={determinant_from_bitstring(modal)}")
        print(f"  det(hf_cudaq)={determinant_from_bitstring(hf_cudaq)}")
        print(f"  det(hf_qiskit)={determinant_from_bitstring(hf_qiskit)}")

        uniq_freq = [b for b, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
        e_wrong = qsci_energy_from_bitstrings(record, uniq_freq)
        e_rev = qsci_energy_from_bitstrings(record, [b[::-1] for b in uniq_freq])
        print(f"  Pauli unique as-is (old): {e_wrong:.10f}  ΔE={mha(e_wrong):.3f} mHa")
        print(f"  Pauli reversed (cudaq→LSB): {e_rev:.10f}  ΔE={mha(e_rev):.3f} mHa")

        # N-filter then reverse
        n_a, n_b = mol.nelec
        kept = []
        for bits, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            a, b = determinant_from_bitstring(bits)
            if bin(a).count("1") == n_a and bin(b).count("1") == n_b:
                kept.append(bits)
        e_n = qsci_energy_from_bitstrings(record, [b[::-1] for b in kept])
        print(f"  Pauli N-filter+rev ({len(kept)} dets): {e_n:.10f}  ΔE={mha(e_n):.3f} mHa")

        from demo.qsci import symmetry_complete, post_select

        ordered = post_select(counts, mol.nelec)
        completed = symmetry_complete(ordered, 2000)
        completed_bits = []
        norb = mol.norb
        for a, b in completed:
            abits = format(a, f"0{norb}b")[::-1]
            bbits = format(b, f"0{norb}b")[::-1]
            completed_bits.append("".join(x + y for x, y in zip(abits, bbits)))
        e_spin = qsci_energy_from_bitstrings(record, [b[::-1] for b in completed_bits])
        print(
            f"  Pauli N+spin+rev ({len(completed_bits)} dets): "
            f"{e_spin:.10f}  ΔE={mha(e_spin):.3f} mHa"
        )

    # Pool singles that were best under the broken scorer
    pool_json = json.loads((ROOT / "results/tin_ab/pool_sno14q.json").read_text())
    by_id = {int(t["token_id"]): t for t in pool_json["tokens"]}
    sampler = NumpySampler(pool, mol.n_qubits, sum(mol.nelec), 8192, seed=42)
    best_mha = 1e9
    best_name = None
    below = []
    for tid in (157, 162, 199, 204, 241, 115, 1, 73, 189):
        tok = by_id[tid]
        counts = sampler.sample([tid])
        e, dets, n_valid = circuit_qsci_energy(engine, counts, mol.nelec, max_dim=2000)
        err = mha(e)
        print(
            f"  single token {tid} word={tok['word']} ang={tok['angle']:+.4f} "
            f"ΔE={err:.3f} mHa N_valid={n_valid} N_sub={len(dets)}"
        )
        if err < best_mha:
            best_mha = err
            best_name = tid
        if err < 1.6 and len(dets) < 49:
            below.append((tid, err, len(dets)))

    print(f"\nBest listed single: token {best_name} ΔE={best_mha:.3f} mHa")
    print(f"Pass singles (<1.6 mHa, N<49): {below}")

    union = {}
    for tid in (157, 162, 199, 204, 241, 115, 189, 73, 110, 272):
        counts = sampler.sample([tid])
        for k, v in counts.items():
            union[k] = union.get(k, 0) + v
    e_u, dets_u, n_u = circuit_qsci_energy(engine, union, mol.nelec, max_dim=2000)
    print(
        f"Union listed singles: ΔE={mha(e_u):.3f} mHa N_valid={n_u} N_sub={len(dets_u)}"
    )


if __name__ == "__main__":
    main()
