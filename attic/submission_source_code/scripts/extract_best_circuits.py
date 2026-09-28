#!/usr/bin/env python3
"""Extract best circuits + rebuild MAP-Elites archives from an RL checkpoint.

Use when training was killed before save_all() — archives lived only in RAM.
Falls back to:
  1. metrics.best_energies from the checkpoint (energy targets / reference)
  2. Re-sampling from the RL-tuned policy + CUDA-Q / energy-cache eval
  3. Inserting into fresh PerMoleculeArchives and writing map_elites_*.json

Usage:
    source scripts/env_gpu.sh
    python scripts/extract_best_circuits.py \
        --checkpoint results/train/h_cgqe_model_qbraid_rl.pt \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --molecules h2 lih ... \
        --energy-cache results/train/rl_energy_cache.sqlite \
        --archive-dir results/train/h_cgqe_model_qbraid_rl_map_elites \
        --out results/train/h_cgqe_model_qbraid_rl_best_circuits.json \
        --n-samples 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import get_active_electron_count
from src.gqe.models.h_cgqe_transformer import HcGQEModel
from src.gqe.models.train_rl_dapo import (
    evaluate_energies_batch,
    load_molecule_data,
    sample_sequences_with_logprobs,
    _get_cached_spin_ham,
    _ensure_cudaq,
)
from src.gqe.rl.energy_cache import PersistentEnergyCache, resolve_energies_with_cache
from src.gqe.rl.map_elites import PerMoleculeArchives


DEFAULT_MOLECULES = [
    "h2", "h2_0.5", "h2_1.0", "h2_1.5", "h2_2.0",
    "anisole_cas12", "benzene_cas12", "diarylethene_frag_cas12", "hf",
    "imeph_cas12", "iodobenzene_cas12", "lih", "lih_1.2", "lih_2.0", "lih_3.0",
    "methyl_iodide_cas12", "ocresol_cas12", "phenol_cas12", "toluene_cas12",
    "beh2", "beh2_1.0", "beh2_1.6", "h2o", "h2o_1.0_631g_cas8",
    "n2_1.1_631g_cas8", "nh3", "ch4", "co", "n2", "n2_1.8", "n2_2.5",
    "lih_1.6_631g",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hamiltonians", type=Path, required=True)
    p.add_argument("--molecules", nargs="+", default=DEFAULT_MOLECULES)
    p.add_argument("--energy-cache", type=Path, default=None)
    p.add_argument("--archive-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="Also dump checkpoint metrics JSON (default: <ckpt>_rl_metrics.json)")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--theta", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--explore-eps", type=float, default=0.1)
    p.add_argument("--force-entanglement", action="store_true", default=True)
    p.add_argument("--max-repeat", type=int, default=4)
    p.add_argument("--freq-penalty", type=float, default=1.0)
    p.add_argument("--max-qubits", type=int, default=22)
    p.add_argument("--max-terms", type=int, default=128)
    p.add_argument("--max-pauli-len", type=int, default=24)
    p.add_argument("--qd-n-bins-entanglement", type=int, default=10)
    p.add_argument("--qd-n-bins-depth", type=int, default=10)
    p.add_argument("--target", type=str, default="nvidia")
    p.add_argument("--target-option", type=str, default="fp32")
    p.add_argument("--eval-async-chunk", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-bf16", action="store_true", default=True)
    p.add_argument("--cache-only", action="store_true", default=False,
                   help="Never call CUDA-Q; uncached circuits get HF penalty")
    return p.parse_args()


def _strip_compile_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """Undo torch.compile `_orig_mod.` prefixes written into RL checkpoints."""
    if not any("._orig_mod." in k for k in state):
        return state
    return {k.replace("._orig_mod.", "."): v for k, v in state.items()}


def _dump_checkpoint_metrics(ckpt: dict[str, Any], path: Path) -> None:
    metrics = ckpt.get("metrics", {})
    payload = {
        "source": "checkpoint_extract",
        "note": (
            "MAP-Elites may be rebuilt by extract_best_circuits.py if training "
            "exited before save_all()."
        ),
        "n_epochs_completed": len(metrics.get("train_log", [])),
        "best_energies": metrics.get("best_energies", {}),
        "train_log": metrics.get("train_log", []),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote checkpoint metrics → {path}", flush=True)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    inv_vocab = ckpt["inv_vocab"]
    config = ckpt["config"]
    ckpt_best = dict(ckpt.get("metrics", {}).get("best_energies", {}))

    metrics_out = args.metrics_out or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_rl_metrics.json"
    )
    _dump_checkpoint_metrics(ckpt, metrics_out)

    model = HcGQEModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config.get("dropout", 0.1),
        max_pauli_len=config.get("max_pauli_len", args.max_pauli_len),
        max_seq_len=config.get("max_seq_len", 64),
    )
    state = _strip_compile_prefix(ckpt["model_state"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing:
            print(f"    missing sample: {missing[:5]}", flush=True)
        if unexpected:
            print(f"    unexpected sample: {unexpected[:5]}", flush=True)
            raise RuntimeError("Unexpected keys remain after stripping torch.compile prefixes")
    model.to(device)
    model.eval()
    max_seq_len = int(config.get("max_seq_len", 64))
    amp_dtype = torch.bfloat16 if (args.use_bf16 and device.type == "cuda") else None

    molecules: list[dict[str, Any]] = []
    for name in args.molecules:
        mol = load_molecule_data(
            args.hamiltonians, name, vocab,
            args.max_terms, args.max_pauli_len, max_seq_len,
        )
        if mol["n_qubits"] > args.max_qubits:
            print(f"  skip {name}: {mol['n_qubits']}q > max {args.max_qubits}", flush=True)
            continue
        molecules.append(mol)
    molecules.sort(key=lambda m: m["n_qubits"])
    print(f"Molecules: {len(molecules)}", flush=True)

    if not args.cache_only:
        cq = _ensure_cudaq()
        if cq is None:
            raise RuntimeError("cudaq unavailable; pass --cache-only to skip CUDA-Q")
        opts: dict[str, Any] = {}
        if args.target_option:
            opts["option"] = args.target_option
        cq.set_target(args.target, **opts)
        print(f"CUDA-Q target={args.target} option={args.target_option}", flush=True)
        for mol in molecules:
            try:
                mol["spin_ham"] = _get_cached_spin_ham(mol["record"], cache_key=mol["name"])
            except Exception as e:
                print(f"  WARNING spin_ham {mol['name']}: {e}", flush=True)
                mol["spin_ham"] = None
    else:
        for mol in molecules:
            mol["spin_ham"] = None

    energy_cache = PersistentEnergyCache(args.energy_cache) if args.energy_cache else None
    if energy_cache is not None:
        print(f"Energy cache: {args.energy_cache} ({energy_cache.stats()['n_entries']} entries)", flush=True)

    archives = PerMoleculeArchives(
        n_bins_entanglement=args.qd_n_bins_entanglement,
        n_bins_depth=args.qd_n_bins_depth,
        max_seq_len=max_seq_len,
    )

    best_circuits: dict[str, Any] = {}
    t0 = time.time()

    for mi, mol in enumerate(molecules):
        name = mol["name"]
        nq = mol["n_qubits"]
        n_electrons = get_active_electron_count(mol["record"])
        hf = mol["hf_energy"] if mol["hf_energy"] is not None else 0.0
        print(f"\n[{mi+1}/{len(molecules)}] {name} ({nq}q) sampling {args.n_samples}...", flush=True)
        t_mol = time.time()

        pauli_ids = mol["pauli_ids"].unsqueeze(0).to(device)
        coeffs = mol["coeffs"].unsqueeze(0).to(device)
        term_mask = mol["term_mask"].unsqueeze(0).to(device)

        _, _, operator_lists, _ = sample_sequences_with_logprobs(
            model, pauli_ids, coeffs, term_mask,
            n_samples=args.n_samples,
            max_seq_len=max_seq_len,
            temperature=args.temperature,
            vocab=vocab,
            inv_vocab=inv_vocab,
            n_qubits=nq,
            force_entanglement=args.force_entanglement,
            max_repeat=args.max_repeat,
            device=device,
            top_p=args.top_p,
            explore_eps=args.explore_eps,
            freq_penalty=args.freq_penalty,
            amp_dtype=amp_dtype,
        )
        # Drop empties / EOS-only
        valid = [ops for ops in operator_lists if ops]
        if not valid:
            print(f"  WARNING: no valid circuits for {name}", flush=True)
            continue

        def _eval(ops_batch: list[list[str]]) -> list[float]:
            return evaluate_energies_batch(
                ops_batch,
                mol["record"],
                theta=args.theta,
                spin_ham=mol.get("spin_ham"),
                eval_async=True,
                async_chunk=args.eval_async_chunk,
                show_progress=nq >= 16,
                mol_name=name,
            )

        energies, stats = resolve_energies_with_cache(
            valid,
            molecule_id=name,
            n_qubits=nq,
            n_electrons=n_electrons,
            theta=args.theta,
            eval_fn=_eval,
            cache=energy_cache,
            cache_only=args.cache_only,
            miss_penalty=float(hf),
        )

        best_i = int(min(range(len(energies)), key=lambda i: energies[i]))
        best_e = float(energies[best_i])
        best_ops = list(valid[best_i])
        ckpt_e = ckpt_best.get(name)

        for ops, e in zip(valid, energies):
            archives.insert(name, ops, float(e), nq, metadata={"source": "resample"})

        elites = archives.get(name).get_elite_circuits()
        if elites:
            top = min(elites, key=lambda e: e["energy"])
            if float(top["energy"]) < best_e:
                best_e = float(top["energy"])
                best_ops = list(top["operators"])

        best_circuits[name] = {
            "n_qubits": nq,
            "n_electrons": n_electrons,
            "energy": best_e,
            "operators": best_ops,
            "n_ops": len(best_ops),
            "checkpoint_best_energy": ckpt_e,
            "energy_delta_vs_checkpoint": (
                None if ckpt_e is None else best_e - float(ckpt_e)
            ),
            "hf_energy": mol["hf_energy"],
            "fci_energy": mol["fci_energy"],
            "cache_stats": stats,
            "n_valid_samples": len(valid),
            "archive_summary": archives.get(name).summary(),
        }
        dt = time.time() - t_mol
        print(
            f"  E={best_e:.6f}  ckpt={ckpt_e}  "
            f"cache hit={stats['hits']}/{len(valid)}  "
            f"QD={archives.get(name).summary()}  {dt:.1f}s",
            flush=True,
        )

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    archives.save_all(str(args.archive_dir))
    print(f"\nMAP-Elites archives → {args.archive_dir}/", flush=True)
    print(f"  {archives.summary()}", flush=True)

    payload = {
        "checkpoint": str(args.checkpoint),
        "n_samples": args.n_samples,
        "theta": args.theta,
        "elapsed_s": time.time() - t0,
        "checkpoint_best_energies": ckpt_best,
        "archive_dir": str(args.archive_dir),
        "archive_summary": archives.summary(),
        "best_circuits": best_circuits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Best circuits → {args.out}", flush=True)

    if energy_cache is not None:
        energy_cache.close()


if __name__ == "__main__":
    main()
