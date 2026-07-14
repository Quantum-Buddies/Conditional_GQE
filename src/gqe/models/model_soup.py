#!/usr/bin/env python3
"""Model Soup: Weight averaging for H-cGQE checkpoints.

Implements the "Model Soups" approach (Wortsman et al., ICML 2022) — averaging
the weights of multiple fine-tuned checkpoints to improve generalization without
any additional training or inference cost.

For H-cGQE, this is applied across RAFT rounds: the model from each round
captures different circuit patterns, and averaging them produces a more robust
policy that generalizes better to unseen molecules.

Usage:
    python src/gqe/models/model_soup.py \
        --checkpoints results/train/h_cgqe_raft_round_1.pt \
                      results/train/h_cgqe_raft_round_2.pt \
                      results/train/h_cgqe_raft_round_3.pt \
        --out results/train/h_cgqe_star_soup.pt

    # Uniform averaging (default)
    python src/gqe/models/model_soup.py --checkpoints ckpt1.pt ckpt2.pt --out soup.pt

    # Greedy soup: add checkpoint only if it improves validation metric
    python src/gqe/models/model_soup.py --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt \
        --out soup.pt --greedy --hamiltonians results/data/hamiltonians.json
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from tqdm import tqdm


def uniform_soup(
    checkpoint_paths: list[Path],
    out_path: Path,
) -> None:
    """Average model weights uniformly across all checkpoints."""
    print(f"Uniform soup: averaging {len(checkpoint_paths)} checkpoints")

    # Load first checkpoint as base
    base_ckpt = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    base_state = base_ckpt["model_state_dict"]
    soup_state = {k: v.clone().float() for k, v in base_state.items()}

    # Accumulate remaining checkpoints
    for ckpt_path in tqdm(checkpoint_paths[1:], desc="Averaging"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        for k in soup_state:
            if k in state:
                soup_state[k] += state[k].float()

    # Divide by N
    n = len(checkpoint_paths)
    for k in soup_state:
        soup_state[k] /= n
        # Cast back to original dtype
        orig_dtype = base_state[k].dtype
        soup_state[k] = soup_state[k].to(orig_dtype)

    # Save
    base_ckpt["model_state_dict"] = soup_state
    base_ckpt["metadata"] = base_ckpt.get("metadata", {})
    base_ckpt["metadata"]["model_soup"] = True
    base_ckpt["metadata"]["soup_components"] = [str(p) for p in checkpoint_paths]
    base_ckpt["metadata"]["soup_method"] = "uniform"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base_ckpt, out_path)
    print(f"Model soup saved to: {out_path}")
    print(f"  Components: {n}")
    print(f"  Method: uniform")


def greedy_soup(
    checkpoint_paths: list[Path],
    out_path: Path,
    hamiltonians_path: Path | None = None,
) -> None:
    """Greedy soup: add each checkpoint only if it improves the held-out metric.

    Requires evaluating energy on a validation set. If no Hamiltonians provided,
    falls back to uniform soup.
    """
    if hamiltonians_path is None or not hamiltonians_path.exists():
        print("No Hamiltonians provided for greedy soup, falling back to uniform")
        uniform_soup(checkpoint_paths, out_path)
        return

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from src.gqe.common.hamiltonian_utils import load_hamiltonian_records
    from src.gqe.eval.evaluate_h_cgqe import evaluate_model_on_molecules

    print(f"Greedy soup: evaluating {len(checkpoint_paths)} checkpoints")
    ham_records = load_hamiltonian_records(hamiltonians_path)

    # Evaluate each checkpoint individually
    best_energy = float("inf")
    best_ckpt_path = None
    individual_energies = {}

    for ckpt_path in tqdm(checkpoint_paths, desc="Evaluating individual"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        energy = evaluate_model_on_molecules(ckpt, ham_records, max_qubits=24)
        individual_energies[str(ckpt_path)] = energy
        if energy < best_energy:
            best_energy = energy
            best_ckpt_path = ckpt_path

    print(f"Best individual: {best_ckpt_path} (energy={best_energy:.6f})")

    # Greedy addition
    soup_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    soup_state = {k: v.clone().float() for k, v in soup_ckpt["model_state_dict"].items()}
    current_n = 1
    current_energy = best_energy

    for ckpt_path in tqdm(checkpoint_paths, desc="Greedy addition"):
        if ckpt_path == best_ckpt_path:
            continue

        # Try adding this checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        candidate_state = {k: v.clone().float() for k, v in soup_state.items()}
        for k in candidate_state:
            if k in ckpt["model_state_dict"]:
                candidate_state[k] = (candidate_state[k] * current_n + ckpt["model_state_dict"][k].float()) / (current_n + 1)

        # Evaluate candidate
        candidate_ckpt = copy.deepcopy(soup_ckpt)
        candidate_ckpt["model_state_dict"] = {k: v.to(soup_ckpt["model_state_dict"][k].dtype) for k, v in candidate_state.items()}
        candidate_energy = evaluate_model_on_molecules(candidate_ckpt, ham_records, max_qubits=24)

        if candidate_energy < current_energy:
            print(f"  Added {ckpt_path.name}: energy {current_energy:.6f} → {candidate_energy:.6f}")
            soup_state = candidate_state
            current_n += 1
            current_energy = candidate_energy
        else:
            print(f"  Skipped {ckpt_path.name}: energy {candidate_energy:.6f} >= {current_energy:.6f}")

    # Save
    soup_ckpt["model_state_dict"] = {k: v.to(soup_ckpt["model_state_dict"][k].dtype) for k, v in soup_state.items()}
    soup_ckpt["metadata"] = soup_ckpt.get("metadata", {})
    soup_ckpt["metadata"]["model_soup"] = True
    soup_ckpt["metadata"]["soup_components"] = [str(p) for p in checkpoint_paths]
    soup_ckpt["metadata"]["soup_method"] = "greedy"
    soup_ckpt["metadata"]["soup_final_energy"] = current_energy

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup_ckpt, out_path)
    print(f"Greedy soup saved to: {out_path}")
    print(f"  Final energy: {current_energy:.6f}")
    print(f"  Components: {current_n}/{len(checkpoint_paths)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Soup: weight averaging for H-cGQE")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                        help="Checkpoints to average")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output soup checkpoint path")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy soup (add checkpoint only if it improves metric)")
    parser.add_argument("--hamiltonians", type=Path, default=None,
                        help="Hamiltonians JSON for greedy soup evaluation")
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        print("WARNING: Only one checkpoint provided, soup = copy")
        import shutil
        shutil.copy(args.checkpoints[0], args.out)
        return

    if args.greedy:
        greedy_soup(args.checkpoints, args.out, args.hamiltonians)
    else:
        uniform_soup(args.checkpoints, args.out)


if __name__ == "__main__":
    main()
