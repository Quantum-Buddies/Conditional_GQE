#!/usr/bin/env python3
"""Reinforcement Learning from Quantum Feedback (RLQF) for H-cGQE.

Fine-tunes the H-cGQE Transformer using REINFORCE with energy rewards.
For each sampled Hamiltonian, the policy generates operator sequences, each
sequence is scored by its fixed-theta energy (fast CUDA-Q proxy), and the
policy is updated to prefer sequences with lower energy.

Usage:
    python src/gqe/models/train_rlqf_h_cgqe.py \
        --checkpoint results/train/h_cgqe_model.pt \
        --hamiltonians results/data/hamiltonians.json \
        --out results/train/h_cgqe_model_rlqf.pt \
        --steps 1000 --rollouts 8 --use-cuda
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    hamiltonian_to_spin_operator,
    get_active_electron_count,
)
from src.gqe.models.h_cgqe_transformer import (
    HcGQEModel,
    tokenize_hamiltonian,
    SPECIAL_TOKENS,
)

try:
    import cudaq
    from cudaq import spin
except ImportError:
    cudaq = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pad_pauli_word(word: str, n_qubits: int) -> str:
    """Pad or truncate a Pauli word to match n_qubits.
    
    The shared operator vocabulary spans molecules of different sizes,
    so a generated word may be shorter or longer than the target qubit count.
    Padding with 'I' (identity) preserves the operator's action on the
    relevant qubits. Truncation drops trailing identity qubits.
    """
    if len(word) == n_qubits:
        return word
    if len(word) < n_qubits:
        return word + "I" * (n_qubits - len(word))
    return word[:n_qubits]


def _build_kernel(
    n_qubits: int,
    n_electrons: int,
    operators: list[str],
) -> tuple[Any, Any, Any]:
    """Build a CUDA-Q kernel for the given operator sequence."""
    if cudaq is None:
        raise RuntimeError("CUDA-Q not available")

    @cudaq.kernel
    def kernel(
        n_qubits: int,
        n_electrons: int,
        pauli_words: list[cudaq.pauli_word],
        thetas: list[float],
    ):
        q = cudaq.qvector(n_qubits)
        for i in range(n_electrons):
            x(q[i])
        for i in range(len(pauli_words)):
            exp_pauli(thetas[i], q, pauli_words[i])

    words = ["".join(ops) for ops in operators]
    # Pad/truncate to match n_qubits (vocabulary spans multiple molecule sizes)
    words = [_pad_pauli_word(w, n_qubits) for w in words]
    return kernel, words, n_qubits


def _evaluate_fixed_theta_energy(
    molecule_record: dict[str, Any],
    operators: list[str],
    theta: float = 0.01,
) -> float:
    """Fixed-theta energy proxy used as the RL reward signal."""
    if cudaq is None:
        return float("inf")

    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    spin_ham = hamiltonian_to_spin_operator(molecule_record)
    kernel, words, _ = _build_kernel(n_qubits, n_electrons, operators)
    thetas = np.full(len(operators), theta, dtype=float)

    return float(
        cudaq.observe(kernel, spin_ham, n_qubits, n_electrons, words, thetas.tolist()).expectation()
    )


def _compute_log_probs(
    model: nn.Module,
    pauli_ids: torch.Tensor,
    coeffs: torch.Tensor,
    term_mask: torch.Tensor,
    tgt_tokens: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """Compute log probabilities for generated sequences under the model."""
    model.train()
    tgt_input = tgt_tokens[:, :-1]
    tgt_labels = tgt_tokens[:, 1:]
    logits = model(pauli_ids, coeffs, tgt_input, term_mask=term_mask)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=tgt_labels.unsqueeze(-1)).squeeze(-1)
    mask = tgt_labels != pad_id
    return (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _rollout_sequences(
    model: nn.Module,
    vocab: dict[str, int],
    inv_vocab: dict[int, str],
    molecule_records: list[dict[str, Any]],
    max_terms: int,
    max_pauli_len: int,
    max_seq_len: int,
    rollouts: int,
    device: torch.device,
    temperature: float = 1.0,
    force_entanglement: bool = True,
) -> list[tuple[dict[str, Any], list[str], float]]:
    """Generate and score sequences for a batch of molecules."""
    model.eval()
    results: list[tuple[dict[str, Any], list[str], float]] = []

    for rec in molecule_records:
        terms = [
            (t["term"], float(t.get("real", 0.0)))
            for t in rec.get("terms", [])
        ]
        terms.sort(key=lambda x: abs(x[1]), reverse=True)
        ham_tokens = tokenize_hamiltonian(terms, vocab, max_terms, max_pauli_len)
        pauli_ids = ham_tokens["pauli_ids"].unsqueeze(0).to(device)
        coeffs = ham_tokens["coeffs"].unsqueeze(0).to(device)
        tmask = ham_tokens["term_mask"].unsqueeze(0).to(device)

        n_qubits = int(rec.get("n_qubits", 0))

        for _ in range(rollouts):
            with torch.no_grad():
                generated = model.generate(
                    pauli_ids,
                    coeffs,
                    tmask,
                    bos_id=SPECIAL_TOKENS["<BOS>"],
                    eos_id=SPECIAL_TOKENS["<EOS>"],
                    max_len=max_seq_len,
                    temperature=temperature,
                    vocab=vocab,
                    force_entanglement=force_entanglement,
                    max_repeat=4,
                    sample=True,
                    n_qubits=n_qubits,
                )
            tokens = generated[0].tolist()
            words = []
            for tid in tokens[1:]:  # skip BOS
                if tid in (SPECIAL_TOKENS["<EOS>"], SPECIAL_TOKENS["<PAD>"]):
                    break
                word = inv_vocab.get(tid, "<UNK>")
                if word in ("<BOS>", "<UNK>"):
                    continue
                words.append(word)

            # Trim trailing noise (single-qubit Z-only / identity) before scoring
            from src.gqe.models.infer_h_cgqe import _is_trailing_noise
            while words and _is_trailing_noise(words[-1]):
                words.pop()

            if len(words) < 2:
                continue

            energy = _evaluate_fixed_theta_energy(rec, words)
            results.append((rec, words, energy))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="RLQF fine-tuning for H-cGQE")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hamiltonians", type=Path, default=Path("results/data/hamiltonians.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--rollouts", type=int, default=8, help="Sequences per molecule per RL step")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of molecules per RL step")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--force-entanglement", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _seed_everything(args.seed)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if cudaq is None:
        raise RuntimeError("CUDA-Q is required for RLQF energy evaluation")

    # Set CUDA-Q target for GPU acceleration
    try:
        cudaq.set_target("nvidia")
        print("CUDA-Q target: nvidia")
    except RuntimeError:
        try:
            cudaq.set_target("qpp-cpu")
            print("CUDA-Q target: qpp-cpu (fallback)")
        except RuntimeError:
            print("Warning: Could not set CUDA-Q target, using default")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    inv_vocab = ckpt["inv_vocab"]
    config = ckpt["config"]

    model = HcGQEModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_pauli_len=config["max_pauli_len"],
        max_seq_len=config["max_seq_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    records = load_hamiltonian_records(args.hamiltonians)
    train_records = [r for r in records if r.get("split", "train") == "train"]
    if not train_records:
        train_records = records

    max_terms = config["max_pauli_len"]
    max_seq_len = config["max_seq_len"]

    best_mean_reward = float("-inf")
    history: list[dict[str, Any]] = []

    pbar = tqdm(range(args.steps), desc="RLQF step", unit="step")
    for step in pbar:
        molecules = random.sample(train_records, min(args.batch_size, len(train_records)))

        rollouts = _rollout_sequences(
            model, vocab, inv_vocab, molecules,
            max_terms=max_terms,
            max_pauli_len=config["max_pauli_len"],
            max_seq_len=max_seq_len,
            rollouts=args.rollouts,
            device=device,
            temperature=args.temperature,
            force_entanglement=args.force_entanglement,
        )

        if not rollouts:
            continue

        # Reward: lower energy is better. Use a reference (Hartree-Fock baseline) to stabilize.
        energies = np.array([e for _, _, e in rollouts])
        baseline = np.median(energies)
        rewards = -(energies - baseline)  # positive when better than median

        # Tokenize the rollout sequences for log-prob computation
        from src.gqe.models.h_cgqe_transformer import tokenize_operator_sequence
        tgt_list = [
            tokenize_operator_sequence(words, vocab, max_seq_len)
            for _, words, _ in rollouts
        ]
        tgt_tokens = torch.stack(tgt_list).to(device)

        # Build Hamiltonian inputs for each rollout
        pauli_ids_list = []
        coeffs_list = []
        tmask_list = []
        for rec, _, _ in rollouts:
            terms = [(t["term"], float(t.get("real", 0.0))) for t in rec.get("terms", [])]
            terms.sort(key=lambda x: abs(x[1]), reverse=True)
            ham_tokens = tokenize_hamiltonian(terms, vocab, max_terms, config["max_pauli_len"])
            pauli_ids_list.append(ham_tokens["pauli_ids"])
            coeffs_list.append(ham_tokens["coeffs"])
            tmask_list.append(ham_tokens["term_mask"])
        pauli_ids = torch.stack(pauli_ids_list).to(device)
        coeffs = torch.stack(coeffs_list).to(device)
        tmask = torch.stack(tmask_list).to(device)

        log_probs = _compute_log_probs(model, pauli_ids, coeffs, tmask, tgt_tokens)
        rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
        loss = -(log_probs * rewards_t).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mean_reward = float(rewards.mean())
        mean_energy = float(energies.mean())
        history.append({
            "step": step,
            "mean_energy": mean_energy,
            "mean_reward": mean_reward,
            "loss": float(loss.item()),
            "baseline": float(baseline),
        })

        pbar.set_postfix_str(f"mean_energy={mean_energy:.6f} mean_reward={mean_reward:.4f}")

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "inv_vocab": inv_vocab,
                "config": config,
                "rlqf_config": vars(args),
                "best_mean_reward": best_mean_reward,
            }, args.out)

    # Save full history
    history_path = args.out.parent / f"{args.out.stem}_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"RLQF history saved to: {history_path}")
    print(f"Best mean reward: {best_mean_reward:.4f}")
    print(f"Final model saved to: {args.out}")


if __name__ == "__main__":
    main()
