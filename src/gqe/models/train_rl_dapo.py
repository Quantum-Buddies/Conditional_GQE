"""DAPO-based Reinforcement Learning training for H-cGQE Transformer.

Replaces supervised cross-entropy with an RL loop that directly optimizes
for energy via the DAPO algorithm (Decoupled Clip + Dynamic Sampling
Policy Optimization). Key improvements over GPT-QE's GRPO:

1. Clip-Higher: asymmetric clipping prevents entropy collapse (diagonal
   sequence collapse in our case).
2. Dynamic Sampling: skips batches where all sampled circuits have
   identical energy (zero-advantage gradient waste).
3. Token-Level Loss: fair weighting for variable-length circuits.
4. Multi-component reward: energy + entanglement bonus + depth penalty
   + non-commuting fraction bonus.

Usage:
    python src/gqe/models/train_rl_dapo.py \
        --checkpoint results/train/h_cgqe_uccsd_model.pt \
        --hamiltonians results/data/hamiltonians.json \
        --molecules h2_0.74 lih_1.6_full n2_1.1_full beh2_1.3_full \
        --out results/train/h_cgqe_rl_dapo_model.pt \
        --epochs 200 --n-samples 50 --use-cuda --multi-gpu \
        --target nvidia --target-option mqpu
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.optimize import minimize

try:
    import cudaq
except ImportError:
    cudaq = None

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.gqe.models.h_cgqe_transformer import (
    HcGQEModel,
    build_operator_vocab,
    tokenize_hamiltonian,
    tokenize_operator_sequence,
    PAULI_CHAR_VOCAB,
    SPECIAL_TOKENS,
    build_z_only_token_mask,
    build_length_token_mask,
)
from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    hamiltonian_to_spin_operator,
    find_record_by_name,
    get_active_electron_count,
)
from src.gqe.common.operator_pool import _jw_excitation_pauli_words


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """FIFO replay buffer storing (sequence, energy, log_probs, molecule) tuples."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def push(self, sequence: torch.Tensor, energy: float,
             log_probs: torch.Tensor, molecule: str,
             operators: list[str]) -> None:
        self.buffer.append({
            "sequence": sequence.cpu(),
            "energy": energy,
            "log_probs": log_probs.cpu(),
            "molecule": molecule,
            "operators": operators,
        })

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def best_energy(self, molecule: str | None = None) -> float:
        if not self.buffer:
            return float("inf")
        items = [x for x in self.buffer if molecule is None or x["molecule"] == molecule]
        if not items:
            return float("inf")
        return min(x["energy"] for x in items)


# ---------------------------------------------------------------------------
# Sequence sampling with log-probability tracking
# ---------------------------------------------------------------------------

def sample_sequences_with_logprobs(
    model: nn.Module,
    pauli_ids: torch.Tensor,
    coeffs: torch.Tensor,
    term_mask: torch.Tensor,
    n_samples: int,
    max_seq_len: int,
    temperature: float,
    vocab: dict[str, int],
    inv_vocab: dict[int, str],
    n_qubits: int,
    force_entanglement: bool = True,
    max_repeat: int = 4,
    device: torch.device = torch.device("cpu"),
    is_data_parallel: bool = False,
    top_p: float = 1.0,
    min_temp: float = 0.7,
    max_temp: float = 2.0,
    target_entropy: float = 1.5,
    explore_eps: float = 0.0,
    freq_penalty: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]], float]:
    """Sample n_sequences from the model and track per-token log probabilities.

    Returns:
        sequences: (n_samples, seq_len) token IDs
        log_probs: (n_samples, seq_len) per-token log probabilities
        operator_lists: list of n_samples lists of Pauli word strings
        mean_entropy: average per-token entropy across sampling (for adaptive temp)
    """
    model.eval()
    bos_id = SPECIAL_TOKENS["<BOS>"]
    eos_id = SPECIAL_TOKENS["<EOS>"]
    pad_id = SPECIAL_TOKENS["<PAD>"]

    z_only_mask = build_z_only_token_mask(vocab, device=device) if force_entanglement else None
    length_mask = build_length_token_mask(vocab, n_qubits, device=device)

    # Expand input for batch sampling
    pauli_ids_batch = pauli_ids.expand(n_samples, -1, -1).to(device)
    coeffs_batch = coeffs.expand(n_samples, -1).to(device)
    term_mask_batch = term_mask.expand(n_samples, -1).to(device)

    # Get encoder memory once
    with torch.no_grad():
        if is_data_parallel:
            _, memory = model.module.encoder(pauli_ids_batch, coeffs_batch, term_mask_batch)
        else:
            _, memory = model.encoder(pauli_ids_batch, coeffs_batch, term_mask_batch)

    # Autoregressive sampling with log-prob tracking
    sequences = torch.full((n_samples, 1), bos_id, dtype=torch.long, device=device)
    log_probs_list = []
    entropy_accum: list[float] = []
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)
    has_entangler = torch.zeros(n_samples, dtype=torch.bool, device=device)
    repeat_count = torch.zeros(n_samples, dtype=torch.long, device=device)
    last_token = torch.full((n_samples,), -1, dtype=torch.long, device=device)

    with torch.no_grad():
        for step in range(max_seq_len - 1):
            if is_data_parallel:
                logits = model.module.decoder(
                    sequences, memory, term_mask_batch,
                    tgt_key_padding_mask=(sequences == pad_id),
                )[:, -1, :]
            else:
                logits = model.decoder(
                    sequences, memory, term_mask_batch,
                    tgt_key_padding_mask=(sequences == pad_id),
                )[:, -1, :]

            # Frequency penalty: penalize tokens that have already appeared in the sequence.
            # This is the OpenAI frequency penalty approach (additive in logit space):
            #   logit[token] -= freq_penalty * count[token]
            # This prevents mode collapse to a single operator (e.g. XZXI×5).
            # See Keskar et al. (CTRL, 2019) and LZ penalty (arXiv:2504.20131).
            if freq_penalty > 0.0:
                # Count occurrences of each token in current sequences (excluding BOS)
                token_counts = torch.zeros(n_samples, logits.size(-1), device=device)
                for t in range(1, sequences.size(1)):
                    token_counts.scatter_(1, sequences[:, t:t+1], 1.0, reduce='add')
                logits = logits - freq_penalty * token_counts

            if temperature != 1.0:
                logits = logits / temperature

            # Force entanglement: mask Z-only tokens until an entangler is generated
            if force_entanglement and z_only_mask is not None:
                constrain = ~has_entangler
                if constrain.any():
                    logits[constrain] = logits[constrain].masked_fill(z_only_mask, float("-inf"))

            # Length compatibility mask
            if length_mask is not None:
                logits[:, ~length_mask] = float("-inf")

            # Compute probs with optional top-p (nucleus) sampling
            probs = torch.softmax(logits, dim=-1)

            # Top-p truncation: keep only tokens in the top-p cumulative probability mass
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                # Mask tokens beyond the nucleus
                nucleus_mask = cumsum <= top_p
                # Always keep at least the top-1 token
                nucleus_mask[..., 0] = True
                # Scatter mask back to original ordering
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask.scatter_(1, sorted_indices, nucleus_mask)
                probs = probs * mask
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            # Exploration floor: mix with uniform distribution to enforce minimum entropy.
            # When the model is extremely confident (e.g. 99% on one token), temperature
            # scaling alone is insufficient — even T=50 can't flatten a logit gap of 30.
            # Distribution mixing directly controls the entropy floor: with eps=0.3,
            # the top token gets at most 0.7*0.99 + 0.3/V ≈ 0.70, giving H ≈ 1.5+.
            if explore_eps > 0.0:
                uniform = torch.ones_like(probs) / probs.size(-1)
                probs = (1.0 - explore_eps) * probs + explore_eps * uniform
                # Renormalize for safety
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            # Track entropy for adaptive temperature
            step_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (n_samples,)
            entropy_accum.append(step_entropy.mean().item())

            dist = torch.distributions.Categorical(probs=probs)
            next_token = dist.sample()  # (n_samples,)
            token_log_prob = dist.log_prob(next_token)  # (n_samples,)

            sequences = torch.cat([sequences, next_token.unsqueeze(-1)], dim=1)
            log_probs_list.append(token_log_prob)

            next_token_flat = next_token
            same = next_token_flat == last_token
            repeat_count = torch.where(same, repeat_count + 1, torch.zeros_like(repeat_count))
            last_token = next_token_flat
            finished |= next_token_flat == eos_id
            finished |= repeat_count >= max_repeat
            if force_entanglement and z_only_mask is not None:
                has_entangler |= ~z_only_mask[next_token_flat]
            if finished.all():
                break

    # Pad log_probs to match sequence length
    seq_len = sequences.size(1)
    if len(log_probs_list) < seq_len - 1:
        remaining = seq_len - 1 - len(log_probs_list)
        log_probs_list.extend([torch.zeros(n_samples, device=device)] * remaining)

    log_probs = torch.stack(log_probs_list, dim=1)  # (n_samples, seq_len-1)

    # Compute mean entropy across all sampling steps
    mean_entropy = sum(entropy_accum) / max(len(entropy_accum), 1)

    # Decode operator sequences
    operator_lists = []
    for i in range(n_samples):
        words = []
        for tid in sequences[i]:
            tid_val = tid.item()
            if tid_val == eos_id:
                break
            if tid_val == pad_id:
                continue
            word = inv_vocab.get(tid_val, "<UNK>")
            if word not in ["<BOS>", "<UNK>"]:
                words.append(word)
        # Trim trailing noise (identity / single-qubit Z)
        while words and _is_trailing_noise(words[-1]):
            words.pop()
        operator_lists.append(words)

    model.train()
    return sequences, log_probs, operator_lists, mean_entropy


def _is_trailing_noise(word: str) -> bool:
    non_identity = [i for i, ch in enumerate(word) if ch != "I"]
    if not non_identity:
        return True
    if len(non_identity) == 1 and word[non_identity[0]] == "Z":
        return True
    return False


# ---------------------------------------------------------------------------
# Energy evaluation via CUDA-Q
# ---------------------------------------------------------------------------

def _pad_pauli_word(word: str, n_qubits: int) -> str:
    if len(word) == n_qubits:
        return word
    if len(word) < n_qubits:
        return word + "I" * (n_qubits - len(word))
    return word[:n_qubits]


# Module-level CUDA-Q kernel definition.
# MUST be defined here (not inside a function) because cudaq.make_kernel()
# is NOT thread-safe when called inside a loop that also dispatches
# observe_async. See CUDA-Q issues #4359, #2821.
_gqe_kernel = None

def _get_gqe_kernel():
    global _gqe_kernel
    if _gqe_kernel is None and cudaq is not None:
        @cudaq.kernel
        def kernel(n_q: int, n_el: int, pauli_words: list[cudaq.pauli_word], thetas: list[float]):
            q = cudaq.qvector(n_q)
            for i in range(n_el):
                x(q[i])
            for i in range(len(pauli_words)):
                exp_pauli(thetas[i], q, pauli_words[i])
        _gqe_kernel = kernel
    return _gqe_kernel


def _load_pretrain_sequences(
    gqe_json_path: Path,
    molecules: list[str],
    vocab: dict[str, int],
    max_seq_len: int,
) -> dict[str, list[dict[str, Any]]]:
    """Load pre-constructed operator sequences from GQE baseline JSON.

    Returns a dict mapping molecule name -> list of pre-constructed samples,
    each with 'operators', 'sequence' (tokenized), and 'energy'.
    """
    if not gqe_json_path.exists():
        print(f"  Warning: pretrain data not found at {gqe_json_path}")
        return {}

    with gqe_json_path.open("r") as f:
        data = json.load(f)

    results = data.get("results", data) if isinstance(data, dict) else data
    pretrain_data: dict[str, list[dict[str, Any]]] = {}

    for result in results:
        name = result.get("system", result.get("molecule", ""))
        if name not in molecules:
            continue
        ops_raw = result.get("gqe_selected_operators", [])
        if not ops_raw:
            continue
        operators = [op["pauli_word"] for op in ops_raw if "pauli_word" in op]
        if not operators:
            continue
        energy = result.get("baseline_energy", result.get("best_energy"))
        # Tokenize the operator sequence
        tokens = tokenize_operator_sequence(operators, vocab, max_seq_len)
        pretrain_data.setdefault(name, []).append({
            "operators": operators,
            "sequence": tokens,
            "energy": float(energy) if energy is not None else 0.0,
        })

    total = sum(len(v) for v in pretrain_data.values())
    print(f"  Loaded {total} pre-constructed sequences for {len(pretrain_data)} molecules")
    return pretrain_data


def _optimize_theta_quick(
    operators: list[str],
    molecule_record: dict[str, Any],
    initial_theta: float = 0.01,
    max_iters: int = 10,
) -> tuple[float, float]:
    """Quick L-BFGS-B optimization of rotation angles for a single circuit.

    Returns (optimized_energy, best_theta_scalar).
    Uses scipy.optimize.minimize on the CUDA-Q energy function.
    """
    if cudaq is None or not operators:
        return 0.0, initial_theta

    kernel = _get_gqe_kernel()
    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    spin_ham = hamiltonian_to_spin_operator(molecule_record)

    padded = [_pad_pauli_word(w, n_qubits) for w in operators]
    pauli_words = [cudaq.pauli_word(w) for w in padded]

    def energy_fn(thetas_arr):
        thetas = [float(t) for t in thetas_arr]
        try:
            result = cudaq.observe(kernel, spin_ham, n_qubits, n_electrons, pauli_words, thetas)
            return float(result.expectation())
        except Exception:
            return 1e10

    x0 = np.array([initial_theta] * len(pauli_words))
    try:
        opt_result = minimize(
            energy_fn, x0, method="L-BFGS-B",
            options={"maxiter": max_iters, "ftol": 1e-6},
        )
        return float(opt_result.fun), float(np.mean(opt_result.x))
    except Exception:
        return energy_fn(x0), initial_theta


def evaluate_energies_batch(
    operators_batch: list[list[str]],
    molecule_record: dict[str, Any],
    theta: float = 0.01,
) -> list[float]:
    """Evaluate energies for a batch of operator sequences using CUDA-Q."""
    if cudaq is None:
        return [0.0] * len(operators_batch)

    kernel = _get_gqe_kernel()
    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    spin_ham = hamiltonian_to_spin_operator(molecule_record)

    energies = []
    for operators in operators_batch:
        if not operators:
            energies.append(0.0)
            continue
        padded = [_pad_pauli_word(w, n_qubits) for w in operators]
        pauli_words = [cudaq.pauli_word(w) for w in padded]
        thetas = [theta] * len(pauli_words)
        try:
            result = cudaq.observe(kernel, spin_ham, n_qubits, n_electrons, pauli_words, thetas)
            energies.append(float(result.expectation()))
        except Exception as e:
            print(f"  CUDA-Q error: {e}")
            energies.append(0.0)
    return energies


def evaluate_energies_parallel(
    operators_batch: list[list[str]],
    molecule_record: dict[str, Any],
    theta: float = 0.01,
    n_gpus: int = 1,
) -> list[float]:
    """Evaluate energies in parallel across GPUs using CUDA-Q mqpu."""
    if cudaq is None:
        return [0.0] * len(operators_batch)

    kernel = _get_gqe_kernel()
    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    spin_ham = hamiltonian_to_spin_operator(molecule_record)

    # Submit all observations as async futures
    futures = []
    for operators in operators_batch:
        if not operators:
            futures.append(None)
            continue
        padded = [_pad_pauli_word(w, n_qubits) for w in operators]
        pauli_words = [cudaq.pauli_word(w) for w in padded]
        thetas = [theta] * len(pauli_words)
        try:
            qpu_id = len(futures) % n_gpus
            handle = cudaq.observe_async(kernel, spin_ham, n_qubits, n_electrons,
                                         pauli_words, thetas, qpu_id=qpu_id)
            futures.append(handle)
        except Exception as e:
            print(f"  CUDA-Q async error: {e}")
            futures.append(None)

    # Collect results
    energies = []
    for f in futures:
        if f is None:
            energies.append(0.0)
        else:
            try:
                energies.append(float(f.get().expectation()))
            except Exception as e:
                print(f"  CUDA-Q result error: {e}")
                energies.append(0.0)
    return energies


# ---------------------------------------------------------------------------
# Multi-component reward function
# ---------------------------------------------------------------------------

def compute_reward(
    energy: float,
    operators: list[str],
    hf_energy: float | None,
    fci_energy: float | None,
    max_seq_len: int,
    w_energy: float = 1.0,
    w_entangle: float = 0.1,
    w_depth: float = 0.01,
    w_commute: float = 0.05,
    w_diversity: float = 0.2,
    target_len: int = 10,
) -> float:
    """Multi-component reward for a generated circuit.

    R = w1 * (-E / |E_ref|)             # normalized energy (lower is better)
      + w2 * entanglement_fraction       # fraction of operators with X/Y
      + w3 * length_reward               # Gaussian reward peaking at target_len
      + w4 * non_commuting_fraction      # fraction of non-commuting pairs
      + w5 * operator_diversity          # fraction of unique operators

    The diversity component (w5) prevents mode collapse to a single operator.
    Without it, the policy finds shortcuts like [XZXI]×5 which maximizes
    entanglement fraction but has zero expressivity. See DARLING (arXiv:2509.02534)
    and GAPO (EMNLP 2025) for theoretical justification.
    """
    if not operators:
        return -1.0

    # Energy component: normalize by FCI if available, else HF, else use raw energy
    if fci_energy is not None and abs(fci_energy) > 1e-10:
        energy_reward = -energy / abs(fci_energy)
    elif hf_energy is not None and abs(hf_energy) > 1e-10:
        energy_reward = -energy / abs(hf_energy)
    else:
        # No reference: use raw negative energy (scale will be handled by GRPO normalization)
        energy_reward = -energy / max(abs(energy), 1.0)

    # Entanglement fraction: fraction of operators containing X or Y
    n_entangling = sum(1 for w in operators if "X" in w or "Y" in w)
    entangle_frac = n_entangling / len(operators)

    # Length reward: Gaussian peaking at target_len (replaces depth penalty)
    # Old: depth_penalty = -n_gates / max_seq_len (penalized ALL length → premature EOS)
    # New: reward peaks at target_len, decays for too short or too long
    n_ops = len(operators)
    length_reward = np.exp(-0.5 * ((n_ops - target_len) / max(target_len * 0.5, 1.0)) ** 2)

    # Non-commuting fraction (sampled pairs for efficiency)
    if n_ops >= 2:
        n_pairs = min(n_ops * (n_ops - 1) // 2, 50)  # cap for efficiency
        n_commute = 0
        checked = 0
        for i in range(min(n_ops, 20)):
            for j in range(i + 1, min(n_ops, 20)):
                if _words_commute(operators[i], operators[j]):
                    n_commute += 1
                checked += 1
                if checked >= n_pairs:
                    break
            if checked >= n_pairs:
                break
        non_commute_frac = 1.0 - (n_commute / max(checked, 1))
    else:
        non_commute_frac = 0.0

    # Operator diversity: fraction of unique operators in the sequence.
    # [XZXI, XZXI, XZXI] → 1/3 = 0.33 (penalized)
    # [XZXI, YZYI, XZXI] → 2/3 = 0.67 (moderate)
    # [XZXI, YZYI, XYYX] → 3/3 = 1.0 (max diversity)
    unique_frac = len(set(operators)) / len(operators)

    reward = (w_energy * energy_reward
              + w_entangle * entangle_frac
              + w_depth * length_reward
              + w_commute * non_commute_frac
              + w_diversity * unique_frac)
    return reward


def _words_commute(w1: str, w2: str) -> bool:
    """Check if two Pauli words commute (even number of anti-commuting positions)."""
    min_len = min(len(w1), len(w2))
    n_anticommute = 0
    for i in range(min_len):
        c1, c2 = w1[i], w2[i]
        if c1 == "I" or c2 == "I":
            continue
        if c1 != c2:
            n_anticommute += 1
    return n_anticommute % 2 == 0


# ---------------------------------------------------------------------------
# DAPO Loss
# ---------------------------------------------------------------------------

def dapo_loss(
    log_probs_new: torch.Tensor,      # (G, seq_len) new policy log-probs
    log_probs_old: torch.Tensor,      # (G, seq_len) old policy log-probs
    advantages: torch.Tensor,          # (G,) group-relative advantages
    attention_mask: torch.Tensor,      # (G, seq_len) 1 for real tokens, 0 for pad
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    token_level: bool = True,
    entropy_coef: float = 0.0,
    logits: torch.Tensor | None = None,  # (G, seq_len, vocab_size) for entropy bonus
) -> torch.Tensor:
    """DAPO clipped surrogate loss with optional entropy regularization.

    Key differences from GRPO:
    1. Clip-Higher: asymmetric clipping (clip_low < clip_high) prevents entropy collapse
    2. Token-Level Loss: averages over all tokens, not per-sequence
    3. Entropy Bonus: encourages diverse sampling, prevents premature convergence
    """
    # Importance sampling ratio
    ratio = torch.exp(log_probs_new - log_probs_old)  # (G, seq_len)

    # Expand advantages to per-token
    adv = advantages.unsqueeze(1).expand_as(log_probs_new)  # (G, seq_len)

    # Clipped surrogate
    pg_losses1 = -adv * ratio
    pg_losses2 = -adv * torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)  # (G, seq_len)

    # Apply attention mask
    pg_losses = pg_losses * attention_mask

    # Entropy bonus: encourage diverse sampling
    entropy_loss = torch.tensor(0.0, device=log_probs_new.device)
    if entropy_coef > 0.0 and logits is not None:
        probs = torch.softmax(logits, dim=-1)
        log_probs_all = torch.log_softmax(logits, dim=-1)
        # Per-token entropy: H = -sum(p * log(p))
        token_entropy = -(probs * log_probs_all).sum(dim=-1)  # (G, seq_len)
        token_entropy = token_entropy * attention_mask
        entropy_loss = -entropy_coef * token_entropy.sum() / attention_mask.sum().clamp_min(1.0)

    if token_level:
        # Token-level: average over all tokens across all sequences
        pg_loss = pg_losses.sum() / attention_mask.sum().clamp_min(1.0)
    else:
        # Sequence-level: average per-sequence, then average across sequences
        seq_losses = pg_losses.sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)
        pg_loss = seq_losses.mean()

    return pg_loss + entropy_loss


def compute_advantages(
    rewards: np.ndarray,  # (G,) rewards for a group of samples
    use_grpo: bool = True,
    repo_beta: float = 0.0,
    old_log_probs: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute group-relative advantages (GRPO style).

    A_i = (R_i - mean(R)) / (std(R) + eps)

    With REPO modification (arXiv:2603.11682):
    A_REPO = A - beta * (log_pi(a|s) - E[log_pi(a|s)])
    The centered log-prob term penalizes high-probability actions, directly
    pushing the policy toward diversity. beta > 0 increases entropy.
    """
    if use_grpo:
        mean_r = rewards.mean()
        std_r = rewards.std()
        advantages = (rewards - mean_r) / (std_r + 1e-8)
    else:
        advantages = rewards - rewards.mean()
    advantages = torch.tensor(advantages, dtype=torch.float32)

    # REPO: modify advantages with centered log-prob penalty
    if repo_beta > 0.0 and old_log_probs is not None and attention_mask is not None:
        # Ensure advantages on same device as old_log_probs
        advantages = advantages.to(old_log_probs.device)
        # Per-sequence mean log-prob: L_i = mean(log_pi(a_t|s)) over tokens
        seq_log_probs = old_log_probs * attention_mask  # (G, seq_len)
        seq_mean_log_prob = seq_log_probs.sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)  # (G,)
        # Center across the group: L_centered_i = L_i - mean_j(L_j)
        # Samples with higher-than-average log-prob (more confident/deterministic)
        # get penalized; samples with lower log-prob (more diverse) get boosted.
        group_mean_log_prob = seq_mean_log_prob.mean()
        centered_log_prob = seq_mean_log_prob - group_mean_log_prob  # (G,)
        # REPO advantage: A_REPO = A - beta * L_centered
        advantages = advantages - repo_beta * centered_log_prob

    return advantages


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_cuda_context() -> None:
    import ctypes
    import os
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    libcudart = ctypes.CDLL("/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib/libcudart.so")
    libcudart.cudaSetDevice(local_rank)
    d = ctypes.c_void_p()
    libcudart.cudaMalloc(ctypes.byref(d), 4)
    libcudart.cudaFree(d)


def load_molecule_data(ham_path: Path, molecule: str, vocab: dict[str, int],
                       max_terms: int, max_pauli_len: int, max_seq_len: int,
                       ) -> dict[str, Any]:
    """Load and tokenize a molecule's Hamiltonian."""
    records = load_hamiltonian_records(ham_path)
    record = find_record_by_name(records, molecule)
    terms = record.get("terms", [])
    term_list = []
    for t in terms:
        if isinstance(t, dict):
            term_list.append((t.get("term", ""), float(t.get("real", 0.0))))
    term_list.sort(key=lambda x: abs(x[1]), reverse=True)

    ham_tokens = tokenize_hamiltonian(term_list, vocab, max_terms, max_pauli_len)
    n_qubits = int(record.get("n_qubits", 0))

    # Get reference energies (may be missing)
    hf_energy = record.get("hf_energy")
    if hf_energy is not None:
        hf_energy = float(hf_energy)
    fci_energy = record.get("fci_energy")
    if fci_energy is not None:
        fci_energy = float(fci_energy)

    return {
        "name": molecule,
        "record": record,
        "n_qubits": n_qubits,
        "hf_energy": hf_energy,
        "fci_energy": fci_energy,
        "pauli_ids": ham_tokens["pauli_ids"],
        "coeffs": ham_tokens["coeffs"],
        "term_mask": ham_tokens["term_mask"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DAPO-RL training for H-cGQE Transformer")
    # Model
    parser.add_argument("--checkpoint", type=Path, default=None, help="Supervised pretrained checkpoint (skip for --from-scratch)")
    parser.add_argument("--from-scratch", action="store_true", default=False,
                        help="Initialize model randomly (pure RL, no supervised pretraining). "
                             "Builds vocab from UCCSD operator pool. arXiv:2502.19402 shows "
                             "RL from scratch outperforms SFT-then-RL by avoiding imitation bias.")
    # Model config (used when --from-scratch, otherwise loaded from checkpoint)
    parser.add_argument("--d-model", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--encoder-layers", type=int, default=4, help="Encoder layers")
    parser.add_argument("--decoder-layers", type=int, default=6, help="Decoder layers")
    parser.add_argument("--dim-feedforward", type=int, default=1024, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--hamiltonians", type=Path, default=Path("results/data/hamiltonians.json"))
    parser.add_argument("--molecules", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output checkpoint path")
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=50, help="Circuits sampled per molecule per epoch")
    parser.add_argument("--n-iters", type=int, default=1,
                        help="Number of gradient update iterations per epoch on different replay buffer batches "
                             "(GPT-QE paper uses N_iter=5)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Lower LR for RL fine-tuning")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--max-terms", type=int, default=128)
    parser.add_argument("--max-pauli-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    # DAPO
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.28)
    parser.add_argument("--token-level-loss", action="store_true", default=True)
    parser.add_argument("--dynamic-sampling", action="store_true", default=True,
                        help="Skip groups where all energies are identical (std=0)")
    parser.add_argument("--max-resample-attempts", type=int, default=3,
                        help="Max resampling attempts for dynamic sampling before giving up")
    # Exploration
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling threshold. 1.0 = full distribution, 0.9 = nucleus")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient to prevent policy collapse (0 = disabled)")
    parser.add_argument("--adaptive-temp", action="store_true", default=True,
                        help="Dynamically adjust temperature based on sampling entropy")
    parser.add_argument("--min-temp", type=float, default=0.7,
                        help="Minimum temperature for adaptive temp scheduling")
    parser.add_argument("--max-temp", type=float, default=2.0,
                        help="Maximum temperature for adaptive temp scheduling")
    parser.add_argument("--target-entropy", type=float, default=1.5,
                        help="Target per-token entropy for adaptive temperature")
    parser.add_argument("--explore-eps", type=float, default=0.3,
                        help="Exploration floor: mix sampling distribution with uniform (0=off, 0.3=30%% uniform)")
    parser.add_argument("--adaptive-eps", action="store_true", default=True,
                        help="Adaptively tune explore_eps based on observed entropy")
    # Reward weights
    parser.add_argument("--w-energy", type=float, default=1.0)
    parser.add_argument("--w-entangle", type=float, default=0.1)
    parser.add_argument("--w-depth", type=float, default=0.05,
                        help="Weight for length reward (Gaussian peaking at target-len)")
    parser.add_argument("--w-commute", type=float, default=0.05)
    parser.add_argument("--w-diversity", type=float, default=0.2,
                        help="Weight for operator diversity reward (unique operator fraction). "
                             "Prevents mode collapse to a single repeated operator.")
    parser.add_argument("--target-len", type=int, default=10,
                        help="Target sequence length for length reward (Gaussian peak)")
    parser.add_argument("--freq-penalty", type=float, default=1.0,
                        help="Frequency penalty for sampling: subtracts freq_penalty * count[token] "
                             "from logits. Prevents mode collapse to repeated operators.")
    # Replay buffer
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--buffer-batch-size", type=int, default=0,
                        help="Batch size from replay buffer (0 = no replay training)")
    # Pre-constructed data mixing (GPT-QE paper Section 2.2)
    parser.add_argument("--pretrain-data", type=Path, default=None,
                        help="Path to GQE baseline JSON with pre-constructed operator sequences. "
                             "These are mixed into the replay buffer at --pretrain-fraction, "
                             "linearly decaying to 0 over --pretrain-decay-epochs.")
    parser.add_argument("--pretrain-fraction", type=float, default=0.3,
                        help="Initial fraction of pre-constructed data in replay buffer (0.3 = 30%%). "
                             "Linearly decays to 0 over --pretrain-decay-epochs.")
    parser.add_argument("--pretrain-decay-epochs", type=int, default=150,
                        help="Number of epochs to linearly decay pre-constructed data fraction to 0.")
    # Adaptive theta optimization during RL
    parser.add_argument("--adaptive-theta", action="store_true", default=False,
                        help="Run quick L-BFGS-B optimization of theta for the best circuit in each "
                             "batch. Uses optimized energy for reward instead of fixed theta. "
                             "More expensive but gives much better energy signal.")
    parser.add_argument("--adaptive-theta-iters", type=int, default=10,
                        help="Max L-BFGS-B iterations for adaptive theta optimization.")
    # CUDA-Q
    parser.add_argument("--target", type=str, default="nvidia")
    parser.add_argument("--target-option", type=str, default="mqpu")
    parser.add_argument("--theta", type=float, default=0.01, help="Fixed rotation angle for energy eval")
    parser.add_argument("--max-qubits", type=int, default=24, help="Skip molecules with more qubits")
    # Device
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--use-fp16", action="store_true", help="Deprecated: use --use-bf16 instead")
    parser.add_argument("--use-bf16", action="store_true", default=True,
                        help="Use bfloat16 mixed precision (prevents FP16 multiplicative bias entropy collapse)")
    parser.add_argument("--force-entanglement", action="store_true", default=True)
    parser.add_argument("--max-repeat", type=int, default=4)
    # REPO-style advantage modification
    parser.add_argument("--repo-beta", type=float, default=0.05,
                        help="REPO advantage regularization coefficient (0=off, 0.05=mild entropy preservation)")
    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true", default=True,
                        help="Enable curriculum learning: start with small molecules, gradually add larger ones")
    parser.add_argument("--curriculum-warmup", type=int, default=30,
                        help="Number of epochs to train only on smallest molecules before adding larger ones")
    parser.add_argument("--curriculum-steps", type=int, default=3,
                        help="Number of curriculum stages (molecules added in stages)")
    args = parser.parse_args()

    _seed_everything(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model: either from checkpoint or from scratch (pure RL)
    if args.from_scratch or args.checkpoint is None:
        print("\n=== PURE RL FROM SCRATCH (no supervised pretraining) ===")
        print("Building vocabulary from UCCSD operator pool...")
        # Build vocab from all molecules' UCCSD excitation operators.
        # Limit pool size to keep vocab manageable (GPT-QE paper uses 12 operators).
        # Hamiltonian terms are NOT included — they're tokenized at character level
        # (PAULI_CHAR_VOCAB: I/X/Y/Z) for the encoder, not as operator tokens.
        MAX_SINGLES = 10  # 10 single excitations → 20 Pauli words per molecule
        MAX_DOUBLES = 10  # 10 double excitations → 80 Pauli words per molecule
        all_pauli_words: list[str] = []
        ham_records = load_hamiltonian_records(args.hamiltonians)
        for mol_name in args.molecules:
            record = find_record_by_name(ham_records, mol_name)
            if record is None:
                continue
            n_qubits_mol = int(record.get("n_qubits", 0))
            if n_qubits_mol > args.max_qubits:
                continue
            n_electrons_mol = get_active_electron_count(record)
            excitation_words = _jw_excitation_pauli_words(
                n_qubits_mol, n_electrons_mol,
                max_singles=MAX_SINGLES, max_doubles=MAX_DOUBLES,
            )
            for word, _ in excitation_words:
                all_pauli_words.append(word)
        vocab = build_operator_vocab(all_pauli_words)
        inv_vocab = {v: k for k, v in vocab.items()}
        config = {
            "vocab_size": max(vocab.values()) + 1,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "encoder_layers": args.encoder_layers,
            "decoder_layers": args.decoder_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "max_pauli_len": args.max_pauli_len,
            "max_seq_len": args.max_seq_len,
        }
        print(f"Vocab size: {config['vocab_size']} "
              f"({len(all_pauli_words)} Pauli words from UCCSD pool, "
              f"max_singles={MAX_SINGLES}, max_doubles={MAX_DOUBLES} per molecule)")
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
        # Random initialization — no supervised pretraining
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model initialized from scratch: {n_params:,} parameters")
        print("WARNING: Pure RL mode. Entropy collapse prevention is critical.")
        print("  Enabled: distribution mixing, REPO, curriculum, BF16, top-p")
    else:
        print(f"Loading checkpoint from {args.checkpoint}")
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
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {n_params:,} parameters")

    is_dp = False
    if args.multi_gpu and torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        model = nn.DataParallel(model)
        is_dp = True
        print(f"Using nn.DataParallel with {n_gpus} GPUs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # BF16 mixed precision: arXiv:2603.11682 shows FP16 multiplicative bias causes entropy collapse.
    # BF16 has 8 exponent bits (same as FP32) vs FP16's 5, avoiding the multiplicative
    # bias in softmax gradients that systematically reduces entropy.
    use_bf16 = args.use_bf16 or args.use_fp16  # bf16 supersedes fp16
    scaler = torch.amp.GradScaler('cuda') if (args.use_fp16 and not args.use_bf16) else None
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if use_bf16:
        print(f"Using BF16 mixed precision (prevents FP16 entropy collapse)")
        # BF16 doesn't need GradScaler (no overflow risk with 8-bit exponent)
        scaler = None

    # Setup CUDA-Q
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if cudaq and args.target:
        try:
            if args.target == "nvidia" and args.target_option == "mqpu":
                cudaq.set_target("nvidia", option="mqpu")
                print(f"CUDA-Q target: nvidia (mqpu, {n_gpus} GPUs)")
            else:
                cudaq.set_target(args.target)
                print(f"CUDA-Q target: {args.target}")
        except Exception as e:
            print(f"Warning: CUDA-Q target setup failed: {e}")

    # Load molecule data
    print("\nLoading molecule data:")
    molecules_data = []
    for mol_name in args.molecules:
        mol_data = load_molecule_data(
            args.hamiltonians, mol_name, vocab,
            args.max_terms, args.max_pauli_len, args.max_seq_len,
        )
        if mol_data["n_qubits"] > args.max_qubits:
            print(f"  {mol_name}: {mol_data['n_qubits']} qubits — SKIPPING (max {args.max_qubits})")
            continue
        fci_str = f"{mol_data['fci_energy']:.4f}" if mol_data['fci_energy'] is not None else "N/A"
        print(f"  {mol_name}: {mol_data['n_qubits']} qubits, "
              f"HF={mol_data['hf_energy'] or 'N/A'}, "
              f"FCI={fci_str}")
        molecules_data.append(mol_data)

    if not molecules_data:
        print("No molecules to train on!")
        return

    # Sort molecules by qubit count for curriculum learning
    molecules_data.sort(key=lambda m: m["n_qubits"])
    print(f"\nMolecules sorted by qubit count (curriculum order):")
    for i, m in enumerate(molecules_data):
        print(f"  [{i}] {m['name']}: {m['n_qubits']} qubits")

    # Curriculum learning: define stages
    # Stage 0: smallest molecule(s) only
    # Stage 1: add next group
    # ... until all molecules are included
    n_mols = len(molecules_data)
    if args.curriculum and n_mols > 1 and args.curriculum_steps > 0:
        stage_size = max(1, n_mols // args.curriculum_steps)
        curriculum_stages = []
        for s in range(args.curriculum_steps):
            end_idx = min(n_mols, (s + 1) * stage_size)
            curriculum_stages.append(molecules_data[:end_idx])
        # Last stage includes all
        curriculum_stages[-1] = molecules_data
        print(f"Curriculum: {len(curriculum_stages)} stages, warmup={args.curriculum_warmup} epochs")
        for s, stage_mols in enumerate(curriculum_stages):
            print(f"  Stage {s}: {[m['name'] for m in stage_mols]}")
    else:
        curriculum_stages = [molecules_data]
        args.curriculum_warmup = 0

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)

    # Load pre-constructed sequences from GQE baseline (GPT-QE paper Section 2.2)
    pretrain_data: dict[str, list[dict[str, Any]]] = {}
    if args.pretrain_data is not None:
        print(f"\nLoading pre-constructed data from {args.pretrain_data}...")
        pretrain_data = _load_pretrain_sequences(
            args.pretrain_data, args.molecules, vocab, args.max_seq_len,
        )
        if pretrain_data:
            total_pre = sum(len(v) for v in pretrain_data.values())
            print(f"  Pre-constructed mixing: {args.pretrain_fraction*100:.0f}% initial, "
                  f"decaying to 0% over {args.pretrain_decay_epochs} epochs")
            # Pre-fill replay buffer with pre-constructed data
            n_pre = int(args.buffer_size * args.pretrain_fraction)
            all_pre_samples = []
            for mol_name, samples in pretrain_data.items():
                for s in samples:
                    all_pre_samples.append((mol_name, s))
            if all_pre_samples and n_pre > 0:
                # Sample with replacement if needed to fill n_pre slots
                indices = np.random.choice(len(all_pre_samples), size=min(n_pre, len(all_pre_samples)), replace=True)
                for idx in indices:
                    mol_name, s = all_pre_samples[idx]
                    replay_buffer.push(
                        s["sequence"], s["energy"],
                        torch.zeros(args.max_seq_len),  # dummy log_probs
                        mol_name, s["operators"],
                    )
                print(f"  Pre-filled replay buffer with {len(replay_buffer)} pre-constructed samples")

    # Training loop
    best_energy_per_mol = {m["name"]: float("inf") for m in molecules_data}
    train_metrics_log = []
    entropy_history: list[float] = []

    pbar = tqdm(range(args.epochs), desc="RL Epoch", unit="epoch")
    for epoch in pbar:
        # Curriculum: select which molecules to train on this epoch
        if args.curriculum and len(curriculum_stages) > 1:
            stage_idx = min(len(curriculum_stages) - 1, epoch // args.curriculum_warmup)
            active_molecules = curriculum_stages[stage_idx]
            if stage_idx < len(curriculum_stages) - 1 and epoch % args.curriculum_warmup == 0 and epoch > 0:
                print(f"\n  Curriculum stage {stage_idx}: now training on {[m['name'] for m in active_molecules]}")
        else:
            active_molecules = molecules_data
        epoch_energies = []
        epoch_rewards = []
        epoch_losses = []
        epoch_skipped = 0
        epoch_sequences_generated = 0

        for mol_data in active_molecules:
            mol_name = mol_data["name"]
            n_qubits = mol_data["n_qubits"]

            # --- Phase 1: Sample sequences ---
            attempts = 0
            valid_batch = False

            while attempts < args.max_resample_attempts:
                attempts += 1
                # Adaptive exploration: tune explore_eps based on observed entropy.
                # Temperature scaling is insufficient when logits are sharp (gap ~30);
                # distribution mixing directly enforces an entropy floor.
                sample_temp = args.temperature
                sample_eps = args.explore_eps
                if args.adaptive_eps and len(entropy_history) > 0:
                    recent_entropy = np.mean(entropy_history[-10:])
                    if recent_entropy < args.target_entropy * 0.5:
                        # Entropy collapsed — increase exploration mixing
                        sample_eps = min(0.6, args.explore_eps * (args.target_entropy / max(recent_entropy, 0.05)))
                    elif recent_entropy > args.target_entropy * 2.0:
                        # Entropy too high — reduce exploration
                        sample_eps = max(0.0, args.explore_eps * 0.5)

                sequences, old_log_probs, operator_lists, sample_entropy = sample_sequences_with_logprobs(
                    model,
                    mol_data["pauli_ids"].unsqueeze(0),
                    mol_data["coeffs"].unsqueeze(0),
                    mol_data["term_mask"].unsqueeze(0),
                    n_samples=args.n_samples,
                    max_seq_len=args.max_seq_len,
                    temperature=sample_temp,
                    vocab=vocab,
                    inv_vocab=inv_vocab,
                    n_qubits=n_qubits,
                    force_entanglement=args.force_entanglement,
                    max_repeat=args.max_repeat,
                    device=device,
                    is_data_parallel=is_dp,
                    top_p=args.top_p,
                    min_temp=args.min_temp,
                    max_temp=args.max_temp,
                    target_entropy=args.target_entropy,
                    explore_eps=sample_eps,
                    freq_penalty=args.freq_penalty,
                )
                entropy_history.append(sample_entropy)

                # Filter out empty sequences
                valid_indices = [i for i, ops in enumerate(operator_lists) if len(ops) > 0]
                if not valid_indices:
                    print(f"  {mol_name}: all sequences empty, resampling...")
                    continue

                sequences = sequences[valid_indices]
                old_log_probs = old_log_probs[valid_indices]
                operator_lists = [operator_lists[i] for i in valid_indices]

                # --- Phase 2: Evaluate energies ---
                # Sync PyTorch GPU ops before CUDA-Q to avoid context conflicts
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if args.target == "nvidia" and args.target_option == "mqpu" and n_gpus > 1:
                    energies = evaluate_energies_parallel(
                        operator_lists, mol_data["record"],
                        theta=args.theta, n_gpus=n_gpus,
                    )
                else:
                    energies = evaluate_energies_batch(
                        operator_lists, mol_data["record"],
                        theta=args.theta,
                    )

                # --- Phase 3: Compute rewards ---
                # Adaptive theta: optimize coefficients for the best circuit in the batch
                if args.adaptive_theta and len(energies) > 0:
                    best_idx = int(np.argmin(energies))
                    if operator_lists[best_idx]:
                        opt_energy, opt_theta = _optimize_theta_quick(
                            operator_lists[best_idx], mol_data["record"],
                            initial_theta=args.theta,
                            max_iters=args.adaptive_theta_iters,
                        )
                        if opt_energy < energies[best_idx]:
                            energies[best_idx] = opt_energy

                rewards = np.array([
                    compute_reward(
                        e, ops, mol_data["hf_energy"], mol_data["fci_energy"],
                        args.max_seq_len,
                        args.w_energy, args.w_entangle, args.w_depth, args.w_commute,
                        w_diversity=args.w_diversity,
                        target_len=args.target_len,
                    )
                    for e, ops in zip(energies, operator_lists)
                ])

                # --- Phase 4: Dynamic sampling check ---
                if args.dynamic_sampling and rewards.std() < 1e-8:
                    print(f"  {mol_name}: std(rewards)={rewards.std():.2e}, "
                          f"skipping (dynamic sampling)")
                    epoch_skipped += 1
                    continue

                valid_batch = True
                break

            if not valid_batch:
                continue

            # --- Phase 5: Compute advantages (with REPO modification) ---
            pad_id = SPECIAL_TOKENS["<PAD>"]
            attn_mask_for_adv = (sequences[:, 1:] != pad_id).float()
            advantages = compute_advantages(
                rewards,
                repo_beta=args.repo_beta,
                old_log_probs=old_log_probs,
                attention_mask=attn_mask_for_adv,
            )

            # Store in replay buffer
            for i, (ops, e) in enumerate(zip(operator_lists, energies)):
                replay_buffer.push(
                    sequences[i], e, old_log_probs[i], mol_name, ops,
                )
                if e < best_energy_per_mol[mol_name]:
                    best_energy_per_mol[mol_name] = e

            epoch_energies.extend(energies)
            epoch_rewards.extend(rewards.tolist())
            epoch_sequences_generated += len(operator_lists)

            # --- Phase 6: Compute DAPO loss and update (n_iters gradient steps) ---
            model.train()
            bos_id = SPECIAL_TOKENS["<BOS>"]
            pad_id = SPECIAL_TOKENS["<PAD>"]

            for iter_idx in range(max(1, args.n_iters)):
                optimizer.zero_grad()

                if iter_idx == 0:
                    # First iteration: use freshly sampled sequences
                    iter_sequences = sequences
                    iter_old_log_probs = old_log_probs
                    iter_advantages = advantages
                    iter_mol_data = mol_data
                else:
                    # Subsequent iterations: sample from replay buffer
                    if len(replay_buffer) < args.n_samples:
                        break
                    replay_samples = replay_buffer.sample(args.n_samples)
                    iter_sequences = torch.stack([s["sequence"] for s in replay_samples])
                    iter_old_log_probs = torch.stack([s["log_probs"] for s in replay_samples])
                    iter_operators = [s["operators"] for s in replay_samples]
                    iter_mol_name = replay_samples[0]["molecule"]
                    iter_mol_data = next((m for m in molecules_data if m["name"] == iter_mol_name), mol_data)
                    # Recompute advantages from replay energies
                    replay_energies = np.array([s["energy"] for s in replay_samples])
                    replay_rewards = np.array([
                        compute_reward(
                            e, ops, iter_mol_data["hf_energy"], iter_mol_data["fci_energy"],
                            args.max_seq_len,
                            args.w_energy, args.w_entangle, args.w_depth, args.w_commute,
                            w_diversity=args.w_diversity,
                            target_len=args.target_len,
                        )
                        for e, ops in zip(replay_energies, iter_operators)
                    ])
                    if replay_rewards.std() < 1e-8:
                        break  # skip if no advantage signal
                    iter_attn = (iter_sequences[:, 1:] != pad_id).float()
                    iter_advantages = compute_advantages(
                        replay_rewards,
                        repo_beta=args.repo_beta,
                        old_log_probs=iter_old_log_probs,
                        attention_mask=iter_attn,
                    )

                tgt_input = iter_sequences[:, :-1].to(device)
                tgt_labels = iter_sequences[:, 1:].to(device)
                attention_mask = (tgt_labels != pad_id).float().to(device)

                # Expand Hamiltonian input for the batch
                pauli_ids_batch = iter_mol_data["pauli_ids"].unsqueeze(0).expand(
                    iter_sequences.size(0), -1, -1
                ).to(device)
                coeffs_batch = iter_mol_data["coeffs"].unsqueeze(0).expand(
                    iter_sequences.size(0), -1
                ).to(device)
                term_mask_batch = iter_mol_data["term_mask"].unsqueeze(0).expand(
                    iter_sequences.size(0), -1
                ).to(device)

                if scaler is not None:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        logits = model(
                            pauli_ids_batch, coeffs_batch, tgt_input,
                            term_mask=term_mask_batch,
                            tgt_key_padding_mask=(tgt_input == pad_id),
                        )
                        log_probs_new = _compute_sequence_log_probs(
                            logits, tgt_labels, attention_mask,
                        )
                        loss = dapo_loss(
                            log_probs_new, iter_old_log_probs.to(device),
                            iter_advantages.to(device), attention_mask,
                            clip_low=args.clip_low, clip_high=args.clip_high,
                            token_level=args.token_level_loss,
                            entropy_coef=args.entropy_coef,
                            logits=logits,
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                elif use_bf16:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(
                            pauli_ids_batch, coeffs_batch, tgt_input,
                            term_mask=term_mask_batch,
                            tgt_key_padding_mask=(tgt_input == pad_id),
                        )
                        log_probs_new = _compute_sequence_log_probs(
                            logits, tgt_labels, attention_mask,
                        )
                        loss = dapo_loss(
                            log_probs_new, iter_old_log_probs.to(device),
                            iter_advantages.to(device), attention_mask,
                            clip_low=args.clip_low, clip_high=args.clip_high,
                            token_level=args.token_level_loss,
                            entropy_coef=args.entropy_coef,
                            logits=logits,
                        )
                    loss.backward()
                    optimizer.step()
                else:
                    logits = model(
                        pauli_ids_batch, coeffs_batch, tgt_input,
                        term_mask=term_mask_batch,
                        tgt_key_padding_mask=(tgt_input == pad_id),
                    )
                    log_probs_new = _compute_sequence_log_probs(
                        logits, tgt_labels, attention_mask,
                    )
                    loss = dapo_loss(
                        log_probs_new, iter_old_log_probs.to(device),
                        iter_advantages.to(device), attention_mask,
                        clip_low=args.clip_low, clip_high=args.clip_high,
                        token_level=args.token_level_loss,
                        entropy_coef=args.entropy_coef,
                        logits=logits,
                    )
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())

        # --- Replay buffer training (optional) ---
        if args.buffer_batch_size > 0 and len(replay_buffer) >= args.buffer_batch_size:
            replay_samples = replay_buffer.sample(args.buffer_batch_size)
            # TODO: could add replay training here
            pass

        # --- Pre-constructed data mixing (GPT-QE paper Section 2.2) ---
        # Linearly decay the fraction of pre-constructed data to 0
        if pretrain_data and args.pretrain_decay_epochs > 0:
            current_frac = max(0.0, args.pretrain_fraction * (1.0 - epoch / args.pretrain_decay_epochs))
            if current_frac > 0:
                n_inject = int(args.buffer_size * current_frac * 0.05)  # inject 5% of fraction per epoch
                if n_inject > 0:
                    all_pre = []
                    for mol_name, samples in pretrain_data.items():
                        for s in samples:
                            all_pre.append((mol_name, s))
                    if all_pre:
                        indices = np.random.choice(len(all_pre), size=min(n_inject, len(all_pre)), replace=True)
                        for idx in indices:
                            mol_name, s = all_pre[idx]
                            replay_buffer.push(
                                s["sequence"], s["energy"],
                                torch.zeros(args.max_seq_len),
                                mol_name, s["operators"],
                            )

        # Logging
        mean_energy = np.mean(epoch_energies) if epoch_energies else 0.0
        min_energy = np.min(epoch_energies) if epoch_energies else 0.0
        mean_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        recent_entropy = np.mean(entropy_history[-10:]) if entropy_history else 0.0
        pbar.set_postfix_str(
            f"loss={mean_loss:.4f} "
            f"E_mean={mean_energy:.4f} "
            f"E_min={min_energy:.4f} "
            f"R={mean_reward:.4f} "
            f"H={recent_entropy:.2f} "
            f"skip={epoch_skipped} "
            f"buf={len(replay_buffer)}"
        )

        train_metrics_log.append({
            "epoch": epoch,
            "mean_energy": mean_energy,
            "min_energy": min_energy,
            "mean_reward": mean_reward,
            "mean_loss": mean_loss,
            "mean_entropy": recent_entropy,
            "n_skipped": epoch_skipped,
            "n_generated": epoch_sequences_generated,
            "buffer_size": len(replay_buffer),
            "best_energies": dict(best_energy_per_mol),
        })

        # Save best model
        if mean_loss < 1e9:  # always save (could add early stopping)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            save_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                "model_state": save_model.state_dict(),
                "vocab": vocab,
                "inv_vocab": inv_vocab,
                "config": config,
                "metrics": {
                    "best_energies": dict(best_energy_per_mol),
                    "train_log": train_metrics_log,
                },
            }, args.out)

    # Final summary
    print("\n" + "=" * 60)
    print("DAPO-RL Training Complete")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Replay buffer size: {len(replay_buffer)}")
    print("\nBest energies per molecule:")
    for mol_name, e in best_energy_per_mol.items():
        mol = next(m for m in molecules_data if m["name"] == mol_name)
        fci = mol["fci_energy"]
        hf = mol["hf_energy"]
        hf_str = f"{hf:.6f}" if hf is not None else "N/A"
        fci_str = f"{fci:.6f}" if fci is not None else "N/A"
        err_str = ""
        if fci is not None:
            err = abs(e - fci) * 1000  # mHa
            err_str = f"  err={err:.2f} mHa"
        print(f"  {mol_name}: E={e:.6f}  HF={hf_str}  FCI={fci_str}{err_str}")

    # Save metrics JSON
    metrics_path = args.out.parent / f"{args.out.stem}_rl_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "best_energies": dict(best_energy_per_mol),
            "train_log": train_metrics_log,
            "final_buffer_size": len(replay_buffer),
        }, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Model saved to: {args.out}")


def _compute_sequence_log_probs(
    logits: torch.Tensor,      # (G, seq_len, vocab_size)
    labels: torch.Tensor,      # (G, seq_len) target token IDs
    attention_mask: torch.Tensor,  # (G, seq_len) 1 for real, 0 for pad
) -> torch.Tensor:
    """Compute per-token log probabilities of the selected tokens."""
    log_probs = F.log_softmax(logits, dim=-1)  # (G, seq_len, vocab_size)
    # Gather log-probs of the actual tokens
    gathered = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # (G, seq_len)
    return gathered * attention_mask


if __name__ == "__main__":
    main()
