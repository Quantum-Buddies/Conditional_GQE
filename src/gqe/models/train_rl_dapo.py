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

try:
    import cudaq
except ImportError:
    cudaq = None

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.gqe.models.h_cgqe_transformer import (
    HcGQEModel,
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
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]]:
    """Sample n_sequences from the model and track per-token log probabilities.

    Returns:
        sequences: (n_samples, seq_len) token IDs
        log_probs: (n_samples, seq_len) per-token log probabilities
        operator_lists: list of n_samples lists of Pauli word strings
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

            if temperature != 1.0:
                logits = logits / temperature

            # Force entanglement: mask Z-only tokens until an entangler is generated
            if force_entanglement and z_only_mask is not None:
                constrain = ~has_entangler
                if constrain.any():
                    logits[constrain, z_only_mask] = float("-inf")

            # Length compatibility mask
            if length_mask is not None:
                logits[:, ~length_mask] = float("-inf")

            # Sample
            probs = torch.softmax(logits, dim=-1)
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
                # Pad remaining log_probs with zeros
                remaining = max_seq_len - 1 - step - 1
                if remaining > 0:
                    log_probs_list.append(torch.zeros(n_samples, device=device))
                break

    # Pad log_probs to match sequence length
    seq_len = sequences.size(1)
    if len(log_probs_list) < seq_len - 1:
        remaining = seq_len - 1 - len(log_probs_list)
        log_probs_list.extend([torch.zeros(n_samples, device=device)] * remaining)

    log_probs = torch.stack(log_probs_list, dim=1)  # (n_samples, seq_len-1)

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
    return sequences, log_probs, operator_lists


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


def evaluate_energies_batch(
    operators_batch: list[list[str]],
    molecule_record: dict[str, Any],
    theta: float = 0.01,
) -> list[float]:
    """Evaluate energies for a batch of operator sequences using CUDA-Q."""
    if cudaq is None:
        return [0.0] * len(operators_batch)

    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    spin_ham = hamiltonian_to_spin_operator(molecule_record)

    @cudaq.kernel
    def kernel(n_q: int, n_el: int, pauli_words: list[cudaq.pauli_word], thetas: list[float]):
        q = cudaq.qvector(n_q)
        for i in range(n_el):
            x(q[i])
        for i in range(len(pauli_words)):
            exp_pauli(thetas[i], q, pauli_words[i])

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

    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    spin_ham = hamiltonian_to_spin_operator(molecule_record)

    @cudaq.kernel
    def kernel(n_q: int, n_el: int, pauli_words: list[cudaq.pauli_word], thetas: list[float]):
        q = cudaq.qvector(n_q)
        for i in range(n_el):
            x(q[i])
        for i in range(len(pauli_words)):
            exp_pauli(thetas[i], q, pauli_words[i])

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
) -> float:
    """Multi-component reward for a generated circuit.

    R = w1 * (-E / |E_ref|)             # normalized energy (lower is better)
      + w2 * entanglement_fraction       # fraction of operators with X/Y
      + w3 * (-n_gates / max_seq_len)    # circuit depth penalty
      + w4 * non_commuting_fraction      # fraction of non-commuting pairs
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

    # Depth penalty
    depth_penalty = -len(operators) / max_seq_len

    # Non-commuting fraction (sampled pairs for efficiency)
    n_ops = len(operators)
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

    reward = (w_energy * energy_reward
              + w_entangle * entangle_frac
              + w_depth * depth_penalty
              + w_commute * non_commute_frac)
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
) -> torch.Tensor:
    """DAPO clipped surrogate loss.

    Key differences from GRPO:
    1. Clip-Higher: asymmetric clipping (clip_low < clip_high) prevents entropy collapse
    2. Token-Level Loss: averages over all tokens, not per-sequence
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

    if token_level:
        # Token-level: average over all tokens across all sequences
        loss = pg_losses.sum() / attention_mask.sum().clamp_min(1.0)
    else:
        # Sequence-level: average per-sequence, then average across sequences
        seq_losses = pg_losses.sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)
        loss = seq_losses.mean()

    return loss


def compute_advantages(
    rewards: np.ndarray,  # (G,) rewards for a group of samples
    use_grpo: bool = True,
) -> torch.Tensor:
    """Compute group-relative advantages (GRPO style).

    A_i = (R_i - mean(R)) / (std(R) + eps)
    """
    if use_grpo:
        mean_r = rewards.mean()
        std_r = rewards.std()
        advantages = (rewards - mean_r) / (std_r + 1e-8)
    else:
        # Simple baseline subtraction
        advantages = rewards - rewards.mean()
    return torch.tensor(advantages, dtype=torch.float32)


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
    parser.add_argument("--checkpoint", type=Path, required=True, help="Supervised pretrained checkpoint")
    parser.add_argument("--hamiltonians", type=Path, default=Path("results/data/hamiltonians.json"))
    parser.add_argument("--molecules", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output checkpoint path")
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=50, help="Circuits sampled per molecule per epoch")
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
    # Reward weights
    parser.add_argument("--w-energy", type=float, default=1.0)
    parser.add_argument("--w-entangle", type=float, default=0.1)
    parser.add_argument("--w-depth", type=float, default=0.01)
    parser.add_argument("--w-commute", type=float, default=0.05)
    # Replay buffer
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--buffer-batch-size", type=int, default=0,
                        help="Batch size from replay buffer (0 = no replay training)")
    # CUDA-Q
    parser.add_argument("--target", type=str, default="nvidia")
    parser.add_argument("--target-option", type=str, default="mqpu")
    parser.add_argument("--theta", type=float, default=0.01, help="Fixed rotation angle for energy eval")
    parser.add_argument("--max-qubits", type=int, default=24, help="Skip molecules with more qubits")
    # Device
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--force-entanglement", action="store_true", default=True)
    parser.add_argument("--max-repeat", type=int, default=4)
    args = parser.parse_args()

    _seed_everything(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained checkpoint
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
    scaler = torch.amp.GradScaler('cuda') if args.use_fp16 else None

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

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)

    # Training loop
    best_energy_per_mol = {m["name"]: float("inf") for m in molecules_data}
    train_metrics_log = []

    pbar = tqdm(range(args.epochs), desc="RL Epoch", unit="epoch")
    for epoch in pbar:
        epoch_energies = []
        epoch_rewards = []
        epoch_losses = []
        epoch_skipped = 0
        epoch_sequences_generated = 0

        for mol_data in molecules_data:
            mol_name = mol_data["name"]
            n_qubits = mol_data["n_qubits"]

            # --- Phase 1: Sample sequences ---
            attempts = 0
            valid_batch = False

            while attempts < args.max_resample_attempts:
                attempts += 1
                sequences, old_log_probs, operator_lists = sample_sequences_with_logprobs(
                    model,
                    mol_data["pauli_ids"].unsqueeze(0),
                    mol_data["coeffs"].unsqueeze(0),
                    mol_data["term_mask"].unsqueeze(0),
                    n_samples=args.n_samples,
                    max_seq_len=args.max_seq_len,
                    temperature=args.temperature,
                    vocab=vocab,
                    inv_vocab=inv_vocab,
                    n_qubits=n_qubits,
                    force_entanglement=args.force_entanglement,
                    max_repeat=args.max_repeat,
                    device=device,
                    is_data_parallel=is_dp,
                )

                # Filter out empty sequences
                valid_indices = [i for i, ops in enumerate(operator_lists) if len(ops) > 0]
                if not valid_indices:
                    print(f"  {mol_name}: all sequences empty, resampling...")
                    continue

                sequences = sequences[valid_indices]
                old_log_probs = old_log_probs[valid_indices]
                operator_lists = [operator_lists[i] for i in valid_indices]

                # --- Phase 2: Evaluate energies ---
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
                rewards = np.array([
                    compute_reward(
                        e, ops, mol_data["hf_energy"], mol_data["fci_energy"],
                        args.max_seq_len,
                        args.w_energy, args.w_entangle, args.w_depth, args.w_commute,
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

            # --- Phase 5: Compute advantages ---
            advantages = compute_advantages(rewards)

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

            # --- Phase 6: Compute DAPO loss and update ---
            model.train()
            optimizer.zero_grad()

            # Recompute log_probs with current model (gradient tracking)
            bos_id = SPECIAL_TOKENS["<BOS>"]
            pad_id = SPECIAL_TOKENS["<PAD>"]
            tgt_input = sequences[:, :-1].to(device)
            tgt_labels = sequences[:, 1:].to(device)
            attention_mask = (tgt_labels != pad_id).float().to(device)

            # Expand Hamiltonian input for the batch
            pauli_ids_batch = mol_data["pauli_ids"].unsqueeze(0).expand(
                sequences.size(0), -1, -1
            ).to(device)
            coeffs_batch = mol_data["coeffs"].unsqueeze(0).expand(
                sequences.size(0), -1
            ).to(device)
            term_mask_batch = mol_data["term_mask"].unsqueeze(0).expand(
                sequences.size(0), -1
            ).to(device)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(
                        pauli_ids_batch, coeffs_batch, tgt_input,
                        term_mask=term_mask_batch,
                        tgt_key_padding_mask=(tgt_input == pad_id),
                    )
                    log_probs_new = _compute_sequence_log_probs(
                        logits, tgt_labels, attention_mask,
                    )
                    loss = dapo_loss(
                        log_probs_new, old_log_probs.to(device),
                        advantages.to(device), attention_mask,
                        clip_low=args.clip_low, clip_high=args.clip_high,
                        token_level=args.token_level_loss,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
                    log_probs_new, old_log_probs.to(device),
                    advantages.to(device), attention_mask,
                    clip_low=args.clip_low, clip_high=args.clip_high,
                    token_level=args.token_level_loss,
                )
                loss.backward()
                optimizer.step()

            epoch_losses.append(loss.item())

        # --- Replay buffer training (optional) ---
        if args.buffer_batch_size > 0 and len(replay_buffer) >= args.buffer_batch_size:
            replay_samples = replay_buffer.sample(args.buffer_batch_size)
            # TODO: could add replay training here
            pass

        # Logging
        mean_energy = np.mean(epoch_energies) if epoch_energies else 0.0
        min_energy = np.min(epoch_energies) if epoch_energies else 0.0
        mean_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        pbar.set_postfix_str(
            f"loss={mean_loss:.4f} "
            f"E_mean={mean_energy:.4f} "
            f"E_min={min_energy:.4f} "
            f"R={mean_reward:.4f} "
            f"skip={epoch_skipped} "
            f"buf={len(replay_buffer)}"
        )

        train_metrics_log.append({
            "epoch": epoch,
            "mean_energy": mean_energy,
            "min_energy": min_energy,
            "mean_reward": mean_reward,
            "mean_loss": mean_loss,
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
