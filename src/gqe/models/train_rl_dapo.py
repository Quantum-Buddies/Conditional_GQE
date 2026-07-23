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
import os
import random
import sys
import time
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
from src.gqe.rl.map_elites import (
    MAPElitesArchive, DedupCache, PerMoleculeArchives, compute_circuit_features,
)
from src.gqe.rl.energy_cache import PersistentEnergyCache, resolve_energies_with_cache


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
    amp_dtype: torch.dtype | None = None,
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

    use_amp = amp_dtype is not None and device.type == "cuda"
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else torch.autocast(device_type="cuda", enabled=False)
    )

    # Get encoder memory once
    with torch.no_grad(), amp_ctx:
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

    with torch.no_grad(), amp_ctx:
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
            # Sampling math in FP32 for stable entropy / categorical
            logits = logits.float()

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
_spin_ham_cache: dict[str, Any] = {}
_current_cudaq_target: tuple[str, str] | None = None


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


def _get_cached_spin_ham(molecule_record: dict[str, Any], cache_key: str | None = None) -> Any:
    """Build SpinOperator once per molecule; rebuilds dominate large-Hamiltonian wall time."""
    key = cache_key or str(molecule_record.get("name", id(molecule_record)))
    spin_ham = _spin_ham_cache.get(key)
    if spin_ham is None:
        spin_ham = hamiltonian_to_spin_operator(molecule_record)
        _spin_ham_cache[key] = spin_ham
    return spin_ham


def _set_cudaq_target_cached(target: str, option: str = "") -> None:
    """Avoid redundant cudaq.set_target calls (they stall and flush GPU work)."""
    global _current_cudaq_target
    if cudaq is None:
        return
    # Blackwell: prefer explicit fp32 so CUDAQ_ALLOW_FP32_EMULATED BF16x9 path is used
    if target == "nvidia" and (not option or option == ""):
        option = "fp32"
    key = (target, option or "")
    if _current_cudaq_target == key:
        return
    if option:
        cudaq.set_target(target, option=option)
    else:
        cudaq.set_target(target)
    _current_cudaq_target = key


def _enable_blackwell_torch_optimizations(device: torch.device) -> None:
    """Enable B200 / Blackwell PyTorch + cuBLAS paths (BF16 training + BF16x9 FP32 GEMMs)."""
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    try:
        # Prefer TF32 for any residual FP32 matmuls; cuBLAS BF16x9 is env-driven.
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    except Exception:
        pass
    # SDPA backends — Flash / mem-efficient on Blackwell
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    cap = torch.cuda.get_device_capability(0)
    is_blackwell = cap[0] >= 10
    print(
        f"  Blackwell torch opts: TF32=on fp32_precision="
        f"{getattr(torch.backends.cuda.matmul, 'fp32_precision', '?')} "
        f"sm_{cap[0]}{cap[1]} blackwell={is_blackwell}",
        flush=True,
    )
    print(
        f"  cuBLAS BF16x9 env: CUBLAS_EMULATE_SINGLE_PRECISION="
        f"{os.environ.get('CUBLAS_EMULATE_SINGLE_PRECISION', 'unset')} "
        f"STRATEGY={os.environ.get('CUBLAS_EMULATION_STRATEGY', 'unset')}",
        flush=True,
    )
    print(
        f"  CUDA-Q env: ALLOW_FP32_EMULATED={os.environ.get('CUDAQ_ALLOW_FP32_EMULATED', 'unset')} "
        f"FUSION_MAX_QUBITS={os.environ.get('CUDAQ_FUSION_MAX_QUBITS', 'unset')} "
        f"MEMPOOL={os.environ.get('CUDAQ_ENABLE_MEMPOOL', 'unset')}",
        flush=True,
    )


def _warmup_cudaq_observe(n_qubits: int = 4, n_electrons: int = 2) -> None:
    """Force one sync observe so JIT/PTX compile happens before the training loop."""
    if cudaq is None:
        return
    kernel = _get_gqe_kernel()
    # Minimal 4q Z-only Hamiltonian so compile path is exercised without big allocs.
    dummy = {
        "name": "__warmup__",
        "n_qubits": n_qubits,
        "terms": [{"term": "Z0", "real": 1.0, "imag": 0.0}],
    }
    try:
        spin_ham = hamiltonian_to_spin_operator(dummy)
        words = [cudaq.pauli_word("X" + "I" * (n_qubits - 1))]
        thetas = [0.01]
        cudaq.observe(kernel, spin_ham, n_qubits, n_electrons, words, thetas)
        print("  CUDA-Q observe warmup complete (JIT primed)", flush=True)
    except Exception as e:
        print(f"  CUDA-Q warmup skipped: {e}", flush=True)


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
    spin_ham = _get_cached_spin_ham(molecule_record)

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
    execution=None,
    eval_async: bool = True,
    spin_ham: Any | None = None,
    async_chunk: int = 32,
    show_progress: bool = False,
    mol_name: str = "",
) -> list[float]:
    """Evaluate energies for a batch of operator sequences using CUDA-Q.

    Args:
        execution: Optional cudaq.parallel execution mode (e.g. cudaq.parallel.thread)
                   to distribute Hamiltonian terms across multiple GPUs.
        eval_async: When True and execution is None, pipeline circuits via
                    cudaq.observe_async(qpu_id=0) instead of blocking observe().
        spin_ham: Precomputed SpinOperator (avoids rebuild per call).
        async_chunk: Max in-flight observe_async handles. Submitting hundreds at
                     once spawns huge thread pools and stalls with 0% GPU util.
        show_progress: Print chunk progress (helps diagnose stalls).
        mol_name: Label for progress lines.
    """
    if cudaq is None:
        return [0.0] * len(operators_batch)

    kernel = _get_gqe_kernel()
    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    if spin_ham is None:
        spin_ham = _get_cached_spin_ham(molecule_record)

    def _observe_sync(operators: list[str]) -> float:
        padded = [_pad_pauli_word(w, n_qubits) for w in operators]
        pauli_words = [cudaq.pauli_word(w) for w in padded]
        thetas = [theta] * len(pauli_words)
        try:
            if execution is not None:
                result = cudaq.observe(
                    kernel, spin_ham, n_qubits, n_electrons,
                    pauli_words, thetas, execution=execution,
                )
            else:
                result = cudaq.observe(
                    kernel, spin_ham, n_qubits, n_electrons,
                    pauli_words, thetas,
                )
            return float(result.expectation())
        except Exception as e:
            # Single-GPU nvidia rejects parallel.thread — fall back to plain observe.
            if execution is not None and "parallel" in str(e).lower():
                result = cudaq.observe(
                    kernel, spin_ham, n_qubits, n_electrons,
                    pauli_words, thetas,
                )
                return float(result.expectation())
            raise

    # Chunked observe_async: pipeline without flooding the runtime.
    if execution is None and eval_async:
        energies: list[float] = [0.0] * len(operators_batch)
        nonempty = [i for i, ops in enumerate(operators_batch) if ops]
        chunk = max(1, int(async_chunk))
        label = mol_name or molecule_record.get("name", "mol")
        try:
            for start in range(0, len(nonempty), chunk):
                batch_ids = nonempty[start:start + chunk]
                futures: list[tuple[int, Any]] = []
                for i in batch_ids:
                    operators = operators_batch[i]
                    padded = [_pad_pauli_word(w, n_qubits) for w in operators]
                    pauli_words = [cudaq.pauli_word(w) for w in padded]
                    thetas = [theta] * len(pauli_words)
                    handle = cudaq.observe_async(
                        kernel, spin_ham, n_qubits, n_electrons,
                        pauli_words, thetas, qpu_id=0,
                    )
                    futures.append((i, handle))

                for i, handle in futures:
                    try:
                        energies[i] = float(handle.get().expectation())
                    except Exception as e:
                        print(f"  CUDA-Q async result error: {e}", flush=True)
                        energies[i] = 0.0

                if show_progress:
                    done = min(start + chunk, len(nonempty))
                    print(
                        f"    [{label}] energy eval {done}/{len(nonempty)} "
                        f"(chunk={chunk})",
                        flush=True,
                    )
            return energies
        except Exception as e:
            print(f"  observe_async failed, falling back to sync observe: {e}", flush=True)

    energies = []
    for idx, operators in enumerate(operators_batch):
        if not operators:
            energies.append(0.0)
            continue
        try:
            energies.append(_observe_sync(operators))
        except Exception as e:
            print(f"  CUDA-Q error: {e}", flush=True)
            energies.append(0.0)
        if show_progress and (idx + 1) % max(1, async_chunk) == 0:
            print(
                f"    [{mol_name or 'mol'}] sync energy eval "
                f"{idx + 1}/{len(operators_batch)}",
                flush=True,
            )
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
    spin_ham = _get_cached_spin_ham(molecule_record)

    # Chunked multi-GPU async to avoid thread-pool floods
    energies = [0.0] * len(operators_batch)
    nonempty = [i for i, ops in enumerate(operators_batch) if ops]
    chunk = max(1, min(32, max(n_gpus * 4, 8)))
    for start in range(0, len(nonempty), chunk):
        batch_ids = nonempty[start:start + chunk]
        futures: list[tuple[int, Any]] = []
        for j, i in enumerate(batch_ids):
            operators = operators_batch[i]
            padded = [_pad_pauli_word(w, n_qubits) for w in operators]
            pauli_words = [cudaq.pauli_word(w) for w in padded]
            thetas = [theta] * len(pauli_words)
            try:
                qpu_id = j % max(n_gpus, 1)
                handle = cudaq.observe_async(
                    kernel, spin_ham, n_qubits, n_electrons,
                    pauli_words, thetas, qpu_id=qpu_id,
                )
                futures.append((i, handle))
            except Exception as e:
                print(f"  CUDA-Q async error: {e}", flush=True)
        for i, handle in futures:
            try:
                energies[i] = float(handle.get().expectation())
            except Exception as e:
                print(f"  CUDA-Q result error: {e}", flush=True)
    return energies


# ---------------------------------------------------------------------------
# QD-GRPO: Truncated L-BFGS-B energy evaluation with dedup cache
# ---------------------------------------------------------------------------

def evaluate_energies_qd(
    operators_batch: list[list[str]],
    molecule_record: dict[str, Any],
    dedup_cache: DedupCache,
    initial_theta: float = 0.01,
    max_iters: int = 5,
) -> tuple[list[float], dict[str, int]]:
    """Evaluate energies using truncated L-BFGS-B with global dedup cache.

    This replaces the fixed-θ proxy with a fast (5-iteration) L-BFGS-B
    optimization per circuit, cached globally so identical circuits are
    never re-evaluated. The truncated optimization gives a much better
    ranking signal than fixed-θ (Spearman ρ ~0.5 vs ~0.2).

    Args:
        operators_batch: list of operator sequences
        molecule_record: molecule Hamiltonian record
        dedup_cache: global DedupCache for circuit→energy mapping
        initial_theta: starting angle for L-BFGS-B
        max_iters: max L-BFGS-B iterations (5 = fast surrogate)

    Returns:
        (energies, stats) where stats has cache hit/miss counts
    """
    if cudaq is None:
        return [0.0] * len(operators_batch), {"hits": 0, "misses": 0}

    kernel = _get_gqe_kernel()
    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    spin_ham = _get_cached_spin_ham(molecule_record)

    energies = []
    n_hits = 0
    n_misses = 0

    for operators in operators_batch:
        if not operators:
            energies.append(0.0)
            continue

        # Check dedup cache first
        def compute_fn(ops):
            padded = [_pad_pauli_word(w, n_qubits) for w in ops]
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
                return float(opt_result.fun)
            except Exception:
                return energy_fn(x0)

        energy, was_cached = dedup_cache.get_or_compute(operators, compute_fn)
        if was_cached:
            n_hits += 1
        else:
            n_misses += 1
        energies.append(energy)

    return energies, {"hits": n_hits, "misses": n_misses}


# ---------------------------------------------------------------------------
# Multi-component reward function
# ---------------------------------------------------------------------------

def _has_energy_improvement(
    energy: float,
    hf_energy: float | None,
    threshold: float,
) -> bool:
    return hf_energy is not None and energy < hf_energy - threshold


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
    gate_auxiliary_rewards: bool = True,
    energy_improvement_threshold: float = 0.0,
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

    auxiliary_scale = 1.0
    if gate_auxiliary_rewards and not _has_energy_improvement(
        energy, hf_energy, energy_improvement_threshold
    ):
        auxiliary_scale = 0.0

    reward = (w_energy * energy_reward
              + auxiliary_scale * (w_entangle * entangle_frac
                                   + w_depth * length_reward
                                   + w_commute * non_commute_frac
                                   + w_diversity * unique_frac))
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
# Chemeleon2-inspired batch-level rewards (Park & Walsh, Nat. Mach. Intell. 2026)
# ---------------------------------------------------------------------------

def _pauli_histogram_embedding(
    operators: list[str],
    vocab: dict[str, int],
) -> np.ndarray:
    """Embed a circuit as a normalized histogram over the operator vocabulary.

    This gives a fixed-length vector representation suitable for kernel-based
    distributional alignment (MMD) and diversity computation.
    """
    vec = np.zeros(len(vocab), dtype=np.float32)
    for op in operators:
        idx = vocab.get(op, -1)
        if idx >= 0:
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 2,
                       constant: float = 1.0) -> np.ndarray:
    """Polynomial kernel K(x,y) = (x^T y + c)^d."""
    return (x @ y.T + constant) ** degree


def compute_batch_diversity_mmd(
    operator_lists: list[list[str]],
    vocab: dict[str, int],
    ref_operator_lists: list[list[str]] | None = None,
    kernel_degree: int = 2,
) -> np.ndarray:
    """Leave-one-out MMD diversity reward for each circuit in the batch.

    Following Chemeleon2 (Park & Walsh 2026), this computes the negative
    unbiased MMD between the generated batch and a reference set, then
    attributes credit to each sample via leave-one-out:

        r̂(z_m) = MMD(X) - MMD(X \\ {z_m})

    Samples that increase diversity get positive credit; redundant samples
    get negative credit. This directly prevents mode collapse (their Fig. 2c
    ablation shows severe collapse without this reward).

    Args:
        operator_lists: G circuits in the rollout group
        vocab: operator vocabulary for histogram embedding
        ref_operator_lists: reference set (e.g. from replay buffer). If None,
            only within-batch diversity is computed.
        kernel_degree: polynomial kernel degree

    Returns:
        (G,) array of per-circuit diversity rewards, normalized to [0, 1]
    """
    G = len(operator_lists)
    if G == 0:
        return np.array([])

    # Embed all circuits
    X_gen = np.stack([_pauli_histogram_embedding(ops, vocab) for ops in operator_lists])

    # Reference embeddings (default: use the batch itself)
    if ref_operator_lists and len(ref_operator_lists) > 0:
        X_ref = np.stack([_pauli_histogram_embedding(ops, vocab) for ops in ref_operator_lists])
    else:
        X_ref = X_gen

    N = X_ref.shape[0]

    # Compute full MMD between batch and reference
    K_gen_gen = _polynomial_kernel(X_gen, X_gen, kernel_degree)
    K_ref_ref = _polynomial_kernel(X_ref, X_ref, kernel_degree)
    K_gen_ref = _polynomial_kernel(X_gen, X_ref, kernel_degree)

    # Unbiased MMD estimator: MMD^2 = 1/(G(G-1)) * sum_{i!=j} K(z_i,z_j)
    #                             - 2/(G*N) * sum_{i,j} K(z_i, x_j)
    #                             + 1/(N(N-1)) * sum_{i!=j} K(x_i, x_j)
    # We use negative MMD as the reward (higher = more diverse)
    if G > 1:
        gen_term = (K_gen_gen.sum() - np.trace(K_gen_gen)) / (G * (G - 1))
    else:
        gen_term = 0.0
    cross_term = K_gen_ref.mean()
    if N > 1:
        ref_term = (K_ref_ref.sum() - np.trace(K_ref_ref)) / (N * (N - 1))
    else:
        ref_term = 0.0
    mmd_full = gen_term - 2 * cross_term + ref_term

    # Leave-one-out: for each sample m, compute MMD without it
    # Chemeleon2 uses negative MMD as the diversity reward.
    # LOO marginal credit: r(z_m) = (-MMD_full) - (-MMD_loo) = MMD_loo - MMD_full
    # Samples that increase diversity (removing them increases MMD) get negative
    # credit, which is correct — they make the batch LESS diverse.
    # Samples that decrease diversity (removing them decreases MMD) get positive
    # credit — they make the batch MORE diverse.
    loo_rewards = np.zeros(G, dtype=np.float32)
    for m in range(G):
        mask = np.ones(G, dtype=bool)
        mask[m] = False
        X_loo = X_gen[mask]
        G_loo = G - 1
        if G_loo > 1:
            K_loo_loo = _polynomial_kernel(X_loo, X_loo, kernel_degree)
            gen_loo = (K_loo_loo.sum() - np.trace(K_loo_loo)) / (G_loo * (G_loo - 1))
        else:
            gen_loo = 0.0
        K_loo_ref = _polynomial_kernel(X_loo, X_ref, kernel_degree)
        cross_loo = K_loo_ref.mean()
        mmd_loo = gen_loo - 2 * cross_loo + ref_term
        # Marginal contribution to -MMD (Chemeleon2's diversity reward)
        loo_rewards[m] = mmd_loo - mmd_full

    # Normalize to [0, 1] via min-max scaling
    r_min, r_max = loo_rewards.min(), loo_rewards.max()
    if r_max - r_min > 1e-10:
        loo_rewards = (loo_rewards - r_min) / (r_max - r_min)
    else:
        loo_rewards = np.ones(G, dtype=np.float32) * 0.5

    return loo_rewards


def _normalized_edit_distance(ops1: list[str], ops2: list[str]) -> float:
    """Normalized edit distance between two operator sequences (0=same, 1=disjoint)."""
    if not ops1 and not ops2:
        return 0.0
    # Use set-based Jaccard distance as a fast proxy for edit distance
    s1, s2 = set(ops1), set(ops2)
    union = s1 | s2
    if not union:
        return 0.0
    return 1.0 - len(s1 & s2) / len(union)


def compute_creativity_batch(
    operator_lists: list[list[str]],
    seen_operators: set[tuple[str, ...]] | None = None,
) -> np.ndarray:
    """Creativity reward: uniqueness (in-batch) + novelty (vs reference set).

    Following Chemeleon2's continuous formulation:
    - 1.0 if circuit is both unique (in batch) and novel (vs reference)
    - 0.0 if neither unique nor novel
    - Otherwise: continuous Jaccard distance to nearest match (smooth gradient)

    Args:
        operator_lists: G circuits in the rollout group
        seen_operators: set of frozen operator tuples from replay buffer / training data

    Returns:
        (G,) array of creativity rewards in [0, 1]
    """
    G = len(operator_lists)
    rewards = np.zeros(G, dtype=np.float32)

    # Convert to tuples for hashing
    op_tuples = [tuple(ops) for ops in operator_lists]

    for i in range(G):
        ops_i = operator_lists[i]

        # Uniqueness: is this circuit duplicated in the batch?
        is_unique = sum(1 for j in range(G) if op_tuples[j] == op_tuples[i]) == 1

        # Novelty: has this circuit been seen before?
        is_novel = seen_operators is None or op_tuples[i] not in seen_operators

        if is_unique and is_novel:
            rewards[i] = 1.0
        elif not is_unique and not is_novel:
            rewards[i] = 0.0
        else:
            # Borderline: continuous distance to nearest match
            min_dist = 1.0
            # Check against batch duplicates
            for j in range(G):
                if j != i and op_tuples[j] == op_tuples[i]:
                    dist = 0.0
                else:
                    dist = _normalized_edit_distance(ops_i, operator_lists[j])
                min_dist = min(min_dist, dist)
            # Check against seen set (sample a subset for efficiency)
            if seen_operators:
                seen_list = list(seen_operators)
                for seen_ops in seen_list[-200:]:  # last 200 for efficiency
                    dist = _normalized_edit_distance(ops_i, list(seen_ops))
                    min_dist = min(min_dist, dist)
            # Chemeleon2 creativity: reward = min_dist (higher distance = more creative)
            rewards[i] = min_dist

    return rewards


def compute_msun_metric(
    energies: list[float],
    operator_lists: list[list[str]],
    hf_energy: float | None,
    convergence_threshold: float = 0.1,
) -> dict[str, Any]:
    """mSUN-style metric for circuits: Metastable, Unique, Novel fraction.

    Mirrors Chemeleon2's evaluation: a circuit is 'mSUN' if it:
    1. Converges below HF energy + threshold (metastable analog)
    2. Is unique within the batch
    3. Is novel (not a trivial repeated single-operator circuit)

    Args:
        energies: list of circuit energies
        operator_lists: corresponding operator sequences
        hf_energy: Hartree-Fock reference energy
        convergence_threshold: energy threshold above HF (in Hartree)

    Returns:
        Dict with mSUN fraction and component metrics
    """
    G = len(energies)
    if G == 0:
        return {"msun": 0.0, "converged": 0.0, "unique": 0.0, "novel": 0.0}

    # Converged: energy below HF + threshold
    if hf_energy is not None:
        converged = [e < hf_energy + convergence_threshold for e in energies]
    else:
        converged = [True] * G  # no reference, assume converged

    # Unique: non-duplicated in batch
    op_tuples = [tuple(ops) for ops in operator_lists]
    unique = [sum(1 for t in op_tuples if t == op_tuples[i]) == 1 for i in range(G)]

    # Novel: not a trivial circuit (more than 1 unique operator, has entangling ops)
    novel = []
    for ops in operator_lists:
        has_entangler = any("X" in w or "Y" in w for w in ops)
        has_diversity = len(set(ops)) > 1
        novel.append(has_entangler and has_diversity)

    msun = sum(c and u and n for c, u, n in zip(converged, unique, novel))

    return {
        "msun": msun / G,
        "converged": sum(converged) / G,
        "unique": sum(unique) / G,
        "novel": sum(novel) / G,
        "n_total": G,
        "n_msun": msun,
    }


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
    ref_log_probs: torch.Tensor | None = None,  # (G, seq_len) reference policy log-probs for KL
    kl_coef: float = 0.0,            # β: KL penalty weight (Chemeleon2 uses 1.0)
) -> torch.Tensor:
    """DAPO clipped surrogate loss with KL regularization and entropy bonus.

    Following Chemeleon2 (Park & Walsh 2026), the total loss is:

        L = L_clipped + β·KL[π_θ || π_ref] - γ·H(π_θ)

    Key components:
    1. Clip-Higher: asymmetric clipping (clip_low < clip_high) prevents entropy collapse
    2. Token-Level Loss: averages over all tokens, not per-sequence
    3. KL Penalty: anchors to pretrained reference policy, preserving valid-circuit grammar
    4. Entropy Bonus: encourages diverse sampling, prevents premature convergence

    The KL uses the k3 estimator (http://joschu.net/blog/kl-approx.html):
        KL ≈ exp(Δ) - 1 - Δ, where Δ = log π_ref - log π_θ
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

    # KL divergence to reference policy (k3 estimator)
    kl_loss = torch.tensor(0.0, device=log_probs_new.device)
    if kl_coef > 0.0 and ref_log_probs is not None:
        ref_log_probs = ref_log_probs.to(log_probs_new.device)
        log_ratio = ref_log_probs - log_probs_new  # Δ = log π_ref - log π_θ
        kl_per_token = torch.exp(log_ratio) - 1.0 - log_ratio  # k3 estimator
        kl_per_token = kl_per_token * attention_mask
        kl_loss = kl_coef * kl_per_token.sum() / attention_mask.sum().clamp_min(1.0)

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

    return pg_loss + kl_loss + entropy_loss


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
    libcudart = ctypes.CDLL(os.environ.get("CUDAQ_CUDART", "libcudart.so"))
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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=64, help="Circuits sampled per molecule per epoch")
    parser.add_argument("--n-iters", type=int, default=5,
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
    parser.add_argument("--dynamic-sampling", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip groups where all energies are identical (std=0). Use --no-dynamic-sampling to disable.")
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
    parser.add_argument("--gate-auxiliary-rewards", action=argparse.BooleanOptionalAction, default=True,
                        help="Grant non-energy rewards only after beating Hartree-Fock.")
    parser.add_argument("--energy-improvement-threshold", type=float, default=0.0,
                        help="Required energy improvement below Hartree-Fock in Hartree.")
    parser.add_argument("--freq-penalty", type=float, default=1.0,
                        help="Frequency penalty for sampling: subtracts freq_penalty * count[token] "
                             "from logits. Prevents mode collapse to repeated operators.")
    # Replay buffer
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--buffer-batch-size", type=int, default=64,
                        help="Batch size from replay buffer for off-policy training (0 = disabled)")
    # Off-policy sample reuse (arXiv:2505.22257)
    parser.add_argument("--reuse-iters", type=int, default=3,
                        help="Number of gradient steps per rollout batch (off-policy GRPO). "
                             "Reuses each sample μ times with importance sampling correction. "
                             "1=standard (no reuse), 2-4=recommended for reducing simulation cost.")
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
    parser.add_argument("--adaptive-theta", action=argparse.BooleanOptionalAction, default=True,
                        help="Run quick L-BFGS-B optimization of theta for the best circuit in each "
                             "batch. Uses optimized energy for reward instead of fixed theta. "
                             "More expensive but gives much better energy signal.")
    parser.add_argument("--adaptive-theta-iters", type=int, default=10,
                        help="Max L-BFGS-B iterations for adaptive theta optimization.")
    # CUDA-Q
    parser.add_argument("--target", type=str, default="nvidia",
                        help="CUDA-Q target: nvidia (statevector), tensornet-mps (MPS for 40+ qubits)")
    parser.add_argument("--target-option", type=str, default="mqpu",
                        help="Target option: mqpu (multi-GPU pooling), mps (MPS backend)")
    parser.add_argument("--single-gpu", action="store_true",
                        help="Force single-GPU evaluation (avoids L40S PCIe IPC segfault with mqpu)")
    parser.add_argument("--theta", type=float, default=0.01, help="Fixed rotation angle for energy eval")
    parser.add_argument("--eval-async", action=argparse.BooleanOptionalAction, default=True,
                        help="Pipeline circuit energy evaluations with cudaq.observe_async (qpu_id=0). "
                             "Falls back to sync observe on failure. Disable with --no-eval-async.")
    parser.add_argument("--eval-async-chunk", type=int, default=32,
                        help="Max in-flight observe_async jobs. Larger floods CUDA-Q threads "
                             "and stalls (0%% GPU). 16–48 is typical on B200.")
    parser.add_argument("--energy-cache", type=Path, default=None,
                        help="SQLite path for persistent circuit→energy cache. "
                             "Hits skip CUDA-Q observe. Precompute with "
                             "src/gqe/data/precompute_rl_energy_cache.py")
    parser.add_argument("--cache-only", action="store_true", default=False,
                        help="Never call CUDA-Q on cache miss (offline RL). "
                             "Misses get energy 0.0; prefer precomputing first.")
    parser.add_argument("--max-qubits", type=int, default=30,
                        help="Skip molecules with more qubits. Default 30 (H200 single-GPU). "
                             "Use 24 for L40S (cuStateVec distributed threshold=25).")
    parser.add_argument("--mps-threshold", type=int, default=24,
                        help="Auto-switch to tensornet-mps for molecules above this qubit count")
    parser.add_argument("--mps-bond", type=int, default=64,
                        help="MPS max bond dimension (higher=more accurate, more memory)")
    # Device
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--use-fp16", action="store_true", help="Deprecated: use --use-bf16 instead")
    parser.add_argument("--use-bf16", action="store_true", default=True,
                        help="Use bfloat16 mixed precision (prevents FP16 multiplicative bias entropy collapse)")
    parser.add_argument("--use-nvfp4", action="store_true", default=False,
                        help="Deprecated on this stack (Transformer Engine ABI broken). "
                             "Prefer --use-bf16. Kept for optional TE installs only.")
    parser.add_argument("--nvfp4-bf16-tail", type=int, default=2,
                        help="Number of final transformer layers to keep in BF16 when using NVFP4 "
                             "(NVIDIA recipe: last 15%% of layers. Default 2 for 6-layer decoder.)")
    parser.add_argument("--force-entanglement", action="store_true", default=True)
    parser.add_argument("--max-repeat", type=int, default=4)
    # REPO-style advantage modification
    parser.add_argument("--repo-beta", type=float, default=0.05,
                        help="REPO advantage regularization coefficient (0=off, 0.05=mild entropy preservation)")
    # Chemeleon2-inspired rewards (Park & Walsh, Nat. Mach. Intell. 2026)
    parser.add_argument("--kl-coef", type=float, default=0.0,
                        help="KL divergence penalty weight to reference policy (β). "
                             "Chemeleon2 uses β=1.0 (strong anchoring to pretrained model). "
                             "0=disabled (backward compatible).")
    parser.add_argument("--w-creativity", type=float, default=0.0,
                        help="Weight for creativity reward (uniqueness + novelty with continuous "
                             "edit distance). Chemeleon2 uses w=1.0. 0=disabled.")
    parser.add_argument("--w-mmd-diversity", type=float, default=0.0,
                        help="Weight for leave-one-out MMD diversity reward. "
                             "Chemeleon2's critical anti-mode-collapse component (Fig. 2c ablation). "
                             "0=disabled, 0.5=recommended starting point.")
    parser.add_argument("--chemeleon2-mode", action="store_true", default=False,
                        help="Preset conservative Chemeleon2 hyperparameters: "
                             "kl_coef=1.0, w_creativity=1.0, w_mmd_diversity=1.0, "
                             "clip_low=0.001, clip_high=0.001, entropy_coef=1e-5, "
                             "repo_beta=0.0. Overrides individual args if set.")
    parser.add_argument("--msun-threshold", type=float, default=0.1,
                        help="Energy convergence threshold for mSUN metric (in Hartree above HF)")
    # Curriculum learning
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable curriculum learning: start with small molecules, gradually add larger ones")
    parser.add_argument("--curriculum-warmup", type=int, default=30,
                        help="Number of epochs to train only on smallest molecules before adding larger ones")
    parser.add_argument("--curriculum-steps", type=int, default=3,
                        help="Number of curriculum stages (molecules added in stages)")
    # QD-GRPO: Quality-Diversity RL with MAP-Elites archive
    parser.add_argument("--qd-mode", action="store_true", default=False,
                        help="Enable Quality-Diversity GRPO: MAP-Elites archive + novelty bonus + "
                             "truncated L-BFGS-B surrogate with dedup cache. "
                             "Replaces fixed-θ proxy with a physically meaningful reward signal.")
    parser.add_argument("--qd-novelty-weight", type=float, default=1.0,
                        help="Initial weight for novelty (intrinsic) reward. "
                             "Decays adaptively as archive fills (see --qd-lambda-final).")
    parser.add_argument("--qd-lambda-final", type=float, default=0.1,
                        help="Final novelty weight after archive reaches coverage threshold.")
    parser.add_argument("--qd-coverage-threshold", type=float, default=0.5,
                        help="Archive coverage fraction at which novelty weight reaches --qd-lambda-final.")
    parser.add_argument("--qd-n-bins-entanglement", type=int, default=10,
                        help="Number of bins for entanglement density axis in MAP-Elites grid.")
    parser.add_argument("--qd-n-bins-depth", type=int, default=10,
                        help="Number of bins for circuit depth axis in MAP-Elites grid.")
    parser.add_argument("--qd-lbfgs-iters", type=int, default=5,
                        help="Truncated L-BFGS-B iterations per circuit for surrogate energy. "
                             "5 iters is ~50x faster than full optimization while giving "
                             "Spearman ρ ~0.5 vs final energy (vs ρ ~0.2 for fixed-θ proxy).")
    parser.add_argument("--qd-archive-path", type=Path, default=None,
                        help="Path to save MAP-Elites archive JSON at end of training.")
    args = parser.parse_args()

    # Apply Chemeleon2 preset (Park & Walsh 2026, conservative regime)
    if args.chemeleon2_mode:
        print("\n=== CHEMELEON2 MODE ENABLED (Park & Walsh, Nat. Mach. Intell. 2026) ===")
        print("  Conservative regime: strong KL anchoring, MMD diversity, creativity reward")
        args.kl_coef = 1.0
        args.w_creativity = 1.0
        args.w_mmd_diversity = 1.0
        args.clip_low = 0.001
        args.clip_high = 0.001
        args.entropy_coef = 1e-5
        args.repo_beta = 0.0
        print(f"  kl_coef={args.kl_coef}, w_creativity={args.w_creativity}, "
              f"w_mmd_diversity={args.w_mmd_diversity}")
        print(f"  clip=[{args.clip_low}, {args.clip_high}], "
              f"entropy_coef={args.entropy_coef}, repo_beta={args.repo_beta}")

    _seed_everything(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        _enable_blackwell_torch_optimizations(device)

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
            use_transformer_engine=bool(args.use_nvfp4),
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
            use_transformer_engine=args.use_nvfp4,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {n_params:,} parameters")

    # Create frozen reference model for KL divergence (Chemeleon2 §Methods)
    ref_model = None
    if args.kl_coef > 0.0 and not args.from_scratch and args.checkpoint is not None:
        import copy
        ref_model = copy.deepcopy(model)
        ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print(f"Reference model created (frozen, for KL divergence, β={args.kl_coef})")
    elif args.kl_coef > 0.0 and (args.from_scratch or args.checkpoint is None):
        print("WARNING: --kl-coef > 0 but no checkpoint loaded. KL penalty disabled "
              "(reference model requires a pretrained checkpoint).")
        args.kl_coef = 0.0

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
    use_bf16 = (args.use_bf16 or args.use_fp16) and torch.cuda.is_available()  # bf16 supersedes fp16
    scaler = torch.amp.GradScaler('cuda') if (args.use_fp16 and not args.use_bf16) else None
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # NVFP4 mixed precision for Blackwell GPUs (B200/B300)
    te = None
    fp4_recipe = None
    use_nvfp4 = args.use_nvfp4 and torch.cuda.is_available()
    if use_nvfp4:
        try:
            import transformer_engine
            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe

            te_version = getattr(transformer_engine, "__version__", "unknown")
            recipe_name = "NVFP4BlockScaling"
            try:
                fp4_recipe = recipe.NVFP4BlockScaling()
            except AttributeError:
                recipe_name = "DelayedScaling(Format.NVFP4)"
                fp4_recipe = recipe.DelayedScaling(
                    fp8_format=recipe.Format.NVFP4,
                    amax_history_len=16,
                )
            amp_dtype = torch.float8_e4m3fn  # TE handles FP4 internally
            print(f"Using NVFP4 mixed precision (Blackwell, {args.nvfp4_bf16_tail} tail layers in BF16)")
            print(f"  transformer_engine {te_version}, recipe={recipe_name}")
            print(f"  Expected: ~1.59x throughput, ~4x memory savings vs BF16")
            print("  te.Linear layers enabled (Blackwell FP4 GEMMs)")
            use_bf16 = False  # NVFP4 supersedes BF16
            scaler = None
        except Exception as exc:
            print(f"WARNING: --use-nvfp4 requires transformer_engine ({exc}). Falling back to BF16.")
            print("  Install: pip install --no-build-isolation 'transformer_engine[pytorch]'")
            print("  Note: TE wheel must match your PyTorch/CUDA version (see GitHub releases).")
            use_nvfp4 = False
            use_bf16 = True
            amp_dtype = torch.bfloat16
            te = None
            fp4_recipe = None
    elif use_bf16:
        print(f"Using BF16 mixed precision (prevents FP16 entropy collapse)")
        scaler = None

    # Setup CUDA-Q
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if args.single_gpu:
        n_gpus = 1
        print("Single-GPU mode forced (L40S PCIe IPC workaround)")
    if cudaq and args.target:
        try:
            if args.target == "nvidia" and args.target_option == "mqpu":
                _set_cudaq_target_cached("nvidia", "mqpu")
                print(f"CUDA-Q target: nvidia (mqpu, {n_gpus} GPUs)")
            elif args.target == "tensornet-mps":
                _set_cudaq_target_cached("tensornet-mps")
                print(f"CUDA-Q target: tensornet-mps (MPS, bond={args.mps_bond})")
            elif args.target == "nvidia-mqpu-mps":
                _set_cudaq_target_cached("nvidia-mqpu-mps")
                print(f"CUDA-Q target: nvidia-mqpu-mps (MPS + mqpu, {n_gpus} GPUs)")
            else:
                # Explicit fp32 on Blackwell → cuStateVec BF16 FP32-emulation kernels
                opt = args.target_option or ("fp32" if args.target == "nvidia" else "")
                _set_cudaq_target_cached(args.target, opt)
                print(f"CUDA-Q target: {args.target}" + (f" ({opt})" if opt else ""))
            _warmup_cudaq_observe()
            print(
                "  CUDA-Q Blackwell: FP32 statevector + ALLOW_FP32_EMULATED "
                "(BF16 tensor-core emulation) + mempool + gate fusion",
                flush=True,
            )
        except Exception as e:
            print(f"Warning: CUDA-Q target setup failed: {e}")

    # Set MPS bond dimension if using MPS
    if args.mps_bond != 64:
        import os
        os.environ["CUDAQ_MPS_MAX_BOND"] = str(args.mps_bond)
        print(f"  MPS max bond dimension set to {args.mps_bond}")

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
        # Pre-build SpinOperator once (large Hamiltonians are expensive to rebuild)
        try:
            mol_data["spin_ham"] = _get_cached_spin_ham(mol_data["record"], cache_key=mol_name)
        except Exception as e:
            print(f"    WARNING: spin_ham cache failed for {mol_name}: {e}")
            mol_data["spin_ham"] = None
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

    # Persistent circuit→energy cache (hybrid online/offline speedup)
    energy_cache: PersistentEnergyCache | None = None
    if args.energy_cache is not None:
        energy_cache = PersistentEnergyCache(args.energy_cache)
        cstats = energy_cache.stats()
        print(f"\nEnergy cache: {args.energy_cache} ({cstats['n_entries']} entries)")
        if args.cache_only:
            print("  cache-only mode: CUDA-Q disabled on misses (offline RL)")
        else:
            print("  write-through mode: misses evaluate via CUDA-Q and are stored")
    elif args.cache_only:
        print("WARNING: --cache-only without --energy-cache; ignoring cache-only")
        args.cache_only = False

    # Initialize QD-GRPO components: per-molecule MAP-Elites archives + dedup cache
    map_elites = None
    dedup_cache = None
    if args.qd_mode:
        map_elites = PerMoleculeArchives(
            n_bins_entanglement=args.qd_n_bins_entanglement,
            n_bins_depth=args.qd_n_bins_depth,
            max_seq_len=args.max_seq_len,
        )
        dedup_cache = {}  # molecule_name → DedupCache (created per-molecule)
        print(f"\n=== QD-GRPO MODE ENABLED (MAP-Elites × GRPO) ===")
        print(f"  Archives: per-molecule {args.qd_n_bins_entanglement}×{args.qd_n_bins_depth} grids")
        print(f"  Novelty weight: {args.qd_novelty_weight} → {args.qd_lambda_final} "
              f"(coverage threshold: {args.qd_coverage_threshold})")
        print(f"  Surrogate: truncated L-BFGS-B ({args.qd_lbfgs_iters} iters) + per-molecule dedup cache")
        print(f"  Features: entanglement_density (multi-qubit X/Y) × circuit_depth")

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

    # Build molecule lookup for replay buffer training
    mol_data_by_name = {m["name"]: m for m in molecules_data}
    pad_id = SPECIAL_TOKENS["<PAD>"]

    # Training loop
    best_energy_per_mol = {m["name"]: float("inf") for m in molecules_data}
    train_metrics_log = []
    entropy_history: list[float] = []

    pbar = tqdm(range(args.epochs), desc="RL Epoch", unit="epoch")
    for epoch in pbar:
        # Curriculum: select which molecules to train on this epoch
        if args.curriculum and len(curriculum_stages) > 1:
            warmup = max(1, args.curriculum_warmup)
            stage_idx = min(len(curriculum_stages) - 1, epoch // warmup)
            active_molecules = curriculum_stages[stage_idx]
            if stage_idx < len(curriculum_stages) - 1 and epoch % warmup == 0 and epoch > 0:
                print(f"\n  Curriculum stage {stage_idx}: now training on {[m['name'] for m in active_molecules]}")
        else:
            active_molecules = molecules_data
        epoch_energies = []
        epoch_rewards = []
        epoch_losses = []
        epoch_skipped = 0
        epoch_sequences_generated = 0
        epoch_msun_metrics: list[dict[str, Any]] = []
        epoch_qd_stats: list[dict[str, Any]] = []
        epoch_cache_hits = 0
        epoch_cache_misses = 0
        epoch_cache_skipped = 0

        if energy_cache is not None:
            energy_cache.reset_session_counters()

        for mol_data in active_molecules:
            mol_name = mol_data["name"]
            n_qubits = mol_data["n_qubits"]
            mol_t0 = time.perf_counter()
            print(
                f"  [{mol_name}] sampling {args.n_samples} circuits "
                f"({n_qubits}q)...",
                flush=True,
            )

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
                    amp_dtype=torch.bfloat16 if (use_bf16 or use_nvfp4) else None,
                )
                entropy_history.append(sample_entropy)

                # Filter out empty sequences
                valid_indices = [i for i, ops in enumerate(operator_lists) if len(ops) > 0]
                if not valid_indices:
                    print(f"  {mol_name}: all sequences empty, resampling...", flush=True)
                    continue

                sequences = sequences[valid_indices]
                old_log_probs = old_log_probs[valid_indices]
                operator_lists = [operator_lists[i] for i in valid_indices]
                print(
                    f"  [{mol_name}] sampled {len(operator_lists)} valid "
                    f"in {time.perf_counter() - mol_t0:.1f}s → energy eval...",
                    flush=True,
                )

                # --- Phase 2: Evaluate energies ---
                # Sync PyTorch GPU ops before CUDA-Q to avoid context conflicts
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # QD-GRPO: use truncated L-BFGS-B with per-molecule dedup cache
                if args.qd_mode and dedup_cache is not None:
                    # Get or create per-molecule dedup cache with molecule context
                    if mol_name not in dedup_cache:
                        mol_nq = int(mol_data["record"]["n_qubits"])
                        mol_ne = get_active_electron_count(mol_data["record"])
                        dedup_cache[mol_name] = DedupCache(
                            molecule_id=mol_name,
                            n_qubits=mol_nq,
                            n_electrons=mol_ne,
                            optimizer_iters=args.qd_lbfgs_iters,
                            initial_theta=args.theta,
                        )
                    energies, cache_stats = evaluate_energies_qd(
                        operator_lists, mol_data["record"],
                        dedup_cache=dedup_cache[mol_name],
                        initial_theta=args.theta,
                        max_iters=args.qd_lbfgs_iters,
                    )
                else:
                    # Auto-switch to MPS for large qubit counts (B200: SV ~<=28–32q,
                    # MPS for larger). Key off qubit count only — do not require mqpu.
                    mol_nqubits = int(mol_data["record"]["n_qubits"])
                    use_mps = mol_nqubits > args.mps_threshold
                    n_terms = len(
                        mol_data["record"].get("terms")
                        or mol_data["record"].get("hamiltonian")
                        or []
                    )
                    # Term-parallel observe() needs mqpu multi-QPU. On single-GPU
                    # nvidia/fp32 (B200) it raises and energies become 0.0 — disable.
                    use_term_parallel = (
                        cudaq is not None
                        and hasattr(cudaq, "parallel")
                        and n_gpus > 1
                        and args.target == "nvidia"
                        and args.target_option == "mqpu"
                        and n_terms >= 200
                        and not use_mps
                    )
                    spin_ham = mol_data.get("spin_ham")
                    # Adaptive async depth: small SV jobs tolerate larger in-flight batches;
                    # large/MPS jobs need smaller chunks to avoid CUDA-Q thread floods.
                    if use_mps or mol_nqubits >= 28:
                        async_chunk = min(args.eval_async_chunk, 12)
                    elif mol_nqubits >= 16:
                        async_chunk = min(args.eval_async_chunk, 24)
                    elif mol_nqubits <= 8:
                        async_chunk = max(args.eval_async_chunk, 48)
                    else:
                        async_chunk = args.eval_async_chunk
                    eval_kwargs = dict(
                        molecule_record=mol_data["record"],
                        theta=args.theta,
                        spin_ham=spin_ham,
                        async_chunk=async_chunk,
                        show_progress=True,
                        mol_name=mol_name,
                    )
                    if use_mps and cudaq and args.target in ("nvidia", "tensornet-mps", "nvidia-mqpu-mps", ""):
                        try:
                            _set_cudaq_target_cached("tensornet-mps")
                            print(f"  Auto-switched to tensornet-mps for {mol_name} ({mol_nqubits} qubits)", flush=True)
                        except Exception as e:
                            print(f"  WARNING: MPS switch failed for {mol_name}: {e}", flush=True)

                        def _eval_fn(ops_batch, _ek=eval_kwargs):
                            return evaluate_energies_batch(
                                ops_batch, **_ek, eval_async=args.eval_async,
                            )
                    elif args.target == "nvidia" and args.target_option == "mqpu" and n_gpus > 1 and not use_mps:
                        try:
                            _set_cudaq_target_cached("nvidia", "mqpu")
                        except Exception:
                            pass

                        def _eval_fn(ops_batch, _ek=eval_kwargs):
                            return evaluate_energies_batch(
                                ops_batch, **_ek,
                                execution=cudaq.parallel.thread,
                                eval_async=False,
                            )
                    else:
                        # Ensure SV backend for molecules at/below mps threshold
                        if cudaq and args.target == "nvidia" and not use_mps:
                            try:
                                if args.target_option == "mqpu":
                                    _set_cudaq_target_cached("nvidia", "mqpu")
                                else:
                                    _set_cudaq_target_cached(
                                        "nvidia",
                                        args.target_option or "fp32",
                                    )
                            except Exception:
                                pass
                        _exec = cudaq.parallel.thread if use_term_parallel else None
                        _async = args.eval_async and not use_term_parallel

                        def _eval_fn(ops_batch, _ek=eval_kwargs, _ex=_exec, _as=_async):
                            return evaluate_energies_batch(
                                ops_batch, **_ek, execution=_ex, eval_async=_as,
                            )

                    mol_ne = get_active_electron_count(mol_data["record"])
                    energies, ec_stats = resolve_energies_with_cache(
                        operator_lists,
                        molecule_id=mol_name,
                        n_qubits=mol_nqubits,
                        n_electrons=mol_ne,
                        theta=args.theta,
                        eval_fn=_eval_fn,
                        cache=energy_cache,
                        cache_only=args.cache_only,
                    )
                    epoch_cache_hits += ec_stats["hits"]
                    epoch_cache_misses += ec_stats["misses"]
                    epoch_cache_skipped += ec_stats["skipped"]
                    if energy_cache is not None:
                        print(
                            f"  [{mol_name}] cache hits={ec_stats['hits']} "
                            f"misses={ec_stats['misses']} "
                            f"skipped={ec_stats['skipped']}",
                            flush=True,
                        )
                    print(
                        f"  [{mol_name}] energies done in "
                        f"{time.perf_counter() - mol_t0:.1f}s "
                        f"(best={min(energies) if energies else float('nan'):.6f})",
                        flush=True,
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
                        gate_auxiliary_rewards=args.gate_auxiliary_rewards,
                        energy_improvement_threshold=args.energy_improvement_threshold,
                    )
                    for e, ops in zip(energies, operator_lists)
                ])

                # Chemeleon2 batch-level rewards: MMD diversity + creativity
                if args.w_mmd_diversity > 0.0 and len(operator_lists) > 1:
                    # Reference set from replay buffer (last 100 samples)
                    ref_ops = [s["operators"] for s in list(replay_buffer.buffer)[-100:]]
                    mmd_rewards = compute_batch_diversity_mmd(
                        operator_lists, vocab,
                        ref_operator_lists=ref_ops if ref_ops else None,
                    )
                    mmd_eligible = np.array([
                        _has_energy_improvement(
                            energy, mol_data["hf_energy"], args.energy_improvement_threshold
                        )
                        for energy in energies
                    ], dtype=float)
                    if not args.gate_auxiliary_rewards:
                        mmd_eligible.fill(1.0)
                    rewards = rewards + args.w_mmd_diversity * mmd_rewards * mmd_eligible

                if args.w_creativity > 0.0 and len(operator_lists) > 1:
                    # Build seen set from replay buffer for novelty check
                    seen_set = {tuple(s["operators"]) for s in replay_buffer.buffer}
                    creativity_rewards = compute_creativity_batch(operator_lists, seen_set)
                    creativity_eligible = np.array([
                        _has_energy_improvement(
                            energy, mol_data["hf_energy"], args.energy_improvement_threshold
                        )
                        for energy in energies
                    ], dtype=float)
                    if not args.gate_auxiliary_rewards:
                        creativity_eligible.fill(1.0)
                    rewards = rewards + args.w_creativity * creativity_rewards * creativity_eligible

                # --- Phase 3b: QD-GRPO novelty bonus + archive insertion ---
                if args.qd_mode and map_elites is not None:
                    # Compute novelty bonus using per-molecule archive
                    novelty_bonuses = map_elites.compute_novelty_batch(
                        mol_name, operator_lists, n_qubits,
                    )
                    # Adaptive λ: decay novelty weight as this molecule's archive fills
                    lam = map_elites.adaptive_lambda(
                        mol_name,
                        initial_lambda=args.qd_novelty_weight,
                        final_lambda=args.qd_lambda_final,
                        coverage_threshold=args.qd_coverage_threshold,
                    )
                    # Add novelty bonus to rewards (intrinsic motivation)
                    rewards = rewards + lam * novelty_bonuses

                    # Insert circuits into per-molecule MAP-Elites archive
                    n_new_cells = 0
                    n_improvements = 0
                    for i, (ops, e) in enumerate(zip(operator_lists, energies)):
                        if not ops:
                            continue
                        insert_result = map_elites.insert(
                            mol_name, ops, e, n_qubits,
                            metadata={"molecule": mol_name, "epoch": epoch},
                        )
                        if insert_result["is_new_cell"]:
                            n_new_cells += 1
                        if insert_result["is_improvement"]:
                            n_improvements += 1

                    mol_archive = map_elites.get(mol_name)
                    qd_stats = {
                        "molecule": mol_name,
                        "archive_coverage": mol_archive.coverage(),
                        "archive_size": len(mol_archive),
                        "n_new_cells": n_new_cells,
                        "n_improvements": n_improvements,
                        "novelty_mean": float(np.mean(novelty_bonuses)),
                        "lambda": lam,
                        "cache_hits": cache_stats.get("hits", 0) if args.qd_mode else 0,
                        "cache_misses": cache_stats.get("misses", 0) if args.qd_mode else 0,
                    }
                    epoch_qd_stats.append(qd_stats)

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

            # Store in replay buffer (pad to max_seq_len for consistent stacking)
            pad_id_rb = SPECIAL_TOKENS["<PAD>"]
            for i, (ops, e) in enumerate(zip(operator_lists, energies)):
                seq = sequences[i]
                lp = old_log_probs[i]
                if seq.size(0) < args.max_seq_len:
                    seq = F.pad(seq, (0, args.max_seq_len - seq.size(0)), value=pad_id_rb)
                if lp.size(0) < args.max_seq_len - 1:
                    lp = F.pad(lp, (0, args.max_seq_len - 1 - lp.size(0)), value=0.0)
                replay_buffer.push(
                    seq, e, lp, mol_name, ops,
                )
                if e < best_energy_per_mol[mol_name]:
                    best_energy_per_mol[mol_name] = e

            epoch_energies.extend(energies)
            epoch_rewards.extend(rewards.tolist())
            epoch_sequences_generated += len(operator_lists)

            # Compute mSUN metric (Chemeleon2-style: Metastable, Unique, Novel)
            msun = compute_msun_metric(
                list(energies), operator_lists, mol_data["hf_energy"],
                convergence_threshold=args.msun_threshold,
            )
            msun["molecule"] = mol_name
            epoch_msun_metrics.append(msun)

            # --- Phase 6: Compute DAPO loss and update (off-policy reuse) ---
            # Off-policy GRPO (arXiv:2505.22257): reuse each rollout batch for
            # μ=args.reuse_iters gradient steps. The first iteration uses the
            # freshly sampled sequences. Subsequent iterations recompute
            # log_probs_new on the same batch — the importance sampling ratio
            # in dapo_loss automatically corrects for the policy drift.
            # This cuts CUDA-Q simulation cost by μ× without degrading quality.
            model.train()
            bos_id = SPECIAL_TOKENS["<BOS>"]
            pad_id = SPECIAL_TOKENS["<PAD>"]

            n_reuse = max(1, args.reuse_iters)
            for iter_idx in range(n_reuse):
                optimizer.zero_grad()

                if iter_idx == 0:
                    # First iteration: use freshly sampled sequences
                    iter_sequences = sequences
                    iter_old_log_probs = old_log_probs
                    iter_advantages = advantages
                    iter_mol_data = mol_data
                else:
                    # Off-policy reuse: same batch, recompute with updated policy
                    iter_sequences = sequences
                    iter_old_log_probs = old_log_probs  # original sampling log_probs
                    iter_advantages = advantages
                    iter_mol_data = mol_data

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

                # Compute reference policy log-probs for KL divergence (Chemeleon2)
                ref_log_probs = None
                if ref_model is not None and args.kl_coef > 0.0:
                    with torch.no_grad():
                        ref_logits = ref_model(
                            pauli_ids_batch, coeffs_batch, tgt_input,
                            term_mask=term_mask_batch,
                            tgt_key_padding_mask=(tgt_input == pad_id),
                        )
                        ref_log_probs = _compute_sequence_log_probs(
                            ref_logits, tgt_labels, attention_mask,
                        )

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
                            ref_log_probs=ref_log_probs,
                            kl_coef=args.kl_coef,
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                elif use_nvfp4:
                    with te.autocast(enabled=True, recipe=fp4_recipe):
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
                            ref_log_probs=ref_log_probs,
                            kl_coef=args.kl_coef,
                        )
                    loss.backward()
                    optimizer.step()
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
                            ref_log_probs=ref_log_probs,
                            kl_coef=args.kl_coef,
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
                        ref_log_probs=ref_log_probs,
                        kl_coef=args.kl_coef,
                    )
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())

        # --- Replay buffer off-policy training ---
        # Sample stale sequences from the buffer and do additional gradient
        # updates.  The importance-sampling ratio in dapo_loss automatically
        # corrects for the policy drift between sampling time and now.
        # This gives 'free' extra gradient steps without new CUDA-Q simulations.
        if args.buffer_batch_size > 0 and len(replay_buffer) >= args.buffer_batch_size:
            for _rb_iter in range(args.n_iters):
                replay_samples = replay_buffer.sample(args.buffer_batch_size)

                # Group samples by molecule for per-molecule advantage computation
                by_mol: dict[str, list[dict[str, Any]]] = {}
                for s in replay_samples:
                    by_mol.setdefault(s["molecule"], []).append(s)

                model.train()
                for mol_name_rb, samples_rb in by_mol.items():
                    mol_data_rb = mol_data_by_name.get(mol_name_rb)
                    if mol_data_rb is None:
                        continue

                    # Stack sequences and old log-probs
                    seqs = torch.stack([s["sequence"] for s in samples_rb]).to(device)
                    old_lps = torch.stack([s["log_probs"] for s in samples_rb]).to(device)
                    energies_rb = np.array([s["energy"] for s in samples_rb], dtype=float)

                    # Group-relative advantages (GRPO-style)
                    rewards_rb = energies_rb  # raw energy as reward proxy for replay
                    advantages_rb = compute_advantages(
                        rewards_rb,
                        use_grpo=True,
                        repo_beta=args.repo_beta,
                        old_log_probs=old_lps,
                        attention_mask=(seqs[:, 1:] != pad_id).float(),
                    )

                    tgt_input_rb = seqs[:, :-1].to(device)
                    tgt_labels_rb = seqs[:, 1:].to(device)
                    attn_mask_rb = (tgt_labels_rb != pad_id).float().to(device)

                    pauli_ids_rb = mol_data_rb["pauli_ids"].unsqueeze(0).expand(
                        seqs.size(0), -1, -1
                    ).to(device)
                    coeffs_rb = mol_data_rb["coeffs"].unsqueeze(0).expand(
                        seqs.size(0), -1
                    ).to(device)
                    term_mask_rb = mol_data_rb["term_mask"].unsqueeze(0).expand(
                        seqs.size(0), -1
                    ).to(device)

                    optimizer.zero_grad()
                    if use_nvfp4:
                        with te.autocast(enabled=True, recipe=fp4_recipe):
                            logits_rb = model(
                                pauli_ids_rb, coeffs_rb, tgt_input_rb,
                                term_mask=term_mask_rb,
                                tgt_key_padding_mask=(tgt_input_rb == pad_id),
                            )
                            log_probs_new_rb = _compute_sequence_log_probs(
                                logits_rb, tgt_labels_rb, attn_mask_rb,
                            )
                            loss_rb = dapo_loss(
                                log_probs_new_rb, old_lps,
                                advantages_rb, attn_mask_rb,
                                clip_low=args.clip_low, clip_high=args.clip_high,
                                token_level=args.token_level_loss,
                                entropy_coef=args.entropy_coef,
                                logits=logits_rb,
                                ref_log_probs=None,
                                kl_coef=0.0,
                            )
                        loss_rb.backward()
                        optimizer.step()
                    elif use_bf16:
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            logits_rb = model(
                                pauli_ids_rb, coeffs_rb, tgt_input_rb,
                                term_mask=term_mask_rb,
                                tgt_key_padding_mask=(tgt_input_rb == pad_id),
                            )
                            log_probs_new_rb = _compute_sequence_log_probs(
                                logits_rb, tgt_labels_rb, attn_mask_rb,
                            )
                            loss_rb = dapo_loss(
                                log_probs_new_rb, old_lps,
                                advantages_rb, attn_mask_rb,
                                clip_low=args.clip_low, clip_high=args.clip_high,
                                token_level=args.token_level_loss,
                                entropy_coef=args.entropy_coef,
                                logits=logits_rb,
                                ref_log_probs=None,
                                kl_coef=0.0,
                            )
                        loss_rb.backward()
                        optimizer.step()
                    else:
                        logits_rb = model(
                            pauli_ids_rb, coeffs_rb, tgt_input_rb,
                            term_mask=term_mask_rb,
                            tgt_key_padding_mask=(tgt_input_rb == pad_id),
                        )
                        log_probs_new_rb = _compute_sequence_log_probs(
                            logits_rb, tgt_labels_rb, attn_mask_rb,
                        )
                        loss_rb = dapo_loss(
                            log_probs_new_rb, old_lps,
                            advantages_rb, attn_mask_rb,
                            clip_low=args.clip_low, clip_high=args.clip_high,
                            token_level=args.token_level_loss,
                            entropy_coef=args.entropy_coef,
                            logits=logits_rb,
                            ref_log_probs=None,
                            kl_coef=0.0,
                        )
                        loss_rb.backward()
                        optimizer.step()

                    epoch_losses.append(loss_rb.item())

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

        # Aggregate mSUN across all molecules this epoch
        if epoch_msun_metrics:
            mean_msun = np.mean([m["msun"] for m in epoch_msun_metrics])
            mean_converged = np.mean([m["converged"] for m in epoch_msun_metrics])
            mean_unique = np.mean([m["unique"] for m in epoch_msun_metrics])
            mean_novel = np.mean([m["novel"] for m in epoch_msun_metrics])
        else:
            mean_msun = mean_converged = mean_unique = mean_novel = 0.0

        recent_entropy = np.mean(entropy_history[-10:]) if entropy_history else 0.0

        # QD-GRPO logging
        qd_cov = 0.0
        qd_size = 0
        qd_lambda = 0.0
        qd_cache_hit_rate = 0.0
        if epoch_qd_stats:
            qd_cov = np.mean([s["archive_coverage"] for s in epoch_qd_stats])
            qd_size = epoch_qd_stats[-1]["archive_size"]
            qd_lambda = epoch_qd_stats[-1]["lambda"]
            total_hits = sum(s["cache_hits"] for s in epoch_qd_stats)
            total_misses = sum(s["cache_misses"] for s in epoch_qd_stats)
            qd_cache_hit_rate = total_hits / max(total_hits + total_misses, 1)

        postfix = (f"loss={mean_loss:.4f} "
                   f"E_mean={mean_energy:.4f} "
                   f"E_min={min_energy:.4f} "
                   f"R={mean_reward:.4f} "
                   f"H={recent_entropy:.2f} "
                   f"mSUN={mean_msun:.2f} "
                   f"skip={epoch_skipped} "
                   f"buf={len(replay_buffer)}")
        if energy_cache is not None:
            ec_total = epoch_cache_hits + epoch_cache_misses
            ec_rate = epoch_cache_hits / max(ec_total, 1)
            postfix += f" ecache={ec_rate:.0%}({epoch_cache_hits}/{ec_total})"
        if args.qd_mode:
            postfix += (f" QD={qd_size}({qd_cov:.0%}) "
                        f"λ={qd_lambda:.2f} "
                        f"cache={qd_cache_hit_rate:.0%}")
        pbar.set_postfix_str(postfix)

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
            "msun": mean_msun,
            "msun_converged": mean_converged,
            "msun_unique": mean_unique,
            "msun_novel": mean_novel,
            "msun_per_mol": epoch_msun_metrics,
            "qd_coverage": qd_cov,
            "qd_archive_size": qd_size,
            "qd_lambda": qd_lambda,
            "qd_cache_hit_rate": qd_cache_hit_rate,
            "qd_stats": epoch_qd_stats,
            "energy_cache_hits": epoch_cache_hits,
            "energy_cache_misses": epoch_cache_misses,
            "energy_cache_skipped": epoch_cache_skipped,
            "energy_cache_hit_rate": (
                epoch_cache_hits / max(epoch_cache_hits + epoch_cache_misses, 1)
            ),
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
    if energy_cache is not None:
        final_cstats = energy_cache.stats()
        print(f"Energy cache: {final_cstats['n_entries']} entries @ {final_cstats['path']}")
        print(f"  session hit rate: {final_cstats['session_hit_rate']:.1%} "
              f"({final_cstats['session_hits']} hits / "
              f"{final_cstats['session_hits'] + final_cstats['session_misses']} lookups)")
        energy_cache.close()
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

    # Final mSUN summary
    if train_metrics_log:
        final_msun = train_metrics_log[-1]
        print(f"\nFinal mSUN metrics (last epoch):")
        print(f"  mSUN={final_msun.get('msun', 0.0):.3f}  "
              f"converged={final_msun.get('msun_converged', 0.0):.3f}  "
              f"unique={final_msun.get('msun_unique', 0.0):.3f}  "
              f"novel={final_msun.get('msun_novel', 0.0):.3f}")

    # Save MAP-Elites archives if QD mode
    if args.qd_mode and map_elites is not None:
        archive_dir = args.qd_archive_path or args.out.parent / f"{args.out.stem}_map_elites"
        map_elites.save_all(str(archive_dir))
        print(f"\nMAP-Elites archives saved to: {archive_dir}/")
        print(f"  Archive summary: {map_elites.summary()}")
        if dedup_cache is not None:
            for mol_name, cache in dedup_cache.items():
                print(f"  Dedup cache ({mol_name}): {cache.stats()}")

    # Save metrics JSON
    metrics_path = args.out.parent / f"{args.out.stem}_rl_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "best_energies": dict(best_energy_per_mol),
            "train_log": train_metrics_log,
            "final_buffer_size": len(replay_buffer),
            "qd_summary": map_elites.summary() if args.qd_mode and map_elites is not None else None,
            "dedup_stats": dedup_cache.stats() if args.qd_mode and dedup_cache is not None else None,
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
