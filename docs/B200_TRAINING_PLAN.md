# B200 Training Strategy — Conditional-GQE

## Architecture Decision: Supervised Warm-start → DAPO RL

**Main B200 run = SFT warm-start → DAPO RL (NOT direct RL from scratch)**

The repository contains both paths. `train_h_cgqe.py` learns from existing GQE operator sequences, while `train_rl_dapo.py` can load that checkpoint or use `--from-scratch`.

The warm start is preferable because the RL reward requires CUDA-Q circuit evaluation and L-BFGS-B refinement. On larger and unseen molecules, a randomly initialized policy has high-variance rewards and is more likely to collapse into invalid, repetitive, or diagonal operator sequences. The warm start supplies a valid UCCSD-derived operator vocabulary and useful initial distribution; DAPO then moves beyond imitation using energy feedback.

Direct RL from scratch is retained as an **ablation** for the paper (flag: `--from-scratch`). It is viable on small systems (4–20q) but collapses on large molecules where the policy never finds a low-energy circuit to bootstrap from. This was confirmed in our ablation runs on qBraid.

---

## Hardware: NVIDIA B200 (Blackwell)

| Spec | Value |
|---|---|
| GPU | 1× NVIDIA B200 (sm_100, Blackwell) |
| VRAM | 180 GB HBM3e |
| Memory bandwidth | 8 TB/s (9.3× L40S's 864 GB/s) |
| Compute | NVFP4 tensor cores (4th-gen Transformer Engine) |
| Interconnect | NVLink (vs L40S PCIe-only — no IPC segfault) |
| Max SV qubits (single GPU) | ~32q (2^32 × 16B = 64 GB, fits in 180 GB) |
| Max SV qubits (4× B200) | ~36q (768 GB pooled, NVLink distribution) |
| MPS qubits | 60+ (tensor network, bond-dimension limited) |

### Why B200 over L40S

1. **L40S 24-qubit ceiling**: PCIe-only L40S cluster segfaults at 25q in CUDA-Q's distributed statevector mode due to broken CUDA IPC in Open MPI's `smcuda` BTL. B200's NVLink eliminates this.
2. **9.3× memory bandwidth**: CUDA-Q energy evaluation (the RL bottleneck) is memory-bandwidth-bound. 8 TB/s vs 864 GB/s translates to near-linear speedup for statevector operations.
3. **NVFP4 tensor cores**: Optional 1.59× training throughput and 4× memory savings via Transformer Engine FP4 GEMMs (requires `transformer_engine` install).
4. **180 GB VRAM**: Fits 32q statevector on a single GPU — no multi-GPU distribution needed for the GIC 35-molecule set (max 28q).

---

## Blackwell Environment Setup

### Environment Variables (`scripts/env_b200_blackwell.sh`)

The B200 requires specific environment variables to be sourced **before** any Python that imports `cudaq` or launches PyTorch training:

```bash
source scripts/env_b200_blackwell.sh
```

| Variable | Default | Purpose |
|---|---|---|
| `CUBLAS_EMULATE_SINGLE_PRECISION` | `1` | cuBLAS BF16x9 FP32 emulation (Tensor Core path for FP32 GEMMs) |
| `CUBLAS_EMULATION_STRATEGY` | `performant` | cuBLAS emulation strategy |
| `CUDAQ_ALLOW_FP32_EMULATED` | `1` | CUDA-Q cuStateVec Blackwell FP32→BF16 emulation |
| `CUDAQ_ENABLE_MEMPOOL` | `1` | CUDA-Q memory pool (reduces allocation overhead) |
| `CUDAQ_FUSION_MAX_QUBITS` | `5` | B200 default gate fusion threshold for FP32 |
| `CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS` | `-1` | Unlimited diagonal gate fusion |
| `CUDAQ_FUSION_NUM_HOST_THREADS` | `16` | Circuit preprocessing threads |
| `CUDAQ_MAX_GPU_MEMORY_GB` | `NONE` | Use full B200 HBM for statevectors |
| `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` | `1` | PyTorch TF32/cuBLAS override |

### PyTorch Blackwell Optimizations (in `train_rl_dapo.py`)

The function `_enable_blackwell_torch_optimizations()` automatically:
- Enables TF32 matmul precision (`torch.backends.cuda.matmul.allow_tf32 = True`)
- Enables cuDNN TF32 + benchmark mode
- Sets `torch.set_float32_matmul_precision("high")`
- Enables Flash SDPA, memory-efficient SDPA, and math SDPA backends
- Detects Blackwell (sm_100+) and prints diagnostic info

### CUDA-Q Target on B200

```python
# Blackwell: explicit fp32 option triggers cuStateVec BF16x9 FP32-emulation kernels
cudaq.set_target("nvidia", option="fp32")
```

The `_set_cudaq_target_cached()` function in `train_rl_dapo.py` automatically sets `option="fp32"` when target is `nvidia` with no explicit option, ensuring Blackwell-optimized kernels are used.

### NVFP4 Mixed Precision (Optional)

For additional throughput on Blackwell:

```bash
pip install --no-build-isolation transformer_engine[pytorch]
```

Then use `--use-nvfp4` in `train_rl_dapo.py`. This enables:
- NVFP4 block-scaled FP4 GEMMs via Transformer Engine
- ~1.59× throughput vs BF16
- ~4× memory savings
- Last 2 decoder layers kept in BF16 (NVIDIA recipe: last 15% of layers)

If Transformer Engine is unavailable, falls back to BF16 automatically.

### CUDA Library Path Resolution

The qBraid container provides system CUDA 13.2 but CUDA libs are accessed via pip packages. The launcher (`launch_b200_training.sh`) auto-resolves `LD_LIBRARY_PATH` from the `nvidia` site-packages tree:

```bash
NVIDIA_SITE="$(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia"
# Iterates cu13, cublas, cudnn, cufft, curand, cusolver, cusparse, nccl, cuda_runtime
```

### PyTorch Install for B200

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

cu126 = CUDA 12.6 build; sm_100/B200 kernels are compiled in via PTX JIT from PyTorch 2.7+.

---

## Molecule Inventory

| Dataset file | Count | Qubit range | Purpose |
|---|---:|---:|---|
| `hamiltonians_gic2026/hamiltonians.json` | 35 | 4–28q | Main RL training (GIC challenge molecules) |
| `hamiltonians_merged.json` | 21 | 4–40q | SFT training + scaling baselines |
| `hamiltonians_40plus/hamiltonians.json` | 10 | 4–40q | RL XL scaling run (benzene 40q, N₂ 40q) |
| `hamiltonians.json` | 5 | 4–20q | Legacy baseline set |
| `hamiltonians_iodobenzene.json/` | 2 | 8–12q | Iodobenzene CAS variants |

**GIC 2026 molecules (35):** h2, lih, beh2, n2, imeph_cas12, iodobenzene_cas12, methyl_iodide_cas12, phenol_cas12, ocresol_cas12, anisole_cas12, benzene_cas12, toluene_cas12, h2o, nh3, ch4, ethylene, formaldehyde, acetylene, hf, co, h2_0.5/1.0/1.5/2.0, lih_1.2/2.0/3.0, n2_1.8/2.5, beh2_1.0/1.6, lih_1.6_631g, n2_1.1_631g_cas8, h2o_1.0_631g_cas8, diarylethene_frag_cas12

**40q targets:** benzene_cas20 (40q), n2_ccpvdz_cas20 (40q), n2_ccpvdz (32q), beh2_ccpvdz (32q), ethylene (28q), formaldehyde (24q)

---

## Training Pipeline

### Stage 1: Supervised Fine-Tuning (Warm-Start)

Trains the H-cGQE Transformer via teacher-forced cross-entropy on UCCSD operator sequences with commutator loss penalty.

```bash
python src/gqe/models/train_h_cgqe.py \
    --dataset results/train/gqe_supervised_dataset.pt \
    --out results/train/h_cgqe_model_b200_sft.pt \
    --epochs 500 --batch-size 1024 --lr 5e-4 \
    --d-model 256 --nhead 8 --enc-layers 4 --dec-layers 4 --dim-ff 1024 \
    --dropout 0.1 --train-split 0.8 --use-cuda --use-bf16 \
    --val-every 2 --commutator-weight 0.1 --commutator-ramp-epochs 100 \
    --label-smoothing 0.1 --patience 60 --min-delta 1e-4
```

**SFT Results (B200, 500 epochs, BF16):**
- Model: 7.79M params, vocab size 317
- Best validation loss: 1.037
- Final validation accuracy: **96.2%**
- Training converged after ~200 epochs with early stopping (patience=60)

**Output:** `results/train/h_cgqe_model_b200_sft.pt` (31 MB)

### Stage 2: DAPO RL Fine-Tuning (Main Run)

Fine-tunes the SFT checkpoint with DAPO (Decoupled Clip + Dynamic Sampling Policy Optimization) on the 35 GIC challenge molecules.

```bash
python src/gqe/models/train_rl_dapo.py \
    --checkpoint results/train/h_cgqe_model_b200_sft.pt \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --molecules $MOLECULES \
    --out results/train/h_cgqe_model_b200_rl_main.pt \
    --epochs 300 --n-samples 64 --n-iters 5 --lr 1e-5 \
    --clip-low 0.2 --clip-high 0.28 \
    --use-cuda --use-bf16 --single-gpu \
    --target nvidia --target-option fp32 \
    --max-qubits 28 --mps-threshold 24 --mps-bond 64 \
    --curriculum --curriculum-warmup 30 \
    --force-entanglement --entropy-coef 0.01 \
    --adaptive-temp --explore-eps 0.3 \
    --kl-coef 0.05 --w-creativity 0.1
```

**Key RL design decisions:**
- LR = 1e-5 (10× lower than SFT) for stable RL fine-tuning
- Temperature = 1.0 with adaptive scheduling (0.7–2.0 range)
- `--force-entanglement` prevents diagonal sequence collapse
- `--single-gpu` avoids L40S PCIe IPC issues (B200 has no such limit but keeps config portable)
- `--target-option fp32` triggers Blackwell BF16x9 FP32 emulation kernels
- `--gate-auxiliary-rewards` gates non-energy rewards on beating Hartree-Fock (prevents reward hacking)
- `--curriculum` starts with smallest molecules, adds larger ones in 3 stages over 30 warmup epochs

**Output:** `results/train/h_cgqe_model_b200_rl_main.pt` (40 MB)

### Stage 2 XL: DAPO RL on 40q Scaling Systems

Extends RL training to 40-qubit molecules using MPS backend:

```bash
python src/gqe/models/train_rl_dapo.py \
    --checkpoint results/train/h_cgqe_model_b200_rl_main.pt \
    --hamiltonians results/data/hamiltonians_40plus/hamiltonians.json \
    --molecules $MOLECULES_40 \
    --out results/train/h_cgqe_model_b200_rl_40q.pt \
    --epochs 150 --n-samples 32 --n-iters 3 --lr 5e-6 \
    --use-cuda --use-bf16 --single-gpu \
    --target nvidia --target-option fp32 \
    --max-qubits 40 --mps-threshold 24 --mps-bond 128 \
    --curriculum --force-entanglement
```

**Output:** `results/train/h_cgqe_model_b200_rl_40q.pt`

### Stage 2 Ablation: Direct RL from Scratch

Tests whether RL alone (no SFT) can learn circuit generation. Uses `--from-scratch` flag with compact policy:

```bash
python src/gqe/models/train_rl_dapo.py \
    --from-scratch \
    --hamiltonians results/data/hamiltonians_merged.json \
    --molecules $MOLECULES \
    --out results/train/h_cgqe_model_b200_rl_scratch.pt \
    --epochs 200 --n-samples 128 --n-iters 8 --reuse-iters 16 \
    --lr 1e-4 --d-model 256 --nhead 8 \
    --encoder-layers 4 --decoder-layers 4 --dim-feedforward 1024 \
    --use-cuda --use-bf16 --single-gpu \
    --target nvidia --target-option fp32 \
    --max-qubits 40 --mps-threshold 28 --mps-bond 64 \
    --force-entanglement --no-adaptive-theta \
    --curriculum --curriculum-warmup 15 --curriculum-steps 4 \
    --no-dynamic-sampling --eval-async --eval-async-chunk 24 \
    --energy-cache results/train/rl_energy_cache.sqlite
```

**Ablation smoke test results (2 epochs, H2 + LiH, B200):**
- H2 best energy: -1.1165 Ha (vs FCI -1.1173 Ha → 0.8 mHa error)
- LiH best energy: 0.0 Ha (policy collapsed — no valid circuits found)
- Mean entropy: 3.13 → 2.86 (declining, indicating exploration narrowing)
- mSUN = 1.0 (all samples unique/novel, but none converged for LiH)

**Conclusion:** Direct RL from scratch works for small molecules (H2) but collapses on larger ones (LiH), confirming the need for SFT warm-start.

**Output:** `results/train/h_cgqe_model_b200_rl_scratch.pt`

### qBraid H200 Smoke Test

Validated the SFT→RL pipeline on qBraid H200 instance before the B200 full run:

**Config:** B200 SFT checkpoint → DAPO RL on H2 + LiH, 2 epochs, 16 samples/epoch, QD-GRPO mode
**Results:**
- H2 best energy: -1.1220 Ha (vs FCI -1.1173 Ha → **4.7 mHa below FCI** — likely fixed-θ proxy artifact)
- LiH best energy: -7.8619 Ha
- Dynamic sampling skipped 2/4 batches (identical energies in group)
- Mean entropy: 3.30 (healthy exploration)
- Replay buffer: 1,136 entries (pre-populated from SFT checkpoint)

---

## Energy Cache Precomputation

The RL training bottleneck is CUDA-Q energy evaluation. A persistent SQLite cache (`rl_energy_cache.sqlite`) stores circuit→energy mappings so repeated circuits skip CUDA-Q entirely.

```bash
bash scripts/launch_b200_training.sh cache
```

Precomputes 512 circuits/molecule (≤28q) and 128 circuits/molecule (>28q MPS) using async CUDA-Q observe with chunked evaluation (24 in-flight jobs).

For resuming interrupted cache computation (32–40q only):

```bash
bash scripts/launch_b200_training.sh cache-remaining
```

The cache is **append-safe** — existing entries are preserved. A backup is created before appending.

---

## B200 Launcher

The unified launcher script `scripts/launch_b200_training.sh` orchestrates all stages:

```bash
# Full pipeline (SFT → RL main → RL 40q):
bash scripts/launch_b200_training.sh both

# SFT only:
bash scripts/launch_b200_training.sh sft

# RL only (needs SFT checkpoint):
bash scripts/launch_b200_training.sh rl

# Precompute energy cache (one-time):
bash scripts/launch_b200_training.sh cache

# Resume cache for 32–40q only:
bash scripts/launch_b200_training.sh cache-remaining

# Ablation (direct RL from scratch):
bash scripts/launch_b200_training.sh ablation

# Ablation smoke test (~2 min sanity check):
bash scripts/launch_b200_training.sh ablation-smoke

# Cache + ablation in sequence:
bash scripts/launch_b200_training.sh cache+ablation
```

### Portable Single-B200 Script

For environments without the full launcher (e.g., custom qBraid instances):

```bash
SKIP_SUPERVISED=0 RL_EPOCHS=500 RL_SAMPLES=64 RL_ITERS=5 \
MAX_QUBITS=30 MAX_TERMS=256 \
bash scripts/run_b200_training.sh
```

### Checkpoint Outputs

| File | Size | Description |
|---|---|---|
| `h_cgqe_model_b200_sft.pt` | 31 MB | SFT warm-start checkpoint |
| `h_cgqe_model_b200_rl_main.pt` | 40 MB | DAPO RL on 35 GIC molecules |
| `h_cgqe_model_b200_rl_40q.pt` | 40 MB | DAPO RL extended to 40q |
| `h_cgqe_model_b200_rl_scratch.pt` | 40 MB | Ablation (scratch RL) |
| `rl_energy_cache.sqlite` | varies | Persistent circuit→energy cache |

All checkpoints are hosted on [Hugging Face](https://huggingface.co/Quantum-Buddies/Conditional-GQE-models) and auto-download via `ensure_checkpoint()`.

---

## 40-Qubit Scaling Pipeline

The `scripts/run_40q_scaling_pipeline.sh` script demonstrates end-to-end convergence from 4q to 40q on a single B200:

**Stages:**
1. Hamiltonian generation (4q → 40q)
2. RL training with QD-GRPO + MAP-Elites (B200, NVFP4 optional)
3. AI circuit synthesis (inference with trained model)
4. L-BFGS-B coefficient optimization
5. GPU validation (exact SV ≤32q, MPS 32–40q)
6. QPU submission (IQM Emerald / Rigetti Cepheus)
7. Plot generation and scaling report

**B200 advantages for 40q:**
- 32q exact statevector on single GPU (2^32 × 4B = 16 GB, fits in 180 GB)
- 8 TB/s HBM = 9.3× faster CUDA-Q eval vs L40S
- NVFP4 for 1.59× training throughput, 4× memory savings
- 256 samples/epoch (2× L40S baseline)

---

## Critical Import Order: torch.compile → CUDA-Q

CUDA-Q and Triton (used by `torch.compile`) both embed LLVM. Importing `cudaq` **before** calling `torch.compile` causes:

```text
CommandLine Error: Option 'debug-counter' registered more than once!
LLVM ERROR: inconsistency in registered CommandLine options
```
Exit code 134 (SIGABRT). Confirmed on qBraid H200 and B200.

**Fix (implemented in `train_rl_dapo.py`):**
1. **Lazy CUDA-Q import** via `_ensure_cudaq()` — only imports cudaq AFTER `torch.compile` has loaded Triton's LLVM
2. **`--cache-only` skips CUDA-Q entirely** — no LLVM clash during cache-warmup
3. **AR-safe decoder compile path** — encoder uses `reduce-overhead` (CUDA graphs, fixed shapes); decoder uses `default` mode with `dynamic=True` (CUDA graphs can't handle growing autoregressive sequences)

**Rule: Always `torch.compile` first → `cudaq` import second. Never reverse.**

---

## DAPO RL Training Details

### Reward Function

Multi-component reward with auxiliary gating:

```text
r = w₁·(-E/|E_ref|) + w₂·entanglement_frac + w₃·(-depth/max_len) + w₄·non_commute_frac + w₅·diversity
```

Auxiliary rewards (w₂–w₅) are **gated** on energy improvement over Hartree-Fock. If `E >= E_HF - threshold`, auxiliary rewards are zeroed — preventing reward hacking where the model optimizes structural metrics without lowering energy.

### Key RL Features

| Feature | Implementation | Purpose |
|---|---|---|
| DAPO asymmetric clipping | `clip_low=0.2, clip_high=0.28` | Prevents entropy collapse (Clip-Higher) |
| Dynamic sampling | Skip groups where `std(rewards) < 1e-8` | Avoids wasted gradient on identical circuits |
| Token-level loss | Per-token PPO loss (not sequence-level) | Finer-grained credit assignment |
| Off-policy GRPO (μ-reuse) | `--reuse-iters 3` | 3× gradient steps per rollout, importance sampling correction |
| Replay buffer | FIFO, size 2000 | Off-policy sample reuse |
| Curriculum learning | 3 stages, 30-epoch warmup | Small → large molecule progression |
| Force entanglement | Reject diagonal-only sequences | Prevents diagonal sequence collapse |
| Adaptive temperature | 0.7–2.0 range, target entropy 1.5 | Auto-tune exploration |
| Top-p nucleus sampling | `top_p=0.9` | Focused but diverse sampling |
| Frequency penalty | `freq_penalty=1.0` | Prevents mode collapse to repeated operators |
| REPO regularization | `repo_beta=0.05` | Mild entropy preservation |
| QD-GRPO (optional) | MAP-Elites archive + novelty bonus | Quality-diversity exploration |

### Physicist Verification: Fixed-θ Proxy Limitation

A physicist reviewer identified that the fixed-θ (θ=0.01) proxy energy provides near-zero gradient signal for RL on large molecules. Spearman rank correlation between proxy and converged energy: **0.227** (p=0.416). Most circuits evaluate to exactly the Hartree-Fock baseline with variations only at the 11th decimal place.

**Fix implemented:** `--adaptive-theta` runs truncated L-BFGS-B (10 iterations) on the best circuit in each batch, using the optimized energy for reward instead of the fixed-θ proxy. This gives a physically meaningful reward signal at ~50× the cost of fixed-θ but with Spearman ρ ~0.5 vs final energy.

---

## Recommended Training Stages

1. **SFT warm-start** from `gqe_supervised_dataset.pt` (500 epochs, BF16, batch 1024)
2. **DAPO RL** on H2, LiH, BeH2, N2 (curriculum warmup, 300 epochs)
3. **Add unseen 12–24q GIC molecules** for generalization testing
4. **Evaluate 28–40q molecules** with MPS/statevector safeguards
5. **Optimize coefficients** (L-BFGS-B), validate on free simulator, then submit selected shallow circuits to QPU

---

## References

- **DAPO**: arXiv:2503.14476 — Clip-Higher, Dynamic Sampling, Token-level loss
- **REPO**: arXiv:2603.11682 — Regulated Entropy Policy Optimization
- **Off-policy GRPO (μ-reuse)**: arXiv:2505.22257
- **Chemeleon2**: Park & Walsh, Nat. Mach. Intell. 2026, arXiv:2511.07158
- **RL from scratch**: arXiv:2502.19402 — shows RL-only can outperform SFT-then-RL (tested as ablation)
- **CUDA-Q Blackwell**: NVIDIA CUDA-Q v0.10+ supports B200/GB200 with FP32 emulation
- **NVFP4**: NVIDIA Transformer Engine 4th-gen, Blackwell FP4 block-scaled GEMMs
