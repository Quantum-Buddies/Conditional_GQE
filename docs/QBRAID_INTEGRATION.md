# H-cGQE & qBraid Hybrid HPC-Quantum-AI Integration Guide

This document provides a comprehensive guide to the hybrid HPC-Quantum-AI workflow designed for the GIC 2026 Phase 3 pipeline. It coordinates local HPC cluster runs (using Slurm on AIRE) with remote cloud quantum execution and post-training alignment on the qBraid platform.

---

## 1. Setup & Authentication

The qBraid CLI and SDK are installed inside the target conda environment:
`/scratch/kcwp264/.conda_envs/cudaq-env/`

### Credentials File & Shell Environment
Your API key is configured permanently in both your shell profile and the native qBraid CLI settings:
1. **Shell Profile**: Pre-configured in your `~/.bashrc` to auto-export `QBRAID_API_KEY`.
2. **qBraid Credentials File**: Stored in `~/.qbraid/qbraidrc` to authenticate all CLI and SDK invocations.

Verify credentials and check active devices:
```bash
/scratch/kcwp264/.conda_envs/cudaq-env/bin/qbraid devices list
```

---

## 2. High-Performance GPU training (HPC-AI Stage)

For heavy training, such as the Stage 1 Reinforcement Learning (DAPO) or the Stage 2 RAFT alignment, you can leverage qBraid's larger GPU instances:

### Instance Guide

| Instance | GPU | Memory (VRAM) | Credits/Hour | Speedup vs L40S | Use Case |
|---|---|---|---|---|---|
| `gpu-l40s` | 1x L40S | 48 GB | 228 cr | $1.0\times$ | Standard runs / debugging |
| `gpu-gh200` | 1x Grace Hopper | 96 GB | 287 cr | **$1.6\times - 1.9\times$** | **Recommended** for RL & sweeps |
| `gpu-h100-sxm` | 1x H100 | 80 GB | 537 cr | **$1.7\times - 2.0\times$** | Large matrix product states |
| `gpu-h100-2x` | 2x H100 | 160 GB | 1,048 cr | **$3.2\times$** | Massive candidate searches |

Launch compute instance via CLI:
```bash
qbraid compute up gpu-gh200
```

---

## 3. Post-Training Alignment (RAFT Stage)

We implement **Rejection Sampling Fine-Tuning (RAFT)** to align the model prior. The script samples candidate circuits from your base policy, evaluates them classically using L-BFGS-B optimization on CUDA-Q, and fine-tunes the transformer model on the top-performing (lowest energy) sequences.

Execute the alignment loop locally or inside qBraid Lab:
```bash
python scripts/train_post_alignment.py \
    --checkpoint results/train/h_cgqe_rl_warmstart.pt \
    --hamiltonians results/data/hamiltonians_scaling.json/hamiltonians.json \
    --out results/train/h_cgqe_raft_aligned.pt \
    --epochs 50 \
    --n-samples 50 \
    --top-k 5 \
    --use-cuda \
    --target nvidia \
    --target-option mqpu
```

To run as a cluster batch job:
```bash
sbatch jobs/train_post_alignment.slurm
```

---

## 4. Asynchronous Batch QPU Execution (Quantum QPU Stage)

Measuring individual Hamiltonian terms sequentially wastes credits on separate task fees (30 cr each). Instead, we package all measurement circuits into a single batch and submit them **asynchronously** to the physical QPU. This frees up expensive local GPU nodes while waiting in the quantum queue.

### QPU Cost Comparison (Individual vs Batch)

| Molecule | Qubits | Terms | QPU | Individual (per-term) | Batch (1 task) | Savings |
|---|---|---|---|---|---|---|
| **H2** | 4 | 15 | Rigetti Cepheus | 1,083 cr | **683 cr** | **400 cr** |
| **LiH** | 8 | 185 | Rigetti Cepheus | 6,286 cr | **816 cr** | **5,470 cr** |
| **BeH2** | 14 | 731 | Rigetti Cepheus | 24,914 cr | **3,137 cr** | **21,777 cr** |

---

## 5. Unified Workflow Orchestration

We bind all steps together using the orchestrator script `[run_hpc_qbraid_workflow.sh](file:///scratch/kcwp264/Conditional-GQE_materials/scripts/run_hpc_qbraid_workflow.sh)`.

### Step 1: Submit training/pre-processing to Slurm
```bash
bash scripts/run_hpc_qbraid_workflow.sh --hpc-submit
```

### Step 2: Track Slurm job status
```bash
bash scripts/run_hpc_qbraid_workflow.sh --hpc-status
```

### Step 3: Submit optimized circuits to qBraid QPU (Asynchronous)
Once local optimizations are finished on the cluster, submit the batch QPU jobs:
```bash
bash scripts/run_hpc_qbraid_workflow.sh --qpu-submit
```
This writes the QPU job tracking identifiers to `results/eval/qbraid_job_metadata_*.json`.

### Step 4: Check QPU status
```bash
bash scripts/run_hpc_qbraid_workflow.sh --qpu-status
```

### Step 5: Retrieve QPU results
Once the jobs are completed on the QPU, retrieve them to compute the final expectation energy:
```bash
bash scripts/run_hpc_qbraid_workflow.sh --qpu-retrieve
```

---

## 5b. Alternative: CUDA-Q Native Target Execution

If you prefer to bypass Qiskit circuit translation and run CUDA-Q kernels natively on the remote QPU:
1. Ensure `cudaq` is installed in your active environment.
2. In your python script, set the target device:
   ```python
   import cudaq
   cudaq.set_target("qbraid", machine="aws:rigetti:qpu:cepheus-1-108q")
   # Your standard cudaq kernel execution remains unchanged:
   # energy = cudaq.observe(kernel, hamiltonian, *args).expectation()
   ```

For C++ programs, compile with the `--target` flag:
```bash
nvq++ --target qbraid --qbraid-machine "aws:rigetti:qpu:cepheus-1-108q" kernel.cpp -o kernel
export QBRAID_API_KEY="your_api_key"
./kernel
```
*Note: Ensure `QBRAID_API_KEY` is exported in the environment where the binary runs.*

---

## 6. Simulator Validation (Zero-Credit Verification)

For judge verification, we provide a validation script that runs all generated/aligned circuits on qBraid's free simulator target (`qbraid:qbraid:sim:qir-sv` - up to 30 qubits):

```bash
python scripts/validate_on_qbraid.py \
    --hamiltonians results/data/hamiltonians_scaling.json/hamiltonians.json \
    --optimized results/eval/h_cgqe_uccsd_optimized.json \
    --out results/eval/qbraid_validation_report.json
```
This generates a validation report containing energy differences and execution statuses for all systems.

---

## 7. Post-Training Methods (STaR + Model Soup + Off-Policy GRPO)

We implement four post-training techniques inspired by DeepSeek-R1, Google Gemini, and NVIDIA's post-training scaling laws, adapted for quantum circuit generation:

### 7a. Iterative RAFT (STaR Loop)

Self-Taught Reasoner (STaR) / DeepSeek-R1 Stage 2 approach: each RAFT round produces a better model that generates higher-quality candidates for the next round.

```bash
bash scripts/run_iterative_raft.sh \
    --rounds 3 \
    --checkpoint results/train/h_cgqe_rl.pt \
    --n-samples 100 --top-k 10 \
    --adaptive-n --use-cuda
```

Each round: sample N circuits → optimize (L-BFGS-B) → filter top-k → SFT → next round. Temperature decays 0.9× per round for progressive exploitation.

### 7b. Model Soup

Weight averaging across RAFT rounds (Wortsman et al., ICML 2022). Free performance boost, no extra inference cost:

```bash
python src/gqe/models/model_soup.py \
    --checkpoints results/train/h_cgqe_raft_round_1.pt \
                  results/train/h_cgqe_raft_round_2.pt \
                  results/train/h_cgqe_raft_round_3.pt \
    --out results/train/h_cgqe_star_soup.pt
```

Supports uniform averaging (default) and greedy soup (add checkpoint only if it improves validation energy).

### 7c. Off-Policy GRPO (μ-Reuse)

Reuses each rollout batch for μ gradient steps with importance sampling correction (arXiv:2505.22257). Cuts CUDA-Q simulation cost by μ×:

```bash
python src/gqe/models/train_rl_dapo.py \
    --reuse-iters 3 \
    --target nvidia --target-option mqpu \
    --use-cuda --use-bf16 \
    ...
```

The `dapo_loss` function's importance sampling ratio `exp(log_probs_new - log_probs_old)` automatically corrects for policy drift across reuse iterations.

### 7d. Adaptive Test-Time Compute

Following Snell et al. 2024 (Google DeepMind), allocates more samples to harder molecules (4× more efficient than uniform Best-of-N):

```bash
python scripts/train_post_alignment.py \
    --adaptive-n-samples \
    --n-samples 50 \
    ...
```

Scaling: `N_effective = N_base × max(1, n_qubits ÷ 4)`. H2 (4q) gets 50 samples, N2 (20q) gets 250, benzene (32q) gets 400.

---

## 8. B200/H200 Large-Qubit Scaling

The AIRE L40S cluster is capped at 24 qubits due to PCIe IPC segfault in CUDA-Q's distributed statevector mode. qBraid's B200 (192GB) and H200 (141GB) instances have proper NVLink interconnects, enabling 26-40 qubit simulations.

### Instance Comparison

| Instance | GPU | VRAM | Credits/Hr | Max Qubits (SV) | Max Qubits (MPS) |
|---|---|---|---|---|---|
| `gpu-l40s` | L40S | 48 GB | 228 | 24* | 40+ |
| `gpu-gh200` | Grace Hopper | 96 GB | 287 | 28 | 50+ |
| `gpu-h200` | H200 | 141 GB | 549 | 30 | 60+ |
| `gpu-b200` | B200 | 192 GB | 874 | 32 | 60+ |
| `gpu-b200-4x` | 4× B200 | 768 GB | 3,395 | 36 | 80+ |

*L40S 24-qubit limit is due to PCIe IPC, not VRAM.

### Scaling Config

`configs/experiment_scaling_b200.yaml` defines molecules across 4 qubit tiers:

| Tier | Qubits | Examples | L40S? | H200? | B200? |
|---|---|---|---|---|---|
| Small | 4-12 | H2, LiH | ✅ | ✅ | ✅ |
| Medium | 12-24 | BeH2, N2 | ✅ | ✅ | ✅ |
| Large | 24-32 | N2/6-31g, H2O/6-31g, ethylene/6-31g | ❌ | ✅ | ✅ |
| XL | 32-40 | benzene/CAS12, methanol/6-31g, acetylene/CAS14 | ❌ | ✅ (MPS) | ✅ |

### Running the Full Scaling Pipeline

```bash
# On qBraid H200 (recommended — best cost/performance for 28-32 qubit range)
bash scripts/run_qbraid_scaling.sh --stage all --instance gpu-h200

# On qBraid B200 (for 32-36 qubit statevector)
bash scripts/run_qbraid_scaling.sh --stage all --instance gpu-b200

# Only run RAFT post-training on existing checkpoint
bash scripts/run_qbraid_scaling.sh --stage raft --instance gpu-h200

# Only validate on free simulator
bash scripts/run_qbraid_scaling.sh --stage validate
```

### Credit Budget (11,000 credits)

| Scenario | Instance | Duration | Credits | Remaining |
|---|---|---|---|---|
| Full pipeline (H200) | gpu-h200 | 10 hrs | ~5,490 | ~5,510 |
| Full pipeline (B200) | gpu-b200 | 7 hrs | ~6,118 | ~4,882 |
| RAFT only (H200) | gpu-h200 | 3 hrs | ~1,647 | ~9,353 |
| 36-qubit eval (B200x4) | gpu-b200-4x | 2 hrs | ~6,790 | ~4,210 |
| QPU runs (Rigetti) | — | — | ~683-2,049 | varies |
