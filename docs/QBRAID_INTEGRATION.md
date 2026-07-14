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

## 6. Simulator Validation (Zero-Credit Verification)

For judge verification, we provide a validation script that runs all generated/aligned circuits on qBraid's free simulator target (`qbraid:qbraid:sim:qir-sv` - up to 30 qubits):

```bash
python scripts/validate_on_qbraid.py \
    --hamiltonians results/data/hamiltonians_scaling.json/hamiltonians.json \
    --optimized results/eval/h_cgqe_uccsd_optimized.json \
    --out results/eval/qbraid_validation_report.json
```
This generates a validation report containing energy differences and execution statuses for all systems.
