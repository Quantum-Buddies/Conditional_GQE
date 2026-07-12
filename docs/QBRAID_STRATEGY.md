# qBraid Execution Strategy — GIC 2026 Phase 3

## Overview

This document outlines how to use qBraid credits (11,000 available) for:
1. GPU-based training and simulation on qBraid Lab
2. Real QPU execution of H-cGQE generated circuits
3. Reproducibility validation for competition judges

---

## qBraid GPU Instances

| Instance | GPU | Credits/min | Credits/hour | ~Hours with 11,000 cr |
|---|---|---|---|---|
| `gpu-l40s` | 1x L40S 48GB | 3.80 | 228 | ~48h |
| `gpu-h100-sxm` | 1x H100 80GB | 8.95 | 537 | ~20h |
| `gpu-a100-sxm` | 1x A100 80GB | 4.15 | 249 | ~44h |
| `gpu-gh200` | 1x GH200 96GB | 4.78 | 287 | ~38h |
| `gpu-rtx-4090` | 1x RTX 4090 24GB | 1.45 | 87 | ~126h |

Launch via qBraid CLI:
```bash
qbraid compute up gpu-l40s
```

Or via the qBraid Lab dashboard → On-Demand tab.

---

## qBraid QPU Access

| QPU | Provider | Per-task | Per-shot | Max qubits | Feasible for us? |
|---|---|---|---|---|---|
| IonQ Forte-1 | IonQ | 30 cr | 8 cr | ~30q | Good for 4-12q molecules |
| IonQ Forte-Enterprise-1 | IonQ | 30 cr | 8 cr | ~30q | Same |
| IQM Emerald | AWS | 30 cr | 0.16 cr | 54q | Cheapest per-shot |
| IQM Garnet | AWS | 30 cr | 0.145 cr | 20q | Good for 8-12q |
| Rigetti Cepheus-1-108Q | AWS | 30 cr | 0.0425 cr | 108q | Cheapest shots, most qubits |
| AQT IBEX Q1 | AQT | 30 cr | 2.35 cr | ~24q | Expensive shots |
| QuEra Aquila | QuEra | 30 cr | 1 cr | 256q | Analog Hamiltonian, not gate-based |

**Free simulator**: `qbraid:qbraid:sim:qir-sv` — up to 30 qubits, 2000 shots, zero credits.

---

## CUDA-Q + qBraid Integration

qBraid is now a native CUDA-Q cloud target (June 2026). Compile with:
```bash
nvq++ --target qbraid kernel.cpp -o kernel
```

Set the machine via:
```bash
export QBRAID_API_KEY=<your-key>
# Default: free state-vector simulator
# For QPU: --qbraid-machine aws:rigetti:qpu:cepheus-1-108q
```

This means our CUDA-Q kernels can run on qBraid QPUs with zero code changes.

---

## Recommended Budget Allocation (11,000 credits)

| Activity | Instance/QPU | Est. credits | Est. time | What it gets us |
|---|---|---|---|---|
| Full pipeline run | `gpu-l40s` | ~2,000 | ~8h | Reproducible results on qBraid |
| RL training (Chemeleon2, 500 epochs) | `gpu-l40s` | ~1,500 | ~6h | RL-tuned model |
| 40+ qubit MPS benchmark | `gpu-h100-sxm` | ~500 | ~1h | Scaling data for 40q claim |
| H2 (4q) on Rigetti Cepheus QPU | Rigetti | ~700 | 1 task | Real quantum hardware energy |
| LiH (8q) on Rigetti (reduced shots) | Rigetti | ~1,170 | 1 task | 8q on real hardware |
| Free qBraid simulator validation | `qbraid:sim:qir-sv` | 0 | — | Reproducibility for judges |
| **Total** | | **~5,870** | | Leaves ~5,130 for iteration |

---

## QPU Execution Plan

### Step 1: H2 on Rigetti Cepheus-1-108Q (cheapest per-shot)

H2 has 4 qubits and 15 Pauli terms in the Hamiltonian.

```bash
python src/gqe/eval/qbraid_backend.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --generated results/inference/h_cgqe_generated_uccsd.json \
    --optimized results/eval/h_cgqe_uccsd_optimized.json \
    --molecule h2_0.74 \
    --device aws:rigetti:qpu:cepheus-1-108q \
    --shots 1024 \
    --out results/eval/qbraid_h2_rigetti.json
```

Cost: 30 cr (task) + 15 terms × 1024 shots × 0.0425 cr/shot = 30 + 653 = **~683 credits**

### Step 2: LiH on Rigetti (reduced shots)

LiH (active space) has 8 qubits and ~185 Pauli terms.

```bash
python src/gqe/eval/qbraid_backend.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --generated results/inference/h_cgqe_generated_uccsd.json \
    --optimized results/eval/h_cgqe_uccsd_optimized.json \
    --molecule lih_1.6_full \
    --device aws:rigetti:qpu:cepheus-1-108q \
    --shots 100 \
    --out results/eval/qbraid_lih_rigetti.json
```

Cost: 30 cr (task) + 185 terms × 100 shots × 0.0425 cr/shot = 30 + 789 = **~819 credits**

### Step 3: Free simulator validation (all molecules)

```bash
python src/gqe/eval/qbraid_backend.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --generated results/inference/h_cgqe_generated_uccsd.json \
    --optimized results/eval/h_cgqe_uccsd_optimized.json \
    --molecule h2_0.74 \
    --device qbraid:qbraid:sim:qir-sv \
    --shots 2000 \
    --out results/eval/qbraid_h2_sim.json
```

Cost: **0 credits** (free simulator, up to 30 qubits)

---

## qBraid Lab Setup

### 1. Launch GPU instance

From qBraid Lab dashboard → On-Demand tab → Launch `gpu-l40s`.

### 2. Clone and install

```bash
git clone https://github.com/Quantum-Buddies/Conditional_GQE.git
cd Conditional_GQE
pip install -r requirements-qbraid.txt
```

### 3. Run full pipeline

```bash
bash scripts/run_gic2026_scaling.sh
```

Or step-by-step:
```bash
# 1. Generate Hamiltonians
python src/gqe/data/generate_hamiltonians.py \
    --config configs/experiment_scaling_gic2026.yaml \
    --out-dir results/data

# 2. RL training from scratch
python src/gqe/models/train_rl_dapo.py \
    --from-scratch \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --molecules h2_0.74 lih_1.6_full n2_1.1_full beh2_1.3_full \
    --out results/train/h_cgqe_rl.pt \
    --epochs 300 --lr 3e-4 --n-samples 50 \
    --use-bf16 --curriculum --curriculum-warmup 30 \
    --explore-eps 0.3 --adaptive-eps --top-p 0.9 \
    --force-entanglement

# 3. Inference
python src/gqe/models/infer_h_cgqe.py \
    --checkpoint results/train/h_cgqe_rl.pt \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/inference/h_cgqe_generated.json

# 4. Stage 2 optimization
python src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated results/inference/h_cgqe_generated.json \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/eval/h_cgqe_optimized.json

# 5. Evaluate
python src/gqe/eval/evaluate_h_cgqe.py \
    --generated results/inference/h_cgqe_generated.json \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/eval/h_cgqe_evaluation.json
```

### 4. QPU submission

```bash
# Set qBraid API key
export QBRAID_API_KEY=<your-key>

# Run H2 on Rigetti Cepheus (real quantum hardware)
python src/gqe/eval/qbraid_backend.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --generated results/inference/h_cgqe_generated.json \
    --optimized results/eval/h_cgqe_optimized.json \
    --molecule h2_0.74 \
    --device aws:rigetti:qpu:cepheus-1-108q \
    --shots 1024 \
    --out results/eval/qbraid_h2_rigetti.json
```

---

## Existing qBraid Infrastructure in This Repo

| File | Purpose |
|---|---|
| `src/gqe/eval/qbraid_backend.py` | Qiskit circuit translation + qBraid device submission |
| `requirements-qbraid.txt` | All dependencies for qBraid Lab |
| `environment-qbraid.yml` | Conda environment spec |
| `qbraid_skill/` | Full qBraid skill package with install scripts |
| `scripts/run_h_cgqe_qbraid.sh` | Shell script for qBraid execution |

---

## Phase 3 Submission Impact

- **Platform Use**: qBraid GPU + QPU execution demonstrates full cloud pipeline
- **Phase 3 Execution**: Real hardware results (even H2 only) differentiate from teams that only simulate
- **Reproducibility**: Judges can clone repo, launch qBraid Lab, and re-run everything
- **Scalability**: Free qBraid simulator (30q) + GPU instances for MPS (40q+) covers all qubit ranges

---

## References

- [qBraid CUDA-Q integration](https://www.qbraid.com/blog-posts/qbraid-cudaq-integration) (June 2026)
- [qBraid GPU pricing](https://docs.qbraid.com/v2/home/pricing)
- [qBraid quantum devices](https://docs.qbraid.com/v2/lab/user-guide/quantum-devices)
- [CUDA-Q qBraid target docs](https://nvidia.github.io/cuda-quantum/latest/using/backends/cloud/qbraid.html)
