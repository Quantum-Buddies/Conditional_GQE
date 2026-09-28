# qBraid Execution Strategy — GIC 2026 Phase 3

This document outlines how to use qBraid credits (11,000 available) in a tight-knit HPC-Quantum-AI workflow for:
1. GPU-based training and simulation on qBraid Lab On-Demand instances (B200/H200)
2. Asynchronous batch QPU execution of H-cGQE generated circuits with QWC grouping
3. Reproducibility validation for competition judges using the free simulator target

---

## 1. qBraid GPU Instances (HPC-AI Stage)

| Instance | GPU | VRAM | Credits/min | Credits/hour | Max SV Qubits | ~Hours with 11,000 cr |
|---|---|---|---|---|---|---|
| `gpu-l40s` | 1× L40S | 48 GB | 3.80 | 228 | 24* | ~48h |
| `gpu-a100-sxm` | 1× A100 | 80 GB | 4.15 | 249 | 26 | ~44h |
| `gpu-gh200` | 1× GH200 | 96 GB | 4.78 | 287 | 28 | ~38h |
| `gpu-h100-sxm` | 1× H100 | 80 GB | 8.95 | 537 | 26 | ~20h |
| `gpu-h200` | 1× H200 | 141 GB | 9.15 | 549 | 30 | ~20h |
| `gpu-b200` | 1× B200 | 192 GB | 14.57 | 874 | 32 | ~12.5h |
| `gpu-b200-4x` | 4× B200 | 768 GB | 56.58 | 3,395 | 36 | ~3.2h |

*L40S 24-qubit limit is due to PCIe IPC segfault, not VRAM.

**Recommended instance by task:**

| Task | Instance | Rationale |
|---|---|---|
| RL training (4–28q) | `gpu-h200` | Best cost/performance for 28–30q SV range |
| RL training (4–32q) | `gpu-b200` | 32q SV on single GPU, NVFP4 optional |
| RL training (4–40q) | `gpu-b200-4x` | 36q SV with NVLink distribution |
| MPS evaluation | `gpu-h200` | 30–40q approximate simulation |
| Noise simulation | `gpu-b200` | 32q SV with noise model |
| QPU submission | Any | Device-dependent, per-shot billing |
| FMO recombination | CPU | Classical post-processing, 0 credits |

On-demand instances are billed per minute and can be launched via the qBraid Lab dashboard or via the qBraid CLI:
```bash
qbraid compute up gpu-h200
```

### B200 Blackwell Setup on qBraid

The qBraid container provides system CUDA 13.2 but CUDA libs are accessed via pip packages. The launcher auto-resolves `LD_LIBRARY_PATH` from the `nvidia` site-packages tree. Before training:

```bash
# Source Blackwell environment (MUST be before importing cudaq)
source scripts/env_b200_blackwell.sh

# Install PyTorch with Blackwell support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# Optional: NVFP4 for 1.59x throughput
pip install --no-build-isolation transformer_engine[pytorch]
```

See [B200_TRAINING_PLAN.md](B200_TRAINING_PLAN.md) for full Blackwell environment details.

---

## 2. qBraid QPU Access (Quantum QPU Stage)

| QPU | Provider | Per-task | Per-shot | Max qubits | Feasible for us? |
|---|---|---|---|---|---|
| IonQ Forte-1 | IonQ | 30 cr | 8 cr | ~30q | Good for 4-12q molecules |
| IonQ Forte-Enterprise-1 | IonQ | 30 cr | 8 cr | ~30q | Same |
| IQM Emerald | AWS | 30 cr | 0.16 cr | 54q | Cheapest per-shot |
| IQM Garnet | AWS | 30 cr | 0.145 cr | 20q | Good for 8-12q |
| Rigetti Cepheus-1-108Q | AWS | 30 cr | 0.0425 cr | 108q | Cheapest shots, most qubits |
| AQT IBEX Q1 | AQT | 30 cr | 2.35 cr | ~24q | Expensive shots |
| QuEra Aquila | QuEra | 30 cr | 1 cr | 256q | Analog Hamiltonian, not gate-based |

**Recommended**: Use Rigetti Cepheus for most experiments (cheapest per-shot, most qubits). Use IQM Garnet for 8-12q molecules. Use IonQ for portability demo only (expensive shots).

### Free/Cheap Simulators

| Device ID | Qubits | Cost | Notes |
|---|---|---|---|
| `qbraid:qbraid:sim:qir-sv` | 30 | 7.5 cr/min + 0.5 cr/task | Sparse statevector, for judge validation |
| `ionq:ionq:sim:simulator` | 29 | $0.00/min | Free but rate-limited (429s) |
| `aws:aws:sim:sv1` | 34 | 7.5 cr/min | Free first min/task, no batch support |
| `aws:aws:sim:dm1` | 17 | 7.5 cr/min | Density matrix (noise simulation) |

### QWC Term Grouping: 3-5× Circuit Reduction

We implement qubit-wise commuting (QWC) Pauli term grouping in `src/gqe/eval/qbraid_backend.py` to reduce the number of measurement circuits needed. Terms that commute qubit-wise share the same measurement basis and can be evaluated from a single circuit.

| Molecule | Qubits | Pauli Terms | QWC Groups | Reduction |
|---|---|---|---|---|
| **H2** | 4 | 15 | 5 | 3× |
| **LiH** | 12 | 631 | 180 | 3.5× |
| **N2** | 20 | 2,951 | 1,308 | 2.3× |

**Validation results (AWS SV1, 4096 shots):**
- H2: Simulator energy -1.1182 Ha vs GPU -1.1167 Ha = **1.477 mHa** difference (shot noise)

---

## 3. Asynchronous Batch Submission: The 90%+ Cost Saving

To avoid paying the **30-credit per-task fee** for each Pauli term in the molecular Hamiltonian, we package all QWC measurement circuits into a single batch and submit them **asynchronously** to the QPU. This frees up expensive GPU compute nodes while waiting in the quantum queue.

### QPU Cost Comparison (Individual vs Batch with QWC Grouping)

| Molecule | Qubits | QWC Groups | QPU | Individual (per-term) | Batch (1 task) | Savings |
|---|---|---|---|---|---|---|
| **H2** | 4 | 5 | Rigetti Cepheus | 15×30 + 15×1024×0.0425 = 1,083 cr | 1×30 + 5×1024×0.0425 = **248 cr** | **835 cr** |
| **LiH** | 8 | 180 | Rigetti Cepheus | 185×30 + 185×100×0.0425 = 6,286 cr | 1×30 + 180×100×0.0425 = **795 cr** | **5,491 cr** |
| **BeH2** | 14 | ~250 | Rigetti Cepheus | 731×30 + 731×100×0.0425 = 24,914 cr | 1×30 + 250×100×0.0425 = **1,093 cr** | **23,821 cr** |

### Async HPC→QPU Workflow

The workflow decouples HPC compute from QPU queue time:

1. **HPC exports QWC manifest** (operators, thetas, groups, QASM) — no QPU needed
2. **Submit manifest to QPU asynchronously** — returns immediately with job IDs
3. **Retrieve results separately** — poll until COMPLETED, then parse

```bash
# Export manifest only (no QPU submission)
python scripts/submit_qpu_async.py --export-only \
    --hamiltonians results/data/hamiltonians.json \
    --optimized results/eval/h_cgqe_optimized.json

# Submit to QPU (returns immediately)
python scripts/submit_qpu_async.py \
    --device aws:rigetti:qpu:cepheus-1-108q \
    --manifest results/eval/qwc_manifest.json

# Retrieve results (poll until complete)
python scripts/submit_qpu_async.py --retrieve results/eval/qpu_manifest_meta.json
```

---

## 4. HPC-Quantum-AI Tight-Knit Workflow

We orchestrate the local HPC cluster development and remote qBraid QPU execution using the orchestrator script `scripts/run_hpc_qbraid_workflow.sh`.

### Step 1: Submit Pre-processing & RL Training to Slurm

*Paths/partition names shown are for the University of Leeds AIRE cluster — adapt for your site.*
Submit the local GPU scaling workflow directly to Slurm from your repository directory:
```bash
bash scripts/run_hpc_qbraid_workflow.sh --hpc-submit
```
Monitor queue status:
```bash
bash scripts/run_hpc_qbraid_workflow.sh --hpc-status
```

### Step 2: Submit Circuits to qBraid QPU Asynchronously
Once the local optimizations have completed on the HPC cluster, dispatch the best-predicted circuits to Rigetti Cepheus asynchronously:
```bash
bash scripts/run_hpc_qbraid_workflow.sh --qpu-submit
```
This saves job submission metadata files in `results/eval/` and exits immediately, releasing the local GPU allocation.

### Step 3: Poll and Retrieve QPU Ground State Energy
Monitor the queue status of the QPU jobs:
```bash
bash scripts/run_hpc_qbraid_workflow.sh --qpu-status
```
Once all jobs return `COMPLETED`, execute the retrieval command to parse the parities, compute the term expectations, and save the final energy values:
```bash
bash scripts/run_hpc_qbraid_workflow.sh --qpu-retrieve
```

---

## 5. Manual Execution Details

### Asynchronous QPU Submission
```bash
python src/gqe/eval/qbraid_backend.py \
    --hamiltonians results/data/hamiltonians.json \
    --generated results/inference/h_cgqe_uccsd_inference.json \
    --optimized results/eval/h_cgqe_uccsd_optimized.json \
    --molecule h2_0.74 \
    --device aws:rigetti:qpu:cepheus-1-108q \
    --shots 1024 \
    --submit-only \
    --out results/eval/qbraid_h2_rigetti.json
```

### Retrieval of Completed Results
```bash
python src/gqe/eval/qbraid_backend.py \
    --retrieve results/eval/qbraid_job_metadata_h2_0.74_aws_rigetti_qpu_cepheus-1-108q.json \
    --out results/eval/qbraid_h2_rigetti.json
```

### Free Simulator Validation (0 Credits)
```bash
python src/gqe/eval/qbraid_backend.py \
    --hamiltonians results/data/hamiltonians.json \
    --generated results/inference/h_cgqe_uccsd_inference.json \
    --optimized results/eval/h_cgqe_uccsd_optimized.json \
    --molecule h2_0.74 \
    --device qbraid:qbraid:sim:qir-sv \
    --shots 2000 \
    --out results/eval/qbraid_h2_sim.json
```

---

## 6. Credit Budget (11,000 credits)

| Scenario | Instance | Duration | Credits | Remaining |
|---|---|---|---|---|
| Full pipeline (H200) | gpu-h200 | 10 hrs | ~5,490 | ~5,510 |
| Full pipeline (B200) | gpu-b200 | 7 hrs | ~6,118 | ~4,882 |
| RAFT only (H200) | gpu-h200 | 3 hrs | ~1,647 | ~9,353 |
| 36-qubit eval (B200x4) | gpu-b200-4x | 2 hrs | ~6,790 | ~4,210 |
| QPU runs (Rigetti) | — | — | ~683-2,049 | varies |

**Optimized budget**: Use H200 for RL training (best cost/performance), B200 for 32q SV evaluation, Rigetti for QPU. Total ~10,500 credits.

---

## 7. Rigetti Cepheus-1-108Q Hardware Details

| Spec | Value |
|---|---|
| Architecture | 12 × 9-qubit chiplets, square lattice |
| Connectivity | 4-fold nearest-neighbor (tunable couplers + IMCs) |
| Native gates | RX, RY, CZ (adiabatic) |
| 2Q gate fidelity | 99.1% median (target: 99.5% by end of 2026) |
| 1Q gate fidelity | 99.9% median |
| Gate speed | ~60 ns |
| T1 / T2 | 25 μs / 10 μs |
| Availability | 20 hrs/day |

### Connectivity implications
- H2 (4q): fits within a single chiplet (3×3 grid), minimal SWAP overhead
- LiH (12q): spans ~2 chiplets, moderate SWAP overhead
- N2 (20q): spans ~3 chiplets, significant inter-chiplet routing needed

---

## 8. QPU Safeguards Implemented

- **Circuit complexity preflight**: `_circuit_complexity()` computes depth and 2q gate count before submission
- **ZNE auto-skip**: ZNE skipped if 2q gates > 20 (gate folding would make circuit too deep)
- **REM auto-skip**: Full REM calibration skipped if qubits > 10 (2^n × 2^n is exponential)
- **Configurable**: `--max-zne-two-qubit-gates 20 --max-rem-qubits 10`
- **Retry logic**: 6 attempts with exponential backoff for transient 404 errors

---

## References

- [qBraid CLI documentation](https://docs.qbraid.com/v2/cli/api-reference/qbraid)
- [qBraid SDK program execution](https://docs.qbraid.com/v2/sdk/user-guide/programs)
- [NVIDIA CUDA-Q qBraid target guide](https://nvidia.github.io/cuda-quantum/latest/using/backends/cloud/qbraid.html)
- [B200 Training Plan](B200_TRAINING_PLAN.md) — Full B200/Blackwell setup and training details
- [qBraid Integration Guide](QBRAID_INTEGRATION.md) — Comprehensive HPC-Quantum-AI integration
