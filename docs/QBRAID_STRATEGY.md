# qBraid Execution Strategy — GIC 2026 Phase 3

This document outlines how to use qBraid credits (11,000 available) in a tight-knit HPC-Quantum-AI workflow for:
1. GPU-based training and simulation on qBraid Lab On-Demand instances
2. Asynchronous batch QPU execution of H-cGQE generated circuits on Rigetti Cepheus
3. Reproducibility validation for competition judges using the free simulator target

---

## 1. qBraid GPU Instances (HPC-AI Stage)

| Instance | GPU | Credits/min | Credits/hour | ~Hours with 11,000 cr |
|---|---|---|---|---|
| `gpu-l40s` | 1x L40S 48GB | 3.80 | 228 | ~48h |
| `gpu-h100-sxm` | 1x H100 80GB | 8.95 | 537 | ~20h |
| `gpu-a100-sxm` | 1x A100 80GB | 4.15 | 249 | ~44h |
| `gpu-gh200` | 1x GH200 96GB | 4.78 | 287 | ~38h |
| `gpu-rtx-4090` | 1x RTX 4090 24GB | 1.45 | 87 | ~126h |

On-demand instances are billed per minute and can be launched via the qBraid Lab dashboard or via the qBraid CLI:
```bash
qbraid compute up gpu-l40s
```

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

**Free simulator**: `qbraid:qbraid:sim:qir-sv` — up to 30 qubits, 2000 shots, zero credits.

---

## 3. Asynchronous Batch Submission: The 90%+ Cost Saving

To avoid paying the **30-credit per-task fee** for each Pauli term in the molecular Hamiltonian (15 terms for $H_2$, 185 terms for $LiH$), we submit the circuits as a single batch using `as_batch=True` via the qBraid SDK. 

Further, QPU queue times can be long. Instead of waiting synchronously and tying up expensive GPU compute nodes on our local HPC cluster, we submit jobs **asynchronously** and retrieve them later.

### QPU Cost Comparison (Batch vs Individual Execution)

| Molecule | Qubits | Terms | QPU | Individual Execution Cost | Batch Execution Cost | Savings |
|---|---|---|---|---|---|---|
| **H2** | 4 | 15 | Rigetti Cepheus | 15×30 + 15×1024×0.0425 = 1,083 cr | 1×30 + 15×1024×0.0425 = **683 cr** | **400 cr** |
| **LiH** | 8 | 185 | Rigetti Cepheus | 185×30 + 185×100×0.0425 = 6,286 cr | 1×30 + 185×100×0.0425 = **816 cr** | **5,470 cr** |
| **BeH2** | 14 | 731 | Rigetti Cepheus | 731×30 + 731×100×0.0425 = 24,914 cr | 1×30 + 731×100×0.0425 = **3,137 cr** | **21,777 cr** |

---

## 4. HPC-Quantum-AI Tight-Knit Workflow

We orchestrate the local HPC cluster development and remote qBraid QPU execution using the orchestrator script `[run_hpc_qbraid_workflow.sh](file:///scratch/kcwp264/Conditional-GQE_materials/scripts/run_hpc_qbraid_workflow.sh)`.

### Step 1: Submit Pre-processing & RL Training to Slurm
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

If you prefer to call the modules manually:

### Asynchronous QPU Submission
```bash
python src/gqe/eval/qbraid_backend.py \
    --hamiltonians results/data/hamiltonians_scaling.json/hamiltonians.json \
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
    --hamiltonians results/data/hamiltonians_scaling.json/hamiltonians.json \
    --generated results/inference/h_cgqe_uccsd_inference.json \
    --optimized results/eval/h_cgqe_uccsd_optimized.json \
    --molecule h2_0.74 \
    --device qbraid:qbraid:sim:qir-sv \
    --shots 2000 \
    --out results/eval/qbraid_h2_sim.json
```

---

## References

- [qBraid CLI documentation](https://docs.qbraid.com/v2/cli/api-reference/qbraid)
- [qBraid SDK program execution](https://docs.qbraid.com/v2/sdk/user-guide/programs)
- [NVIDIA CUDA-Q qBraid target guide](https://nvidia.github.io/cuda-quantum/latest/using/backends/cloud/qbraid.html)
