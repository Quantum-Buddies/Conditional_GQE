# Scaling Quantum Simulations with CUDA-Q TensorNet / Multi-GPU

## Objective

Scale GQE and H-cGQE quantum simulations to larger qubit counts (12–22+ qubits) using CUDA-Q backends on AIRE HPC L40S GPUs. Move beyond the Phase 3 limit of 8 qubits by leveraging `tensornet`, `nvidia-mqpu`, and `nvidia-mgpu` backends.

## Environment

- **HPC:** AIRE cluster, 3× NVIDIA L40S GPUs per node (48 GB HBM3 each, PCIe-only, no NVLink)
- **CUDA-Q:** v0.14.2, installed in conda env `/mnt/scratch/kcwp264/.conda_envs/cudaq-env`
- **Project root:** `/scratch/kcwp264/Conditional-GQE_materials`
- **Python:** `/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python`
- **SLURM allocation:** `srun --pty -p gpu --gres=gpu:3 --cpus-per-task=8 --mem=64G -t 20:00:00 /bin/bash`

## CUDA-Q Backends Used

| Backend | Flag | Use Case | Qubit Limit (L40S) |
|---|---|---|---|
| `nvidia` | `--target nvidia` | Single GPU statevector | ~26 qubits (fp32, 48 GB) |
| `nvidia-mqpu` | `--target nvidia --target-option mqpu` | 3 parallel QPUs (independent simulations) | ~26 qubits per QPU |
| `nvidia-mgpu` | `--target nvidia --target-option mgpu,fp32` | 3 GPUs pooled memory for one statevector | ~35 qubits (144 GB total) |
| `tensornet` | `--target tensornet` | Exact tensor network, single GPU | 50+ qubits (shallow circuits) |
| `tensornet-mps` | `--target tensornet-mps` | Approximate MPS, single GPU | 100+ qubits |

## Files Created

### Config
- `configs/experiment_scaling.yaml` — Molecule definitions for scaling experiments

### Scripts
- `scripts/run_scaling_3gpu.sh` — Full pipeline for 3× L40S (mqpu parallel)
- `scripts/run_scaling.sh` — Full pipeline for single GPU (tensornet)
- `scripts/run_scaling_tensornet.py` — Python orchestrator (alternative to shell scripts)

### Results (generated)
- `results/data/hamiltonians_scaling.json/hamiltonians.json` — 11 Hamiltonians, 4–22 qubits
- `results/baselines/cudaq_gqe_scaling.json` — GQE baseline energies
- `results/inference/h_cgqe_generated_scaling.json` — H-cGQE generated operator sequences
- `results/eval/h_cgqe_optimized_scaling.json` — L-BFGS-B optimized coefficients + energies
- `results/eval/h_cgqe_evaluation_scaling.json` — Final evaluation with energy errors

## Molecules in Scaling Config

| Molecule | Basis | Qubits | Electrons | Hamiltonian Terms | Notes |
|---|---|---|---|---|---|
| `h2_0.74` | STO-3G | 4 | 2 | 15 | Reference / sanity check |
| `lih_1.6_full` | STO-3G | 12 | 4 | 631 | Full active space |
| `n2_1.1_full` | STO-3G | 20 | 14 | 2951 | Full active space — largest SV |
| `beh2_1.3_full` | STO-3G | 14 | 6 | 666 | Full active space |
| `iodobenzene_cas12` | STO-3G | 12 | 6 | 471 | EUV-relevant |
| `methyl_iodide_cas12` | STO-3G | 12 | 6 | 923 | EUV-relevant |
| `imeph_cas12` | STO-3G | 12 | 6 | 923 | EUV-relevant (test) |
| `phenol_cas12` | STO-3G | 12 | 6 | 923 | EUV-relevant (test) |
| `lih_1.6_631g` | 6-31G | 22 | 4 | 8758 | **New basis — largest system** |
| `n2_1.1_631g_cas8` | 6-31G | 16 | 8 | 1177 | New basis + active space |
| `h2o_1.0_631g_cas8` | 6-31G | 16 | 8 | 3057 | New molecule |

## Pipeline Steps

### Step 1: Generate Hamiltonians
```bash
$PY src/gqe/data/generate_hamiltonians.py \
    --config configs/experiment_scaling.yaml \
    --out results/data/hamiltonians_scaling.json
```
Output: `results/data/hamiltonians_scaling.json/hamiltonians.json`

### Step 2: GQE Baseline (nvidia-mqpu, 3 parallel GPUs)
```bash
$PY src/gqe/baselines/run_cudaq_gqe.py \
    --ham results/data/hamiltonians_scaling.json/hamiltonians.json \
    --out results/baselines/cudaq_gqe_scaling.json \
    --target nvidia --target-option mqpu \
    --max-qubits 25
```
Runtime: ~9 min for 11 molecules on 3× L40S

### Step 3: H-cGQE Inference (RLQF model)
```bash
$PY src/gqe/models/infer_h_cgqe.py \
    --checkpoint results/train/h_cgqe_model_rlqf_phase3.pt \
    --hamiltonians results/data/hamiltonians_scaling.json/hamiltonians.json \
    --out results/inference/h_cgqe_generated_scaling.json \
    --n-samples 50 --sample --use-cuda \
    --max-pauli-len 22 --max-seq-len 64 \
    --molecules h2_0.74 lih_1.6_full n2_1.1_full beh2_1.3_full \
    iodobenzene_cas12 methyl_iodide_cas12 \
    imeph_cas12 phenol_cas12 \
    lih_1.6_631g n2_1.1_631g_cas8 h2o_1.0_631g_cas8
```
Generates 50 operator sequences per molecule using the trained RLQF model.

### Step 4: Optimize Coefficients (nvidia-mqpu, 3 GPUs)
```bash
$PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated results/inference/h_cgqe_generated_scaling.json \
    --hamiltonians results/data/hamiltonians_scaling.json/hamiltonians.json \
    --out results/eval/h_cgqe_optimized_scaling.json \
    --top-k 5 \
    --target nvidia --target-option mqpu
```
L-BFGS-B optimization of rotation angles for top-5 generated sequences.

### Step 5: Evaluate H-cGQE (nvidia-mqpu, 3 GPUs)
```bash
$PY src/gqe/eval/evaluate_h_cgqe.py \
    --generated results/inference/h_cgqe_generated_scaling.json \
    --baseline results/baselines/cudaq_gqe_scaling.json \
    --hamiltonians results/data/hamiltonians_scaling.json/hamiltonians.json \
    --out results/eval/h_cgqe_evaluation_scaling.json \
    --target nvidia --target-option mqpu
```
Computes energy expectation values with fixed θ=0.01 for all generated circuits.

## Results Summary

### Evaluation (fixed θ=0.01)

| Molecule | Qubits | Ref (Ha) | GQE (Ha) | H-cGQE (Ha) | Err vs Ref (mHa) |
|---|---|---|---|---|---|
| h2_0.74 | 4 | -1.1373 | -1.1169 | -1.1168 | 20.5 |
| lih_1.6_full | 12 | -7.8823 | -7.8619 | -7.8583 | 24.0 |
| n2_1.1_full | 20 | N/A | -107.497 | -107.432 | — |
| beh2_1.3_full | 14 | -15.5950 | -15.5613 | -15.5555 | 39.6 |
| iodobenzene_cas12 | 12 | -7078.040 | -7078.014 | -7078.009 | 31.0 |
| methyl_iodide_cas12 | 12 | -6889.853 | -6889.842 | -6889.831 | 22.4 |
| imeph_cas12 | 12 | -7190.357 | -7190.337 | -7190.331 | 25.8 |
| phenol_cas12 | 12 | -301.658 | -301.613 | -301.611 | 46.8 |
| lih_1.6_631g | 22 | N/A | -7.9793 | -7.9570 | — |
| n2_1.1_631g_cas8 | 16 | N/A | -108.868 | -108.854 | — |
| h2o_1.0_631g_cas8 | 16 | N/A | -75.979 | -75.959 | — |

### Optimized Coefficients (L-BFGS-B, top-5 sequences)

| Molecule | Qubits | Best Optimized E (Ha) | Operators |
|---|---|---|---|
| h2_0.74 | 4 | -1.1168 | 4 ops (entangled) |
| lih_1.6_full | 12 | -7.8619 | 19 ops (entangled) |
| n2_1.1_full | 20 | -107.459 | 19 ops (entangled) |
| beh2_1.3_full | 14 | -15.5613 | 19 ops (entangled) |
| iodobenzene_cas12 | 12 | -7078.009 | 1 op (IZIZ — collapsed) |
| methyl_iodide_cas12 | 12 | -6889.840 | 19 ops (entangled) |
| imeph_cas12 | 12 | -7190.331 | 1 op (IZIZ — collapsed) |
| phenol_cas12 | 12 | -301.611 | 12 ops (mixed) |
| lih_1.6_631g | 22 | -7.9793 | 19 ops (entangled) |
| n2_1.1_631g_cas8 | 16 | -108.856 | 19 ops (entangled) |
| h2o_1.0_631g_cas8 | 16 | -75.974 | 19 ops (entangled) |

## Key Findings

1. **Scaled from 8 → 22 qubits** — 2.75× increase from Phase 3's max of 8 qubits
2. **6-31G basis works** — 3 new molecules at 16–22 qubits, first time using larger basis set
3. **N₂ at 20 qubits (full STO-3G)** — largest exact statevector simulation in the project
4. **H-cGQE within 20–47 mHa of reference** on molecules with FCI reference
5. **GQE baseline slightly outperforms H-cGQE** (expected — H-cGQE uses fixed θ=0.01, GQE optimizes coefficients)
6. **Diagonal sequence collapse** on iodobenzene and imeph — model generates only 1-operator `IZIZ` sequences
7. **RLQF model transfers entangling operators** to larger systems — 19-operator sequences with X/Y terms (`XYYX`, `XXYY`, `YXXY`) on most molecules
8. **3× L40S mqpu parallelism** — all 50 samples per molecule evaluated across 3 GPUs in parallel

## Known Issues & Fixes Applied

### Pauli Word Padding
The shared RLQF model vocabulary was trained on 8-qubit molecules. When evaluating on larger qubit counts (12–22), Pauli words need padding to match the molecule's qubit count.

**Fix:** Added `_pad_pauli_word()` helper in:
- `src/gqe/models/train_rlqf_h_cgqe.py`
- `src/gqe/eval/optimize_h_cgqe_coefficients.py`
- `src/gqe/eval/evaluate_h_cgqe.py`

```python
def _pad_pauli_word(word: str, n_qubits: int) -> str:
    if len(word) == n_qubits:
        return word
    if len(word) < n_qubits:
        return word + "I" * (n_qubits - len(word))
    return word[:n_qubits]
```

### PyTorch 2.6+ Checkpoint Loading
`torch.load` defaults to `weights_only=True` in PyTorch 2.6+, breaking loading of checkpoints with `PosixPath` objects.

**Fix:** Added `weights_only=False` to all `torch.load` calls in:
- `src/gqe/models/infer_h_cgqe.py`
- `src/gqe/models/train_h_cgqe.py`
- `src/gqe/models/train_rlqf_h_cgqe.py`
- `src/gqe/eval/optimize_h_cgqe_coefficients.py`

### MPI Oversubscribe
When running `mpiexec -np 3` on AIRE with `--cpus-per-task=8`, MPI may complain about insufficient slots.

**Fix:** Use `mpiexec --oversubscribe -np 3 ...`

### Hamiltonian Output Path
`generate_hamiltonians.py --out results/data/hamiltonians_scaling.json` creates a **directory** with `hamiltonians.json` inside it. All downstream commands must use:
```
results/data/hamiltonians_scaling.json/hamiltonians.json
```

## Next Steps

### Option A: Multi-GPU mgpu for 30+ qubits
Use `nvidia-mgpu` to pool 3× L40S memory (144 GB) for single large statevector:
```bash
mpiexec --oversubscribe -np 3 $PY script.py --target nvidia --target-option mgpu,fp32
```
Target molecules: full N₂ in 6-31G (36 qubits), full iodobenzene (40+ qubits)

### Option B: TensorNet for 50+ qubits
Use `tensornet` or `tensornet-mps` for shallow-circuit large-qubit validation:
```bash
$PY script.py --target tensornet
```

### Option C: Retrain RLQF on scaling dataset
Fix diagonal collapse on iodobenzene/imeph by retraining with the larger molecule set.

### Option D: Generate scaling report
Create plots comparing Phase 3 (8q) vs scaling (12–22q) results and update README.

## How to Reproduce

```bash
# On AIRE GPU node with 3× L40S:
export PY=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python
cd /scratch/kcwp264/Conditional-GQE_materials

# Run full pipeline (skips Hamiltonian generation if already exists):
bash scripts/run_scaling_3gpu.sh

# Or run individual steps — see Pipeline Steps above
```

## Related Files in the Codebase

- `src/gqe/data/generate_hamiltonians.py` — Hamiltonian generation from YAML config
- `src/gqe/baselines/run_cudaq_gqe.py` — GQE baseline with CUDA-Q (supports `--target`, `--target-option`, `--max-qubits`)
- `src/gqe/models/infer_h_cgqe.py` — H-cGQE inference (supports `--molecules`, `--max-pauli-len`)
- `src/gqe/eval/optimize_h_cgqe_coefficients.py` — L-BFGS-B coefficient optimization (supports `--top-k`, `--target`)
- `src/gqe/eval/evaluate_h_cgqe.py` — Final evaluation (requires `--generated`, `--baseline`, `--hamiltonians`, `--out`)
- `configs/experiment_phase3.yaml` — Phase 3 config (8 qubits max, for comparison)
- `configs/experiment_scaling.yaml` — Scaling config (12–22 qubits)
