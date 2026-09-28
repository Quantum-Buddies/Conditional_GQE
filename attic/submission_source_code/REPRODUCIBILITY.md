# Reproducibility Guide — Quantum-Buddies GIC 2026 Phase 3

## Quick Start

```bash
# Clone and install
git clone https://github.com/Quantum-Buddies/Conditional_GQE.git
cd Conditional_GQE
pip install -r requirements.txt

# Download model checkpoints (~70 MB essential)
python scripts/download_models.py --only essential

# Run full pipeline (GPU stages only, no QPU credits)
bash scripts/run_all_reproducible.sh all

# Or run individual stages
bash scripts/run_all_reproducible.sh setup
bash scripts/run_all_reproducible.sh chem
bash scripts/run_all_reproducible.sh train
bash scripts/run_all_reproducible.sh eval
bash scripts/run_all_reproducible.sh baselines
bash scripts/run_all_reproducible.sh figures
bash scripts/run_all_reproducible.sh report

# QPU submission (uses ~250 of ~600 remaining credits)
bash scripts/run_all_reproducible.sh qpu
```

## Environment

| Component | Version | Notes |
|---|---|---|
| Python | 3.11+ | Tested on 3.11 and 3.12 |
| PyTorch | 2.7+ (CUDA 12.6) | `pip install torch --index-url https://download.pytorch.org/whl/cu126` |
| CUDA-Q | 0.10+ | `pip install cudaq` (pre-installed on qBraid) |
| OpenFermion | 1.7+ | Fermionic → qubit mapping |
| PySCF | 2.13+ | HF, CCSD, FCI reference energies |
| qiskit | 2.0+ | ADAPT-VQE baseline |
| qBraid SDK | 0.9+ | QPU job submission |

- **Conda env (local)**: `cudaq-env` at `/scratch/kcwp264/.conda_envs/cudaq-env/`
- **Lock script**: `bash scripts/lock_environment.sh` captures git commit, Python/pip versions, GPU info

### Critical Import Order

**Always import `torch` before `cudaq`.** Both embed LLVM; reversing causes SIGABRT.

```python
import torch          # FIRST — loads Triton's LLVM
import cudaq          # SECOND — safe after torch.compile
```

## Hardware

- **Development**: AIRE HPC, 3× NVIDIA L40S (48GB, PCIe-only, no NVLink)
- **qBraid GPU profiles**: L40S, H200, B200 available on-demand
- **QPU access**: Rigetti Cepheus, IQM Emerald via qBraid SDK
- **Statevector limit**: 24 qubits on L40S (cuStateVec distribution threshold = 25)
- **MPS backend**: `tensornet-mps` for >24 qubit systems (single-GPU mode on L40S)

## Determinism

- All experiments use seed=42
- PyTorch deterministic mode enabled for inference
- CUDA-Q simulator is deterministic for fixed seed
- L-BFGS-B optimization uses deterministic initialization

## Pipeline Stages

### Stage 1: Hamiltonian Generation

```bash
python src/gqe/data/generate_hamiltonians.py \
    --config configs/gic2026_molecules.yaml \
    --out results/data/hamiltonians_gic2026/
```

**Input:** `configs/gic2026_molecules.yaml` (21 molecules, STO-3G basis)
**Output:** `results/data/hamiltonians_gic2026/hamiltonians.json`
**Expected:** 21 molecules, 4–14 qubits, with Pauli terms + FCI/HF reference energies
**Runtime:** ~5 minutes (CPU, PySCF SCF calculations)

### Stage 2: Supervised Training (SFT)

```bash
python src/gqe/models/train_supervised.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --config configs/gic2026_molecules.yaml \
    --out results/train/h_cgqe_model_sft.pt \
    --use-cuda
```

**Output:** `results/train/h_cgqe_model_sft.pt` (31 MB)
**Runtime:** ~10 minutes on 1× L40S

### Stage 3: DAPO Reinforcement Learning

```bash
python src/gqe/models/train_rl_dapo.py \
    --checkpoint results/train/h_cgqe_model_sft.pt \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/train/h_cgqe_model_rl.pt \
    --n-epochs 50 \
    --target nvidia \
    --max-qubits 24
```

**Output:** `results/train/h_cgqe_model_rl.pt` (40 MB) + MAP-Elites archive
**Runtime:** ~2 hours on 1× L40S (3× L40S with `nvidia-mqpu` target)
**Key flags:** `--gate-auxiliary-rewards` (prevents reward hacking), `--force-entanglement` (prevents Z-only collapse)

### Stage 4: Evaluation + Coefficient Optimization

```bash
python src/gqe/eval/evaluate_h_cgqe.py \
    --checkpoint results/train/h_cgqe_model_rl.pt \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/eval/h_cgqe_evaluation.json

python src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --evaluation results/eval/h_cgqe_evaluation.json \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --target nvidia-mqpu \
    --out results/eval/h_cgqe_optimized.json
```

**Expected:** Chemical accuracy (<1.6 mHa) on H2 and CH3I
**Runtime:** ~30 minutes on 3× L40S

### Stage 5: Classical + VQE Baselines

```bash
# CUDA-Q GQE baseline
python src/gqe/baselines/run_cudaq_gqe.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/baselines/cudaq_gqe.json

# HEA-VQE (hardware-efficient ansatz)
python src/gqe/baselines/run_cudaq_vqe.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --mode cudaq_hwe_vqe \
    --out results/baselines/cudaq_vqe.json

# ADAPT-VQE (small molecules only, ≤4q)
python src/gqe/baselines/run_cudaq_vqe.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --mode adapt_vqe \
    --max-qubits 4 \
    --out results/baselines/adapt_vqe_h2.json
```

**Expected results (error vs FCI in mHa):**

| Molecule | HF | HEA-VQE | ADAPT-VQE | GQE | H-cGQE |
|---|---|---|---|---|---|
| H2 (0.74 Å) | 20.5 | 606,511 | 0.0002 | 20.4 | 0.15 |
| LiH (1.6 Å) | 20.5 | 1,822,901 | 0.0001 | 1.8 | 1.85 |
| CH3I | — | 987.8 | — | 2.6 | 0.63 |
| Iodobenzene | 252.0 | 626.4 | — | 2.0 | 2.97 |
| BeH2 | 202.1 | 2,181,723 | — | 33.8 | 34.8 |

**Key observations:**
- HEA-VQE suffers barren plateaus on >4q (errors >600,000 mHa)
- ADAPT-VQE is near-exact but only feasible on ≤4q (gradient-based operator selection)
- H-cGQE matches or improves on CUDA-Q GQE with a learned (not random) operator pool

### Stage 6: QPU Submission (uses qBraid credits)

```bash
# Export SQD manifests (free)
python scripts/rl_optimize_and_submit.py \
    --checkpoint results/train/h_cgqe_model_rl.pt \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --molecules h2 lih beh2 methyl_iodide iodobenzene \
    --export-only

# Submit to Rigetti Cepheus (~50 credits/molecule)
python scripts/submit_sqd_to_cepheus.py \
    --manifests results/qpu/sqd_manifests/ \
    --device aws:rigetti:qpu:cepheus-1-108q \
    --shots 4096

# Retrieve + SQD post-process
python scripts/retrieve_and_sqd.py \
    --ledger results/qpu/qpu_ledger.sqlite \
    --out results/qpu/cepheus_rl_sqd_results.json
```

**Credit budget:** ~1,925 remaining. Cost per molecule: 30 credits/task + 0.0425 credits/shot. At 8192 shots = ~378 credits/molecule (~5 more possible). At 4096 shots = ~204 credits/molecule (~9 more possible).
**Expected:** Best QPU error 13.9 mHa (methyl iodide). Particle-number preservation 8–71%.

### Stage 7: Report Generation

```bash
python scripts/plot_phase3_report_figures.py
python scripts/generate_phase3_report_docx.py
```

**Output:** `proposals/Ryoushi_Quantum_Buddies_Phase3_Report.docx` (9 figures, 3 tables)

## Model Checkpoints

Hosted on Hugging Face: [Quantum-Buddies/Conditional-GQE-models](https://huggingface.co/Quantum-Buddies/Conditional-GQE-models)

| File | Size | Description |
|---|---|---|
| `h_cgqe_model_b200_sft.pt` | 31 MB | SFT warm-start checkpoint |
| `h_cgqe_model_rl_qd_scratch.pt` | 40 MB | RL trained model (main checkpoint) |
| `h_cgqe_model.pt` | 6.1 MB | Base H-cGQE model |
| `chemistry_encoder.pt` | 114 KB | Chemistry GNN encoder |
| `gqe_supervised_dataset.pt` | 15 MB | Supervised training dataset |

## Key Result Files

| File | Description |
|---|---|
| `results/phase3_final/consolidated_results_gic2026.json` | Main benchmark (17 molecules, GPU + QPU) |
| `results/phase3_final/classical_baseline_comparison.json` | HF, HEA-VQE, ADAPT-VQE, GQE, H-cGQE comparison |
| `results/baselines/cudaq_vqe.json` | HEA-VQE on 5 molecules |
| `results/baselines/adapt_vqe_h2.json` | ADAPT-VQE on H2/LiH |
| `results/baselines/exact_diagonalization.json` | FCI reference energies |
| `results/data/fragments/ccsd_refs.json` | CCSD/CCSD(T) for FMO2 fragments |
| `results/qpu/cepheus_sqd_energies.json` | QPU SQD energies (12 molecules) |
| `results/phase3_final/ablation_sft_vs_rl.json` | SFT vs RL ablation |
| `results/phase3_final/noise_mitigation_summary.json` | QWC grouping summary |
| `results/phase3_final/figures/` | All report PNG figures |
| `proposals/Ryoushi_Quantum_Buddies_Phase3_Report.docx` | Editable Word report |

## Result Manifests

Every result JSON in `results/phase3_final/` contains a manifest with:
- `git_commit`: Full SHA at time of generation
- `timestamp_utc`: ISO 8601 timestamp
- `molecule`, `geometry`, `basis`, `active_electrons`, `active_spatial_orbitals`, `qubits`
- `backend`, `device_id`, `shots`, `seed`
- `logical_depth`, `transpiled_depth`, `two_qubit_gates`
- `energy_hartree`, `reference_energy_hartree`, `error_mha`
- `wall_time_seconds`, `status`

Run `python scripts/lock_environment.sh` to capture environment metadata before experiments.

## QPU Preflight

Before any QPU execution:
```bash
python scripts/qpu_preflight.py --dry-run
```
This lists available devices, estimates credit costs, and saves a sanitized manifest. No QPU credits are spent in dry-run mode.

## Cached Results

Expensive runs (QPU, large MPS) have cached results in `results/phase3_final/`. Judges can reproduce from cached data without re-running on QPU.

## Known Issues

1. **CUDA-Q + torch.compile LLVM conflict:** Always `import torch` before `import cudaq`. Fixed via lazy `_ensure_cudaq()` in `train_rl_dapo.py`.
2. **L40S distributed statevector segfault:** Keep `--max-qubits 24` on PCIe-only L40S. cuStateVec distributed mode (threshold=25) segfaults due to broken CUDA IPC.
3. **Operator pool collapse:** Original GQE baselines used Hamiltonian's own Pauli terms → Z-only diagonal collapse. Fixed with UCCSD operator pool in `src/gqe/common/operator_pool.py`.
4. **QWC bit ordering:** Pauli position q maps to Qiskit qubit n-1-q, which is bitstring index q. Fixed in `src/gqe/eval/qbraid_backend.py`.
5. **Fixed-theta proxy flatness:** RL reward at fixed theta=0.01 produces no ranking signal. Spearman ρ=0.23 (p=0.42). Fix: truncated L-BFGS-B (3-5 steps) during reward computation.
6. **N₂ (12-20 qubits):** Not converged — strongly correlated system, requires larger active space or different operator pool.
7. **BeH₂ (14 qubits):** Not converged — similar diagonal collapse issue.
8. **IMePh (8 qubits, test set):** 24.63 mHa error — unseen EUV molecule, not used in training.
9. **MPS on L40S:** Single-GPU only (pip-installed CUDA-Q does not support MPI tensornet).

## Verification Commands

```bash
# Verify all figures exist
ls results/phase3_final/figures/fig_*.png | wc -l  # should be 9+

# Verify checkpoint integrity
python scripts/download_models.py --check

# Verify Hamiltonian data
python -c "import json; d=json.load(open('results/data/hamiltonians_gic2026/hamiltonians.json')); print(f'{len(d)} molecules')"

# Verify consolidated results
python -c "import json; d=json.load(open('results/phase3_final/consolidated_results_gic2026.json')); print(d.get('summary',{}))"

# Verify classical baseline comparison
python -c "import json; d=json.load(open('results/phase3_final/classical_baseline_comparison.json')); print(f'{len(d.get(\"molecules\",[]))} molecules compared')"
```
