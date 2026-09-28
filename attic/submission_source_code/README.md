# Quantum-Buddies | H-cGQE: Hierarchical Conditional Generative Quantum Eigensolver

**Team Name:** Quantum-Buddies  
**Project Title:** H-cGQE: Hierarchical Conditional Generative Quantum Eigensolver for Electronic Structure Calculations  
**Challenge Track:** Mitsubishi Chemical & AIST — GIC 2026 Phase 3  
**GitHub:** https://github.com/Quantum-Buddies/Conditional_GQE  

---

## Launch on qBraid

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/Quantum-Buddies/Conditional_GQE.git)

Click the button above to clone this repository into qBraid Lab and reproduce all results. qBraid provides CUDA-Q and PyTorch pre-installed — no external configuration needed.

---

## Overview

H-cGQE is a two-stage pipeline for quantum chemistry on NISQ hardware:

1. **Autoregressive Circuit Synthesis:** A GPT-2 style transformer generates Pauli operator sequences conditioned on molecular Hamiltonian features (Pauli term embeddings + coefficients).
2. **Classical Coefficient Optimization:** L-BFGS-B optimizes rotation angles (thetas) for the generated operators using CUDA-Q statevector simulation on NVIDIA GPUs.

Training uses supervised fine-tuning followed by **DAPO** (Decoupled Clip + Dynamic Sampling Policy Optimization) reinforcement learning with a **MAP-Elites** quality-diversity archive. The RL reward directly optimizes for molecular ground-state energy.

### Pipeline Architecture

```
Molecular Hamiltonian (OpenFermion + PySCF)
         │
         ▼
  ┌──────────────────┐
  │  H-cGQE Transformer │  Stage 1: Autoregressive circuit synthesis
  │  (GPT-2 style)       │  Input: Pauli terms + coefficients
  │  SFT → DAPO RL       │  Output: Operator sequence (Pauli words)
  └──────────┬───────┘
             │
             ▼
  ┌──────────────────┐
  │  L-BFGS-B Optimizer  │  Stage 2: Classical coefficient optimization
  │  CUDA-Q statevector  │  Input: Operator sequence + Hamiltonian
  │  nvidia-mqpu target  │  Output: Optimal thetas + ground-state energy
  └──────────┬───────┘
             │
             ▼
  ┌──────────────────┐
  │  QWC Grouping + SQD  │  Measurement reduction + noise mitigation
  │  QASM export         │  Output: QPU-ready circuits + SQD energies
  └──────────────────┘
```

### Key Results

| Experiment | Metric | Value |
|---|---|---|
| H-cGQE on CH3I (8q, GPU) | Error vs FCI | **0.629 mHa** (chemical accuracy) |
| H-cGQE on H2@0.74 (4q, GPU) | Error vs FCI | **0.15 mHa** |
| Cepheus QPU SQD | Molecules | **12 molecules** (8–28q) on Rigetti Cepheus-1-108Q |
| Cepheus Best SQD | Error vs FCI | **13.9 mHa** (methyl iodide, 12q) |
| Cepheus EUV Photoresist | Molecules | 8 (methyl iodide, iodobenzene, phenol, o-cresol, anisole, toluene, benzene, imeph) |
| QSCI Scaling | Max qubits | **40 qubits** (benzene CAS(20e,20o)) |
| IQM Emerald QPU | State fidelity | **87.5%** |
| FMO2 Iodobenzene | Solver error | 26.25 mHa |
| Classical: HF vs H-cGQE (H2) | Error reduction | 20.5 mHa → 0.15 mHa |
| VQE: HEA-VQE vs H-cGQE (CH3I) | Error reduction | 987.8 mHa → 0.63 mHa |
| VQE: ADAPT-VQE vs H-cGQE (H2) | Comparable | 0.0002 mHa vs 0.15 mHa |

---

## Setup Instructions

### Environment Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | >= 3.11 | Tested on 3.11 and 3.12 |
| PyTorch | >= 2.7 (CUDA 12.6) | Transformer training, `torch.compile` |
| CUDA-Q | >= 0.10 | Statevector simulation, GQE solvers |
| OpenFermion | >= 1.7 | Fermionic → qubit mapping (Jordan-Wigner) |
| OpenFermionPySCF | >= 0.5 | PySCF backend for OpenFermion |
| PySCF | >= 2.13 | HF, CCSD, FCI reference energies |
| qiskit | >= 2.0 | Circuit construction, ADAPT-VQE baseline |
| qbraid-sdk | >= 0.9 | QPU job submission (Rigetti, IQM) |
| huggingface_hub | >= 0.24 | Model checkpoint auto-download |
| matplotlib | >= 3.9 | Report figures |
| fpdf | >= 2.8 | PDF report generation |
| tqdm | >= 4.66 | Progress bars |
| pyyaml | >= 6.0 | Config parsing |

**Full list in `requirements.txt`.** Install with:

```bash
pip install -r requirements.txt
```

### Installation on qBraid (zero external configuration)

qBraid Lab comes with CUDA-Q and PyTorch pre-installed. The only setup needed:

```bash
# 1. Clone (or click "Launch on qBraid" above)
git clone https://github.com/Quantum-Buddies/Conditional_GQE.git
cd Conditional_GQE

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download model checkpoints (~70 MB essential, auto-downloads on first use)
python scripts/download_models.py --only essential

# 4. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cudaq; print(f'CUDA-Q {cudaq.__version__}')"
python -c "import openfermion; print(f'OpenFermion {openfermion.__version__}')"
```

### Critical Import Order

**Always import `torch` before `cudaq`.** Both embed LLVM; reversing causes SIGABRT (exit code 134).

```python
import torch          # FIRST — loads Triton's LLVM
# ... torch.compile() calls here ...
import cudaq          # SECOND — safe after Triton LLVM is loaded
```

This is handled automatically in `train_rl_dapo.py` via lazy `_ensure_cudaq()` import.

### Download Model Checkpoints

All trained model checkpoints are hosted on Hugging Face:

[https://huggingface.co/Quantum-Buddies/Conditional-GQE-models](https://huggingface.co/Quantum-Buddies/Conditional-GQE-models)

```bash
# Download all models (~450 MB)
python scripts/download_models.py

# Download only essential models (~70 MB)
python scripts/download_models.py --only essential

# Check which models are missing locally
python scripts/download_models.py --check
```

**Key checkpoints:**

| File | Size | Description |
|---|---|---|
| `h_cgqe_model_b200_sft.pt` | 31 MB | SFT warm-start checkpoint |
| `h_cgqe_model_rl_qd_scratch.pt` | 40 MB | RL trained model (main checkpoint) |
| `h_cgqe_model.pt` | 6.1 MB | Base H-cGQE model |
| `chemistry_encoder.pt` | 114 KB | Chemistry GNN encoder |
| `gqe_supervised_dataset.pt` | 15 MB | Supervised training dataset |

Scripts that load checkpoints will **auto-download from Hugging Face** if the local file is missing. No manual download required unless you prefer to pre-fetch.

---

## Step-by-Step Instructions to Run on qBraid

### Quick Start: Full Pipeline (one command)

```bash
bash scripts/run_all_reproducible.sh all
```

This runs all stages: setup → chemistry → training → evaluation → baselines → figures → report. GPU stages are free on qBraid. The QPU stage is skipped by default (run separately with `bash scripts/run_all_reproducible.sh qpu`).

### Individual Stages

#### Stage 1: Generate Molecular Hamiltonians

```bash
python src/gqe/data/generate_hamiltonians.py \
    --config configs/gic2026_molecules.yaml \
    --out results/data/hamiltonians_gic2026/
```

- **Input:** `configs/gic2026_molecules.yaml` (molecular geometries, basis sets, active spaces)
- **Output:** `results/data/hamiltonians_gic2026/hamiltonians.json` — Pauli term lists, qubit counts, FCI/HF reference energies
- **Time:** ~5 minutes on CPU
- **Cost:** Free

#### Stage 2: Train H-cGQE Transformer (Supervised + RL)

```bash
# 2a. Supervised fine-tuning (warm-start)
python src/gqe/models/train_supervised.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --config configs/gic2026_molecules.yaml \
    --out results/train/h_cgqe_model_sft.pt \
    --use-cuda

# 2b. DAPO reinforcement learning (energy-based reward)
python src/gqe/models/train_rl_dapo.py \
    --checkpoint results/train/h_cgqe_model_sft.pt \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/train/h_cgqe_model_rl.pt \
    --n-epochs 50 \
    --target nvidia \
    --max-qubits 24
```

- **Input:** Hamiltonian JSON + SFT checkpoint
- **Output:** RL-trained model checkpoint + MAP-Elites archive + training metrics
- **Time:** ~2 hours (SFT) + ~8 hours (RL, 50 epochs) on single GPU
- **Cost:** Free (uses local GPU or qBraid free tier)
- **Note:** The `--target nvidia` flag uses CUDA-Q's single-GPU statevector backend. For multi-GPU, use `--target nvidia-mqpu` (requires MPI).

#### Stage 3: Evaluate and Optimize Circuits

```bash
# 3a. Generate circuits from trained model
python src/gqe/eval/evaluate_h_cgqe.py \
    --checkpoint results/train/h_cgqe_model_rl.pt \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/eval/h_cgqe_evaluation.json

# 3b. L-BFGS-B coefficient optimization
python src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --evaluation results/eval/h_cgqe_evaluation.json \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --target nvidia-mqpu \
    --out results/eval/h_cgqe_optimized.json
```

- **Input:** Trained checkpoint + Hamiltonian data
- **Output:** Optimized energies, operator sequences, theta values per molecule
- **Time:** ~30 minutes (evaluation) + ~1 hour (optimization) on GPU
- **Cost:** Free

#### Stage 4: Validate on qBraid Simulator (free, no credits)

```bash
python scripts/validate_on_qbraid.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --optimized results/eval/h_cgqe_optimized.json \
    --out results/eval/qbraid_validation_report.json \
    --shots 2000
```

- **Input:** Optimized results JSON + Hamiltonian data
- **Output:** Validation report with simulator energies vs GPU energies
- **Time:** ~10 minutes
- **Cost:** Free (qBraid QIR simulator, up to 30 qubits)

#### Stage 5: QPU Submission + SQD Post-Processing (uses credits)

```bash
# 5a. Export SQD manifests (no credits needed)
python scripts/rl_optimize_and_submit.py \
    --checkpoint results/train/h_cgqe_model_rl.pt \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --molecules h2 lih beh2 methyl_iodide iodobenzene \
    --export-only

# 5b. Submit to Rigetti Cepheus (AWS Braket via qBraid)
python scripts/submit_sqd_to_cepheus.py \
    --manifests results/qpu/sqd_manifests/ \
    --device aws:rigetti:qpu:cepheus-1-108q \
    --shots 4096

# 5c. Retrieve results and run SQD post-processing
python scripts/retrieve_and_sqd.py \
    --ledger results/qpu/qpu_ledger.sqlite \
    --out results/qpu/cepheus_rl_sqd_results.json
```

- **Input:** SQD manifests with circuit QASM + measurement groupings
- **Output:** QPU bitstring counts + SQD-recovered energies with error analysis
- **Cost:** ~204 qBraid credits/molecule at 4096 shots (30 credits/task + 0.0425 credits/shot)
- **Note:** Steps 5b and 5c can be run separately (async submission + retrieval workflow)

#### Stage 6: QSCI Scaling (4→40 qubits)

```bash
python src/gqe/eval/qsci.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --backend nvidia \
    --max-qubits 40 \
    --out results/phase3_final/qsci_scaling.json
```

- **Input:** Hamiltonian records for molecules 4–40 qubits
- **Output:** QSCI energies, bitstring counts, runtime per molecule
- **Time:** ~20 minutes on GPU
- **Cost:** Free

#### Stage 7: FMO2 Fragmentation

```bash
# 7a. Generate fragment + dimer Hamiltonians
python scripts/generate_fmo2_fragments.py

# 7b. Run FMO2 exact (classical reference)
python src/gqe/eval/run_fmo2.py \
    --fragments results/data/fragments/monomers.json \
    --dimers results/data/fragments/dimers.json \
    --method exact \
    --out results/phase3_final/fmo/fmo2_exact.json

# 7c. Run FMO2 with H-cGQE
python src/gqe/eval/run_fmo2.py \
    --fragments results/data/fragments/monomers.json \
    --dimers results/data/fragments/dimers.json \
    --method hcgqe \
    --checkpoint results/train/h_cgqe_model_rl.pt \
    --out results/phase3_final/fmo/fmo2_hcgqe.json
```

- **Input:** Monomer + dimer Hamiltonian files
- **Output:** FMO2-reconstructed energies with error decomposition
- **Time:** ~15 minutes
- **Cost:** Free

#### Stage 8: Classical + VQE Baselines

```bash
# CUDA-Q GQE baseline
python src/gqe/baselines/run_cudaq_gqe.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/baselines/cudaq_gqe.json

# HEA-VQE baseline
python src/gqe/baselines/run_cudaq_vqe.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --mode cudaq_hwe_vqe \
    --out results/baselines/cudaq_vqe.json

# ADAPT-VQE baseline (small molecules only)
python src/gqe/baselines/run_cudaq_vqe.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --mode adapt_vqe \
    --max-qubits 4 \
    --out results/baselines/adapt_vqe_h2.json

# Exact diagonalization (FCI reference)
python src/gqe/baselines/run_exact_diagonalization.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/baselines/exact_diagonalization.json

# Build comparison table
python scripts/build_gic_benchmark.py \
    --consolidated results/phase3_final/consolidated_results_gic2026.json \
    --vqe results/baselines/cudaq_vqe.json \
    --adapt results/baselines/adapt_vqe_h2.json \
    --gqe results/baselines/cudaq_gqe.json \
    --out results/phase3_final/classical_baseline_comparison.json
```

- **Input:** Hamiltonian JSON
- **Output:** Baseline energies in JSON format
- **Time:** ~1 hour total
- **Cost:** Free

#### Stage 9: Export QASM Circuits for Quaggle/IBM

```bash
# Export all GIC molecules as OpenQASM 2.0 for Quaggle or IBM QPU
python scripts/export_quaggle_demo.py \
    --optimized results/eval/h_cgqe_optimized_gic2026.json \
    --molecule h2_0.74 \
    --hamiltonians results/data/hamiltonians_merged.json \
    --out results/quaggle/
```

- **Input:** Optimized results JSON (operators + thetas)
- **Output:** `{molecule}_gqe_demo.qasm` (with measurements) + `{molecule}_gqe_ansatz.qasm` (without) + `{molecule}_gqe_metadata.json`
- **Available circuits:** H2 (5 bond lengths + 20-op deep), LiH (4 bond lengths), N2, methyl_iodide, iodobenzene, phenol, IMePh — all in `results/quaggle/`

#### Stage 10: Generate Report

```bash
# PDF report
python scripts/generate_submission_pdf.py

# DOCX report
python scripts/generate_phase3_report_docx.py

# Figures
python scripts/plot_phase3_report_figures.py
```

- **Output:** `submission/Quantum-Buddies_Mitsubishi_Chemical_Phase3.pdf`, `submission/Quantum-Buddies_Phase3_Write-Up.docx`, `results/phase3_final/figures/*.png`

---

## Expected Inputs/Outputs

| Step | Input | Output | File |
|---|---|---|---|
| Hamiltonian generation | `configs/gic2026_molecules.yaml` | Pauli terms + FCI/HF energies | `results/data/hamiltonians_gic2026/hamiltonians.json` |
| Supervised training | Hamiltonian JSON | SFT checkpoint | `results/train/h_cgqe_model_sft.pt` |
| RL training | SFT checkpoint + Hamiltonians | RL checkpoint + MAP-Elites archive | `results/train/h_cgqe_model_rl.pt` |
| Evaluation | RL checkpoint + Hamiltonians | Operator sequences + raw energies | `results/eval/h_cgqe_evaluation.json` |
| Coefficient optimization | Evaluation JSON | Optimized thetas + energies | `results/eval/h_cgqe_optimized.json` |
| qBraid validation | Optimized JSON + Hamiltonians | Simulator validation report | `results/eval/qbraid_validation_report.json` |
| QPU submission | SQD manifests | QPU counts + SQD energies | `results/qpu/cepheus_rl_sqd_results.json` |
| QSCI scaling | Hamiltonian JSON | QSCI energies (4–40q) | `results/phase3_final/qsci_scaling.json` |
| FMO2 | Fragment/dimer Hamiltonians | FMO2-reconstructed energies | `results/phase3_final/fmo/fmo2_exact.json` |
| Baselines | Hamiltonian JSON | HF/VQE/GQE/FCI comparison | `results/phase3_final/classical_baseline_comparison.json` |
| QASM export | Optimized JSON | OpenQASM 2.0 circuits | `results/quaggle/{molecule}_gqe_demo.qasm` |
| Report | All result JSONs | PDF + DOCX | `submission/Quantum-Buddies_Mitsubishi_Chemical_Phase3.pdf` |

---

## Source Code Organization

```
Conditional_GQE/
├── src/gqe/
│   ├── models/                    # H-cGQE transformer + training
│   │   ├── h_cgqe_transformer.py  # GPT-2 style transformer (encoder-decoder)
│   │   ├── train_supervised.py    # Supervised fine-tuning (SFT warm-start)
│   │   ├── train_rl_dapo.py       # DAPO RL training (energy-based reward)
│   │   ├── infer_h_cgqe.py        # Autoregressive inference + sampling
│   │   ├── chemistry_encoder.py   # GNN encoder for molecular graphs
│   │   └── train_chemistry_encoder.py
│   ├── data/
│   │   ├── generate_hamiltonians.py  # OpenFermion + PySCF → Pauli terms
│   │   ├── fragmentation.py          # FMO2 fragment generation
│   │   ├── graph_dataset.py          # Atom-level graph samples
│   │   └── prepare_gqe_dataset.py    # Training data preparation
│   ├── eval/
│   │   ├── evaluate_h_cgqe.py           # Circuit evaluation (CUDA-Q)
│   │   ├── optimize_h_cgqe_coefficients.py  # L-BFGS-B optimization
│   │   ├── qbraid_backend.py            # QWC grouping + qBraid QPU submission
│   │   ├── qsci.py                      # QSCI scaling (4→40 qubits)
│   │   ├── sqd.py                       # Sample-based Quantum Diagonalization
│   │   ├── run_fmo2.py                  # FMO2 fragmentation evaluation
│   │   ├── submit_qpu.py               # QPU submission with preflight checks
│   │   └── run_mps_scaling.py          # MPS backend for >24q
│   ├── baselines/
│   │   ├── run_cudaq_gqe.py            # NVIDIA CUDA-Q GQE baseline
│   │   ├── run_cudaq_vqe.py            # HEA-VQE baseline
│   │   ├── run_adapt_vqe.py            # ADAPT-VQE baseline
│   │   └── run_exact_diagonalization.py  # FCI reference
│   ├── common/
│   │   ├── operator_pool.py            # UCCSD fermionic excitation pool
│   │   ├── hamiltonian_utils.py        # Hamiltonian loading/matching
│   │   ├── ensure_checkpoint.py        # Auto-download from HuggingFace
│   │   └── tapering.py                 # Z2 symmetry tapering
│   └── rl/
│       ├── map_elites.py               # MAP-Elites quality-diversity archive
│       └── energy_cache.py             # Cached energy evaluations
├── scripts/
│   ├── run_all_reproducible.sh         # One-command full pipeline
│   ├── download_models.py              # HuggingFace checkpoint download
│   ├── export_quaggle_demo.py          # Export QASM for Quaggle/IBM
│   ├── validate_on_qbraid.py           # Free simulator validation
│   ├── submit_sqd_to_cepheus.py        # Rigetti Cepheus QPU submission
│   ├── retrieve_and_sqd.py             # Async QPU result retrieval + SQD
│   ├── rl_optimize_and_submit.py       # RL checkpoint → QPU manifest
│   ├── generate_fmo2_fragments.py      # FMO2 fragment Hamiltonian generation
│   ├── compute_classical_baselines.py  # HF/CCSD/FCI via PySCF
│   ├── generate_submission_pdf.py      # PDF report generation
│   ├── generate_phase3_report_docx.py  # DOCX report generation
│   ├── plot_phase3_report_figures.py   # All report figures
│   └── launch_b200_training.sh         # B200/GB200 training launcher
├── configs/
│   ├── gic2026_molecules.yaml          # 35 GIC molecules (4–28q)
│   └── experiment.yaml                 # Model hyperparameters
├── results/                             # All result JSONs, figures, QASM
│   ├── data/                            # Hamiltonians, fragments
│   ├── eval/                            # Evaluation + optimization results
│   ├── train/                           # Model checkpoints
│   ├── baselines/                       # VQE/GQE/FCI baselines
│   ├── qpu/                             # QPU counts + SQD energies
│   ├── quaggle/                         # Exported QASM circuits
│   └── phase3_final/                    # Consolidated results + figures
├── tests/                               # Unit tests
├── requirements.txt                     # Python dependencies
└── REPRODUCIBILITY.md                   # Full reproducibility guide
```

---

## Known Limitations and Assumptions

1. **Diagonal Sequence Collapse:** On larger molecules (LiH 12q, BeH2 14q, N2 20q), the transformer sometimes under-generates entangling X/Y operators and produces commuting Z-only sequences that get trapped at the Hartree-Fock energy baseline. Mitigated by UCCSD operator pool (forces X/Y in every operator), `force_entanglement` decoding constraint, and DAPO RL with entanglement-fraction reward component.

2. **NISQ Noise on QPU:** Rigetti Cepheus QPU results show SQD errors ranging from 13.9 mHa (methyl iodide, 12q) to 130.0 mHa (N2, 20q), correlating with circuit depth and qubit count. Particle number preservation varies from 8% to 71% due to NISQ noise.

3. **Statevector Limit:** Exact statevector simulation is capped at 24 qubits on L40S GPUs (cuStateVec distributed mode segfaults on PCIe-only systems at 25+ qubits). MPS backend used for >24q scaling experiments.

4. **FMO2 Fragmentation:** Current FMO2 demonstration uses 2 fragments (iodobenzene), where the dimer equals the parent molecule. Code for 3+ fragment scaling is implemented but not yet executed at scale.

5. **QSCI in HF Regime:** QSCI results at 28–40 qubits recover Hartree-Fock energy rather than correlated ground state. Deeper entangling circuits are needed for post-HF accuracy at scale.

6. **Transfer Learning:** Evaluation is limited to embedding similarity analysis. Full end-to-end transfer experiments on unseen molecules are planned but not included in this submission.

7. **CUDA-Q + PyTorch LLVM Conflict:** Importing CUDA-Q before PyTorch's `torch.compile` causes LLVM symbol conflicts (SIGABRT, exit code 134). Always import torch first. Handled automatically in `train_rl_dapo.py` via lazy `_ensure_cudaq()`.

8. **qBraid Credit Budget:** ~1,925 credits remaining. 12 QPU submissions completed (8192 shots each, ~4,538 credits spent) on Rigetti Cepheus-1-108Q via AWS Braket (30 credits/task + 0.0425 credits/shot). Each new molecule at 4096 shots costs ~204 credits.

9. **Classical/VQE Baselines:** Hartree-Fock (HF), hardware-efficient VQE (HEA-VQE), ADAPT-VQE, and CCSD/CCSD(T) results are included in `results/phase3_final/classical_baseline_comparison.json`. See `REPRODUCIBILITY.md` for the full comparison table.

---

## Pre-Generated QASM Circuits

The repository includes 16 pre-generated OpenQASM 2.0 circuits in `results/quaggle/` ready for immediate QPU execution:

| File | Molecule | Qubits | Gates | Depth | Energy (Ha) |
|---|---|---|---|---|---|
| `h2_0.74_gqe_demo.qasm` | H2 equilibrium | 4 | 26 | 16 | -1.137 |
| `h2_gqe_demo.qasm` | H2 (20-op deep) | 4 | 148 | 90 | -1.135 |
| `lih_1.6_gqe_demo.qasm` | LiH equilibrium | 8 | 25 | 16 | -7.862 |
| `methyl_iodide_gqe_demo.qasm` | CH3I (EUV resist) | 8 | 21 | 12 | -6889.8 |
| `iodobenzene_gqe_demo.qasm` | C6H5I (EUV resist) | 8 | 29 | 18 | -7078.0 |
| `phenol_gqe_demo.qasm` | C6H5OH (EUV resist) | 8 | 21 | 12 | -301.6 |
| `n2_1.1_gqe_demo.qasm` | N2 | 12 | 47 | 30 | -107.5 |

Import these into Quaggle's Circuit Builder or Qiskit to run on IBM `ibm_fez` or any QPU.

---

## Reproducibility

All results can be reproduced by running `bash scripts/run_all_reproducible.sh all` or individual stages as listed above. Key result files included in the submission:

| File | Description |
|---|---|
| `results/phase3_final/consolidated_results_gic2026.json` | Consolidated benchmark + QPU results |
| `results/eval/h_cgqe_optimized_gic2026.json` | H-cGQE L-BFGS-B optimized energies (17 molecules) |
| `results/eval/h_cgqe_rl_optimized.json` | RL-optimized circuit energies |
| `results/qpu/cepheus_sqd_energies.json` | Corrected Cepheus QPU SQD energies (12 molecules) |
| `results/phase3_final/classical_baseline_comparison.json` | HF, HEA-VQE, ADAPT-VQE, GQE, H-cGQE comparison |
| `results/baselines/cudaq_vqe.json` | HEA-VQE baseline (5 molecules) |
| `results/baselines/adapt_vqe_h2.json` | ADAPT-VQE baseline (H2/LiH) |
| `results/baselines/exact_diagonalization.json` | FCI reference energies |
| `results/data/fragments/ccsd_refs.json` | CCSD/CCSD(T) for FMO2 fragments |
| `results/phase3_final/fmo/` | FMO2 exact and H-cGQE results |
| `results/phase3_final/qpu/qpu_validation_consolidated.json` | QPU validation summary |
| `results/phase3_final/figures/` | All report figures as PNG |
| `results/quaggle/*.qasm` | 16 pre-generated QASM circuits for QPU execution |

**Platform:** qBraid (CUDA-Q + PyTorch pre-installed) + Rigetti Cepheus QPU via AWS Braket.  
**Full guide:** See `REPRODUCIBILITY.md` for detailed environment setup, lock scripts, and per-stage verification.

---

## License

MIT License — see LICENSE file for details.

## Team

- **Quantum-Buddies** — GIC 2026 Phase 3 Submission

## References

- GPT-QE: arXiv:2401.09253 — Original Generative Quantum Eigensolver
- CUDA-Q GQE: NVIDIA CUDA-QX solver examples
- DAPO: Decoupled Clip + Dynamic Sampling Policy Optimization
- MAP-Elites: Quality-diversity optimization for RL
- SQD: Sample-based Quantum Diagonalization (IBM Research)
- QWC Grouping: Qubit-wise commuting Pauli term measurement grouping
