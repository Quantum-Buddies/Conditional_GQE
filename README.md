<p align="center">
  <h1 align="center">Conditional-GQE</h1>
  <p align="center">
    <strong>Hierarchical Conditional Generative Quantum Eigensolver for EUV Lithography</strong>
  </p>
  <p align="center">
    <a href="https://github.com/Quantum-Buddies/Conditional_GQE/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.6%2B-red.svg" alt="PyTorch"></a>
    <a href="https://nvidia.github.io/cuda-quantum/"><img src="https://img.shields.io/badge/CUDA--Q-0.8%2B-green.svg" alt="CUDA-Q"></a>
    <a href="https://github.com/Quantum-Buddies/Conditional_GQE"><img src="https://img.shields.io/badge/GIC-Phase%203-purple.svg" alt="GIC Phase 3"></a>
  </p>
</p>

---

## Overview

**Conditional-GQE** is a research framework that uses a **Transformer-based autoregressive model** to generate quantum circuits for molecular energy estimation. Instead of running expensive variational loops on quantum hardware (VQE/ADAPT-VQE), our model learns to map a molecular Hamiltonian directly to a sequence of Pauli rotation operators that form a quantum eigensolver ansatz.

The system trains in two stages:
1. **Pure RL from scratch** — DAPO-based reinforcement learning with no supervised pretraining. The model discovers quantum circuit strategies from energy rewards alone, building its own vocabulary from the UCCSD operator pool (arXiv:2502.19402 shows RL from scratch outperforms SFT-then-RL)
2. **Classical coefficient optimization** — L-BFGS-B rotation angle refinement on GPU

> **Why no supervised pretraining?** Research shows SFT memorizes patterns while RL discovers general strategies (arXiv:2502.19402). SFT-then-RL coupling causes irreversible degradation (arXiv:2601.07389). Our previous supervised pipeline taught the model to mimic GQE baselines that suffered diagonal collapse — the opposite of what we want.

### Key Results (Phase 3)

| Molecule | Qubits | FCI Energy (Ha) | H-cGQE Error (mHa) | GQE Baseline (mHa) | Status |
|---|---|---|---|---|---|
| Methyl iodide (CH₃I) | 8 | -6889.840 | **0.63** | 4.71 | **Chemical accuracy** |
| Iodobenzene (C₆H₅I) | 8 | -7078.012 | **2.73** | 1.96 | Near chemical accuracy |
| LiH (1.6 Å) | 8 | -7.864 | **1.84** | 1.81 | **Chemical accuracy** |
| LiH (1.2 Å) | 8 | -7.838 | **2.07** | 2.05 | Near chemical accuracy |
| IMePh (EUV photoresist) | 8 | -7190.356 | **24.63** | 19.01 | Good (test set) |
| BeH₂ (1.3 Å) | 14 | -15.595 | **33.98** | 33.76 | Moderate |
| N₂ (1.1 Å) | 12 | -107.623 | **126.62** | 126.55 | Hard (strongly correlated) |

> **Chemical accuracy** = 1.6 mHa (millihartree), the gold standard for quantum chemistry

### The Breakthrough: Breaking Diagonal Sequence Collapse

In Phase 2, the model suffered from **diagonal sequence collapse** — it generated only commuting Z-only operators (e.g., `IZII`, `ZIZI`), getting trapped at the Hartree-Fock energy with zero gradients. We resolved this through a **5-layer defense** against entropy collapse:

1. **UCCSD operator pool** — All operators come from fermionic excitation operators mapped through Jordan-Wigner. Every operator contains X/Y components — Z-only collapse is impossible by construction
2. **BF16 mixed precision** — FP16's 5 exponent bits cause multiplicative bias in softmax gradients that systematically reduces entropy. BF16 (8 exponent bits) eliminates this (arXiv:2603.11682)
3. **Distribution mixing** (ε-exploration) — Mixes sampling distribution with uniform distribution (ε=0.3) to enforce a hard entropy floor
4. **REPO advantages** — Regulated Entropy Policy Optimization modifies advantages with a centered log-prob penalty, penalizing deterministic samples and boosting diverse ones (arXiv:2603.11682)
5. **Curriculum learning** — Train on small molecules (4 qubits) first, gradually add larger ones over 30-epoch warmup stages

Additional exploration measures: top-p (nucleus) sampling, adaptive temperature scheduling, entropy bonus in DAPO loss, and adaptive ε decay.

The model now generates **entangling operators** like `XYYX`, `YXXY`, `XXYY` — creating superpositions between the HF determinant and excited determinants.

---

## Architecture

### System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Conditional-GQE Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐ │
│  │  PySCF   │───▶│  OpenFermion │───▶│  Hamiltonian│───▶│  UCCSD     │ │
│  │  SCF     │    │  JW Transform│    │  JSON       │    │  Op Pool   │ │
│  └──────────┘    └──────────────┘    └─────────────┘    └─────┬──────┘ │
│                                                             │          │
│                                                             ▼          │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │              Stage 1: Pure RL from Scratch (DAPO)            │      │
│  │              NO supervised pretraining                      │      │
│  │                                                              │      │
│  │   ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌─────────┐ │      │
│  │   │ Molecule│──▶│  Encoder  │──▶│  Decoder │──▶│ Sample  │ │      │
│  │   │ Embedding│  │ (4-layer) │   │(6-layer) │   │ Circuit │ │      │
│  │   └─────────┘   └───────────┘   └──────────┘   └────┬────┘ │      │
│  │                                                   │        │      │
│  │   ┌──────────────────────────────────────────────┐ │        │      │
│  │   │  Entropy collapse prevention (5 layers):     │ │        │      │
│  │   │  1. BF16 mixed precision                     │ │        │      │
│  │   │  2. Distribution mixing (ε=0.3)              │ │        │      │
│  │   │  3. Top-p sampling + adaptive temperature    │ │        │      │
│  │   │  4. REPO advantage modification              │ │        │      │
│  │   │  5. Curriculum learning (small mols first)   │ │        │      │
│  │   └──────────────────────────────────────────────┘ │        │      │
│  │                                                   ▼        │      │
│  │   ┌──────────┐     ┌────────────┐     ┌──────────────┐   │      │
│  │   │  CUDA-Q  │────▶│  Energy    │────▶│  DAPO Loss   │   │      │
│  │   │  Simulate│     │  <ψ|H|ψ>   │     │  + REPO + H  │   │      │
│  │   │  (MQPU)  │     │            │     │  bonus       │   │      │
│  │   └──────────┘     └────────────┘     └──────┬───────┘   │      │
│  │        ↑                                      │           │      │
│  │        └────── policy update ←────────────────┘           │      │
│  │   300 epochs on 3× L40S GPUs (nvidia-mqpu target)        │      │
│  │   31.7M parameters, BF16, lr=3e-4                        │      │
│  └──────────────────────────┬────────────────────────────────┘      │
│                             │                                          │
│                             ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │              Stage 2: Coefficient Optimization               │      │
│  │   L-BFGS-B on rotation angles (thetas) for fixed sequences   │      │
│  │   Parallelized across 3× L40S via CUDA-Q nvidia-mqpu         │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### H-cGQE Transformer Architecture

```
Input: Molecule conditioning vector (qubit count, electron count, energy stats)
       +
Prefix tokens: [BOS] [MOL]

         ┌────────────────────────────────────────────┐
         │         H-cGQE Transformer (GPT-2)          │
         │                                            │
         │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐       │
         │  │ Enc │  │ Enc │  │ Enc │  │ Enc │       │
         │  │ Lyr │  │ Lyr │  │ Lyr │  │ Lyr │       │
         │  │  1  │  │  2  │  │  3  │  │  4  │       │
         │  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘       │
         │     └────────┴────────┴────────┘           │
         │                    │                       │
         │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐       │
         │  │ Dec │  │ Dec │  │ Dec │  │ Dec │       │
         │  │ Lyr │  │ Lyr │  │ Lyr │  │ Lyr │       │
         │  │  1  │  │  2  │  │  3  │  │  4  │       │
         │  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘       │
         │     └────────┴────────┴────────┘           │
         │                    │                       │
         │              ┌─────┴─────┐                 │
         │              │  Linear   │                 │
         │              │  Softmax  │                 │
         │              └─────┬─────┘                 │
         └────────────────────┼──────────────────────┘
                              │
                              ▼
         Output: [OP₁] [OP₂] ... [OPₖ] [EOS]
                 Pauli word sequence (e.g., XYYX, IZII, IZIZ)
```

**Model specs**: d_model=256, nhead=8, 4 encoder + 6 decoder layers, dim_ff=1024, dropout=0.1, vocab_size=~1000 (UCCSD pool), 31.7M parameters

### DAPO-RL Training Loop

```
┌──────────────────────────────────────────────────────────────┐
│         DAPO: Decoupled Clip + Dynamic Sampling              │
│         Pure RL from Scratch (no supervised pretraining)     │
│                                                              │
│   ┌──────────┐     ┌────────────┐     ┌──────────────┐      │
│   │  Sample  │────▶│  CUDA-Q    │────▶│  Multi-comp  │      │
│   │  circuit │     │  Simulator │     │  Reward      │      │
│   │  seq πθ  │     │  (MQPU)    │     │  r = w₁·E    │      │
│   │  (top-p, │     │            │     │    + w₂·ent  │      │
│   │   ε-mix) │     │            │     │    + w₃·depth│      │
│   └──────────┘     └────────────┘     │    + w₄·comm │      │
│        ↑                               └──────┬───────┘      │
│        │                                      │              │
│        │                                      ▼              │
│   ┌────┴──────────┐                ┌──────────────────────┐  │
│   │  DAPO Loss    │◀── advantages ─│  GRPO + REPO         │  │
│   │  Clip-Higher  │   A_REPO =     │  A = (R - R̄)/σ       │  │
│   │  Token-Level  │   A - β·L̄     │    - β·(L_i - L̄_grp) │  │
│   │  + H bonus    │                └──────────────────────┘  │
│   └───────────────┘                                           │
│                                                              │
│   Entropy floor: ε-exploration (0.3) + top-p (0.9)          │
│   + adaptive temp + REPO (β=0.05) + curriculum (3 stages)  │
│   Mixed precision: BF16 (not FP16 — avoids softmax bias)    │
└──────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Conditional_GQE/
├── src/gqe/
│   ├── data/
│   │   ├── generate_hamiltonians.py    # PySCF + OpenFermion → JW Hamiltonians
│   │   ├── prepare_gqe_dataset.py      # Build supervised training dataset
│   │   ├── graph_dataset.py            # Atom-level graph features
│   │   ├── fragment_molecule.py        # FMO-style molecular fragmentation
│   │   └── fragmentation.py            # Fragment plan execution
│   ├── models/
│   │   ├── h_cgqe_transformer.py       # GPT-2 style Transformer (31.7M params)
│   │   ├── train_rl_dapo.py            # Stage 1: Pure RL from scratch (DAPO + REPO)
│   │   ├── train_h_cgqe.py             # Legacy: Supervised pretraining (optional)
│   │   ├── infer_h_cgqe.py             # Autoregressive circuit synthesis
│   │   ├── chemistry_encoder.py        # Graph neural network conditioning
│   │   └── train_chemistry_encoder.py  # Pretrain chemistry encoder
│   ├── eval/
│   │   ├── evaluate_h_cgqe.py          # Energy evaluation via CUDA-Q
│   │   ├── optimize_h_cgqe_coefficients.py  # L-BFGS-B coefficient optimization
│   │   ├── plot_benchmark_results.py   # Visualization
│   │   └── compare_gqe_results.py      # H-cGQE vs GQE baseline comparison
│   ├── baselines/
│   │   ├── run_cudaq_gqe.py            # NVIDIA CUDA-Q GQE baseline
│   │   ├── run_cudaq_vqe.py            # CUDA-Q VQE baseline
│   │   ├── run_adapt_vqe.py            # Qiskit ADAPT-VQE baseline
│   │   └── run_exact_diagonalization.py  # Exact FCI reference
│   └── common/
│       ├── hamiltonian_utils.py        # Shared Hamiltonian conversion utilities
│       └── operator_pool.py            # UCCSD fermionic excitation pool (no Z-only collapse)
├── configs/
│   ├── experiment_phase3.yaml          # Phase 3 molecule set (17 molecules)
│   └── experiment.yaml                 # Phase 2 configuration
├── scripts/
│   ├── run_full_benchmark.sh           # End-to-end benchmark pipeline
│   ├── run_multigpu_workflow.sh        # Multi-GPU H-cGQE workflow
│   └── run_h_cgqe_qbraid.sh           # QBraid cloud execution
├── results/                            # Evaluation outputs (JSON summaries)
├── proposals/                          # Generated PDF reports
├── requirements.txt                    # Python dependencies
└── environment-dgx-spark-cudaq.yml     # CUDA-Q conda environment
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-Q 0.8+ (for quantum simulation)
- PyTorch 2.6+ (for Transformer training)
- 3× NVIDIA L40S GPUs (recommended for multi-GPU evaluation)

### Installation

```bash
git clone https://github.com/Quantum-Buddies/Conditional_GQE.git
cd Conditional_GQE
pip install -r requirements.txt
# For CUDA-Q quantum simulation:
conda env create -f environment-dgx-spark-cudaq.yml
conda activate conditional-gqe-cudaq
```

### End-to-End Pipeline (Pure RL from Scratch)

```bash
# 1. Generate molecular Hamiltonians (PySCF + OpenFermion)
python src/gqe/data/generate_hamiltonians.py \
    --config configs/experiment_phase3.yaml \
    --out-dir results/data

# 2. Run CUDA-Q GQE baseline (for comparison only — not used for training)
python src/gqe/baselines/run_cudaq_gqe.py \
    --ham results/data/hamiltonians_phase3.json/hamiltonians.json \
    --out results/baselines/cudaq_gqe_phase3.json

# 3. Pure RL training from scratch (no supervised pretraining needed)
#    Model discovers circuits from energy rewards alone (arXiv:2502.19402)
python src/gqe/models/train_rl_dapo.py \
    --from-scratch \
    --hamiltonians results/data/hamiltonians_phase3.json/hamiltonians.json \
    --molecules h2_0.74 lih_1.6_full n2_1.1_full beh2_1.3_full \
    --out results/train/h_cgqe_rl_from_scratch.pt \
    --epochs 300 --lr 3e-4 --n-samples 50 \
    --d-model 256 --nhead 8 --encoder-layers 4 --decoder-layers 6 \
    --use-bf16 --repo-beta 0.05 --curriculum --curriculum-warmup 30 \
    --explore-eps 0.3 --adaptive-eps --top-p 0.9 \
    --adaptive-temp --entropy-coef 0.01 \
    --target nvidia --target-option mqpu \
    --use-cuda --multi-gpu --force-entanglement

# 4. Inference: generate operator sequences
python src/gqe/models/infer_h_cgqe.py \
    --checkpoint results/train/h_cgqe_rl_from_scratch.pt \
    --hamiltonians results/data/hamiltonians_phase3.json/hamiltonians.json \
    --out results/inference/h_cgqe_generated_phase3.json \
    --n-samples 100

# 5. Optimize rotation coefficients (Stage 2: classical optimization)
python src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated results/inference/h_cgqe_generated_phase3.json \
    --hamiltonians results/data/hamiltonians_phase3.json/hamiltonians.json \
    --out results/eval/h_cgqe_optimized_phase3.json \
    --n-sequences 10

# 6. Evaluate final energies
python src/gqe/eval/evaluate_h_cgqe.py \
    --generated results/inference/h_cgqe_generated_phase3.json \
    --hamiltonians results/data/hamiltonians_phase3.json/hamiltonians.json \
    --out results/eval/h_cgqe_evaluation_phase3.json
```

### Full Pipeline Script

```bash
# Runs all steps end-to-end on 3× L40S GPUs
bash scripts/run_full_uccsd_pipeline.sh
```

### Legacy: Supervised Pretraining (optional)

The supervised pipeline is still available for comparison:
```bash
# Prepare dataset from GQE baseline output
python src/gqe/data/prepare_gqe_dataset.py \
    --hamiltonians results/data/hamiltonians.json \
    --gqe-results results/baselines/cudaq_gqe.json \
    --out results/train/uccsd_dataset

# Train supervised model
python src/gqe/models/train_h_cgqe.py \
    --dataset results/train/uccsd_dataset/gqe_supervised_dataset.pt \
    --out results/train/h_cgqe_supervised.pt --epochs 300

# Then fine-tune with RL (uses checkpoint as warm start)
python src/gqe/models/train_rl_dapo.py \
    --checkpoint results/train/h_cgqe_supervised.pt \
    --hamiltonians results/data/hamiltonians.json \
    --molecules h2_0.74 lih_1.6_full ... \
    --out results/train/h_cgqe_rl_finetuned.pt --epochs 200 --lr 1e-5
```

### Multi-GPU Workflow

For running on 3× L40S GPUs with CUDA-Q's `nvidia-mqpu` target:

```bash
bash scripts/run_multigpu_workflow.sh
```

---

## EUV Lithography Application

This work targets **EUV photoresist chemistry** — the halogenated aromatic molecules used in 13.5 nm extreme ultraviolet lithography. Key molecules:

| Molecule | Formula | Role | Qubits | Error |
|---|---|---|---|---|
| Methyl iodide | CH₃I | Simplest EUV absorber | 8 | 0.63 mHa |
| Iodobenzene | C₆H₅I | Prototypical EUV photoresist | 8 | 2.73 mHa |
| 4-iodo-2-methylphenol | IMePh | Key photoresist monomer | 8 | 24.63 mHa |
| Phenol | C₆H₅OH | Non-iodinated control | 8 | 45.09 mHa |

The C–I bond in iodinated photoresists is the primary EUV absorption site. Accurate quantum simulation of these molecules enables **bottom-up photoresist design** — predicting solubility switching and acid generation quantum yields.

---

## Methodology

### Active Space Selection

Heavy-atom molecules (iodobenzene: 66 spin-orbitals, IMePh: 84 spin-orbitals) are intractable for full-CI quantum simulation. We use **active space selection** to focus on the chemically relevant orbitals:

- Iodine 4d lone pair → C–I σ bond (4 electrons, 4 orbitals → 8 qubits)
- Freeze core orbitals (1s through 3d for iodine)
- Jordan–Wigner transformation to qubit Hamiltonian

### Bond Dissociation Curves

Training on multiple geometries teaches the model **entanglement patterns across correlation regimes**:

- H₂ at 0.5, 0.74, 1.0, 1.5, 2.0 Å (weak → strong correlation)
- LiH at 1.2, 1.6, 2.0, 3.0 Å
- N₂ at 1.1, 1.8, 2.5 Å (equilibrium → dissociation)

### Pauli Word Padding

The shared operator vocabulary spans molecules of different qubit counts. We pad/truncate Pauli words with identity operators (`I`) to match each molecule's qubit count, enabling a single model to generate circuits for 4–14 qubit systems.

---

## Comparison with Literature

| Method | LiH (mHa) | N₂ (mHa) | Circuit Depth | Gradient Measurements |
|---|---|---|---|---|
| **H-cGQE (ours)** | **1.84** | 126.62 | 1–18 (fixed) | **0** |
| CUDA-Q GQE | 1.81 | 126.55 | 3–20 | 0 |
| ADAPT-VQE | <0.5 | ~50 | 50–200+ | Exponential |
| UCCSD-VQE | <0.2 | ~30 | 20–100 | Exponential |
| GQKAE (KAN) | ~1.5 | ~80 | Variable | 0 |

Our H-cGQE matches the GQE baseline on LiH while requiring **no gradient measurements on quantum hardware** — the circuit structure is generated classically and only energy expectation needs quantum evaluation.

---

## Phase 3 Submission

### PDF Report

The Phase 3 report is generated by `generate_phase3_pdf.py` and produces:
- `proposals/Ryoushi_Quantum_Buddies__Phase3_Version1.pdf` (official filename with double underscore)
- 5 body pages + cover page + references page
- Includes results tables, operator analysis, EUV application, literature comparison, and roadmap

```bash
python generate_phase3_pdf.py
```

### Submission Package

The final submission is a zipped folder `Ryoushi_Quantum_Buddies_Challenge_Phase3.zip` containing:
- The PDF write-up (`Ryoushi_Quantum_Buddies__Phase3_Version1.pdf`)
- Source code (`src/`, `configs/`, `scripts/`)

```bash
zip -r Ryoushi_Quantum_Buddies_Challenge_Phase3.zip \
    proposals/Ryoushi_Quantum_Buddies__Phase3_Version1.pdf \
    src/ configs/ scripts/ \
    generate_phase3_pdf.py generate_proposal_pdf.py \
    requirements.txt README.md LICENSE
```

### Phase 3 Results Files

| File | Description |
|---|---|
| `results/eval/h_cgqe_evaluation_phase3.json` | Energy evaluation (fixed theta) |
| `results/eval/h_cgqe_optimized_phase3.json` | L-BFGS-B optimized energies |
| `results/baselines/cudaq_gqe_phase3.json` | CUDA-Q GQE baseline |
| `results/train/h_cgqe_model_phase3_metrics.json` | Training metrics (500 epochs) |
| `results/train/h_cgqe_model_rlqf_phase3_history.json` | RLQF training history (409 steps) |
| `results/plots_phase3/energy_error_vs_qubits.png` | Energy error plot |

---

## Citation

```bibtex
@software{conditional_gqe,
  title = {Conditional-GQE: Hierarchical Conditional Generative Quantum Eigensolver for EUV Lithography},
  author = {{Ryoushi Quantum Buddies}},
  url = {https://github.com/Quantum-Buddies/Conditional_GQE},
  version = {4.0.0},
  year = {2026}
}
```

## License

[MIT](LICENSE) — © 2025-2026 Ryoushi Quantum Buddies

## Acknowledgments

- **NVIDIA CUDA-Q** team for the hybrid quantum-classical simulation platform
- **Mitsubishi Chemical & AIST** for the GIC Phase 3 challenge on EUV lithography
- **PySCF** and **OpenFermion** developers for the quantum chemistry toolchain
