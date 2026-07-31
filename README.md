---
language: en
license: mit
library_name: pytorch
tags:
- quantum-computing
- generative-quantum-eigensolver
- reinforcement-learning
- cuda-q
- quantum-chemistry
- circuit-design
- GQE
- DAPO
- MAP-Elites
- QSCI
- FMO2
- GIC2026
datasets:
- custom
base_model: Ryukijano/h-cgqe-gic2026
metrics:
- energy-error
- chemical-accuracy
model-index:
- name: H-cGQE (Conditional-GQE)
  results:
  - task:
      type: quantum-ground-state
      name: Ground State Energy Estimation
    dataset:
      type: custom
      name: GIC 2026 Molecule Suite
    metrics:
      - type: energy-error
        value: 0.63
        name: CH3I Error (mHa)
      - type: energy-error
        value: 1.48
        name: H2 GPU-Simulator Gap (mHa)
---

<p align="center">
  <h1 align="center">⚛️ Conditional-GQE (H-cGQE)</h1>
  <p align="center">
    <strong>AI-Driven Generative Quantum Circuit Design for Molecular & Materials Discovery</strong><br>
    <em>Generative AI × Reinforcement Learning × CUDA-Q × Quantum Hardware</em>
  </p>
  <p align="center">
    <strong>Mitsubishi Chemical Group & AIST Quantum Challenge (GIC 2026)</strong>
  </p>
  <p align="center">
    <a href="https://github.com/Quantum-Buddies/Conditional_GQE/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.7+-red.svg" alt="PyTorch"></a>
    <a href="https://nvidia.github.io/cuda-quantum/"><img src="https://img.shields.io/badge/CUDA--Q-0.8+-green.svg" alt="CUDA-Q"></a>
    <a href="https://huggingface.co/Ryukijano/h-cgqe-gic2026"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Model%20Card-yellow.svg" alt="Hugging Face"></a>
    <a href="https://account.qbraid.com?gitHubUrl=https://github.com/Quantum-Buddies/Conditional_GQE.git"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" alt="Launch on qBraid" height="20"></a>
  </p>
</p>

<p align="center">
  <img src="docs/gic2026_journey_loop.gif" alt="H-cGQE GIC 2026 Journey — From molecular input to quantum circuit synthesis" width="800">
</p>

### 🎬 Animated Visual Overview

<table align="center" border="0" cellpadding="8">
  <tr>
    <td align="center"><b>RL Training Loop (DAPO/GRPO)</b></td>
    <td align="center"><b>Transformer Architecture</b></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/rl_training_loop.gif" alt="RL training loop: sample → evaluate → reward → MAP-Elites → DAPO update → replay" width="400"></td>
    <td align="center"><img src="docs/transformer_architecture.gif" alt="Transformer architecture: GNN encoder → Hamiltonian encoder → cross-attention decoder → constrained sampling" width="400"></td>
  </tr>
  <tr>
    <td align="center"><b>VQE vs H-cGQE Comparison</b></td>
    <td align="center"><b>HPC ↔ QPU Async Workflow</b></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/vqe_vs_gqe.gif" alt="VQE vs GQE: fixed ansatz, barren plateaus, diagonal collapse vs AI-generated, chemical accuracy" width="400"></td>
    <td align="center"><img src="docs/hpc_qpu_workflow.gif" alt="HPC to QPU: QWC grouping, manifest export, qBraid routing, multi-vendor QPU, async retrieval" width="400"></td>
  </tr>
</table>

---

## 🌟 Executive Summary

**Conditional-GQE (H-cGQE)** is an artificial intelligence framework that **automatically designs quantum computing circuits** for chemistry and materials science. 

Traditional Quantum Virtual Eigensolvers (VQEs) rely on manual, human-designed quantum circuits that are either too deep for real quantum hardware or get trapped in mathematical dead-ends called **barren plateaus** and **diagonal collapse**. 

H-cGQE pairs a **Chemical Graph Neural Network (GNN)** and a **Transformer** with **Quality-Diversity Reinforcement Learning (QD-GRPO)** to amortize ansatz design: conditioned on molecular structure and the electronic Hamiltonian, the model proposes compact operator sequences whose continuous angles are refined classically (L-BFGS-B). The goal is chemical accuracy ($\le 1.6 \text{ mHa}$) within stated active spaces on simulators, with selective hardware checks.

```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│   Molecular Structure  │ ───► │  AI Transformer Agent  │ ───► │ Compact Quantum Circuit│
│ (Atoms, Bonds, Energy) │      │ (GNN + QD-GRPO Policy) │      │ (Optimized for QPUs)   │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

> 🔗 **Model Weights & Artifacts:** Hosted on HuggingFace at [`Ryukijano/h-cgqe-gic2026`](https://huggingface.co/Ryukijano/h-cgqe-gic2026)

---

## 🏆 Key Results & Benchmark Scores

| Benchmark / Molecule | Active Space / Qubits | Reported Metric | Baseline / Context | Highlight / Caveat |
|---|---|---|---|---|
| **Methyl Iodide ($\text{CH}_3\text{I}$)** | **8q** CAS(4e,4o) | **$0.63 \text{ mHa}$** vs active-space CASCI/FCI | $2.65 \text{ mHa}$ (CUDA-Q GQE)<br>$988 \text{ mHa}$ (HEA-VQE) | Sub-chemical accuracy ($\le 1.6 \text{ mHa}$) in a controlled 8q comparison (`results/phase3_final/benchmark_ch3i_consolidated.json`). Distinct from the larger GIC `methyl_iodide_cas12` Hamiltonian. |
| **Hydrogen ($\text{H}_2$)** | 4 Qubits | **$1.48 \text{ mHa}$** GPU↔AWS SV1 gap | Shot-noise simulator cross-check | Cross-backend energy discrepancy on a shallow circuit (`results/eval/simulator_validation.json`); **not** an FCI error. |
| **IQM Emerald QPU** | 8 Qubits | **$87.5\%$ state fidelity** | Ideal bitstring target | 1024-shot hardware run; fidelity is to the expected computational-basis string, **not** energy accuracy on QPU. |
| **Benzene ($\text{C}_6\text{H}_6$)** | **40q** (QSCI/MPS path) | QSCI subspace estimate (~**19 s**) | Full SV OOM / infeasible | Runtime scaling demonstration; **not** a claim of exact FCI/CASCI match. See `results/phase3_final/qsci/`. |
| **Ethylene ($\text{C}_2\text{H}_4$)** | **28 Qubits** | MPS bond-dim. sweep ($D=32\ldots256$) | Full SV stressful / often impractical | ~300 s single-GPU MPS convergence study; accuracy depends on bond dimension. |
| **GIC molecule suite** | 4q – 28q | Conditioned generation + collapse mitigations | Diagonal / commuting collapse is a known GQE failure mode | UCCSD pools + entanglement constraints are designed to suppress Z-only collapse. **Suite-wide chemical-accuracy rates should be reported from eval JSONs, not assumed 100%.** |

*Notes:* Chemical accuracy ≈ $1.6 \text{ mHa}$ ($\approx 1 \text{ kcal/mol}$). Where “exact” references appear below, they mean **CASCI/FCI within the stated active space**, not full CI in the complete basis. Held-out / zero-shot molecule generalization remains an open evaluation item.

---

## 📐 Visual Architecture & Dataflow

### Diagram 1 — End-to-End Pipeline (High-Level)

<img src="docs/mermaid_svgs/diagram_01.png" alt="Diagram 1" width="100%">

### Diagram 2 — Internal Transformer Architecture (Technical)

<img src="docs/mermaid_svgs/diagram_02.png" alt="Diagram 2" width="100%">

### Diagram 3 — RL Training Loop & Reward Decomposition

<img src="docs/mermaid_svgs/diagram_03.png" alt="Diagram 3" width="100%">

### Diagram 4 — VQE vs C-GQE Comparison

<img src="docs/mermaid_svgs/diagram_04.png" alt="Diagram 4" width="100%">

### Diagram 5 — Qubit Scaling Spectrum

```
  4q         12q         20q         28q         40q
  │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼
┌─────┐   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
│ H₂  │   │ LiH │    │ N₂  │    │ C₂H₄│    │C₆H₆│
│ (4q)│   │(12q)│    │(20q)│    │(28q)│    │(40q)│
└──┬──┘   └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘
   │         │          │          │          │
   ▼         ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────┐
│  CUDA-Q Statevector (nvidia-mqpu, 3× L40S)          │
│  ● Fast: <1s per circuit    ● RL training rewards   │
│  ● B200 SQLite Cache (24k+ entries)                 │
│  ● QPU Validation (qBraid: IQM, Rigetti)            │
└──────────────────────┬──────────────────────────────┘
                       │ 24q threshold (PCIe L40S)
                       ▼
┌─────────────────────────────────────────────────────┐
│  MPS Tensor Network (bond dim D=32…256)             │
│  ● Ethylene 28q: ~300s on single L40S               │
│  ● Bond convergence sweep required for accuracy     │
└──────────────────────┬──────────────────────────────┘
                       │ 28q threshold
                       ▼
┌─────────────────────────────────────────────────────┐
│  QSCI + FMO2 (28–40q)                               │
│  ● Benzene 40q: ~19s QSCI/MPS path (subspace estimate) │
│  ● FMO2: fragment → evaluate → reassemble parent    │
│  ● NOT brute-force statevector (scientifically wrong│
│    for 32–40q JW chemistry circuits)                │
└─────────────────────────────────────────────────────┘
```

---

## 🔬 In-Depth Nuances & Technical Pillars

### 1. The Chemistry GNN Encoder (`ChemistryEncoder`)

Unlike standard NLP transformers, C-GQE features an **Edge-Aware Message-Passing Graph Neural Network** (`src/gqe/models/chemistry_encoder.py`) that encodes the physical topology of the molecule — conceptually analogous to how AlphaFold's Evoformer processes structural relationships:

<img src="docs/mermaid_svgs/diagram_05.png" alt="Diagram 5" width="100%">

- **Node Features**: Atomic numbers, hybridization states, formal charges, valence, aromaticity.
- **Edge Features**: Chemical bond types, 3D interatomic distances $R_{ij}$, conjugation, ring membership.
- **Global Invariants**: Active space qubit count $N_q$, total electron count $N_e$, spin multiplicity $2S+1$.
- **Mechanism**: 3 layers of edge-weighted message passing → dual pooling (mean + max) → projection to **soft prompt tokens** that prefix-condition the decoder's cross-attention.
- **Why GNN?** The molecular graph topology (bond connectivity, atom types) determines which fermionic excitations are chemically relevant. A flat MLP on atom counts would miss the graph structure — the GNN captures local chemical environments (e.g., "this carbon is in an aromatic ring with two neighbors") that inform operator selection.

### 2. Solving "Diagonal Sequence Collapse"

In early GQE implementations, AI agents discovered a "lazy shortcut": generating commuting $Z$-basis operators (e.g., $IZIZ$, $ZZII$). Because these operators commute with the Hartree-Fock state, their energy gradients are identically zero ($\frac{\partial E}{\partial \theta} = 0$). Classical optimizers trap them at $E = E_{\text{HF}}$, killing training gradient variance (`std(rewards) = 0`).

```
  ❌ COLLAPSED SEQUENCE (Z-only, commuting)         ✅ ENTANGLED SEQUENCE (UCCSD, non-commuting)
  ┌─────────────────────────────────┐               ┌─────────────────────────────────┐
  │  A₁ = IZIZ  (Z-only)            │               │  A₁ = YZXI  (X+Y entangling)    │
  │  A₂ = ZZII  (Z-only)            │               │  A₂ = XZYI  (X+Y entangling)    │
  │  A₃ = IZIZ  (duplicate)         │               │  A₃ = IYZX  (X+Y entangling)    │
  │  A₄ = ZIIZ  (Z-only)            │               │  A₄ = ZXIY  (X+Y entangling)    │
  │                                 │               │                                 │
  │  [A_i, A_j] = 0 ∀ i,j           │               │  [A_i, A_j] ≠ 0 (non-commuting) │
  │  ∂E/∂θ_i = 0 (zero gradient)    │               │  ∂E/∂θ_i ≠ 0 (non-zero grad)    │
  │  E = E_HF (trapped at baseline) │               │  E < E_HF (energy improvement)  │
  │  std(rewards) = 0 → no learning │               │  std(rewards) > 0 → RL learns   │
  └─────────────────────────────────┘               └─────────────────────────────────┘
```

- **Our Solution**:
  1. **UCCSD Operator Pool**: Built from fermionic single/double excitations mapped via Jordan-Wigner, guaranteeing entangling $X/Y$ operations. Zero Z-only operators by construction.
  2. **Entanglement Enforcement**: `force_entanglement=True` in the decoder masks Z-only tokens during sampling, ensuring at least one multi-qubit entangler per sequence.
  3. **Commutator Penalty**: Explicit reward penalty $w_4 \cdot \text{frac}([A_i, A_j] \neq 0)$ for commuting operator sequences.

<img src="docs/mermaid_svgs/diagram_06.png" alt="Diagram 6" width="100%">

**Verified pool statistics**: H₂ (4q): 16 Pauli words, 0 Z-only, 192 pool entries. LiH (12q): 1,408 Pauli words, 0 Z-only. N₂ (20q): 11,088 Pauli words, 0 Z-only. BeH₂ (14q): 3,456 Pauli words, 0 Z-only.

### 3. Quality-Diversity RL: QD-GRPO with MAP-Elites
Standard Policy Gradient methods (PPO/GRPO) suffer from mode collapse, finding only one circuit structure. We implement **MAP-Elites QD-GRPO** (`src/gqe/rl/map_elites.py`):
- **2D Feature Space**: The archive space is discretized into a 10×10 grid indexed by **Entanglement Density** (ratio of multi-qubit $X/Y$ terms) and **Circuit Depth**.
- **Adaptive Novelty Bonus**: Rewards the policy not just for low energy, but for filling unvisited cells in feature space:
  $$\text{Reward} = w_1 \cdot \left(-\frac{E}{|E_{\text{ref}}|}\right) + w_2 \cdot \text{Entanglement} + \lambda \cdot \text{Novelty}$$
- As coverage exceeds $50\%$, $\lambda$ decays adaptively to shift focus to energy refinement.

<img src="docs/mermaid_svgs/diagram_07.png" alt="Diagram 7" width="100%">

### 4. L-BFGS-B Angle Fine-Tuning

For a generated sequence $[A_1, A_2, \dots, A_k]$, each operator $A_i = e^{i\theta_i \hat{P}_i}$ requires a continuous rotation angle $\theta_i \in \mathbb{R}$. The energy landscape is:
$$E(\boldsymbol{\theta}) = \langle \psi_0 | U_{j_k}^\dagger \cdots U_{j_1}^\dagger \hat{H} U_{j_1} \cdots U_{j_k} | \psi_0 \rangle$$

<img src="docs/mermaid_svgs/diagram_08.png" alt="Diagram 8" width="100%">

- **Truncated mode (RL training)**: 3–5 iterations, $\theta_0 = 0.01$, Spearman $\rho \approx 0.5$ with converged energies, $50\times$ faster than full opt.
- **Full mode (final evaluation)**: 200 iterations, $\text{ftol} = 10^{-10}$, machine-precision convergence.
- **Why L-BFGS-B?** BFGS approximates the inverse Hessian $H^{-1}$ using rank-2 updates from gradient evaluations — no explicit Hessian computation needed. The bounded variant (L-BFGS-B) handles box constraints on $\theta_i \in [-\pi, \pi]$.
- **DedupCache**: MD5 hash of operator sequence → energy. Identical circuits are never re-evaluated. SQLite-backed for persistence across training runs.

### 5. B200 Energy Cache & Offline RL Pretraining

<img src="docs/mermaid_svgs/diagram_09.png" alt="Diagram 9" width="100%">

- **SQLite Cache**: 24,000+ entries keyed by MD5 hash of operator sequence (`results/train/rl_energy_cache.sqlite`).
- **Offline Pretraining**: `src/gqe/data/cache_to_pretrain.py` recovers 17,408 (operators, energy) pairs by replaying deterministic circuit generation. This allows **replay-buffer mixing** of known-good circuits without CUDA-Q.
- **Cache-only mode**: `--cache-only` returns HF energy for cache misses (no CUDA-Q). Useful for buffer imitation, but **on-policy rollouts rarely hit the fixed cache** → flat rewards → DAPO advantage collapse. For real RL, use **write-through** (drop `--cache-only`) so misses are evaluated and stored. See `bash scripts/train_rl.sh full`.

### 6. Scaling to 40 Qubits: QSCI & FMO2

Direct statevector simulation breaks above 28 qubits ($2^{28} \approx 268$M amplitudes). To tackle 32–40 qubit systems required by the GIC challenge, we deploy two scientific scaling pillars:

<img src="docs/mermaid_svgs/diagram_10.png" alt="Diagram 10" width="100%">

- **QSCI (Quantum Selected Configuration Interaction)**: Samples circuits to build a determinant subspace, then classically diagonalizes a reduced Hamiltonian. Used here as a **scaling path** for ~40q systems (e.g. benzene) when full statevector is infeasible — report subspace energies and wall time, **not** “exact FCI match,” unless an independent CASCI/FCI reference is provided.
- **FMO2 (Fragment Molecular Orbital)**: Fragments large macromolecules into 8–12 qubit sub-units, evaluates them on quantum hardware, and reassembles parent energies via pairwise additive correction:
  $$E_{\text{FMO2}} = \sum_i E_i - \sum_{i<j} (E_{ij} - E_i - E_j)$$

### 7. Comparative Architectural Analysis: H-cGQE vs. SpinGQE & GPT-QE

To contextualize C-GQE against contemporary generative quantum eigensolvers, the table below compares **H-cGQE** with **GPT-QE** (*NVIDIA / U. Toronto / St. Jude, arXiv:2401.09253*) and **SpinGQE** (*Mindbeam AI, March 2026, arXiv:2603.24298*):

| Technical Dimension | **GPT-QE** (NVIDIA/Toronto, 2024) | **SpinGQE** (Mindbeam AI, March 2026) | **Our H-cGQE** (Quantum-Buddies, 2026) |
|---|---|---|---|
| **Target Systems** | Single-molecule Fermionic UCCSD ($H_2, LiH, N_2, CO_2$) | 4-qubit Heisenberg Spin Model | **35 GIC 2026 Molecular Hamiltonians** (4q–28q, extended to 40q) |
| **Model Topology** | Unconditional Decoder-Only (GPT-2) | Unconditional Decoder-Only (GPT-2) | **Conditional Encoder-Decoder Transformer** |
| **Conditioning Mode** | None (1 model per fixed molecule) | None (1 model per fixed Hamiltonian) | **Chemistry GNN + Hamiltonian Term Cross-Attention** |
| **Training Objective** | Softmax Boltzmann weighting $\exp(-\beta E)$ | Weighted MSE Loss: $w(E) \cdot (l_t - E_t)^2$ | **DAPO Policy Gradient (GRPO)** + Asymmetric Clipping |
| **Parameterization** | Discretized evolution times $e^{i P t_k}$ | Discretized evolution times / angle refinement | **Two-Stage: Discrete Topology $\rightarrow$ L-BFGS-B Continuous $\vec{\theta}$ Optimization** |
| **Exploration & Diversity** | Inverse temperature schedule $\beta$ | Inverse temperature schedule $\beta$ | **MAP-Elites Quality-Diversity Archive (QD-GRPO)** |
| **Diagonal Collapse Mitigation** | None | Temperature tuning | **UCCSD Excitations + Commutator Loss + Entropy Floor** |
| **Generalization** | Single instance | Single instance | **Conditioned for cross-molecule generation** (held-out energy tables still needed) |

#### Core Methodological Advances over SpinGQE & GPT-QE

1. **Cross-Molecule Conditioning via Encoder-Decoder**:
   - *SpinGQE & GPT-QE Limit*: Decoder-only models are typically trained for a single fixed Hamiltonian; changing geometry often means retraining.
   - *H-cGQE Approach*: `HamiltonianEncoder` + `ChemistryEncoder` (MPNN) condition a shared policy on $(H, \text{graph})$. This **enables** multi-molecule amortization; rigorous leave-one-family-out energy evaluation is the right test of whether that conditioning generalizes.

2. **Policy Optimization (DAPO RL) vs. Weighted MSE Loss**:
   - *SpinGQE Limit*: SpinGQE uses a heuristic weighted MSE loss $L = \sum w(E) \cdot (\text{logits}_t - E_t)^2$ to force discrete categorical token logits to regress onto continuous energy values. This leads to vanishing gradients near energy plateaus.
   - *H-cGQE Solution*: We frame circuit design as pure Reinforcement Learning via **DAPO (Decoupled Clip + Dynamic Sampling Policy Optimization)** with group-relative advantage $A_i = \frac{R_i - \mu_R}{\sigma_R}$. Asymmetric clipping ($\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28$) and token-level loss stabilize RL updates without surrogate MSE regression.

3. **Decoupled Two-Stage Optimization (Topology vs. Continuous Rotation Angles)**:
   - *SpinGQE & GPT-QE Limit*: Both models discretize continuous evolution times into discrete vocabulary tokens ($e^{i P_j t_k}$ for $t_k \in \{0.01, 0.05, 0.1, \dots\}$), causing vocabulary explosion and limiting expressivity.
   - *H-cGQE Solution*: We decouple discrete structural topology from continuous parameterization. Stage 1 (Transformer) generates the discrete operator sequence $(P_{j_1}, P_{j_2}, \dots)$. Stage 2 (L-BFGS-B) optimizes the continuous rotation angles $\vec{\theta}$ over the exact CUDA-Q expectation landscape using `nvidia-mqpu`.

4. **Quality-Diversity Archive (MAP-Elites) preventing Diagonal Collapse**:
   - *SpinGQE & GPT-QE Limit*: Autoregressive transformers naturally collapse into generating commuting, single-qubit, or Z-only operators (diagonal sequence collapse) because they carry zero entanglement overhead.
   - *H-cGQE Solution*: We maintain a 2D MAP-Elites archive (*Entanglement Density* $\times$ *Circuit Depth*). Rollouts discovering unoccupied topological niches receive intrinsic novelty bonuses, forcing the agent to learn non-commuting $X/Y$ entangling operators.

---

## 🧪 Comprehensive Molecule Inventory (35 GIC Molecules)

The framework is benchmarked across the complete GIC 2026 challenge molecule suite:

| Category | Molecules Included | Qubit Range |
|---|---|---|
| **Small Diatomics / Hydrides** | $\text{H}_2$ (4 bond lengths), $\text{LiH}$ (4 bond lengths), $\text{BeH}_2$ (3 bond lengths), $\text{HF}$ | 4q – 14q |
| **Organic & Volatile Compounds** | $\text{H}_2\text{O}$, $\text{NH}_3$, $\text{CH}_4$, Formaldehyde, Acetylene, Ethylene | 14q – 28q |
| **Aromatic & Heteroaromatic Systems** | Benzene, Toluene, Anisole, o-Cresol, Phenol | 12q – 24q |
| **Heavy-Atom & CAS Systems** | Methyl Iodide ($\text{CH}_3\text{I}$), Iodobenzene, IMePh, Diarylethene fragment | 12q – 24q |
| **Challenge 40q Scaling Set** | Benzene / $\text{N}_2$ large active spaces (QSCI/MPS path) | **40q** (subspace / TN estimates) |

---

## Quick start (qBraid)

### 1. Clone and one-shot setup

```bash
git clone https://github.com/Quantum-Buddies/Conditional_GQE.git
cd Conditional_GQE
bash scripts/setup_env.sh
```

`setup_env.sh` handles everything — no sudo or system conda needed:
- Downloads and installs git-lfs binary to `$HOME/.local/bin` (prebuilt, no root)
- Pulls all LFS-tracked assets (checkpoints, energy cache, pretrain data)
- Installs Python dependencies via `python3 -m pip` (qBraid-safe)
- Verifies GPU, CUDA-Q, and audits critical files

**LFS artifacts on `main`:**

| File | Purpose |
|---|---|
| `results/train/h_cgqe_model_b200_sft.pt` | SFT warm-start checkpoint |
| `results/train/gqe_supervised_dataset.pt` | Supervised training dataset |
| `results/train/rl_energy_cache.sqlite` | 25K circuit→energy cache (4–28q) |
| `results/train/rl_pretrain_from_cache.json` | 24K pretrain bootstrap circuits |

### 2. Environment

On qBraid Lab, use the **Launch on qBraid** button or:

[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png)](https://account.qbraid.com?gitHubUrl=https://github.com/Quantum-Buddies/Conditional_GQE.git)

For local or HPC setups:

```bash
conda env create -f environment-dgx-spark-cudaq.yml
conda activate conditional-gqe-cudaq
pip install -r requirements-qbraid.txt
```

### 3. Smoke test

```bash
bash scripts/train_rl.sh smoke
```

### 4. Training

```bash
bash scripts/train_rl.sh full          # write-through RL from SFT (skips cache-only)
bash scripts/train_rl.sh online-rl     # 50 epochs, write-through cache + CUDA-Q misses
bash scripts/train_rl.sh cache-warmup  # buffer imitation only (weak on-policy RL)
```

### 5. Evaluation

```bash
bash scripts/evaluate_rl.sh all        # infer → eval → optimize → report
bash scripts/evaluate_rl.sh infer      # generate circuits from checkpoint
bash scripts/evaluate_rl.sh eval       # CUDA-Q energy evaluation
bash scripts/evaluate_rl.sh optimize   # L-BFGS-B coefficient optimization
```

### 6. QPU validation

```bash
python scripts/qpu_preflight.py --dry-run --device qbraid:qbraid:sim:qir-sv
bash scripts/run_hpc_qbraid_workflow.sh --qpu-submit
bash scripts/run_hpc_qbraid_workflow.sh --qpu-retrieve
```

---

## Training launcher

### Portable qBraid scripts (zero hardcoded paths)

| Script | Purpose |
|---|---|
| [`scripts/setup_env.sh`](scripts/setup_env.sh) | One-shot setup: git-lfs, pip deps, GPU verify (no sudo) |
| [`scripts/env_gpu.sh`](scripts/env_gpu.sh) | Auto-detect GPU, set CUDA-Q gate fusion / mempool env vars |
| [`scripts/train_rl.sh`](scripts/train_rl.sh) | Write-through RL from SFT (`smoke` / `cache-warmup` / `online-rl` / `full`) |
| [`scripts/evaluate_rl.sh`](scripts/evaluate_rl.sh) | Evaluation pipeline (infer / eval / optimize / report / all) |

```bash
bash scripts/train_rl.sh smoke          # 2 epochs, 2 molecules (~2 min)
bash scripts/train_rl.sh full           # write-through RL from SFT (skips cache-only)
bash scripts/evaluate_rl.sh all         # infer → eval → optimize → report
# Optional: MAX_QUBITS_OVERRIDE=28 bash scripts/train_rl.sh full
```

**`train_rl.sh` modes:** `full` / `online-rl` use **write-through** caching (CUDA-Q evaluates misses and stores them). Prefer these for real learning. `full` always starts from the SFT checkpoint (ignores any stale `*_rl_warmup.pt`). `cache-warmup` (`--cache-only`) is kept for buffer-imitation experiments only — on-policy samples almost never hit the precomputed MD5 keys, so misses get a flat HF penalty and DAPO/GRPO advantages collapse.

GPU auto-detection: `env_gpu.sh` reads compute capability and sets CUDA-Q gate fusion level (Hopper CC 9.0 → fusion 5, Blackwell CC 10.0 → +FP32 emulation, Ampere CC 8.0 → fusion 4). Molecule lists are auto-generated from the Hamiltonians JSON filtered by GPU-specific qubit limits (`train_rl.sh` defaults to ≤22q on H200).

**Import order note:** Triton (`torch.compile`) and CUDA-Q both embed LLVM. `train_rl_dapo.py` lazy-imports CUDA-Q **after** `torch.compile`. Do not `import cudaq` before compiling the model in the same process.

### B200 / Blackwell launcher (legacy)

Portable entry point: [`scripts/launch_b200_training.sh`](scripts/launch_b200_training.sh)

```bash
bash scripts/launch_b200_training.sh sft          # supervised warm-start
bash scripts/launch_b200_training.sh ablation       # RL from scratch (ablation)
bash scripts/launch_b200_training.sh cache          # precompute energy cache (≤28q only)
bash scripts/launch_b200_training.sh both           # SFT → RL main pipeline
```

**Energy cache:** SQLite-backed circuit→energy store for fast RL. Default cap **`CACHE_MAX_QUBITS=28`**. Do not precompute 32–40q SV caches — use QSCI/FMO2 instead. `train_rl.sh` defaults to ≤22q on H200 (override with `MAX_QUBITS_OVERRIDE`).

```bash
# Optional: one-time cache fill (append-safe, skips existing keys)
bash scripts/launch_b200_training.sh cache
```

Blackwell / B200 env knobs: [`scripts/env_b200_blackwell.sh`](scripts/env_b200_blackwell.sh) (source before `import cudaq`). GPU auto-env for H100/H200/etc.: [`scripts/env_gpu.sh`](scripts/env_gpu.sh).

---

## Datasets

| File | Molecules | Qubits | Purpose |
|---|---|---|---|
| `results/data/hamiltonians_gic2026/` | 35 | 4–28 | GIC challenge set |
| `results/data/hamiltonians_rl_b200/` | 51 | 4–40 | RL scaling curriculum |
| `results/data/hamiltonians_merged.json` | 21 | 4–40 | SFT + baselines |
| `results/data/fragments/fmo_hamiltonians.json` | — | 4–12 | FMO2 fragments |

Generate new Hamiltonians:

```bash
python src/gqe/data/generate_hamiltonians.py --help
```

---

## QPU guidelines (qBraid)

- Target **4–12 qubit** molecules for hardware (`h2`, `iodobenzene`, `imeph_cas12`).
- Preflight skips **ZNE** if two-qubit gates > 20; skips **REM** if qubits > 10.
- Use **Pauli expectation** energy (`cudaq.observe`), not raw state probability.
- **FMO dimers** (8–12q) are the best “large system + real QPU” story — not 40q full Hamiltonians on hardware.

```bash
python scripts/qpu_preflight.py --dry-run
python src/gqe/eval/submit_qpu.py --help
```

---

## Repository layout

```
Conditional_GQE/
├── README.md                          # This file
├── QUICKSTART.md                      # Short reproduction guide
├── AGENTS.md                          # Canonical training decisions
├── docs/B200_TRAINING_PLAN.md         # B200 / Blackwell notes
├── scripts/
│   ├── launch_b200_training.sh        # SFT / RL / cache launcher
│   ├── run_hpc_qbraid_workflow.sh      # HPC → QPU orchestration
│   └── phase3/                        # Experiment scripts (01–09)
├── src/gqe/
│   ├── models/                        # Transformer, train_rl_dapo.py
│   ├── eval/                          # evaluate, QSCI, FMO2, submit_qpu
│   ├── rl/                            # MAP-Elites, energy_cache
│   └── data/                          # Hamiltonians, precompute cache
└── results/
    ├── train/                         # Checkpoints (LFS), metrics, cache
    └── phase3_final/                  # Published experiment artifacts
```

---

## Safeguards

| Safeguard | What it prevents |
|---|---|
| `--gate-auxiliary-rewards` | Reward hacking without energy improvement |
| `--statevector-max-qubits 24` | GPU OOM on L40S |
| MPS bond sweep (D=32,64,128,256) | False accuracy from single bond dim |
| QPU preflight (ZNE/REM limits) | Infeasible mitigation on deep circuits |
| RL cache cap at 28q | Wasting GPU weeks on 32q+ SV observe loops |

---

## Hardware notes

| Platform | Statevector | MPS | QPU validation |
|---|---|---|---|
| **qBraid L40S** | ≤24q | 28q+ | Primary dev target |
| **qBraid B200** | ≤32q (reference only) | 28–40q | Optional local CUDA-Q |
| **AIRE 3× L40S** | ≤24q (MQPU task-parallel) | 28q | Slurm jobs |

> L40S is PCIe-only: keep `n_qubits ≤ 24` for `nvidia-mqpu` to avoid distributed statevector segfaults.

---

## Phase 3 Submission — Quick Start for Judges

### Verify the Pipeline (Single Command)

```bash
bash scripts/phase3/00_smoke_test.sh
```

This runs 5 verification tests: DedupCache SQLite persistence, offline RL cache-only mode, FMO2 exact reconstruction, QPU manifest generation (QWC grouping), and code import sanity.

### Full Pipeline

The Phase 3 pipeline is a 3-stage hybrid GPU→GPU→QPU workflow:

<img src="docs/mermaid_svgs/diagram_11.png" alt="Diagram 11" width="100%">

| Stage | Hardware | What Happens | Script |
|---|---|---|---|
| **1. Precompute** | B200 GPU (qBraid) | Generate Hamiltonians, run H-cGQE inference, cache energies to SQLite | `scripts/launch_b200_training.sh` |
| **2. Offline RL Training** | L40S GPU (HPC) | Buffer-imitation / cache lookups; prefer write-through on CUDA-Q GPUs for real RL | `train_rl_dapo.py --energy-cache ...` (± `--cache-only`) |
| **3. QPU Validation** | Rigetti Cepheus (qBraid) | Execute QWC-grouped measurement circuits on 108q QPU | `scripts/phase3/generate_qpu_manifests.py` |

### Stage 1: Energy Cache Precompute (B200)

```bash
# On qBraid B200 instance — generates rl_energy_cache.sqlite
bash scripts/launch_b200_training.sh cache
```

### Stage 2: Offline RL Training (L40S, no CUDA-Q required)

```bash
python src/gqe/models/train_rl_dapo.py \
    --molecules h2_0.74 lih_1.6_full \
    --qd-mode \
    --energy-cache results/train/rl_energy_cache.sqlite \
    --cache-only \
    --epochs 50 \
    --out results/train/h_cgqe_rl_dapo_phase3.pt
```

Key flags:
- **`--energy-cache`**: Path to SQLite file from Stage 1. DedupCache / PersistentEnergyCache loads precomputed energies.
- **`--cache-only`**: Skips CUDA-Q; uncached circuits get HF penalty. Prefer **without** `--cache-only` (write-through) when CUDA-Q is available so novel circuits get real energies. On qBraid: `bash scripts/train_rl.sh full`.

### Stage 3: FMO2 3-Fragment Scaling (Genuine Qubit Reduction)

```bash
# Generate 3-fragment iodobenzene Hamiltonians (monomers 4q, dimers 8q, parent 12q)
python scripts/generate_fmo2_fragments.py

# Run FMO2 exact + H-cGQE + L-BFGS-B
python scripts/run_fmo2_scaling.py
python scripts/run_fmo2_lbfgs.py

# Submit dimer/monomer circuits to Rigetti Cepheus QPU
python scripts/submit_fmo2_qpu.py --submit

# Retrieve QPU results + SQD post-processing
python scripts/retrieve_and_sqd.py --meta results/qpu/fmo2_cepheus_submission_meta.json \
    --hamiltonians results/data/fragments/dimers.json \
    --out results/qpu/fmo2_cepheus_sqd_results.json
```

**Key result**: 12q parent recovered from max 8q circuits (33% qubit reduction). Fragmentation error: 11.3 mHa (nonzero → non-tautological).

### Stage 4: Bi-Level QPU Validation (L-BFGS-B Optimized Circuits)

```bash
# Submit L-BFGS-B optimized + zero-theta circuits to Cepheus
python scripts/submit_lbfgs_qpu.py --submit --include-zero-theta

# Retrieve + SQD
python scripts/retrieve_and_sqd.py --meta results/qpu/lbfgs_cepheus_submission_meta.json \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/qpu/lbfgs_cepheus_sqd_results.json
```

**Bi-level pipeline**: RL discovers operator topology (outer loop) → L-BFGS-B optimizes continuous angles (inner loop) → QPU executes → SQD recovers energy.

### Stage 5: QPU Manifest Generation (Legacy)

```bash
python scripts/phase3/generate_qpu_manifests.py \
    --molecules h2_0.74 lih_1.6_full \
    --hamiltonians results/data/hamiltonians_merged.json \
    --optimized results/eval/h_cgqe_uccsd_optimized.json \
    --out-dir results/qpu/manifests \
    --shots 4096
```

Outputs per-molecule JSON manifests with QWC-grouped QASM 2.0 measurement circuits, ready for qBraid submission to Rigetti Cepheus.

### Key Components

| Component | File | Description |
|---|---|---|
| **DedupCache (SQLite)** | `src/gqe/rl/map_elites.py` | Persistent energy cache with `from_sqlite()` classmethod for offline loading |
| **Offline / write-through RL** | `src/gqe/models/train_rl_dapo.py` | `--energy-cache`; omit `--cache-only` for write-through CUDA-Q misses |
| **FMO2 Pipeline** | `src/gqe/eval/run_fmo2.py` | Fragment → GQE → reassemble with MAP-Elites archive integration |
| **QPU Manifests** | `scripts/phase3/generate_qpu_manifests.py` | QWC grouping, QASM export, cost estimation for Rigetti Cepheus |
| **Smoke Test** | `scripts/phase3/00_smoke_test.sh` | Single-command verification for judges |

### Reproducibility

- **Energy cache**: SQLite file ensures deterministic rewards across training runs
- **MAP-Elites archives**: JSON-serialized per-molecule elite circuit libraries
- **Chemical accuracy target**: ≤ 1.6 mHa (~1 kcal/mol) vs CASCI/FCI **within the stated active space**
- **QPU cost transparency**: Per-manifest cost estimates (0.0425 credits/shot + 30 credits/task on Cepheus)

---

## Citation

```bibtex
@software{conditional_gqe,
  title  = {Conditional-GQE: Scalable Generative Quantum Eigensolver with RL, QSCI, and FMO2},
  author = {{Ryoushi Quantum Buddies}},
  url    = {https://github.com/Quantum-Buddies/Conditional_GQE},
  year   = {2026}
}
```

## License

[MIT](LICENSE) — © 2025–2026 Ryoushi Quantum Buddies

## Acknowledgments

NVIDIA CUDA-Q · Mitsubishi Chemical Group · AIST · qBraid · PySCF · OpenFermion · Park & Walsh (Chemeleon2, arXiv:2511.07158) · Nakaji et al. (GQE, arXiv:2401.09253)
