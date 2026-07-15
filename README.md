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

**Conditional-GQE** is a hierarchical HPC-AI-Quantum workflow that learns compact molecular ansatze on GPUs, validates selected circuits on qBraid-accessible QPUs, and scales to larger EUV-relevant parent systems through active-space reduction, tensor-network simulation, and fragment molecular orbital decomposition.

The system maps a molecular Hamiltonian directly to a sequence of Pauli rotation operators that form a quantum eigensolver ansatz, using a Transformer encoder-decoder model trained with reinforcement learning from energy feedback.

### The Hierarchy

| Level | Description | Tools |
|---|---|---|
| **1 - Classical chemistry / HPC** | PySCF -> active space -> qubit Hamiltonian -> UCCSD operator pool | PySCF, OpenFermion |
| **2 - AI circuit discovery** | Chemistry-conditioned Transformer -> RL from energy feedback -> diverse candidate operator sequences | PyTorch, DAPO/GRPO |
| **3 - GPU quantum simulation** | Exact state-vector for tractable systems -> MPS for larger systems -> multi-GPU task-parallel evaluation | CUDA-Q (nvidia, nvidia-mqpu, tensornet-mps) |
| **4 - QPU validation** | Hardware-aware transpilation -> execute selected shallow fragment circuits -> compare ideal, noisy and hardware energies | qBraid SDK |
| **5 - Parent-system reconstruction** | FMO monomer/dimer energies -> many-body expansion -> ~40-qubit parent chemistry | FMO2 |

### Version History

| Version | Method | Description |
|---|---|---|
| **v3** | Supervised Conditional-GQE | Imitation learning on GQE baseline circuits |
| **v4** | RLQF collapse-breaking prototype | Supervised warm start + PPO-style RL fine-tuning |
| **v5** | Phase 3 pipeline | Supervised warm start + DAPO RL fine-tuning with collapse prevention. Pure RL from scratch (--from-scratch) available as ablation baseline. |

> Earlier versions used supervised imitation followed by RLQF. Version 5 uses supervised warm start + DAPO RL fine-tuning. The supervised workflow remains as an ablation baseline. The --from-scratch flag enables pure RL without supervised initialization for comparison.

---

## Canonical Pipeline Stages

| Stage | Name | Description |
|---|---|---|
| **0** | Chemistry preprocessing | Hamiltonians, active spaces, fragment plans, operator pools |
| **1** | RL circuit-policy training | DAPO/GRPO with exploration and diversity controls |
| **2** | Post-training and test-time scaling | RAFT/STaR rounds, model soup, off-policy reuse, Best-of-N (if implemented) |
| **3** | Continuous parameter refinement | L-BFGS-B optimization of rotation angles for selected sequences |
| **4** | Backend validation | Exact simulator, MPS, noisy simulation, QPU execution |

> **FMO is not a training stage.** It is the outer scalability layer surrounding Stages 1-4.

---

## Key Results (Phase 3)

### Experiment 1: AI vs Ansatz Benchmark (CH3I)

| Method | Energy (Ha) | Error (mHa) | Runtime (s) |
|---|---|---|---|
| HEA-VQE (COBYLA, 200 iter) | -6888.8526 | 987.79 | 9.68 |
| CUDA-Q GQE (UCCSD pool, 25 iter) | -6889.8430 | 2.65 | <0.01 |
| **H-cGQE RLQF (L-BFGS-B)** | **-6889.8397** | **0.63** | **<0.01** |

H-cGQE with RLQF fine-tuning achieves 0.63 mHa error -- 4x better than CUDA-Q GQE and 1500x better than HEA-VQE.

### Experiment 2: QPU Validation (IQM Emerald)

| Device | Type | Shots | State Fidelity | Cost (credits) |
|---|---|---|---|---|
| qBraid QIR Simulator | Simulator | 2000 | 100.0% | 0 |
| AWS SV1 Simulator | Simulator | 1024 | 100.0% | 0.375 |
| **IQM Emerald (54q)** | **QPU** | **1024** | **87.5%** | **193.84** |

8-qubit H-cGQE circuit (operator XYYX, depth 12, 6 CNOTs) executed on IQM Emerald superconducting QPU. 87.5% of shots measured the expected HF state; remaining 12.5% distributed across 1-bit and 2+ bit errors consistent with gate noise.

### Experiment 3: FMO2 Reconstruction (IMePh)

| Method | FMO2 Energy (Ha) | Error vs Exact FMO2 |
|---|---|---|
| Exact-fragment FMO2 | (reference) | 0.000 mHa |
| H-cGQE FMO2 | (measured) | 26.252 mHa |

Solver error: 26.252 mHa. Fragmentation error: 0.000 mHa (exact by construction with 2 fragments).

### Experiment 4: MPS Scaling Curve

| Molecule | Qubits | SV Time (s) | MPS D=32 | MPS D=256 |
|---|---|---|---|---|
| H2 | 4 | <0.01 | match | match |
| LiH | 12 | 0.1 | match | match |
| CH3I | 8 | <0.01 | match | match |
| BeH2 | 14 | 0.3 | match | match |
| N2 | 20 | 2.1 | match | match |
| C2H4 (ethylene) | 28 | N/A (>24q) | MPS only | MPS only |

MPS breaks the 24-qubit statevector wall on a single L40S GPU. Ethylene (28 qubits) simulated in ~300s. Runtime scales polynomially (~O(n^2)).

### Training Results Across Molecule Set

| Molecule | Qubits | H-cGQE Error (mHa) | GQE Baseline (mHa) | Status |
|---|---|---|---|---|
| Methyl iodide (CH3I) | 8 | **0.63** | 2.65 | Chemical accuracy (<=1.6 mHa) |
| Iodobenzene (C6H5I) | 8 | **2.73** | 1.96 | Near chemical accuracy (1.6-3.0 mHa) |
| LiH (1.6 A) | 8 | **1.84** | 1.81 | Near chemical accuracy (1.6-3.0 mHa) |
| LiH (1.2 A) | 8 | **2.07** | 2.05 | Near chemical accuracy (1.6-3.0 mHa) |
| IMePh (EUV photoresist) | 8 | **24.63** | 19.01 | Unseen EUV test case |
| BeH2 (1.3 A) | 14 | **33.98** | 33.76 | Not converged |
| N2 (1.1 A) | 12 | **126.62** | 126.55 | Not converged - strongly correlated |

### Accuracy Categories

| Error Range | Wording |
|---|---|
| <=1.6 mHa | Chemical accuracy |
| 1.6-3.0 mHa | Near chemical accuracy |
| 3-10 mHa | Sub-chemical accuracy not reached |
| >10 mHa | Challenging / not converged |

### Active-Space Documentation

All reference energies are **CASCI/FCI within the selected active space**, not full-molecule full-basis FCI. The following parameters define each calculation:

| Molecule | Basis | Charge | Multiplicity | Active Electrons | Active Orbitals | Qubits | Frozen Core |
|---|---|---|---|---|---|---|---|
| H2 (0.74 A) | sto-3g | 0 | 1 | 2 (full) | 2 (full) | 4 | None |
| LiH (1.6 A) | sto-3g | 0 | 1 | 2 | 4 | 8 | 1s (Li) |
| BeH2 (1.3 A) | sto-3g | 0 | 1 | 4 (full) | 7 (full) | 14 | None |
| N2 (1.1 A) | sto-3g | 0 | 1 | 6 | 6 | 12 | 1s (N) |
| CH3I | sto-3g | 0 | 1 | 4 | 4 | 8 | 1s-3d (I), 1s (C) |
| Iodobenzene | sto-3g | 0 | 1 | 4 | 4 | 8 | 1s-3d (I), 1s (C) |
| IMePh | sto-3g | 0 | 1 | 4 | 4 | 8 | 1s-3d (I), 1s (C) |

Nuclear repulsion is included in all reported energies. No relativistic treatment or ECP is applied (sto-3g basis).

---

## Architecture

### System Pipeline

```
+-------------------------------------------------------------------------+
|                    Conditional-GQE Pipeline                             |
+-------------------------------------------------------------------------+
|                                                                         |
|  +----------+    +--------------+    +-------------+    +------------+  |
|  |  PySCF   |--->|  OpenFermion |--->|  Hamiltonian|--->|  UCCSD     |  |
|  |  SCF     |    |  JW Transform|    |  JSON       |    |  Op Pool   |  |
|  +----------+    +--------------+    +-------------+    +-----+------+  |
|                                                            |           |
|  +------------------------------------------------------+  |           |
|  |  Stage 0: Chemistry Preprocessing                   |<-+           |
|  +------------------------------------------------------+              |
|                            |                                          |
|                            v                                          |
|  +------------------------------------------------------+             |
|  |  Stage 1: RL Circuit-Policy Training (DAPO)         |             |
|  |  Supervised warm start -> DAPO RL fine-tuning        |             |
|  |                                                      |             |
|  |   +---------+   +-----------+   +----------+        |             |
|  |   | Molecule|--->|  Encoder  |--->|  Decoder |--->  |             |
|  |   | Ham     |   | (4-layer) |   |(4-layer) |       |             |
|  |   +---------+   +-----------+   +----------+        |             |
|  |                                                      |             |
|  |   Entropy collapse prevention:                       |             |
|  |   1. UCCSD operator pool (no Z-only operators)      |             |
|  |   2. BF16 mixed precision                           |             |
|  |   3. Distribution mixing (e-exploration)            |             |
|  |   4. REPO advantage modification                    |             |
|  |   5. Curriculum learning                            |             |
|  |   6. Chemeleon2: MMD diversity + creativity         |             |
|  |   7. Entropy bonus + KL penalty                     |             |
|  |                                                      |             |
|  |   +----------+     +------------+     +----------+   |             |
|  |   |  CUDA-Q  |---->|  Energy    |---->|  DAPO    |   |             |
|  |   |  Simulate|     |  <psi|H|psi>|    |  Loss    |   |             |
|  |   |(MQPU)    |     |            |    |  +REPO+H |   |             |
|  |   +----------+     +------------+    +-----+----+   |             |
|  |        ^                              |            |             |
|  |        +------ policy update <--------+            |             |
|  +--------------------------+---------------------------+             |
|                            |                                          |
|                            v                                          |
|  +------------------------------------------------------+             |
|  |  Stage 3: Coefficient Optimization (L-BFGS-B)       |             |
|  |  Rotation angle refinement for fixed sequences       |             |
|  |  Multi-GPU task-parallel via CUDA-Q nvidia-mqpu      |             |
|  +------------------------------------------------------+             |
|                                                                         |
|  +------------------------------------------------------+             |
|  |  Stage 4: Backend Validation                        |             |
|  |  Exact simulator -> MPS -> noisy sim -> QPU          |             |
|  +------------------------------------------------------+             |
|                                                                         |
|  +------------------------------------------------------+             |
|  |  Scalability Layer: FMO2 Reconstruction             |             |
|  |  Monomer + dimer energies -> parent system           |             |
|  +------------------------------------------------------+             |
|                                                                         |
+-------------------------------------------------------------------------+
```

### H-cGQE Transformer Architecture

The model is a **Transformer encoder-decoder** (not decoder-only GPT-2). The encoder processes the molecular Hamiltonian (Pauli terms + coefficients); the decoder autoregressively generates operator sequences via cross-attention to the encoded Hamiltonian.

```
Input: Hamiltonian Pauli terms + coefficients
       (encoded via char-level Pauli embeddings + coefficient projection)

         +--------------------------------------------+
         |      H-cGQE Transformer Encoder-Decoder     |
         |                                            |
         |  +-----+  +-----+  +-----+  +-----+       |
         |  | Enc |  | Enc |  | Enc |  | Enc |       |
         |  | Lyr |  | Lyr |  | Lyr |  | Lyr |       |
         |  |  1  |  |  2  |  |  3  |  |  4  |       |
         |  +--+--+  +--+--+  +--+--+  +--+--+       |
         |     +-------+-------+-------+              |
         |                    | cross-attn            |
         |  +-----+  +-----+  +-----+  +-----+       |
         |  | Dec |  | Dec |  | Dec |  | Dec |       |
         |  | Lyr |  | Lyr |  | Lyr |  | Lyr |       |
         |  |  1  |  |  2  |  |  3  |  |  4  |       |
         |  +--+--+  +--+--+  +--+--+  +--+--+       |
         |     +-------+-------+-------+              |
         |                    |                       |
         |              +-----+-----+                 |
         |              |  Linear   |                 |
         |              |  Softmax  |                 |
         |              +-----+-----+                 |
         +-------------------+------------------------+
                             |
                             v
         Output: [OP1] [OP2] ... [OPk] [EOS]
                 Pauli word sequence (e.g., XYYX, IZII, IZIZ)
```

**Model specs** (verified from checkpoint metadata):
- d_model=256, nhead=8, **4 encoder + 4 decoder layers**, dim_ff=1024, dropout=0.1
- Vocab size: 78-149 (UCCSD pool, molecule-dependent)
- **~7.8M parameters**
- Max Pauli length: 24, Max sequence length: 64

### Multi-GPU Simulation

CUDA-Q's nvidia-mqpu target provides **multi-GPU task-parallel quantum simulation** -- independent observable evaluations and candidate circuits are distributed across 3x L40S GPUs. This is task parallelism, not distributed state-vector simulation.

| Parallelism Type | Description | Used? |
|---|---|---|
| Task parallelism (MQPU) | Independent circuits/observables on separate GPUs | Yes |
| Hamiltonian-term batching | Split observables across GPUs | Yes (via MQPU) |
| Data-parallel AI training | Batched gradient computation | Yes (PyTorch DDP) |
| Distributed state-vector | Single state-vector across GPUs | No (L40S PCIe limitation) |

> **L40S constraint**: The L40S cluster is PCIe-only (no NVLink). CUDA-Q's distributed state-vector mode (cuStateVec, threshold=25 qubits) segfaults on PCIe-only L40S due to broken CUDA IPC in Open MPI's smcuda BTL. All state-vector simulations are limited to 24 qubits per GPU. For >24 qubits, use the tensornet-mps (MPS) backend in single-GPU mode.

---

## Breaking Diagonal Sequence Collapse

In Phase 2, the model suffered from **diagonal sequence collapse** -- it generated only commuting Z-only operators (e.g., IZII, ZIZI), getting trapped at the Hartree-Fock energy with zero gradients. We resolved this through a **7-layer defense** against entropy collapse:

1. **UCCSD operator pool** -- All operators come from fermionic excitation operators mapped through Jordan-Wigner. Every operator contains X/Y components -- Z-only collapse is impossible by construction
2. **BF16 mixed precision** -- FP16's 5 exponent bits cause multiplicative bias in softmax gradients that systematically reduces entropy. BF16 (8 exponent bits) eliminates this
3. **Distribution mixing** (e-exploration) -- Mixes sampling distribution with uniform distribution (e=0.3) to enforce a hard entropy floor
4. **REPO advantages** -- Regulated Entropy Policy Optimization modifies advantages with a centered log-prob penalty, penalizing deterministic samples and boosting diverse ones
5. **Curriculum learning** -- Train on small molecules (4 qubits) first, gradually add larger ones over 30-epoch warmup stages
6. **Chemeleon2-inspired rewards** (Park & Walsh, arXiv:2511.07158) -- Multi-objective reward with leave-one-out MMD diversity, creativity (uniqueness + novelty via edit distance), and KL penalty to reference policy (k3 estimator)
7. **Entropy bonus** -- Explicit entropy term in DAPO loss (-gamma * H(pi_theta)) encourages diverse sampling

Additional exploration measures: top-p (nucleus) sampling, adaptive temperature scheduling, and adaptive e decay.

The model now generates **entangling operators** like XYYX, YXXY, XXYY -- creating superpositions between the HF determinant and excited determinants.

---

## Pipeline Safeguards (Phase 3)

Four pre-flight safeguards ensure robustness and scientific validity before expensive GPU and QPU runs:

### 1. RL Reward Gating on Energy Improvement

All auxiliary RL rewards (entanglement fraction, circuit depth, non-commutativity, MMD diversity, creativity) are **gated on energy improvement** over the Hartree-Fock reference. If a generated circuit does not improve energy beyond a configurable threshold, auxiliary rewards are zeroed out — preventing reward hacking where the model optimizes for circuit structure metrics without actually lowering energy.

```bash
python src/gqe/models/train_rl_dapo.py \
    --gate-auxiliary-rewards \
    --energy-improvement-threshold 0.0  # mHa; default: any improvement
```

**File**: `src/gqe/models/train_rl_dapo.py` — `_has_energy_improvement()` + `compute_reward()`

### 2. Statevector Simulation Qubit Cap

Exact statevector simulation is explicitly capped at 24 qubits (configurable) to prevent OOM on L40S GPUs. The MPS scaling script automatically skips statevector for molecules exceeding the cap and reports `N/A` instead of crashing.

```bash
python src/gqe/eval/run_mps_scaling.py \
    --statevector-max-qubits 24  # L40S safe limit; use 32 for B200
```

**File**: `src/gqe/eval/run_mps_scaling.py` — `_run_statevector()` with `max_qubits` parameter

### 3. MPS Bond Dimension Convergence Reporting

MPS accuracy claims require **multiple bond dimensions** to demonstrate convergence. The scaling script now computes and reports energy differences across bond dimensions (D=32, 64, 128, 256), flagging whether results have converged or require higher bond dimension. A single bond dimension result is no longer presented as an accuracy claim.

**File**: `src/gqe/eval/run_mps_scaling.py` — convergence metrics in scaling artifact JSON

### 4. QPU Preflight Circuit Complexity Checks

Before QPU submission, the pipeline computes circuit depth and two-qubit gate count via Qiskit decomposition. Error mitigation is automatically skipped when infeasible:

- **ZNE skipped** if two-qubit gate count exceeds threshold (default: 20) — gate folding would make the circuit too deep for meaningful extrapolation
- **REM calibration skipped** if qubit count exceeds threshold (default: 10) — full assignment matrix calibration requires `2^n × 2^n` measurements, exponential in qubit count

```bash
python src/gqe/eval/submit_qpu.py \
    --max-zne-two-qubit-gates 20 \
    --max-rem-qubits 10
```

**File**: `src/gqe/eval/submit_qpu.py` — `_circuit_complexity()` + preflight checks

### Orbital Reordering: Deliberate Exclusion

Orbital reordering was **intentionally not added** to the MPS scaling script. The current benchmark uses a synthetic nearest-neighbor CNOT chain (worst-case entanglement stress test), while the Hamiltonians are already qubit-mapped via Jordan-Wigner. Reordering only the circuit or only the Hamiltonian would change the physical problem. A valid orbital-reordering experiment requires regenerating the fermionic Hamiltonian and operator pool with the same orbital permutation, then remapping both together — this is documented as a future enhancement, not a current limitation.

---

## Hybrid Workflow Advantages

Conditional-GQE reduces the need for repeated structural optimization on the QPU by learning reusable circuit-generation policies classically, while retaining the ability to evaluate selected ansatze on quantum hardware.

Potential advantages to measure include:
- Fewer QPU optimization iterations
- Lower transpiled two-qubit gate count
- Fewer circuit evaluations
- Cross-molecule reuse of learned policies
- Improved warm starts
- Better energy under fixed hardware budget

These are **hybrid workflow advantages**, not proof of computational quantum advantage.

---

## GPU Scaling Capacity

| Configuration | Engineering Estimate | Demonstrated Maximum |
|---|---|---|
| L40S single GPU | 24 qubits (state-vector) | 20 qubits (N2, sto-3g) |
| 3x L40S (MQPU) | 24 qubits per task | 20 qubits (task-parallel) |
| H200 (141 GB) | ~30 qubits (state-vector) | Not yet demonstrated |
| B200 (192 GB) | ~32 qubits (state-vector) | Not yet demonstrated |
| 4x B200 (768 GB) | ~36 qubits (state-vector) | Not yet demonstrated |
| MPS (tensornet-mps) | 40+ qubits | **28 qubits (ethylene, C2H4)** on L40S |

> qBraid offers on-demand GPU profiles across L40S, H200, B200, and multi-GPU configurations. Availability and actual workload limits depend on the selected instance, software stack, memory overhead, and runtime behaviour. Only the "Demonstrated" column reflects actual executed and logged workloads.

---

## Repository Structure

```
Conditional_GQE/
|-- README.md                            # This file
|-- QUICKSTART.md                        # One-command setup and reproduction
|-- REPRODUCIBILITY.md                   # Environment, determinism, and limitations
|-- LICENSE
|-- CITATION.cff
|-- environment-qbraid.yml               # Conda environment for qBraid
|-- requirements-qbraid.txt              # qBraid-compatible dependencies
|
|-- configs/
|   |-- phase3_final/                    # Phase 3 experiment configs
|   |   |-- benchmark.yaml               # Experiment 1: AI vs ansatz benchmark
|   |   |-- qpu_validation.yaml          # Experiment 2: QPU execution
|   |   |-- fmo_imeph.yaml               # Experiment 3: FMO2 reconstruction
|   |   `-- mps_scaling.yaml             # Experiment 4: MPS scaling curve
|   |-- experiment_phase3.yaml           # Phase 3 molecule set
|   |-- experiment_scaling_gic2026.yaml  # GIC 2026 scaling config
|   `-- experiment.yaml                  # Phase 2 configuration
|
|-- scripts/
|   |-- phase3/                          # Phase 3 pipeline scripts
|   |   |-- 00_smoke_test.sh
|   |   |-- 01_generate_hamiltonians.sh
|   |   |-- 02_run_baselines.sh
|   |   |-- 03_run_hcgqe.sh
|   |   |-- 04_run_fmo.sh
|   |   |-- 05_run_mps.sh
|   |   |-- 06_submit_qpu.sh
|   |   |-- 07_collect_qpu.sh
|   |   `-- 08_build_report.sh
|   |-- lock_environment.sh              # Environment lock for reproducibility
|   |-- qpu_preflight.py                 # QPU availability and cost check
|   `-- run_full_uccsd_pipeline.sh       # Full training pipeline
|
|-- src/gqe/
|   |-- data/
|   |   |-- generate_hamiltonians.py     # PySCF + OpenFermion -> JW Hamiltonians
|   |   |-- prepare_gqe_dataset.py       # Build supervised training dataset
|   |   |-- graph_dataset.py             # Atom-level graph features
|   |   |-- fragment_molecule.py         # FMO-style molecular fragmentation
|   |   `-- fragmentation.py             # Fragment plan execution
|   |-- models/
|   |   |-- h_cgqe_transformer.py        # Transformer encoder-decoder (~7.8M params)
|   |   |-- train_rl_dapo.py             # Stage 1: RL training (DAPO + REPO + Chemeleon2 + reward gating)
|   |   |-- model_soup.py                # Weight averaging across RAFT rounds
|   |   |-- train_h_cgqe.py              # Legacy: Supervised pretraining (ablation baseline)
|   |   |-- infer_h_cgqe.py              # Autoregressive circuit synthesis
|   |   |-- chemistry_encoder.py         # Graph neural network conditioning
|   |   `-- train_chemistry_encoder.py   # Pretrain chemistry encoder
|   |-- eval/
|   |   |-- evaluate_h_cgqe.py           # Energy evaluation via CUDA-Q
|   |   |-- optimize_h_cgqe_coefficients.py  # L-BFGS-B coefficient optimization
|   |   |-- submit_qpu.py               # Submit H-cGQE circuit to qBraid QPU
|   |   |-- collect_qpu.py              # Collect QPU job results from qBraid
|   |   |-- consolidate_qpu.py          # Consolidate QPU validation results
|   |   |-- run_fmo2.py                 # FMO2 fragment energy calculation
|   |   |-- run_mps_scaling.py           # MPS vs statevector scaling experiment (with SV cap + convergence)
|   |   |-- fmo2_error_decomposition.py  # FMO2 error breakdown
|   |   |-- qbraid_backend.py            # qBraid batched evaluation backend
|   |   |-- mitigation.py               # REM + ZNE error mitigation
|   |   |-- qsci.py                     # QSCI scaling to 40 qubits
|   |   |-- plot_benchmark_results.py    # Visualization
|   |   `-- compare_gqe_results.py       # H-cGQE vs GQE baseline comparison
|   |-- baselines/
|   |   |-- run_cudaq_gqe.py             # NVIDIA CUDA-Q GQE baseline
|   |   |-- run_cudaq_vqe.py             # CUDA-Q VQE baseline
|   |   |-- run_adapt_vqe.py             # Qiskit ADAPT-VQE baseline
|   |   `-- run_exact_diagonalization.py # Exact FCI reference
|   `-- common/
|       |-- hamiltonian_utils.py         # Shared Hamiltonian conversion utilities
|       |-- operator_pool.py             # UCCSD fermionic excitation pool
|       |-- smiles_encoder.py            # SMILES molecular encoder for transfer learning
|       `-- run_manifest.py              # Reproducibility manifest utilities
|
|-- results/
|   `-- phase3_final/                    # Phase 3 experiment results
|       |-- manifest.json
|       |-- environment.json
|       |-- baselines/
|       |-- hcgqe/
|       |-- fmo/
|       |-- mps/
|       |-- qpu/
|       `-- figures/
|
|-- tests/
|   `-- test_run_manifest.py             # Unit tests for manifest utilities
|
`-- proposals/                           # Generated PDF reports
`-- RESULTS.md                           # Clean results summary
```

---

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for full setup and reproduction instructions.

### Prerequisites

- Python 3.10+
- CUDA-Q 0.8+ (for quantum simulation)
- PyTorch 2.6+ (for Transformer training)
- 3x NVIDIA L40S GPUs (for multi-GPU evaluation)

### Installation

```bash
git clone -b phase3-submission https://github.com/Quantum-Buddies/Conditional_GQE.git
cd Conditional_GQE
conda env create -f environment-dgx-spark-cudaq.yml
conda activate conditional-gqe-cudaq
pip install -r requirements-qbraid.txt
```

### Smoke Test

```bash
bash scripts/phase3/00_smoke_test.sh
```

### Full Pipeline (Supervised Warm Start + RL Fine-tuning)

```bash
# Runs all steps end-to-end on 3x L40S GPUs
bash scripts/run_full_uccsd_pipeline.sh
```

### Phase 3 Experiments

```bash
# Experiment 1: AI vs ansatz benchmark
bash scripts/phase3/02_run_baselines.sh
bash scripts/phase3/03_run_hcgqe.sh

# Experiment 2: QPU validation (requires qBraid API key)
bash scripts/phase3/06_submit_qpu.sh

# Experiment 3: FMO2 reconstruction
bash scripts/phase3/04_run_fmo.sh

# Experiment 4: MPS scaling curve
bash scripts/phase3/05_run_mps.sh

# Build 6-page PDF report
bash scripts/phase3/08_build_report.sh
```

### Ablation: Pure RL from Scratch

The --from-scratch flag disables supervised initialization, training purely from energy rewards:

```bash
python src/gqe/models/train_rl_dapo.py \
    --from-scratch \
    --hamiltonians results/data/hamiltonians_phase3.json/hamiltonians.json \
    --molecules h2_0.74 lih_1.6_full n2_1.1_full \
    --out results/train/h_cgqe_rl_from_scratch.pt \
    --epochs 300 --lr 3e-4 --n-samples 50 \
    --use-bf16 --repo-beta 0.05 --curriculum \
    --target nvidia --target-option mqpu \
    --use-cuda --multi-gpu --force-entanglement
```

---

## Post-Training Methods

Following DeepSeek-R1 and Chemeleon2, we implement post-training techniques adapted for quantum circuit generation:

### Iterative RAFT (STaR Loop)

Self-Taught Reasoner approach: each RAFT round produces a better model that generates higher-quality candidates for the next round.

```bash
bash scripts/run_iterative_raft.sh \
    --rounds 3 \
    --checkpoint results/train/h_cgqe_model_phase3.pt \
    --n-samples 100 --top-k 10
```

### Model Soup

Weight averaging across RAFT rounds (Wortsman et al., ICML 2022):

```bash
python src/gqe/models/model_soup.py \
    --checkpoints results/train/h_cgqe_raft_round_1.pt \
                  results/train/h_cgqe_raft_round_2.pt \
                  results/train/h_cgqe_raft_round_3.pt \
    --out results/train/h_cgqe_star_soup.pt
```

### Off-Policy GRPO (mu-Reuse)

Reuses each rollout batch for mu gradient steps with importance sampling correction (arXiv:2505.22257):

```bash
python src/gqe/models/train_rl_dapo.py \
    --reuse-iters 3 \
    --target nvidia --target-option mqpu \
    --use-cuda --use-bf16
```

### Chemeleon2-Inspired Rewards

Following Park & Walsh (arXiv:2511.07158):

| Reward | Flag | Weight | Purpose |
|---|---|---|---|
| **KL penalty** | --kl-coef | 0.0-1.0 | Anchors to pretrained reference policy |
| **Entropy bonus** | --entropy-coef | 0.0-0.01 | Encourages diverse sampling |
| **MMD diversity** | --w-mmd-diversity | 0.0-1.0 | Leave-one-out MMD -- anti-mode-collapse |
| **Creativity** | --w-creativity | 0.0-1.0 | Uniqueness + novelty via edit distance |
| **Chemeleon2 preset** | --chemeleon2-mode | -- | Conservative regime |

---

## EUV Lithography Application

This work targets **EUV photoresist chemistry** -- the halogenated aromatic molecules used in 13.5 nm extreme ultraviolet lithography:

| Molecule | Formula | Role | Qubits | Error (mHa) | Status |
|---|---|---|---|---|---|
| Methyl iodide | CH3I | Simplest EUV absorber | 8 | 0.63 | Chemical accuracy |
| Iodobenzene | C6H5I | Prototypical EUV photoresist | 8 | 2.73 | Near chemical accuracy |
| 4-iodo-2-methylphenol | IMePh | Key photoresist monomer | 8 | 24.63 | Unseen EUV test case |
| Phenol | C6H5OH | Non-iodinated control | 8 | 45.09 | Not converged |

The C-I bond in iodinated photoresists is the primary EUV absorption site. Accurate quantum simulation of these molecules enables **bottom-up photoresist design** -- predicting solubility switching and acid generation quantum yields.

---

## Methodology

### Active Space Selection

Heavy-atom molecules (iodobenzene: 66 spin-orbitals, IMePh: 84 spin-orbitals) are intractable for full-CI quantum simulation. We use **active space selection** to focus on the chemically relevant orbitals:

- Iodine 4d lone pair -> C-I sigma bond (4 electrons, 4 orbitals -> 8 qubits)
- Freeze core orbitals (1s through 3d for iodine)
- Jordan-Wigner transformation to qubit Hamiltonian

### Bond Dissociation Curves

Training on multiple geometries teaches the model **entanglement patterns across correlation regimes**:

- H2 at 0.5, 0.74, 1.0, 1.5, 2.0 A (weak -> strong correlation)
- LiH at 1.2, 1.6, 2.0, 3.0 A
- N2 at 1.1, 1.8, 2.5 A (equilibrium -> dissociation)

### Pauli Word Padding

The shared operator vocabulary spans molecules of different qubit counts. We pad/truncate Pauli words with identity operators (I) to match each molecule's qubit count, enabling a single model to generate circuits for 4-14 qubit systems.

---

## Comparison with Literature

| Method | LiH (mHa) | N2 (mHa) | Circuit Depth | Gradient Measurements |
|---|---|---|---|---|
| **H-cGQE (ours)** | **1.84** | 126.62 | 1-18 (fixed) | **0** |
| CUDA-Q GQE | 1.81 | 126.55 | 3-20 | 0 |
| ADAPT-VQE | <0.5 | ~50 | 50-200+ | Exponential |
| UCCSD-VQE | <0.2 | ~30 | 20-100 | Exponential |
| GQKAE (KAN) | ~1.5 | ~80 | Variable | 0 |

H-cGQE matches the GQE baseline on LiH while requiring **no gradient measurements on quantum hardware** -- the circuit structure is generated classically and only energy expectation needs quantum evaluation. N2 remains unconverged for both GQE and H-cGQE, indicating a limitation of the current operator pool for strongly correlated systems.

---

## Phase 3 Experiments

### Experiment 1: AI vs Ansatz Benchmark
Controlled comparison of H-cGQE vs hardware-efficient VQE vs UCCSD-derived GQE on CH3I (8 qubits). All methods share the same active space, Hamiltonian, seed, and optimization budget.

### Experiment 2: QPU Validation
8-qubit H-cGQE circuit (operator XYYX) submitted to 3 qBraid devices: qBraid QIR Simulator (2000 shots, 100% fidelity), AWS SV1 Simulator (1024 shots, 100% fidelity), and **IQM Emerald QPU** (54q superconducting, 1024 shots, 87.5% state fidelity, 193.84 credits). Circuit decomposed to depth-12 with 6 CNOTs for QPU compatibility.

### Experiment 3: FMO2 Reconstruction
FMO2 decomposition of IMePh into 2 fragments (4q + 8q). Exact-fragment and H-cGQE fragment energies computed separately. Solver error: 26.252 mHa. Fragmentation error: 0.000 mHa (exact by construction with 2 fragments -- parent = dimer).

### Experiment 4: MPS Scaling Curve
MPS simulation from 4 to 28 qubits with bond dimension sweep (D=32,64,128,256). Statevector reference computed for <=24 qubits (explicit cap, configurable via `--statevector-max-qubits`) using CUDA-Q nvidia backend. MPS extends to 28 qubits (ethylene) on a single L40S GPU, breaking the 24-qubit statevector wall. Bond dimension convergence is reported across all dimensions — a single bond dimension result is never presented as an accuracy claim. Runtime scales polynomially (~O(n^2)).

### Experiment 5: GQE-QSCI Scaling to 40 Qubits (BONUS POINT)

Quantum-Selected Configuration Interaction (QSCI) samples determinants from a quantum state and diagonalizes the subspace Hamiltonian classically, enabling scaling beyond exact diagonalization limits.

| Molecule | Qubits | Terms | Bitstrings | QSCI Energy (Ha) | Time (s) | Backend |
|---|---|---|---|---|---|---|
| H2 | 4 | 15 | 6 | -1.137284 | 0.1 | nvidia |
| LiH | 12 | 631 | 93 | -7.861865 | 0.1 | nvidia |
| BeH2 | 14 | 666 | 129 | -15.561278 | 0.1 | nvidia |
| N2 | 20 | 2951 | 129 | -107.496501 | 0.1 | nvidia |
| Formaldehyde | 24 | 9257 | 129 | -112.352446 | 0.4 | nvidia |
| Ethylene | 28 | 8919 | 131 | -77.070316 | 12.1 | tensornet-mps |
| **Benzene CAS(20e,20o)** | **40** | **29897** | **131** | **-227.890091** | **19.1** | **tensornet-mps** |

H2 QSCI achieves exact FCI energy (0.000 mHa error). Benzene at 40 qubits completes in ~19 seconds on MPS backend, demonstrating beyond-statevector quantum chemistry on a single L40S GPU.

### Experiment 6: Cross-Molecule Transfer Learning

SMILES-based molecular encoder for cross-molecule generalization. Chemistry-aware tokenizer handles multi-character atoms (Cl, Br, Li, Be). 2-layer transformer encoder produces 256-dim molecular embeddings. Dataset includes 10 molecules spanning 4-56 qubits.

### Experiment 7: QPU Error Mitigation

REM (Reference-State Error Mitigation) for readout error correction and ZNE (Zero-Noise Extrapolation) with gate folding at scale factors [1, 2, 3] and Richardson extrapolation. Integrated into QPU submission pipeline via `--mitigate rem,zne` flag.

---

## Launch on qBraid

[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid.svg)](https://account.qbraid.com/)

Click the button above to launch this project on qBraid Lab with pre-configured quantum hardware access.

---

## Citation

```bibtex
@software{conditional_gqe,
  title = {Conditional-GQE: Hierarchical Conditional Generative Quantum Eigensolver for EUV Lithography},
  author = {{Ryoushi Quantum Buddies}},
  url = {https://github.com/Quantum-Buddies/Conditional_GQE},
  version = {5.0.0},
  year = {2026}
}
```

## License

[MIT](LICENSE) -- (c) 2025-2026 Ryoushi Quantum Buddies

## Acknowledgments

- **NVIDIA CUDA-Q** team for the hybrid quantum-classical simulation platform
- **Mitsubishi Chemical & AIST** for the GIC Phase 3 challenge on EUV lithography
- **PySCF** and **OpenFermion** developers for the quantum chemistry toolchain
- **Park & Walsh** (Imperial College London) for Chemeleon2 -- GRPO with creativity, diversity, and KL rewards (arXiv:2511.07158)
- **Wortsman et al.** for Model Soups -- weight averaging for improved generalization (ICML 2022)
- **DeepSeek-R1** team for STaR-style iterative rejection sampling fine-tuning
- **Snell et al.** (Google DeepMind) for compute-optimal test-time scaling research
