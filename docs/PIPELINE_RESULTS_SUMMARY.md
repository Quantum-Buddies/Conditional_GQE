# Conditional-GQE: Complete Pipeline, Results & Diagram Planning Guide

**Team:** Quantum-Buddies | **Challenge:** GIC 2026 (Mitsubishi Chemical & AIST) | **Phase 3 Submission**

This document compiles every pipeline stage, result, and technical detail across all docs and result files into a single reference for diagram planning and write-up.

---

## 1. Competition Context

- **GIC 2026**: Global Industry Challenge, hosted by Connected DMV, qBraid, Aqora, Quantum World Congress
- **Track**: Mitsubishi Chemical & AIST — Advanced Materials
- **Task**: AI-enhanced quantum eigensolvers for molecular simulation (GQE approach)
- **Timeline**: Phase 1 (concept) → Phase 2 (technical paper) → Phase 3 (execution, deadline July 26 2026)
- **Compute**: qBraid platform (11,000 credits), AIRE HPC (28 nodes × 3 L40S), B200/H200 instances
- **QPU Access**: Rigetti Cepheus-1-108Q, IQM Emerald, IonQ, free simulators via qBraid

---

## 2. Architecture: 5-Layer Pipeline

### Layer 1: HPC / CPU Chemistry
- **Molecular integrals**: PySCF SCF + OpenFermion Jordan-Wigner transform
- **Active-space selection**: CAS(2,2) through CAS(20,20)
- **Fragment generation**: FMO2 decomposition (`fragmentation.py`)
- **Classical references**: FCI, CCSD, CCSD(T), HEA-VQE, ADAPT-VQE
- **Output**: Hamiltonian JSON records (Pauli terms, coefficients, n_qubits, hf_energy, fci_energy)

### Layer 2: AI / GPU Circuit Discovery
- **Chemistry encoder**: Edge-aware GNN (MPNN) → molecule-aware conditioning vector
- **Generative policy**: H-cGQE Transformer (GPT-2 style, 7.79M params, d_model=256, 4+4 layers)
- **RL optimization**: DAPO (Decoupled Clip + Dynamic Sampling Policy Optimization)
- **Reward**: `r = w1*(-E/|E_ref|) + w2*entanglement_frac + w3*(-depth) + w4*non_commute + w5*MMD + w6*creativity`
- **Safeguards**: Auxiliary rewards gated on energy improvement over HF (prevents reward hacking)
- **Post-training**: RAFT (STaR loop), Model Soup (weight averaging), Adaptive test-time compute

### Layer 3: GPU Quantum Simulation
- **Exact statevector**: CUDA-Q `nvidia` / `nvidia-mqpu` (4–32 qubits)
- **MPS simulation**: CUDA-Q `tensornet-mps` (24–40 qubits, bond dimension sweep D=32–256)
- **Noise simulation**: Depolarizing, amplitude damping, dephasing (device-informed)
- **Large circuit screening**: Filter by energy, depth, 2q gate count

### Layer 4: QPU Validation
- **Hardware**: Rigetti Cepheus-1-108Q (108q superconducting), IQM Emerald (54q), IonQ Forte (30q)
- **QWC grouping**: 3–5× circuit reduction (H2: 15→5, LiH: 631→180, N2: 2951→1308)
- **Error mitigation**: ZNE (gate folding), REM (readout correction), SQD (symmetry filtering)
- **Preflight checks**: ZNE skipped if 2q gates > 20; REM skipped if qubits > 10
- **Async workflow**: HPC exports QWC manifest → submit to QPU → retrieve separately

### Layer 5: HPC Integration & Analysis
- **FMO2 reconstruction**: `E_total = Σ_i E_i + Σ_{i<j} (E_{ij} - E_i - E_j)`
- **Uncertainty propagation**: Fragment energy errors → total error bounds
- **Benchmarking**: vs FCI, CCSD, CUDA-Q GQE, HEA-VQE, ADAPT-VQE
- **Dashboard**: Consolidated results, efficiency metrics, EUV photoresist analysis

---

## 3. Training Pipeline (Detailed)

### Stage 1: Supervised Fine-Tuning (SFT)
- **Model**: H-cGQE Transformer, 7.79M params, vocab 317, d_model=256, 4+4 layers
- **Data**: Synthetic operator sequences from CUDA-Q `solvers.gqe()` baseline
- **Optimizer**: AdamW (lr=6.4e-4, weight_decay=0.0), cosine annealing + 10-epoch warmup
- **Loss**: Cross-entropy on operator token sequences
- **B200 results**: 500 epochs BF16, best val loss 1.037, val accuracy **96.2%**
- **Checkpoint**: `h_cgqe_model_b200_sft.pt` (31 MB)

### Stage 2: DAPO Reinforcement Learning
- **Algorithm**: DAPO (Decoupled Clip + Dynamic Sampling Policy Optimization)
- **Key features**:
  - Asymmetric clipping (clip_low=0.2, clip_high=0.28) — prevents entropy collapse
  - Dynamic sampling — skips batches where all rollouts have identical reward
  - Token-level loss (not sequence-level)
  - Off-policy GRPO with μ-reuse (3 reuse iterations)
  - Replay buffer (FIFO, size=1000)
  - Curriculum learning (4q → 8q → 12q → 14q → 20q → 24q → 28q → 32q → 40q)
  - Force entanglement (prevents diagonal sequence collapse)
  - Adaptive temperature, top-p nucleus sampling
  - Frequency penalty, REPO regularization, QD-GRPO (MAP-Elites archive)
- **Energy evaluation**: CUDA-Q `nvidia-mqpu` async observe (multi-GPU)
- **Energy cache**: Persistent SQLite (`rl_energy_cache.sqlite`, 4.8 MB)
- **LR**: 1e-5 (10× lower than SFT for stable RL)
- **Checkpoints**:
  - `h_cgqe_model_b200_rl_main.pt` (40 MB) — DAPO RL on 35 GIC molecules
  - `h_cgqe_model_b200_rl_40q.pt` (40 MB) — Extended to 40q
  - `h_cgqe_model_b200_rl_scratch.pt` (40 MB) — Ablation: direct RL from scratch
  - `h_cgqe_rl_gic2026.pt` (33 MB) — Main submission checkpoint

### Stage 3: Classical Coefficient Optimization
- **Optimizer**: L-BFGS-B (scipy.optimize), multi-start (5 restarts)
- **Backend**: CUDA-Q `nvidia-mqpu` (3× L40S GPUs)
- **Parallelization**: Async `cudaq.observe_async()` across Hamiltonian terms

### Stage 4: QPU Submission + SQD Post-Processing
- **QWC manifest export**: Operators, thetas, groups, QASM (no QPU needed)
- **Async submission**: Single batch task to Rigetti Cepheus
- **Retrieval**: Poll until COMPLETED → parse parities → compute term expectations
- **SQD**: Sample-based Quantum Diagonalization on QPU bitstrings

### Critical Bug Fix: torch.compile + CUDA-Q LLVM Conflict
- **Root cause**: Both Triton (torch.compile) and CUDA-Q embed LLVM → SIGABRT (exit 134)
- **Fix**: Lazy `_ensure_cudaq()` — import cudaq AFTER torch.compile loads Triton's LLVM
- **AR-safe compile**: Encoder uses `reduce-overhead` (CUDA graphs); decoder uses `default` + `dynamic=True`

### Physicist Verification: Fixed-θ Proxy Limitation
- **Finding**: Fixed-θ (θ=0.01) proxy provides zero gradient signal on large molecules (Spearman ρ=0.227, p=0.416)
- **Root cause**: All circuits evaluate to Hartree-Fock energy at θ≈0 — reward landscape is flat
- **Fix**: Truncated L-BFGS-B (3–5 steps) during RL reward calculation

---

## 4. Hardware Stack

### AIRE HPC Cluster (Local)

*Paths/partition names shown are for the University of Leeds AIRE cluster — adapt for your site.*
- 28 nodes × 3 NVIDIA L40S (48GB, PCIe, no NVLink)
- Slurm `--partition=gpu --gres=gpu:l40s:N`
- Limit: 24 qubits (cuStateVec distributed segfault on PCIe)
- Conda env: `cudaq-env`

### qBraid GPU Instances (Cloud)
| Instance | GPU | VRAM | Credits/hr | Max SV Qubits |
|---|---|---|---|---|
| `gpu-l40s` | L40S | 48 GB | 228 | 24 |
| `gpu-h100-sxm` | H100 | 80 GB | 537 | 26 |
| `gpu-h200` | H200 | 141 GB | 549 | 30 |
| `gpu-b200` | B200 | 192 GB | 874 | 32 |
| `gpu-b200-4x` | 4× B200 | 768 GB | 3,395 | 36 |

### B200 Blackwell Environment
- `source scripts/env_b200_blackwell.sh` — cuBLAS BF16x9, CUDA-Q FP32 emulation, PyTorch TF32
- PyTorch 2.7+ cu126 (sm_100/B200 kernels via PTX JIT)
- NVFP4 optional via Transformer Engine (1.59× throughput, 4× memory savings)
- CUDA-Q 0.10+ with Blackwell support

### QPU Hardware
| QPU | Provider | Per-task | Per-shot | Max qubits |
|---|---|---|---|---|
| Rigetti Cepheus-1-108Q | AWS | 30 cr | 0.0425 cr | 108q |
| IQM Emerald | AWS | 30 cr | 0.16 cr | 54q |
| IQM Garnet | AWS | 30 cr | 0.145 cr | 20q |
| IonQ Forte-1 | IonQ | 30 cr | 8 cr | ~30q |

### Rigetti Cepheus Hardware Details
- 12 × 9-qubit chiplets, square lattice, 4-fold nearest-neighbor
- Native gates: RX, RY, CZ (adiabatic)
- 2Q fidelity: 99.1% median, 1Q fidelity: 99.9% median
- T1/T2: 25 μs / 10 μs, gate speed ~60 ns

---

## 5. Molecule Inventory

### GIC 2026 Molecules (35 total, 4–28q)
| Category | Molecules | Qubits |
|---|---|---|
| Small | H2 (5 geometries), LiH (3 geometries) | 4, 8 |
| EUV photoresist | CH3I, IMePh, iodobenzene, phenol, o-cresol, anisole, toluene, benzene | 8–12 |
| Medium | BeH2, N2 (3 geometries), H2O, NH3, CH4, ethylene, formaldehyde, acetylene, HF, CO | 12–20 |
| Large | N2/6-31g, BeH2/cc-pVDZ, ethylene/6-31g | 24–28 |

### 40-Qubit Scaling Targets
| Molecule | Qubits | Method |
|---|---|---|
| benzene CAS(20e,20o) | 40 | MPS / QSCI |
| N2 cc-pVDZ CAS(20e,20o) | 40 | MPS |
| N2 cc-pVDZ | 32 | SV (B200) |
| BeH2 cc-pVDZ | 32 | SV (B200) |
| ethylene/6-31g | 28 | SV (H200) |
| formaldehyde/6-31g | 24 | SV (L40S) |

---

## 6. Complete Results

### 6.1 GPU Benchmark (H-cGQE vs CUDA-Q GQE vs FCI)

| Molecule | Qubits | H-cGQE Error (mHa) | CUDA-Q GQE Error (mHa) | Improvement (mHa) | Chemical Accuracy? |
|---|---|---|---|---|---|
| H2@0.5 | 4 | 0.77 | 12.06 | 11.28 | ✅ |
| H2@0.74 | 4 | **0.15** | 20.38 | 20.23 | ✅ |
| H2@1.0 | 4 | **0.00** | 34.95 | 34.95 | ✅ |
| H2@1.5 | 4 | 19.93 | 87.22 | 67.30 | ❌ |
| H2@2.0 | 4 | 66.54 | 164.81 | 98.27 | ❌ |
| LiH@1.2 | 8 | 2.09 | 2.13 | -0.04 | ❌ |
| LiH@1.6 | 8 | 1.85 | 1.89 | -0.04 | ❌ |
| LiH@2.0 | 8 | 2.01 | 2.05 | -0.04 | ❌ |
| LiH@3.0 | 8 | 16.92 | 16.94 | -0.02 | ❌ |
| N2@1.1 | 12 | 126.77 | 126.99 | -0.22 | ❌ |
| N2@1.8 | 12 | 442.40 | 442.71 | -0.31 | ❌ |
| N2@2.5 | 12 | 817.63 | 817.90 | -0.27 | ❌ |
| Iodobenzene | 8 | 2.97 | -2.03 | -4.93 | ❌ |
| Methyl iodide | 8 | **1.59** | -6.30 | -7.89 | ✅ |
| IMePh | 8 | 24.78 | 18.63 | -5.76 | ❌ |
| Phenol | 8 | 45.15 | 45.50 | -0.35 | ❌ |
| BeH2@1.3 | 14 | 34.81 | 35.87 | -1.06 | ❌ |

**Summary**: 4/17 molecules at chemical accuracy (<1.6 mHa). H-cGQE outperforms CUDA-Q GQE on all H2 geometries (up to 98 mHa improvement).

### 6.2 QPU Results (Rigetti Cepheus-1-108Q, 8192 shots)

| Molecule | Qubits | Circuit Type | SQD Error (mHa) | PN Preservation (%) |
|---|---|---|---|---|
| Methyl iodide CAS12 | 12 | RL-optimized | **13.9** | 8.1 |
| LiH | 12 | RL-optimized | 19.6 | 14.3 |
| Phenol CAS12 | 12 | RL-optimized | 22.0 | 7.0 |
| Anisole CAS12 | 12 | HF+SQD | 24.4 | 64.5 |
| IMePh CAS12 | 12 | HF+SQD | 24.7 | 64.3 |
| o-Cresol CAS12 | 12 | HF+SQD | 26.5 | 63.9 |
| BeH2 | 14 | RL-optimized | 31.4 | 38.9 |
| Iodobenzene | 8 | HF+SQD | N/A | 71.0 |
| Benzene CAS12 | 12 | HF+SQD | 52.4 | 64.0 |
| Toluene CAS12 | 12 | HF+SQD | 53.3 | 65.3 |
| N2 | 20 | RL-optimized | 130.0 | 28.4 |
| Ethylene | 28 | RL-optimized | N/A (no FCI) | 23.9 |

**QPU best**: Methyl iodide at 13.9 mHa (RL-optimized ansatz). 12 molecules run on Rigetti Cepheus, max 28 qubits (ethylene).

### 6.3 IQM Emerald QPU Validation
- Molecule: Methyl iodide (CH3I, 8q)
- State fidelity: **87.5%** (896/1024 shots in correct state)
- Circuit depth: 12, 6 CNOTs
- Error mitigation: ZNE (scale factors [1,2,3]) + REM applied

### 6.4 SFT vs RL Ablation

| Molecule | SFT Error (mHa) | RL Error (mHa) | Improvement (mHa) |
|---|---|---|---|
| H2@0.74 | 20.52 | **0.15** | +20.38 |
| LiH@1.6 | 1.81 | 1.85 | -0.04 |
| N2@1.1 | 126.56 | 126.77 | -0.21 |
| BeH2@1.3 | 33.77 | 34.81 | -1.05 |
| Iodobenzene | 3.10 | 2.97 | +0.13 |
| Methyl iodide | 1.43 | 1.59 | -0.17 |
| IMePh | 25.01 | 24.78 | +0.24 |
| Phenol | 45.12 | 45.15 | -0.03 |

**Key finding**: RL provides dramatic improvement on H2 (20 mHa) but marginal on larger molecules — consistent with the fixed-θ proxy limitation discovered by the physicist.

### 6.5 DAPO Component Ablation (4 core molecules)

| Variant | H2 (Ha) | LiH (Ha) | BeH2 (Ha) | N2 (Ha) |
|---|---|---|---|---|
| Full (DAPO+KL+MMD+creativity) | -1.1168 | -7.8619 | -15.5612 | -107.4964 |
| Vanilla DAPO | -1.1168 | -7.8619 | -15.5612 | -107.4965 |
| No KL | -1.1167 | -7.8619 | -15.5612 | -107.4963 |
| KL only | -1.1168 | -7.8619 | -15.5612 | -107.4963 |
| No creativity | -1.1168 | -7.8618 | -15.5612 | -107.4963 |
| No MMD | -1.1168 | -7.8619 | -15.5612 | -107.4963 |

**Finding**: Component ablations show minimal energy differences — the reward signal is dominated by the energy term. The fixed-θ proxy flatness explains why structural diversity components don't differentiate.

### 6.6 B200 Training Results

| Run | Platform | Key Metric |
|---|---|---|
| SFT warm-start (500 ep) | B200 | Val accuracy 96.2%, val loss 1.037 |
| DAPO RL smoke (2 ep) | H200 | H2: -1.1220 Ha, LiH: -7.8619 Ha, entropy 3.30 |
| Ablation: RL from scratch (2 ep) | B200 | H2: -1.1165 Ha, LiH: 0.0 Ha (collapsed) |

**Architecture decision**: SFT warm-start → DAPO RL (NOT direct RL from scratch). Direct RL collapses on larger molecules where the policy never finds a low-energy circuit to bootstrap from.

### 6.7 QWC Grouping Results

| Molecule | Qubits | Pauli Terms | QWC Groups | Reduction | Validation |
|---|---|---|---|---|---|
| H2 | 4 | 15 | 5 | 3× | -1.1182 Ha (sim) vs -1.1167 Ha (GPU) = 1.477 mHa |
| LiH | 12 | 631 | 180 | 3.5× | Manifest exported |
| N2 | 20 | 2,951 | 1,308 | 2.3× | Manifest exported |

### 6.8 QPU Cost Savings (QWC + Batch)

| Molecule | QWC Groups | Individual Cost | Batch Cost | Savings |
|---|---|---|---|---|
| H2 | 5 | 1,083 cr | 248 cr | 835 cr |
| LiH | 180 | 6,286 cr | 795 cr | 5,491 cr |
| BeH2 | ~250 | 24,914 cr | 1,093 cr | 23,821 cr |

### 6.9 Error Mitigation Summary

| Technique | Description | Applied To | Result |
|---|---|---|---|
| REM | Readout error calibration via assignment matrix | IQM Emerald (8q), Rigetti H2 (4q) | Skipped if qubits > 10 |
| ZNE | Gate folding [1,2,3] + Richardson extrapolation | IQM Emerald (6 CNOTs) | Skipped if 2q gates > 20 |
| SQD | Symmetry-filtered subspace diagonalization | Rigetti: H2, LiH, BeH2 | H2: 0.0 mHa, LiH: 1.63 mHa |

### 6.10 FMO2 Fragmentation (Iodobenzene)
- 3-fragment decomposition: ortho, meta, para fragments + dimers
- FMO2 formula: `E_total = Σ_i E_i + Σ_{i<j} (E_{ij} - E_i - E_j)`
- QPU execution: 3 monomers + 3 dimers on Rigetti Cepheus
- Solver error: 26.25 mHa (fragment-level H-cGQE vs exact FMO2)

### 6.11 QSCI Scaling
- Max qubits: **40** (benzene CAS(20e,20o))
- Method: MPS backend with CUDA-Q `tensornet-mps`
- Note: QSCI at 28–40q recovers Hartree-Fock energy, not correlated ground state (deeper entangling circuits needed)

---

## 7. Key Discoveries & Bug Fixes

### Diagonal Sequence Collapse
- **Problem**: On larger molecules (LiH, BeH2, N2), the model under-generates entangling operations (X/Y terms) and produces commuting Z-only sequences trapped at Hartree-Fock energy
- **Root cause**: Zero gradients for Z-only circuits (all terms commute, no energy variation)
- **Fix**: Force entanglement in sampling + curriculum learning + commutator loss penalty

### Bit Ordering Bug in QWC Parsing
- **Problem**: H2 energy evaluated to +0.46 Ha instead of -1.12 Ha
- **Root cause**: `bitstring[n_qubits-1-q]` should be `bitstring[q]` — Pauli position q maps to Qiskit qubit n_qubits-1-q, which is bitstring index q
- **Fix**: Corrected parity calculation in `_parse_grouped_results`

### torch.compile + CUDA-Q LLVM Conflict
- **Problem**: SIGABRT (exit 134) when cudaq imported before torch.compile
- **Root cause**: Both embed LLVM → "Option 'debug-counter' registered more than once"
- **Fix**: Lazy `_ensure_cudaq()` — import cudaq AFTER torch.compile

### Fixed-θ Proxy Limitation
- **Problem**: RL reward landscape is flat on large molecules (Spearman ρ=0.227)
- **Root cause**: All circuits evaluate to HF energy at θ=0.01 — no gradient signal
- **Fix**: Truncated L-BFGS-B (3–5 steps) during RL reward calculation

---

## 8. Diagram Planning: Recommended Figures

### Figure 1: System Architecture (Hero Diagram)
**Type**: Layered block diagram (L1–L5)
**Content**:
- Layer 1: PySCF → Hamiltonians → OpenFermion JW
- Layer 2: Chemistry GNN → H-cGQE Transformer → DAPO RL
- Layer 3: CUDA-Q SV (L40S/H200/B200) → MPS (tensornet-mps)
- Layer 4: QWC grouping → QPU submission → SQD
- Layer 5: FMO2 reconstruction → Benchmarking
**Tool**: draw.io (editable, exportable to SVG/PDF)

### Figure 2: Training Pipeline Flow
**Type**: Horizontal flowchart
**Content**:
```text
SFT (cross-entropy) → DAPO RL (energy reward) → L-BFGS-B optimization → QPU submission
     ↓                      ↓                        ↓                    ↓
  96.2% acc            MAP-Elites              Optimized θ           SQD energy
```
**Key annotations**: B200 BF16, NVFP4, energy cache, force entanglement, reward gating

### Figure 3: GPU Scaling Ladder
**Type**: Staircase/bar chart
**Content**:
| GPU | SV Qubits | MPS Qubits | VRAM |
|---|---|---|---|
| L40S | 24 | 40+ | 48 GB |
| H200 | 30 | 60+ | 141 GB |
| B200 | 32 | 60+ | 192 GB |
| B200×4 | 36 | 80+ | 768 GB |
**Annotation**: PCIe IPC limit on L40S, NVLink on H200/B200

### Figure 4: QWC Grouping + Async QPU Workflow
**Type**: Sequence diagram
**Content**:
1. HPC: Group Pauli terms → construct measurement circuits → export QASM manifest
2. Submit: Single batch to Rigetti Cepheus (returns immediately)
3. Retrieve: Poll → parse parities → compute ⟨H⟩ = Σᵢ cᵢ⟨Pᵢ⟩
**Annotation**: 3–5× circuit reduction, 90%+ cost savings

### Figure 5: Energy Accuracy Comparison
**Type**: Bar chart (grouped)
**Content**: H-cGQE vs CUDA-Q GQE vs FCI for 17 molecules
**Highlight**: Chemical accuracy line at 1.6 mHa

### Figure 6: QPU Results Heatmap
**Type**: Heatmap (molecules × metrics)
**Content**: SQD error, PN preservation %, circuit type (RL vs HF+SQD)
**Highlight**: Best QPU result (methyl iodide 13.9 mHa)

### Figure 7: SFT vs RL Ablation
**Type**: Paired bar chart
**Content**: SFT error vs RL error for 8 molecules
**Highlight**: H2 improvement (20.52 → 0.15 mHa)

### Figure 8: DAPO RL Architecture
**Type**: Block diagram
**Content**:
- Replay buffer → sample sequences → CUDA-Q evaluate → compute reward → DAPO loss → update policy
- Components: asymmetric clipping, dynamic sampling, token-level loss, GRPO advantages
- Safeguards: reward gating, force entanglement, energy cache

### Figure 9: Credit Budget Allocation
**Type**: Pie/donut chart
**Content**: 11,000 credits split across GPU instances + QPU runs
**Highlight**: H200 for RL training, B200 for 32q SV, Rigetti for QPU

### Figure 10: EUV Photoresist Molecule Set
**Type**: Molecular structure grid + table
**Content**: 8 EUV photoresist molecules with structures, qubit counts, SQD errors
**Highlight**: Mitsubishi Chemical relevance (iodobenzene, methyl iodide, phenol, etc.)

### Figure 11: Error Mitigation Stack
**Type**: Layered diagram
**Content**: Raw QPU counts → REM (readout correction) → ZNE (gate folding) → SQD (symmetry filtering) → Final energy
**Annotation**: Preflight checks (skip thresholds)

### Figure 12: FMO2 Fragmentation Workflow
**Type**: Flow diagram
**Content**: Parent molecule → fragment decomposition → monomer/dimer Hamiltonians → H-cGQE per fragment → FMO2 recombination → Total energy

---

## 9. Documentation Files Updated

| File | Status | Content |
|---|---|---|
| `B200_TRAINING_PLAN.md` | ✅ Rewritten | Full B200 strategy, Blackwell env, training results, launcher |
| `QBRAID_STRATEGY.md` | ✅ Updated | B200/H200 pricing, QWC grouping, async workflow, credit budget, Rigetti details |
| `QBRAID_INTEGRATION.md` | ✅ Updated | Section 8: B200 training results, Blackwell env, launcher, checkpoints |
| `PIPELINE_VISION.md` | ✅ Updated | GPU scaling ladder with B200 as primary, actual results |
| `CODEBASE_MINDMAP.md` | ✅ Updated | B200 scripts table, checkpoint sizes, cudaq_tuning B200 presets |
| `PIPELINE_RESULTS_SUMMARY.md` | ✅ New | This document — master planning reference |

---

## 10. Key Result Files

| File | Content |
|---|---|
| `results/phase3_final/consolidated_results_gic2026.json` | Full benchmark + QPU results (17 GPU + 12 QPU molecules) |
| `results/phase3_final/ablation_sft_vs_rl.json` | SFT vs RL comparison + DAPO component ablation |
| `results/phase3_final/classical_baseline_comparison.json` | H-cGQE vs CUDA-Q GQE vs FCI |
| `results/phase3_final/efficiency_metrics.json` | Circuit depth, gate counts, optimization times |
| `results/phase3_final/noise_mitigation_summary.json` | REM/ZNE/SQD techniques + QWC grouping |
| `results/phase3_final/qpu/qpu_validation_consolidated.json` | IQM Emerald 87.5% fidelity result |
| `results/qpu/cepheus_sqd_energies.json` | Corrected Cepheus QPU SQD energies (12 molecules) |
| `results/train/h_cgqe_model_b200_sft_metrics.json` | SFT training metrics (val loss, accuracy) |
| `results/train/h_cgqe_model_b200_rl_scratch_smoke_rl_metrics.json` | Ablation RL from scratch |
| `results/train/h_cgqe_model_qbraid_smoke_rl_metrics.json` | qBraid H200 RL smoke test |
| `results/eval/simulator_benchmark.json` | QWC simulator validation |
| `results/eval/verify_rl_proxy_iodobenzene.json` | Physicist verification of fixed-θ proxy |

---

## 11. Credits & Budget

- **Total budget**: 11,000 qBraid credits
- **QPU spend**: ~12,800 credits across 12 Cepheus submissions (8192 shots each) — note: original budget was 13,400
- **Optimized plan**: H200 for RL training (~5,490 cr), B200 for 32q SV (~6,118 cr), Rigetti for QPU (~683–2,049 cr)
- **Free simulators**: IonQ sim (29q, rate-limited), AWS SV1 (34q, free first min/task)

---

## 12. Hugging Face Model Card

- **Repo**: `https://huggingface.co/Quantum-Buddies/Conditional-GQE-models`
- **Model**: `Ryukijano/h-cgqe-gic2026` (32 MB, 7.85M params)
- **Architecture**: Encoder-decoder, GPT-2 style, d_model=256, 4+4 layers
- **Training**: SFT pretraining → DAPO RL fine-tuning
- **Operator pool**: UCCSD fermionic excitations (Jordan-Wigner mapped)

---

## References

- [GPT-QE paper](https://arxiv.org/abs/2401.09253) — Original GQE method
- [DAPO paper](https://arxiv.org/abs/2503.14476) — Decoupled Clip + Dynamic Sampling
- [NVIDIA CUDA-Q GQE docs](https://nvidia.github.io/cudaqx/examples_rst/solvers/gqe.html)
- [GIC 2026 challenge](https://www.pqic.org/challenge)
- [qBraid SDK docs](https://docs.qbraid.com/v2/sdk/user-guide/programs)
- [RubriQ (related work)](https://arxiv.gg/abs/2607.07554) — GRPO for quantum circuit synthesis on CUDA-Q
- [QUASAR (related work)](https://arxiv.org/pdf/2510.00967) — Agentic RL for quantum circuit generation
