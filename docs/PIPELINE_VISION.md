# Conditional-GQE: HPC + AI + Quantum Molecular Discovery Pipeline

## Architecture Overview

```
Molecular structure
        │
        ▼
┌──────────────────────────────────────────┐
│ Layer 1: HPC / CPU Chemistry             │
│                                          │
│ • Molecular integrals (PySCF)            │
│ • Active-space selection                 │
│ • Fermion-to-qubit mapping (JW/BK)       │
│ • Fragment generation (FMO)              │
│ • Classical reference methods            │
│ • Experiment tracking & job dispatch     │
└──────────────────┬───────────────────────┘
                   │ Hamiltonians + molecular graphs
                   ▼
┌──────────────────────────────────────────┐
│ Layer 2: AI / GPU Circuit Discovery      │
│                                          │
│ • Chemistry graph encoder                │
│ • Conditional circuit generation          │
│ • RL-based circuit optimization (DAPO)   │
│ • Curriculum learning                     │
│ • Hardware-aware conditioning             │
│ • Cross-molecule transfer                 │
│ • Post-training (RAFT, model soup)        │
└──────────────────┬───────────────────────┘
                   │ Candidate circuits
                   ▼
┌──────────────────────────────────────────┐
│ Layer 3: GPU Quantum Simulation          │
│                                          │
│ • Exact state-vector (4→30 qubits)       │
│ • MPS simulation (24→40 qubits)          │
│ • Multi-GPU Hamiltonian batching          │
│ • Noise simulation (depolarizing)         │
│ • Large circuit screening                │
│ • Training reward computation            │
└──────────────────┬───────────────────────┘
                   │ Selected compact circuits
                   ▼
┌──────────────────────────────────────────┐
│ Layer 4: QPU Validation                  │
│                                          │
│ • Hardware-aware transpilation           │
│ • Shot-based observable estimation       │
│ • Multi-device benchmarking              │
│ • Noise/error-mitigation experiments     │
│ • Cross-provider portability             │
└──────────────────┬───────────────────────┘
                   │ Fragment energies
                   ▼
┌──────────────────────────────────────────┐
│ Layer 5: HPC Integration & Analysis      │
│                                          │
│ • FMO2 many-body reconstruction          │
│ • Uncertainty propagation                │
│ • Comparison with classical references   │
│ • Scaling, accuracy and runtime metrics  │
│ • Final dashboard generation             │
└──────────────────────────────────────────┘
```

---

## Layer 1: HPC / CPU Chemistry

### Role
Scientific-computing backbone: molecular parsing, electronic structure, Hamiltonian construction, fragmentation, classical baselines, FMO reconstruction, and job orchestration.

### Components

#### 1.1 Molecular Integrals & Active Space (PySCF)
- **Existing**: `src/gqe/data/generate_hamiltonians.py` — PySCF SCF + OpenFermion JW transform
- **Enhancement**: Support larger basis sets (6-31g, cc-pVDZ) and bigger active spaces (CAS(12,12), CAS(14,14))
- **Output**: Hamiltonian JSON records with terms, coefficients, n_qubits, hf_energy, fci_energy

#### 1.2 Fragment Generation (FMO)
- **Existing**: `src/gqe/data/fragmentation.py` — `build_fragment_records()`, `ActiveSpaceSpec`
- **Existing**: `src/gqe/data/fragment_molecule.py` — CLI for fragment plan generation
- **New**: FMO2 many-body expansion recombination module
  - FMO2 formula: `E_total = Σ_i E_i + Σ_{i<j} (E_{ij} - E_i - E_j)`
  - For N=5 fragments: 5 monomers + 10 dimers = 15 calculations
  - Electrostatic embedding potential from surrounding fragments
  - Pair interaction energy (PIE) decomposition for analysis

#### 1.3 Classical Reference Methods
- **Existing**: `src/gqe/baselines/run_exact_diagonalization.py` — FCI reference
- **Existing**: `src/gqe/baselines/run_cudaq_vqe.py` — UCCSD-VQE baseline
- **Existing**: `src/gqe/baselines/run_adapt_vqe.py` — ADAPT-VQE baseline
- **New**: Fragment-level FCI references for FMO validation

#### 1.4 Experiment Tracking
- Freeze environment: `git rev-parse HEAD`, `python --version`, `pip freeze`, `nvidia-smi`, `cudaq.__version__`
- Log all job metadata (Slurm job ID, qBraid job QRN, device ID, shots, queue time)
- Never store API keys in repo/logs

### CPU Allocation Strategy
All PySCF calculations, OpenFermion transforms, fragment plan generation, FMO recombination, and classical baselines run on CPU. These are embarassingly parallel across fragments and can use OpenMP/multiprocessing.

---

## Layer 2: AI / GPU Circuit Discovery

### Role
Given molecular and Hamiltonian context (and optionally device constraints), generate a compact parameterized circuit expected to prepare a low-energy state.

### Components

#### 2.1 Chemistry Encoder (Existing)
- **File**: `src/gqe/models/chemistry_encoder.py`
- **File**: `src/gqe/data/graph_dataset.py`
- Encodes atoms, bonds, orbitals, geometry into molecule-aware conditioning vector
- Graph neural network → latent prior → prefix/GRU initial states for transformer

#### 2.2 Generative Policy (Existing)
- **File**: `src/gqe/models/h_cgqe_transformer.py`
- GPT-2 style autoregressive transformer (31.7M params)
- Generates Pauli word sequences from UCCSD operator pool
- Conditioned on chemistry encoder output

#### 2.3 RL Optimization (Existing + Enhanced)
- **File**: `src/gqe/models/train_rl_dapo.py`
- DAPO + REPO + Chemeleon2 rewards (KL, MMD diversity, creativity, entropy bonus)
- Off-policy GRPO with μ-reuse for sample efficiency
- Multi-component reward: `r = w₁·(-E/|E_ref|) + w₂·entanglement + w₃·(-depth) + w₄·non_commute + w₅·MMD + w₆·creativity`
- **Safeguard**: Auxiliary rewards (w₂-w₆) are gated on energy improvement over HF. If `E >= E_HF - threshold`, auxiliary rewards are zeroed — preventing reward hacking where the model optimizes structural metrics without lowering energy. Controlled via `--gate-auxiliary-rewards` and `--energy-improvement-threshold`.

#### 2.4 Hardware-Aware Conditioning (NEW)

**Basic version** — Add hardware cost to reward:
```python
r_hardware = -λ_d · depth - λ_2q · n_2q_gates - λ_native · n_non_native - λ_route · n_routing
```

**Strong version** — Condition generation on device descriptor:
```
p(circuit | molecule, Hamiltonian, device)
```

Device descriptor includes:
- Connectivity graph (coupling map)
- Native gate set (e.g., {RZ, RX, RY, CZ} for Rigetti, {RZ, RY, XX} for IonQ)
- Error rates per gate type
- Maximum circuit depth before decoherence

Implementation approach:
1. Extend chemistry encoder to accept device embedding (connectivity graph + gate set one-hot)
2. Add backend-specific allowed gate masks during sampling
3. Post-generation transpilation scoring (Qiskit transpiler level-0 for depth estimation)
4. Device-specific reward penalties during RL

```python
# Pseudocode for device-conditioned generation
device_embedding = encode_device(connectivity_graph, native_gates, error_rates)
molecule_embedding = chemistry_encoder(molecular_graph)
conditioning = concat(molecule_embedding, device_embedding)
circuit_tokens = transformer.decode(conditioning, operator_pool=device_aware_pool)
```

#### 2.5 Noise-Aware Fine-Tuning (NEW)

Four-stage offline-to-online transfer:

**Stage 2a — Ideal GPU pretraining** (existing):
- Train against exact GPU expectation values
- No noise model, full exploration

**Stage 2b — Noise-aware fine-tuning** (new):
- Add depolarizing noise model (CUDA-Q `noise_model`)
- Device-informed noise: gate-specific error rates from QPU calibration data
- Hardware cost reward:
  ```
  R = -Ê - λ_d·d - λ_2q·N_2q - λ_r·N_routing - λ_v·Var(Ê)
  ```
- Penalize circuits that are too deep for hardware coherence time

**Stage 2c — QPU calibration set** (new):
- Execute small diverse circuit set on chosen QPU
- Measure: simulator-to-hardware energy shift, circuit degradation vs 2q gate count, shot variance, hardware ranking consistency
- Fit a hardware-error surrogate model

**Stage 2d — Limited hardware-in-the-loop** (new):
- Only top-K circuits sent to QPU
- QPU results rerank finalists, calibrate surrogate, fine-tune small adapter
- Select final circuit for each fragment

#### 2.6 Curriculum Learning (Existing)
- Start with 4-qubit systems (H₂)
- Progress to 8q (LiH, CH₃I), 12q (N₂), 14q (BeH₂), 20q, 24q, 28q, 32q, 36q, 40q
- 30-epoch warmup per stage

#### 2.7 Post-Training (Existing)
- Iterative RAFT (STaR loop): `scripts/run_iterative_raft.sh`
- Model soup: `src/gqe/models/model_soup.py`
- Adaptive test-time compute: `--adaptive-n-samples`

---

## Layer 3: GPU Quantum Simulation

### Role
High-throughput training environment and digital twin for quantum hardware.

### Simulation Backends

#### 3.1 Exact State-Vector Simulation
- **CUDA-Q target**: `nvidia` (single-GPU) or `nvidia-mqpu` (multi-GPU pooling)
- **Scaling ladder**: 4 → 8 → 12 → 16 → 20 → 24 → 28 → 30 qubits
- **L40S limit**: 24 qubits (PCIe IPC segfault at 25q in distributed statevector)
- **H200 limit**: ~30 qubits (141GB VRAM, NVLink interconnect)
- **B200 limit**: ~32 qubits (192GB VRAM)
- **B200x4 limit**: ~36 qubits (768GB pooled, NVLink)

Memory formula: `2^n qubits × 16 bytes (complex128) = VRAM needed`
- 24q: 256 MB (easy)
- 28q: 4 GB (feasible on H200)
- 30q: 16 GB (feasible on H200/B200)
- 32q: 64 GB (B200 only)
- 36q: 1 TB (B200x4 with distribution)

#### 3.2 MPS Simulation (NEW)
- **CUDA-Q target**: `tensornet-mps`
- **Scaling ladder**: 24 → 28 → 32 → 36 → 40 qubits
- **Configuration**:
  ```bash
  export CUDAQ_MPS_MAX_BOND=256      # Max singular values to keep
  export CUDAQ_MPS_ABS_CUTOFF=1e-5   # Absolute truncation cutoff
  export CUDAQ_MPS_RELATIVE_CUTOFF=1e-5  # Relative truncation cutoff
  export CUDAQ_MPS_SVD_ALGO=GESVDJ   # SVD algorithm (Jacobi for GPU)
  ```
- **Reporting requirements**:
  - Bond dimension χ (actual vs max)
  - Truncation error estimate
  - Runtime per circuit
  - Convergence vs exact (for small cases)
- **Key caveat**: A completed 40-qubit MPS run is NOT automatically an accurate 40-qubit result. Report bond dimension and truncation error.
- **Safeguard implemented**: The scaling script now requires multiple bond dimensions and reports convergence. A single bond dimension result is never presented as an accuracy claim. Statevector is explicitly capped at 24q (`--statevector-max-qubits`).

#### 3.3 Multi-GPU Evaluation (Existing)
- **CUDA-Q**: `nvidia-mqpu` target for Hamiltonian-term batching
- **CUDA-Q**: `cudaq.parallel` for candidate-circuit batching
- Parallel across: Hamiltonian terms, candidate circuits, geometry sweeps, seed sweeps, fragment-level execution

#### 3.4 Noise Simulation (NEW)
- **CUDA-Q**: Built-in noise model support
- **Noise types**: Depolarizing, amplitude damping, dephasing, Kraus operator channels
- **Device-informed**: Load calibration data from QPU provider (gate error rates, T1/T2)
- **Use cases**:
  - Reward computation during noise-aware fine-tuning (Stage 2b)
  - Pre-QPU validation: noisy simulation before hardware submission
  - Computing ideal references for QPU output comparison

#### 3.5 Large Circuit Screening
- Screen thousands of AI-generated candidate circuits
- Filter by: energy < threshold, depth < max_depth, 2q gate count < max_2q
- Rank by energy × hardware cost trade-off
- Output: top-K shallow circuits for QPU submission

### GPU Scaling Ladder

| Mode | Qubits | Backend | GPU | VRAM Needed | Notes |
|------|--------|---------|-----|-------------|-------|
| SV | 4-24 | `nvidia` / `nvidia-mqpu` | L40S | 256MB-256MB | Current capability |
| SV | 4-30 | `nvidia` | H200 | up to 16GB | qBraid H200 |
| SV | 4-32 | `nvidia` | B200 | up to 64GB | qBraid B200 |
| SV | 4-36 | `nvidia-mqpu` | B200x4 | up to 1TB (distributed) | qBraid B200x4 |
| MPS | 24-40 | `tensornet-mps` | any GPU | depends on χ | Approximate, report χ |

---

## Layer 4: QPU Validation

### Role
Hardware validation, device-aware circuit assessment, and selected fragment execution. NOT decorative — every QPU job has a precise scientific purpose.

### QPU Funnel

```
Many AI-generated circuits (1000s)
          │
          ▼
Exact GPU evaluation (all circuits)
          │
          ▼
MPS/noisy simulation (top 100)
          │
          ▼
Hardware-aware scoring (top 20)
          │
          ▼
Top few shallow circuits (3-5)
          │
          ▼
Real QPU execution
```

### qBraid QPU Access

#### Device Discovery
```bash
# List all online QPUs
qbraid devices list --type QPU --status ONLINE

# Inspect a specific device
qbraid devices get <device-id>
```

#### CUDA-Q QPU Submission
```python
import cudaq

# Set qBraid as target with specific QPU
cudaq.set_target("qbraid", machine="<qbraid-device-id>")

# Submit kernel for execution
result = cudaq.sample(kernel, parameters, shots_count=4000)

# Async submission (for queue management)
future = cudaq.sample_async(kernel, parameters, shots_count=4000)
# Persist future to disk, retrieve later
```

#### qBraid SDK Direct Submission
```python
from qbraid.runtime import GroupJobSession, QbraidProvider

provider = QbraidProvider()

with GroupJobSession(name="fragment-cross-device-benchmark") as group:
    device_a = provider.get_device("aws:aws:sim:sv1")
    job1 = device_a.run(circuit_qasm, shots=4000)

    device_b = provider.get_device("ionq:ionq:qpu.aria-1")
    job2 = device_b.run(circuit_qasm, shots=4000)

results = group.results(timeout=300)
```

### QPU Pricing (qBraid Credits)

| Provider | QPU | Per-task | Per-shot | Min shots | Notes |
|----------|-----|----------|----------|-----------|-------|
| IQM | Garnet | 30 cr | 0.145 cr | — | Superconducting, 54q |
| IQM | Emerald | 30 cr | 0.16 cr | — | Superconducting |
| Rigetti (AWS) | Cepheus-1-108Q | 30 cr | 0.0425 cr | — | Superconducting, 108q |
| IonQ | Aria-1 | 30 cr | 3 cr | 2500* | Trapped-ion, all-to-all |
| IonQ | Forte-1 | 30 cr | 8 cr | — | Trapped-ion, highest fidelity |
| AQT | IBEX Q1 | 30 cr | 2.35 cr | — | Trapped-ion |

*IonQ error mitigation requires minimum 2,500 shots per task.

**Budget estimate** (11,000 credits):
- IQM Garnet: 30 + 0.145×4000 = 610 cr per circuit → ~18 circuits
- Rigetti Cepheus: 30 + 0.0425×4000 = 200 cr per circuit → ~55 circuits
- IonQ Aria-1: 30 + 3×2500 = 7,530 cr per circuit → ~1 circuit (expensive!)

**Recommended**: Use Rigetti or IQM for most experiments, IonQ for portability demo only.

### Three QPU Experiments

#### Experiment A — Circuit Portability
**Purpose**: Prove the learned circuit can run on real hardware.

- Select: one 2-4 qubit fragment, shallow AI-generated circuit, small observable set, one accessible QPU
- Compare:
  1. Ideal GPU value (exact statevector)
  2. Noisy simulator value (CUDA-Q noise model)
  3. Raw QPU value
  4. Mitigated QPU value (if available)
- **Success criterion**: QPU energy within 10% of noisy simulator prediction

#### Experiment B — AI Ansatz vs Fixed Ansatz
**Purpose**: Demonstrate AI component improves hardware suitability.

For same fragment and optimizer budget, compare:
1. Hardware-efficient fixed ansatz (Qiskit `EfficientSU2`)
2. UCC-inspired ansatz (standard UCCSD truncated)
3. H-cGQE-generated ansatz

Measure:
- Pre-transpilation depth
- Post-transpilation depth (native gate set)
- Two-qubit gate count
- Circuit fidelity proxy (product of gate fidelities)
- Raw energy error
- Mitigated energy error
- Shot count
- Hardware runtime

**Strongest potential result**: "H-cGQE obtains comparable energy with fewer native two-qubit gates and lower hardware error."

#### Experiment C — Cross-QPU Portability
**Purpose**: Show value of qBraid's multi-provider platform.

Run same small fragment on two different hardware modalities:
- Trapped-ion (IonQ Aria) vs superconducting (IQM Garnet)
- OR two superconducting providers (IQM vs Rigetti)

Record per device:
- Physical qubits selected
- Native gate set
- Compiled depth
- Two-qubit gate count
- Queue time
- Execution time
- Shots
- Raw error
- Mitigated error

Use `GroupJobSession` for unified tracking.

### Anti-Patterns (Do NOT)
- ❌ Put RL training loop directly around remote QPU (too slow, costly, noisy)
- ❌ Send whole 30-40q chemistry experiments to QPUs
- ❌ Submit full experiment batch before validating circuit compilation, parameter binding, qubit ordering, bit ordering, measurement convention, Hamiltonian grouping, result parsing, retry behavior, job persistence
- ❌ Store API keys in repo, notebook, logs, or report

### Safeguards Implemented
- **Circuit complexity preflight**: `_circuit_complexity()` computes depth and two-qubit gate count via Qiskit decomposition before submission
- **ZNE auto-skip**: ZNE is skipped if two-qubit gate count exceeds threshold (default: 20) — gate folding would make the circuit too deep for meaningful extrapolation
- **REM auto-skip**: Full assignment-matrix REM calibration is skipped if qubit count exceeds threshold (default: 10) — `2^n × 2^n` calibration is exponential
- **Configurable**: `--max-zne-two-qubit-gates 20 --max-rem-qubits 10`

### Known Limitation: QPU Energy Evaluation
The current QPU submission pipeline uses an approximate ideal energy proxy (probability of the all-zeros state) rather than full Hamiltonian expectation value measurement. Full Pauli-basis measurement grouping is needed before QPU energy results can be compared to simulator energies. This is the next priority for the QPU pipeline.

---

## Layer 5: HPC Integration & Analysis

### Role
Recombine fragment energies, propagate uncertainties, compare with classical references, and produce final metrics.

### FMO2 Many-Body Expansion

#### Formula
```
E_FMO2 = Σ_i E_i + Σ_{i<j} ΔE_{ij}

where:
  E_i = monomer energy (fragment i in embedding field)
  ΔE_{ij} = E_{ij} - E_i - E_j  (pair interaction energy)
  E_{ij} = dimer energy (fragments i+j in embedding field)
```

For N=5 fragments: 5 monomers + C(5,2)=10 dimers = 15 calculations

#### Implementation Plan
- **New module**: `src/gqe/eval/fmo_recombination.py`
  - `compute_fmo2_energy(monomer_energies, dimer_energies) → E_FMO2`
  - `pair_interaction_energy_decomposition() → PIE analysis`
  - `uncertainty_propagation(errors) → total uncertainty`
  - Support both simulator-only and hybrid GPU/QPU energy inputs

#### FMO Bridge: 40q Parent → Small Fragments

```
40-qubit parent active space
           │
           ▼
Five chemically meaningful fragments
           │
           ▼
Monomer and selected dimer Hamiltonians
           │
           ▼
4–12-qubit fragment circuits
           │
     ┌─────┴──────┐
     ▼            ▼
GPU simulator    QPU
     │            │
     └─────┬──────┘
           ▼
FMO2 energy reconstruction
```

**Correct claim**: "We map a parent active space of approximately 40 qubits into independently solvable monomer and dimer subproblems. AI-generated fragment circuits are optimized and screened using GPU simulation, selected instances are executed on qBraid-accessible QPUs, and fragment energies are recombined through an FMO2 many-body expansion."

### Uncertainty Propagation
- Fragment energy uncertainties from: shot noise, optimization convergence, MPS truncation
- FMO2 total uncertainty: `σ_total = sqrt(Σ σ_i² + Σ σ_{ij}²)`
- Report confidence intervals on final reconstructed energy

### Final Dashboard

| Metric | Value |
|--------|-------|
| Parent qubits | ~40 |
| Max fragment qubits | 4-12 |
| GPU simulator backend | H200 / B200 |
| QPU backend | IQM Garnet / Rigetti / IonQ |
| Generated circuit depth | X |
| Native two-qubit gate count | Y |
| Simulator energy error | Z mHa |
| QPU energy error | W mHa |
| Reconstructed FMO energy | E_FMO2 Ha |
| Total runtime | T hours |

---

## Experiment Priority

### P0 — Lock Computational Environment
```bash
git rev-parse HEAD > environment-freeze/git_commit.txt
python --version > environment-freeze/python_version.txt
pip freeze > environment-freeze/requirements.txt
nvidia-smi > environment-freeze/gpu_info.txt
python -c "import cudaq; print(cudaq.__version__)" > environment-freeze/cudaq_version.txt
qbraid devices list --type QPU --status ONLINE > environment-freeze/available_qpus.txt
```
Never place API key in repository, notebook, logs, or report.

### P1 — GPU Scaling Backbone
- Direct state vector from 4 to largest stable width (24q on L40S, 30q on H200, 32q on B200)
- MPS from overlap region up to 40 qubits (report bond dimension, cutoff, runtime, convergence)
- 1/2/3-GPU throughput measurements
- Curriculum and chemistry-conditioning ablations

### P2 — IMePh FMO Benchmark
For every fragment, save:
- Fragment atoms, active electrons/orbitals, qubit count, Pauli-term count
- Generated circuit, circuit depth, two-qubit gate count
- H-cGQE energy, exact-fragment reference, runtime

Reconstruct:
- Exact-fragment FMO energy (all 15 on GPU)
- H-cGQE simulator FMO energy
- Hybrid QPU/simulator FMO energy (selected fragments on QPU)
- Compare difference

### P3 — QPU Pilot (Smallest Fragment First)
Test before scaling:
- Circuit compilation, parameter binding, qubit ordering, bit ordering
- Measurement convention, Hamiltonian grouping, result parsing
- Retry behavior, job persistence

### P4 — AI vs Conventional Circuits
Run AI-generated circuit + baseline ansatz on:
- Ideal simulation, noisy simulation, one QPU
- Repeat over several seeds/parameter starts

### P5 — Multi-QPU Experiment
If credits, queue times, and device access permit:
- Run best 1-2 circuits on second QPU architecture
- Supports portability claim

---

## Essential Metrics

### Chemistry Accuracy
- `ΔE = E_method - E_reference` in Hartree and mHa
- Chemical accuracy threshold: 1.6 mHa
- Report which reference and active space are used

### AI Performance
- Energy after fixed evaluation budget
- Evaluations to target error
- Success rate across seeds
- Transfer performance on unseen molecules
- Generated depth, two-qubit gate count

### HPC/GPU Performance
- GPU type and count
- Wall-clock time, peak GPU memory
- Circuits evaluated per second
- Hamiltonian terms per second
- 1/2/3-GPU speedup, parallel efficiency
- MPS bond dimension and cutoff

### QPU Performance
- Provider/device identifier
- Physical qubits used, native gate set
- Transpiled depth, two-qubit gate count
- Number of shots, queue time, execution time
- Raw energy, mitigigated energy, uncertainty/CI

### FMO Performance
- Parent active-space qubits
- Maximum executed fragment qubits
- Number of monomers, dimers, expansion order
- Quantum-solver error, fragmentation error, total reconstructed error

---

## Four Claims for Final Submission

### Claim 1 — AI reduces circuit-search cost
Chemistry-conditioned curriculum learning improves circuit quality and convergence on unseen molecular configurations relative to training from scratch and fixed ansatz baselines.
**Requires**: molecule-held-out test set

### Claim 2 — HPC/GPU resources extend the tractable range
GPU state-vector and tensor-network backends provide high-throughput reward evaluation and support direct or approximate simulation across increasing active-space widths.
**Requires**: measured runtime, error, and memory results

### Claim 3 — Fragmentation maps larger chemistry to smaller quantum workloads
FMO2 decomposes a parent active space of approximately 40 qubits into chemically meaningful low-qubit fragment calculations that can be distributed across GPUs and selectively executed on QPUs.
**Requires**: real parent system, fragment definitions, reconstructed energy

### Claim 4 — Generated circuits execute on real hardware
Selected H-cGQE circuits are transpiled and evaluated on qBraid-accessible quantum processors, with accuracy and resource costs compared against ideal simulation and conventional ansätze.
**Requires**: actual job identifiers and result files

---

## Implementation File Plan

### New Files

| File | Layer | Description |
|------|-------|-------------|
| `src/gqe/eval/fmo_recombination.py` | 5 | FMO2 energy reconstruction + uncertainty propagation |
| `src/gqe/models/hardware_encoder.py` | 2 | Device descriptor encoder (connectivity, gate set, errors) |
| `src/gqe/models/train_noise_aware.py` | 2 | Noise-aware fine-tuning stage (Stages 2b-2d) |
| `src/gqe/sim/mps_backend.py` | 3 | MPS simulation wrapper with bond dimension reporting |
| `src/gqe/sim/noise_model.py` | 3 | CUDA-Q noise model construction from device calibration |
| `src/gqe/sim/circuit_screening.py` | 3 | Large-scale circuit screening and ranking |
| `src/gqe/qpu/qbraid_runtime.py` | 4 | qBraid QPU submission, device discovery, GroupJobSession |
| `src/gqe/qpu/transpile.py` | 4 | Hardware-aware transpilation and circuit export (OpenQASM) |
| `src/gqe/qpu/experiments.py` | 4 | Exp A/B/C framework with metrics collection |
| `src/gqe/qpu/mitigation.py` | 4 | Error mitigation wrappers (ZNE, readout correction) |
| `scripts/run_fmo_pipeline.sh` | All | Full FMO pipeline orchestrator |
| `scripts/run_gpu_scaling_benchmark.sh` | 3 | SV + MPS scaling benchmark suite |
| `scripts/run_qpu_experiments.sh` | 4 | QPU experiment orchestrator |
| `scripts/lock_environment.sh` | P0 | Environment freeze script |
| `configs/imoph_fmo.yaml` | 1+2 | IMePh FMO benchmark config (40q parent → 5 fragments) |

### Modified Files

| File | Change |
|------|--------|
| `src/gqe/models/train_rl_dapo.py` | Add `--device-conditioning`, `--noise-model`, `--hardware-cost-reward` flags |
| `src/gqe/models/h_cgqe_transformer.py` | Accept device embedding as additional conditioning input |
| `src/gqe/models/chemistry_encoder.py` | Optional device descriptor concatenation |
| `src/gqe/data/generate_hamiltonians.py` | Support 6-31g basis, larger active spaces, fragment-level generation |
| `src/gqe/data/fragmentation.py` | Add FMO2 dimer generation, electrostatic embedding |
| `src/gqe/eval/evaluate_h_cgqe.py` | Support MPS backend, noise model, multi-fragment evaluation |
| `scripts/run_qbraid_scaling.sh` | Add MPS stage, QPU validation stage, FMO recombination stage |

---

## qBraid Orchestration

### Instance Selection

| Stage | Instance | Credits/Hr | Purpose |
|-------|----------|------------|---------|
| RL Training | H200 | 549 | 28-30q SV reward computation |
| MPS Evaluation | H200 | 549 | 30-40q approximate simulation |
| Noise Simulation | B200 | 874 | Large noise model, 32q SV |
| QPU Submission | Any | per-shot | Device-dependent |
| FMO Recombination | CPU | 0 | Classical post-processing |

### Credit Budget (11,000 credits)

| Component | Credits | Notes |
|-----------|---------|-------|
| RL training (H200, 10hr) | 5,490 | Stage 1 + noise-aware fine-tuning |
| MPS benchmark (H200, 2hr) | 1,098 | 24-40q scaling |
| QPU Exp A (IQM Garnet, 3 circuits) | 1,830 | 3 × (30 + 0.145×4000) |
| QPU Exp B (IQM Garnet, 6 circuits) | 3,660 | 3 ansätze × 2 seeds |
| QPU Exp C (Rigetti, 2 circuits) | 400 | 2 × (30 + 0.0425×4000) |
| **Total** | **12,478** | Slightly over — reduce Exp B to 4 circuits |

Optimized budget: skip Exp C or use fewer shots → ~10,500 credits.

---

## Key References

- **Chemeleon2**: Park & Walsh, Nat. Mach. Intell. 2026, arXiv:2511.07158 — GRPO with creativity, diversity, KL rewards
- **FMO-VQE**: Nakagawa et al., Sci. Rep. 2024 — FMO-based VQE for large systems
- **CUDA-Q MPS**: arXiv:2501.15939 — MPS simulation on Grace Hopper
- **CUDA-Q qBraid target**: NVIDIA/cuda-quantum#4328 — Remote QPU submission
- **qBraid GroupJobSession**: qBraid SDK 0.12.0 — Cross-device job grouping
- **Hardware-aware ansatz**: CutVQA arXiv:2508.03376, HaQGNN arXiv:2506.21161
- **Noise-aware RL**: Quantum noise modeling through RL, IOPscience 2025
- **Model soups**: Wortsman et al., ICML 2022
- **Off-policy GRPO**: arXiv:2505.22257
- **Adaptive test-time compute**: Snell et al., Google DeepMind 2024
- **DeepSeek-R1**: STaR-style iterative rejection sampling
- **DAPO**: arXiv:2503.14476 — Clip-Higher, Dynamic Sampling, Token-level loss
- **REPO**: arXiv:2603.11682 — Regulated Entropy Policy Optimization
