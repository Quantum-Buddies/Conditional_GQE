# H-cGQE Full Pipeline Architecture — Detailed Spec for No-Code Orchestrator

## Overview

The H-cGQE pipeline spans **three compute tiers**: AIRE HPC (classical GPU training), qBraid Lab (quantum simulation + QPU access), and physical QPU hardware (Rigetti Cepheus, IQM Emerald, IonQ). The orchestrator must model each tier as a distinct stage with its own inputs, outputs, cost model, and failure modes.

---

## Tier 1: AIRE HPC (Classical GPU Compute)

### Hardware
- 28 nodes × 3 NVIDIA L40S GPUs (48GB, PCIe, no NVLink)
- Slurm scheduler: `--partition=gpu --gres=gpu:l40s:N`
- Conda env: `cudaq-env` at `/scratch/kcwp264/.conda_envs/cudaq-env/`
- Scratch: `/scratch/kcwp264/`

### Connection
- SSH into AIRE login node → `sbatch` job submission
- Or interactive: `srun --partition=gpu --gres=gpu:l40s:3 --time=04:00:00 --pty bash`

### Stage 1: Hamiltonian Generation
- **Script**: `src/gqe/data/generate_hamiltonians.py`
- **Config**: `configs/gic2026_molecules.yaml` (35 molecules, 4–28 qubits)
- **Input**: Molecular geometry + basis set (STO-3G, 6-31G)
- **Process**: PySCF → Hartree-Fock → OpenFermion → Jordan-Wigner → Pauli term list
- **Output**: `results/data/hamiltonians_gic2026/hamiltonians.json` (10 MB, 35 molecules)
- **Time**: ~5 minutes
- **Cost**: Free (HPC compute)

### Stage 2: Supervised Fine-Tuning (SFT)
- **Script**: `src/gqe/models/train_supervised.py`
- **Input**: Hamiltonians JSON + `configs/experiment.yaml` (model hyperparams)
- **Process**: GPT-2 style transformer trained on (molecule_metadata → UCCSD operator sequence) pairs
- **Output**: `results/train/h_cgqe_model_sft.pt` (~31 MB)
- **Time**: ~30 min on 1 GPU
- **Cost**: Free

### Stage 3: DAPO Reinforcement Learning
- **Script**: `src/gqe/models/train_rl_dapo.py`
- **Input**: SFT checkpoint + Hamiltonians JSON
- **Process**:
  1. Autoregressive sampling: model generates Pauli operator sequences
  2. Energy evaluation: CUDA-Q `nvidia-mqpu` target computes ⟨ψ|H|ψ⟩ on 3 GPUs
  3. Reward computation: `w1*(-E/|E_ref|) + w2*entanglement_frac + w3*(-depth/max_len) + w4*non_commute_frac`
  4. DAPO loss: asymmetric clipping (ε_low=0.2, ε_high=0.28), token-level, dynamic sampling
  5. GRPO advantages: (R_i - mean) / (std + eps)
- **Critical**: `torch.compile` must run BEFORE `import cudaq` (LLVM clash fix)
- **Output**: `results/train/h_cgqe_model_rl.pt` (~32 MB)
- **Time**: ~2-4 hours on 3 GPUs (50 epochs)
- **Cost**: Free
- **Key params**: `--max-qubits 24` (cuStateVec limit), `--target nvidia-mqpu`, `--n-epochs 50`

### Stage 4: Inference + L-BFGS-B Optimization
- **Script 4a**: `src/gqe/eval/evaluate_h_cgqe.py` (autoregressive inference)
- **Script 4b**: `src/gqe/eval/optimize_h_cgqe_coefficients.py` (classical optimization)
- **Input**: RL checkpoint + Hamiltonians JSON
- **Process**:
  1. Model generates operator sequences (Pauli words) for each molecule
  2. Qiskit circuit construction: HF initial state + Pauli rotation gates (CNOT ladders + RZ)
  3. L-BFGS-B optimizes rotation angles (thetas) to minimize ⟨ψ|H|ψ⟩
  4. Energy evaluation via CUDA-Q statevector on GPU
- **Output**: `results/eval/h_cgqe_optimized.json` (best operators + thetas + energy per molecule)
- **Time**: ~10-30 min on 3 GPUs
- **Cost**: Free

### Stage 5: Baselines
- **Scripts**: `src/gqe/baselines/run_cudaq_gqe.py`, `run_cudaq_vqe.py`, `run_exact_diagonalization.py`
- **Output**: `results/baselines/` (GQE, VQE, ADAPT-VQE, FCI reference energies)
- **Time**: ~20 min
- **Cost**: Free

### Stage 6: Consolidation + Figures
- **Scripts**: `scripts/consolidate_results.py`, `scripts/build_gic_benchmark.py`, `scripts/plot_phase3_report_figures.py`
- **Output**: `results/phase3_final/consolidated_results_gic2026.json`, `results/phase3_final/figures/*.png`
- **Time**: ~2 min
- **Cost**: Free

---

## Tier 2: qBraid Lab (Quantum Simulation + QPU Bridge)

### What is qBraid?
qBraid is a cloud-based quantum computing platform that provides:
1. **qBraid Lab**: Browser-based JupyterLab/VSCode with pre-configured quantum environments
2. **qBraid SDK**: Python runtime framework for cross-provider quantum job submission
3. **qBraid Runtime**: Unified API to 34+ quantum devices (QPUs + simulators) from IonQ, Rigetti, IQM, QuEra, AWS Braket, Azure Quantum
4. **GPU Fleet**: 20+ on-demand GPU instance types (including H200, B200) for hybrid workloads
5. **NVIDIA CUDA-Q integration**: qBraid is a remote cloud target in CUDA-Q's `nvq++` compiler

### How qBraid Connects to QPUs
qBraid acts as a **quantum cloud aggregator** — you use a single qBraid API key to access:
- **Rigetti Cepheus-1-108Q** (AWS Braket, superconducting, 108q)
- **IQM Emerald** (Azure Quantum, superconducting)
- **IonQ Forte/Tempo** (AWS/Azure, trapped ion)
- **QuEra Aquila** (AWS Braket, neutral atom)
- **AWS Braket SV1** (free statevector simulator, 34q)
- **qBraid QIR Simulator** (`qbraid:qbraid:sim:qir-sv`, free, 30q, 2000 shots)

### Authentication
```python
from qbraid import QbraidProvider
provider = QbraidProvider()  # uses QBRAID_API_KEY env var
devices = provider.get_devices()  # list all available devices
device = provider.get_device("aws:rigetti:qpu:cepheus-1-108q")
```

### qBraid "Runway" (GPU + Lab orchestration)
qBraid Lab provides on-demand GPU instances (H200, B200, L40S, etc.) accessible from the browser. This is NOT a separate product called "Runway" — it's qBraid Lab's GPU fleet. You launch a GPU instance from `account.qbraid.com`, get a JupyterLab environment with CUDA-Q + PyTorch pre-installed, and run hybrid quantum-classical workloads directly.

### Stage 7: Judge Validation (Free qBraid Simulator)
- **Script**: `scripts/validate_on_qbraid.py`
- **Input**: `results/eval/h_cgqe_optimized.json` + Hamiltonians JSON
- **Process**:
  1. Load optimized operators + thetas
  2. Build Qiskit parameterized circuit (HF state + Pauli rotations)
  3. Bind parameters, decompose to basic gates
  4. Group Hamiltonian Pauli terms by Qubit-Wise Commutativity (QWC)
  5. Build one measurement circuit per QWC group (basis changes: H for X, Sdg-H for Y)
  6. Submit batch to `qbraid:qbraid:sim:qir-sv` (free, 30q, 2000 shots)
  7. Parse counts → per-term expectations → total energy
- **Output**: `results/eval/qbraid_validation_report.json`
- **Cost**: FREE (qBraid QIR simulator)
- **Time**: ~5-15 min per molecule (rate limited)

### QWC Grouping (Key Algorithm)
```
Input: 15 Pauli terms for H2 (e.g., IZIZ, ZIZI, IZZZ, XIXI, ...)
Output: 5 QWC groups (3x reduction)
  Group 0: {ZIZI, IZIZ, IZZI, ZZII, ZZZZ} — all Z-basis, one circuit
  Group 1: {XIXI, IXIX} — X-basis on qubits 0,2, one circuit
  Group 2: {YIYI, IYIY} — Y-basis, one circuit
  ...
Each group → one circuit with combined measurement basis → one job to QPU/simulator
```

---

## Tier 3: QPU Execution (Physical Hardware)

### Stage 8: Manifest Export (HPC → Portable JSON)
- **Script**: `scripts/submit_qpu_async.py --export-only`
- **Input**: Optimized results + Hamiltonian records
- **Process**:
  1. Build Qiskit circuit from operators + thetas
  2. Group Pauli terms by QWC
  3. For each group: compose ansatz + measurement basis changes + measure_all
  4. Export QASM 2.0 string per group
  5. Package into self-contained JSON manifest: `{operators, thetas, groups: [{qasm, basis, terms}]}`
- **Output**: `results/qpu/{molecule}_manifest.json`
- **Why**: Decouples HPC compute from QPU submission. Manifest is portable — can be submitted from any machine with qBraid SDK installed.

### Stage 9: QPU Submission (qBraid → Physical Hardware)
- **Script**: `scripts/submit_sqd_to_cepheus.py` (SQD path) or `scripts/submit_qpu_async.py` (QWC path)
- **Input**: Manifest JSON + qBraid API key
- **Process**:
  1. `QbraidProvider()` authenticates with qBraid
  2. `provider.get_device("aws:rigetti:qpu:cepheus-1-108q")` resolves to Rigetti hardware
  3. qBraid transpiles Qiskit circuit → provider-native format (QIR/Quil/OpenQASM3)
  4. `device.run(circuit, shots=4096)` submits to QPU queue
  5. Returns job ID(s) — async, doesn't wait for completion
- **Ledger**: SQLite-backed `QpuLedger` tracks:
  - Circuit hash (idempotency: don't resubmit same circuit)
  - Job status (submitted → queued → running → completed/failed)
  - Cost estimate (credits per task + per shot)
  - Budget enforcement (hard limit, raises ValueError if exceeded)
  - Error classification (transient vs permanent, auto-retry up to 3x)
- **Output**: `results/qpu/{molecule}_submission_meta.json` (job IDs + group mapping)
- **Cost**: ~204 credits/molecule at 4096 shots (Rigetti Cepheus: 30 credits/task + 0.0425 credits/shot)
- **Budget**: 13,400 qBraid credits total

### Two QPU Execution Paths

#### Path A: SQD (Sample-based Quantum Diagonalization)
- **Measurement**: Computational basis only (Z-basis, no basis changes)
- **Circuits**: 1 per molecule
- **Post-processing**: SQD algorithm takes raw bitstring counts → filters by particle number symmetry → diagonalizes in subspace → variational energy estimate
- **Script**: `scripts/retrieve_and_sqd.py`
- **Advantage**: Fewer circuits, noise-tolerant (variational), works on any QPU
- **Limitation**: Requires n_electrons info, less accurate on noisy hardware

#### Path B: QWC (Qubit-Wise Commuting Grouped Measurement)
- **Measurement**: Per-group basis changes (H for X, Sdg-H for Y, nothing for Z)
- **Circuits**: N_groups per molecule (e.g., 5 for H2, 180 for LiH)
- **Post-processing**: Parse each group's counts → extract per-term expectation via parity bitmask → sum weighted expectations
- **Script**: `scripts/submit_qpu_async.py` + `retrieve_qbraid_job()`
- **Advantage**: Direct energy estimate, no variational post-processing
- **Limitation**: More circuits = more QPU time, shot noise per term

### Stage 10: Result Retrieval + SQD Post-Processing
- **Script**: `scripts/retrieve_and_sqd.py`
- **Input**: Submission metadata JSON (from Stage 9)
- **Process**:
  1. `load_job(job_id)` — retrieves job from qBraid by ID
  2. Check status: if queued/running, return "pending"
  3. If completed: `job.result()` → `result.data.get_counts()` → `{bitstring: count}`
  4. For Rigetti QPU: reverse bit order (qubit 0 is leftmost, opposite of Qiskit)
  5. Run SQD: filter by particle number → subspace diagonalization → energy
  6. Compare to FCI reference energy
- **Output**: `results/qpu/cepheus_sqd_results.json`
- **Cost**: Free (retrieval is always free)

### Device Selection Matrix

| Device ID | Type | Qubits | Cost | Batch Support | Use Case |
|---|---|---|---|---|---|
| `qbraid:qbraid:sim:qir-sv` | Simulator | 30 | Free | Yes | Judge validation |
| `aws:aws:sim:sv1` | Simulator | 34 | Free (1 min/task) | No | Pre-QPU testing |
| `ionq:ionq:sim:simulator` | Simulator | 29 | Free | No | Fallback sim |
| `aws:rigetti:qpu:cepheus-1-108q` | QPU | 108 | 30 cr/task + 0.0425 cr/shot | No | Production QPU runs |
| `azure:iqm:qpu:emerald-1-17q` | QPU | 17 | Varies | No | Small molecule QPU |

### Retry + Error Handling
- **Rate limiting (429)**: Exponential backoff, 6 retries, 5-30s delays
- **Transient errors**: Auto-retry 3x with classification (timeout, 503, connection reset)
- **Permanent errors**: No retry (unauthorized, unsupported gate, too many qubits)
- **Batch fallback**: `as_batch=True` → list mode → individual circuit loop
- **Local sim fallback**: If qBraid API completely fails, fall back to local Qiskit Statevector

---

## Data Flow Summary

```
[PySCF/OpenFermion]
       │
       ▼
[Hamiltonians JSON] ──────────────────────────────────┐
       │                                              │
       ▼                                              │
[SFT Training] ──→ [SFT Checkpoint .pt]               │
       │                                              │
       ▼                                              │
[DAPO RL Training] ──→ [RL Checkpoint .pt]            │
       │                                              │
       ▼                                              │
[Inference: Generate Operators]                       │
       │                                              │
       ▼                                              │
[L-BFGS-B: Optimize Thetas]                           │
       │                                              │
       ▼                                              │
[Optimized Results JSON]                              │
   {molecule, operators, thetas, energy}              │
       │                                              │
       ├──→ [QWC Manifest Export] ──→ [Manifest JSON] │
       │         (portable, self-contained)            │
       │                    │                          │
       │                    ▼                          │
       │         [qBraid QPU Submission]               │
       │         (QbraidProvider → device.run)         │
       │                    │                          │
       │                    ▼                          │
       │         [QPU Execution: Rigetti/IonQ/IQM]     │
       │                    │                          │
       │                    ▼                          │
       │         [Result Retrieval: load_job]          │
       │         (counts: {bitstring: count})          │
       │                    │                          │
       │                    ▼                          │
       │         [SQD Post-Processing]                 │
       │         (counts → subspace → energy)          │
       │                    │                          │
       │                    ▼                          │
       │         [QPU Energy Result JSON]              │
       │                                              │
       ▼                                              │
[qBraid Sim Validation] ◄────────────────────────────┘
   (free QIR simulator, judge reproducibility)
```

---

## Orchestrator Design Notes

### State Machine
Each molecule has a state: `uninitialized → hamiltonian_generated → sft_trained → rl_trained → inferred → optimized → manifest_exported → qpu_submitted → qpu_completed → sqd_processed → validated`

### Idempotency
- Circuit hash (SHA256 of QASM) prevents duplicate QPU submissions
- SQLite ledger enforces idempotency at the database level
- HPC stages use checkpoint files (skip if output exists)

### Cost Tracking
- HPC stages: free (AIRE compute)
- qBraid sim: free (QIR simulator)
- QPU: per-task + per-shot credits, tracked in ledger
- Budget enforcement: hard limit, raises before submission

### Parallelism
- HPC: 3 GPUs via CUDA-Q `nvidia-mqpu` (energy evaluation parallelized)
- QPU: multiple molecules can be submitted simultaneously (different job IDs)
- Retrieval: poll job statuses independently

### Failure Modes
1. **Rate limiting (429)**: Backoff + retry (transient)
2. **QPU offline**: Fallback to simulator or reschedule
3. **CUDA-Q/Triton LLVM clash**: Import order fix (torch.compile before cudaq)
4. **Diagonal sequence collapse**: RL force_entanglement + UCCSD operator pool
5. **cuStateVec segfault on >24q**: Hard cap at 24q for mqpu, MPS for larger
6. **Bit ordering**: Qiskit little-endian vs Rigetti big-endian (reverse_bits flag)

### Key File Paths
```
/scratch/kcwp264/Conditional-GQE_materials/
├── configs/gic2026_molecules.yaml          # 35 molecule definitions
├── configs/experiment.yaml                 # Model hyperparams
├── results/data/hamiltonians_gic2026/      # Generated Hamiltonians
├── results/train/h_cgqe_model_rl.pt        # RL checkpoint (32 MB)
├── results/eval/h_cgqe_optimized.json      # Best operators + thetas
├── results/qpu/{mol}_manifest.json         # Portable QPU manifest
├── results/qpu/{mol}_submission_meta.json  # Job IDs for retrieval
├── results/qpu/qpu_jobs.sqlite             # Ledger database
├── results/phase3_final/                   # Final results + figures
└── results/quaggle/                        # Pre-generated QASM circuits
```

### qBraid Launch Flow (for judges)
1. Judge clicks "Launch on qBraid" button in README
2. Redirects to `https://account.qbraid.com?gitHubUrl=https://github.com/Quantum-Buddies/Conditional_GQE.git`
3. qBraid clones the repo into a new Lab environment
4. Judge runs: `pip install -r requirements.txt && python scripts/download_models.py --only essential`
5. Judge runs: `python scripts/validate_on_qbraid.py` (free simulator, no credits needed)
6. Validation report shows H-cGQE energy vs GPU energy vs FCI reference
