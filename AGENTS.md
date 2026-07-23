## B200 Training Strategy (Canonical Decision — 2026-07-23)

### Architecture Decision: Supervised Warm-start → DAPO RL

**Main B200 run = SFT warm-start → DAPO RL (NOT direct RL from scratch)**

Rationale:
- The reward signal (CUDA-Q energy evaluation) is expensive and sparse on unseen 28–40q systems
- Direct RL from scratch is viable on small systems (4–20q) but collapses on large molecules where the policy never finds a low-energy circuit to bootstrap from
- SFT warm-start gives the policy a valid operator vocabulary and non-collapsed prior — tested in our ablation runs (`h_cgqe_rl_*_metrics.json`)
- Direct RL from scratch is retained as an **ablation** for the paper (flag: `--from-scratch`)

### Molecule Inventory (full, as of 2026-07-23)

| Dataset file | Count | Qubit range | Purpose |
|---|---|---|---|
| `hamiltonians_gic2026/hamiltonians.json` | 35 | 4–28q | Main RL training (GIC challenge molecules) |
| `hamiltonians_merged.json` | 21 | 4–40q | SFT training + scaling baselines |
| `hamiltonians_40plus/hamiltonians.json` | 10 | 4–40q | RL XL scaling run (benzene 40q, N₂ 40q) |
| `hamiltonians.json` | 5 | 4–20q | Legacy baseline set |
| `hamiltonians_iodobenzene.json/` | 2 | 8–12q | Iodobenzene CAS variants |

**GIC 2026 molecules (35):** h2, lih, beh2, n2, imeph_cas12, iodobenzene_cas12, methyl_iodide_cas12, phenol_cas12, ocresol_cas12, anisole_cas12, benzene_cas12, toluene_cas12, h2o, nh3, ch4, ethylene, formaldehyde, acetylene, hf, co, h2_0.5/1.0/1.5/2.0, lih_1.2/2.0/3.0, n2_1.8/2.5, beh2_1.0/1.6, lih_1.6_631g, n2_1.1_631g_cas8, h2o_1.0_631g_cas8, diarylethene_frag_cas12

**40q targets:** benzene_cas20 (40q), n2_ccpvdz_cas20 (40q), n2_ccpvdz (32q), beh2_ccpvdz (32q), ethylene (28q), formaldehyde (24q)

### B200 Launcher

**Portable script:** `scripts/launch_b200_training.sh`

```bash
# Full pipeline (SFT → RL → RL-40q):
bash scripts/launch_b200_training.sh both

# SFT only:
bash scripts/launch_b200_training.sh sft

# RL only (needs SFT checkpoint):
bash scripts/launch_b200_training.sh rl

# Ablation (direct RL from scratch, 4 core molecules):
bash scripts/launch_b200_training.sh ablation
```

Outputs:
- `results/train/h_cgqe_model_b200_sft.pt`      — SFT warm-start checkpoint
- `results/train/h_cgqe_model_b200_rl_main.pt`  — DAPO RL on 35 GIC molecules
- `results/train/h_cgqe_model_b200_rl_40q.pt`   — DAPO RL extended to 40q
- `results/train/h_cgqe_model_b200_rl_scratch.pt`— Ablation (scratch RL)

### B200 CUDA Setup (qBraid env)

The qBraid container provides system CUDA 13.2 but CUDA libs are accessed via pip packages.
The launcher auto-resolves `LD_LIBRARY_PATH` from the `nvidia` site-packages tree.

CUDA PyTorch install:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```
(cu126 = CUDA 12.6 build; sm_100/B200 kernels are compiled in via PTX JIT from PyTorch 2.7+)

For NVFP4 (Blackwell-specific):
```bash
pip install --no-build-isolation transformer_engine[pytorch]
```
Then use `--use-nvfp4` flag in `train_rl_dapo.py`.

---

## Learned User Preferences
- Prioritize short, planning-first phases before running or making substantive edits.
- Use tqdm for progress in data generation, adaptive VQE, and supervised training loops.
- Prefer `cudaq-env` conda environment for all CUDA-Q workloads on AIRE; use `environment-dgx-spark-cudaq.yml` for one-step DGX Spark CUDA-Q setup.
- Keep `--max-qubits 24` for mgpu runs on L40S (cuStateVec distribution threshold is 25; distributed statevector segfaults on PCIe-only L40S due to broken CUDA IPC in Open MPI's smcuda BTL).
- Use `nvidia-mqpu` backend for multi-GPU pooling (not `tensornet` with MPI — pip-installed CUDA-Q doesn't support MPI tensornet).
- Use `nvidia` single-GPU backend for H-cGQE inference (RLQF model doesn't need MPI).

## Learned Workspace Facts

### Project structure
- Project root: `/scratch/kcwp264/Conditional-GQE_materials`
- Upstream GitHub: `Quantum-Buddies/Conditional_GQE` (remote folder name `Conditional-GQE_materials`)
- DGX Spark env manifests: `environment-dgx-spark.yml` (base stack), `environment-dgx-spark-cudaq.yml` (includes `cudaq-solvers[gqe]`)
- Two-stage pipeline: (1) Autoregressive circuit synthesis via H-cGQE Transformer (GPT-2 style), (2) Classical L-BFGS-B optimization of rotation coefficients (thetas) parallelized across GPUs using CUDA-Q's `nvidia-mqpu` target.
- Key files:
  - `src/gqe/baselines/run_cudaq_gqe.py` — CUDA-Q GQE baseline
  - `src/gqe/baselines/run_cudaq_vqe.py` — UCCSD-style VQE baseline
  - `src/gqe/eval/evaluate_h_cgqe.py` — H-cGQE evaluation with `--max-qubits` filtering and MPI support
  - `src/gqe/eval/optimize_h_cgqe_coefficients.py` — L-BFGS-B coefficient optimization
  - `src/gqe/models/train_supervised.py` — Supervised transformer training (defaults to CPU unless `--use-cuda`)
  - `src/gqe/data/generate_hamiltonians.py` — Hamiltonian generation from molecular inputs
  - `configs/experiment.yaml` — Model scale and synthetic sample config
  - `scripts/run_scaling_mgpu.sh` — 2-GPU mgpu scaling (max 24 qubits, non-distributed)
  - `scripts/run_scaling_tn3gpu.sh` — tensornet 3-GPU (WARNING: not supported for pip CUDA-Q, kept as reference)
  - `scripts/run_scaling_3gpu.sh` — 3-GPU mqpu pooling script
  - `jobs/gqe-suite.slurm` — Slurm job script

### Environment
- **Conda env**: `cudaq-env` at `/scratch/kcwp264/.conda_envs/cudaq-env/` (also accessible as `/mnt/scratch/kcwp264/.conda_envs/cudaq-env/`)
- **Python**: via cudaq-env (CUDA-Q, Open MPI 5.0.10 with `--with-cuda`)
- **Platform**: AIRE HPC, 28 nodes × 3 NVIDIA L40S GPUs (PCIe-only, no NVLink), 24 AMD cores, 256GB RAM per node
- **Slurm**: `--partition=gpu --gres=gpu:l40s:N` (max 3 per node)

### Key discoveries and fixes
- **Diagonal sequence collapse**: On larger molecules (LiH, BeH2, N2), the model under-generates entangling operations (X/Y terms) and produces commuting Z-only sequences that get trapped at Hartree-Fock energy baseline due to zero gradients.
- **Proposed Phase 2 solutions**: (1) Symmetry-Preserving Masking (Constrained Decoding), (2) Curriculum Learning for Entanglement with commutator loss penalty, (3) Reinforcement Learning from Quantum Feedback (RLQF via PPO with energy-expectation dense rewards).
- **mgpu segfault fix**: Keep `n_qubits <= 24` to avoid cuStateVec distributed statevector mode (threshold=25) which segfaults on PCIe-only L40S due to broken CUDA IPC in Open MPI's smcuda BTL.
- **tensornet MPI limitation**: Multi-GPI tensornet requires CUDA-Q built from source against CUDA-aware MPI; pip-installed CUDA-Q segfaults with `-np > 1`.
- **CUDA context fix**: `_ensure_cuda_context()` in `evaluate_h_cgqe.py` creates a CUDA context per MPI rank before `MPI_Init()` using `ctypes.CDLL` to call `cudaSetDevice` + `cudaMalloc`.
- **Operator padding**: `evaluate_h_cgqe.py` pads/truncates Pauli words to match `n_qubits` before `cudaq.pauli_word()` conversion.
- **Active electron count**: Replaced `n_qubits // 2` heuristic with `get_active_electron_count()` for accurate electron count in evaluation.

### Phase 3 pipeline safeguards
- **RL reward gating**: Auxiliary rewards gated on energy improvement over HF (`--gate-auxiliary-rewards`, `--energy-improvement-threshold`). Prevents reward hacking. In `train_rl_dapo.py`.
- **Statevector cap**: Explicit 24q cap on exact statevector simulation (`--statevector-max-qubits 24`). MPS scaling script skips SV above cap. In `run_mps_scaling.py`.
- **MPS convergence reporting**: Multiple bond dimensions required for accuracy claims. Energy differences across D=32,64,128,256 reported. In `run_mps_scaling.py`.
- **QPU preflight checks**: ZNE skipped if two-qubit gates > 20 (`--max-zne-two-qubit-gates`); REM skipped if qubits > 10 (`--max-rem-qubits`). In `submit_qpu.py`.
- **Orbital reordering**: Intentionally excluded from MPS scaling. Synthetic CNOT chain + JW-mapped Hamiltonians would be physically inconsistent to reorder independently. Valid reordering requires regenerating fermionic Hamiltonian + operator pool with same orbital permutation.
- **QPU energy limitation**: Current `submit_qpu.py` uses approximate ideal energy proxy (all-zeros probability), not full Hamiltonian expectation. Full Pauli-basis measurement grouping is the next priority.

### Scripts and commands
- Full benchmark: `bash scripts/run_full_benchmark.sh` (optional `RUN_CUDAQ_GQE=1` for CUDA-Q GQE baseline)
- Plots: `bash scripts/plot_benchmarks.sh` (PNG outputs)
- Results aggregation: supports `--cudaq-baseline` and `--gqe-baseline` flags
- Supervised training: `src/gqe/models/train_supervised.py` (defaults to CPU unless `--use-cuda`; model scale from `configs/experiment.yaml` or CLI `--model-hidden`, `--model-layers`, `--model-vocab`, `--seq-samples`)
- CUDA-Q VQE: `run_cudaq_vqe.py` runs UCCSD-style VQE
- CUDA-Q GQE: `run_cudaq_gqe.py` calls `solvers.gqe()` with `cudaq.pauli_word` kernel typing

### Terminology
- The cGQE transformer scaffold is NOT the same as NVIDIA `solvers.gqe`; both coexist in this repository.
- H-cGQE = Hierarchical conditional GQE (autoregressive transformer + coefficient optimization)
- RLQF = Reinforcement Learning from Quantum Feedback

### Skills and workflows
- Devin skills/workflows available from `agent-skills-fresh` repo (38 skills, 38 workflows)
- AIRE-specific skills: `aire-slurm-submit`, `submit-gpu-job`, `conda-env-setup`, `debug-pytorch-gpu`
- No local `.devin` or `.cursor` directory in this project (use global skills from `agent-skills-fresh`)
