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
