#!/bin/bash
# Gate 2 ONLY if Gate 1 pool-native QSCI misses 1.6 mHa.
# 14q vocab from scratch on sno_14q — do NOT load iodine/12-char SFT checkpoint.
# 3x L40S PCIe, no NVLink. NCCL Socket.
#SBATCH --job-name=tin_dapo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:3
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.err

set -euo pipefail
source "${SCRATCH_ROOT:-/scratch/kcwp264}/.aire_scratch_env.sh"
module load cuda/12.6.2
module load miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_ROOT:-/scratch/kcwp264}/.conda_envs/cudaq-env"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

ROOT=${SCRATCH_ROOT:-/scratch/kcwp264}/Conditional-GQE_materials
OUT=${SCRATCH_ROOT:-/scratch/kcwp264}/Conditional-GQE_materials/results/tin_ab
mkdir -p "${OUT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
source "${ROOT}/scripts/env_gpu.sh"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi

# From-scratch RL builds UCCSD vocab for sno_14q (14-char words). No iodine SFT mix.
python src/gqe/models/train_rl_dapo.py \
  --from-scratch \
  --hamiltonians "${OUT}/hamiltonians_tin/hamiltonians.json" \
  --molecules sno_14q \
  --epochs 20 \
  --n-samples 8 \
  --out "${OUT}/h_cgqe_rl_sno14q_fromscratch.pt"

echo "Job finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
