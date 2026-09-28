#!/bin/bash
# 1x L40S: 14-char pool vocab SFT (new embedding; do not load mixed UCCSD ckpt).
# Submit only after tin_qsci_controls gate1_pool_native_pass is true.
#SBATCH --job-name=tin_sft_pool14
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.err

set -euo pipefail
source "${SCRATCH_ROOT:-/scratch/kcwp264}/.aire_scratch_env.sh"
module load cuda/12.6.2
module load miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_ROOT:-/scratch/kcwp264}/.conda_envs/cudaq-env"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

ROOT=${SCRATCH_ROOT:-/scratch/kcwp264}/Conditional-GQE_materials
BASE=${SCRATCH_ROOT:-/scratch/kcwp264}/baselines/gqe-qsci-euv-photoresists/code
OUT=${SCRATCH_ROOT:-/scratch/kcwp264}/Conditional-GQE_materials/results/tin_ab
HAM="${OUT}/hamiltonians_tin/hamiltonians.json"
POOL="${OUT}/pool_sno14q.json"
MATGENQ_RES="${BASE}/results_demo.json"
DS_DIR="${OUT}/sft_pool14"
CKPT="${OUT}/h_cgqe_sft_pool14_${SLURM_JOB_ID}.pt"

mkdir -p "${OUT}" "${DS_DIR}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
source "${ROOT}/scripts/env_gpu.sh"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi

python src/gqe/data/prepare_sno14_sft_dataset.py \
  --pool "${POOL}" \
  --hamiltonians "${HAM}" \
  --matgenq-results "${MATGENQ_RES}" \
  --out-dir "${DS_DIR}" \
  --n-synthetic 2048 \
  --max-seq-len 16

python src/gqe/models/train_h_cgqe.py \
  --dataset "${DS_DIR}/gqe_supervised_dataset_sno14.pt" \
  --out "${CKPT}" \
  --epochs 80 \
  --batch-size 16 \
  --lr 1e-4 \
  --use-cuda \
  --use-bf16 \
  --label-smoothing 0.1 \
  --patience 15

echo "Wrote checkpoint ${CKPT}"
echo "Job finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
