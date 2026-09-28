#!/bin/bash
# 1x L40S: infer 14-char SFT (n_samples>=64) + MatGen-Q QSCI with N_dets cap 32.
#SBATCH --job-name=tin_infer_pool14
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.err

set -euo pipefail
source /scratch/kcwp264/.aire_scratch_env.sh
module load cuda/12.6.2
module load miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/.conda_envs/cudaq-env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

ROOT=/scratch/kcwp264/Conditional-GQE_materials
OUT=/scratch/kcwp264/Conditional-GQE_materials/results/tin_ab
HAM="${OUT}/hamiltonians_tin/hamiltonians.json"
POOL="${OUT}/pool_sno14q.json"
CKPT="${TIN_SFT_CKPT:-${OUT}/h_cgqe_sft_pool14_7431349.pt}"
if [[ ! -f "${CKPT}" ]]; then
  echo "Missing 14-char SFT checkpoint: ${CKPT}"; exit 1
fi
echo "TIN_SFT_CKPT=${CKPT}"
INFER="${OUT}/hcgqe_infer_pool14_${SLURM_JOB_ID}.json"
QSCI="${OUT}/hcgqe_qsci_pool14_${SLURM_JOB_ID}.json"

mkdir -p "${OUT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
source "${ROOT}/scripts/env_gpu.sh"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi

python src/gqe/models/infer_h_cgqe.py \
  --checkpoint "${CKPT}" \
  --hamiltonians "${HAM}" \
  --molecules sno_14q \
  --n-samples 64 \
  --max-seq-len 16 \
  --temperature 1.2 \
  --exact-word-length \
  --pool "${POOL}" \
  --use-cuda \
  --out "${INFER}"

python src/gqe/eval/tin_hcgqe_qsci.py \
  --infer "${INFER}" \
  --out "${QSCI}" \
  --n-shots 8192 \
  --max-union-dets 32

echo "Job finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
