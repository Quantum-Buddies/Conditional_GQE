#!/bin/bash
# Phase 0/1: 1x L40S. Reproduce MatGen-Q Demo A (SnO 14q GATE 1), export the
# identical JW Hamiltonian, then frozen H-cGQE + QSCI on that H.
#SBATCH --job-name=tin_qsci
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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

ROOT=/scratch/kcwp264/Conditional-GQE_materials
BASE=/scratch/kcwp264/baselines/gqe-qsci-euv-photoresists/code
OUT=/scratch/kcwp264/Conditional-GQE_materials/results/tin_ab
CKPT="${ROOT}/results/train/h_cgqe_uccsd_model.pt"
HAM="${OUT}/hamiltonians_tin/hamiltonians.json"
INFER="${OUT}/hcgqe_infer_sno14q_${SLURM_JOB_ID}.json"
mkdir -p "${OUT}" "${OUT}/hamiltonians_tin" "${ROOT}/logs"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi
source "${ROOT}/scripts/env_gpu.sh"

cd "${BASE}"
# GATE 1: CASCI -288.10912104 Ha must assert.
python classical_baseline.py | tee "${OUT}/gpu_classical_${SLURM_JOB_ID}.log"
SMOKE=1 python run_demo.py | tee "${OUT}/matgenq_demo_a_smoke_${SLURM_JOB_ID}.log"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python src/gqe/data/export_matgenq_hamiltonian.py \
  --matgenq-code "${BASE}" \
  --out "${HAM}" \
  --skip-jw-check | tee "${OUT}/export_sno14q_${SLURM_JOB_ID}.log"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing ${CKPT}; infer/QSCI skipped."
  echo "Job finished at $(date)"
  command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
  exit 0
fi

python src/gqe/models/infer_h_cgqe.py \
  --checkpoint "${CKPT}" \
  --hamiltonians "${HAM}" \
  --molecules sno_14q \
  --n-samples 16 \
  --use-cuda \
  --out "${INFER}" | tee "${OUT}/hcgqe_infer_${SLURM_JOB_ID}.log"

python src/gqe/eval/qsci.py \
  --hamiltonians "${HAM}" \
  --molecules sno_14q \
  --optimized "${INFER}" \
  --n-samples 1000 5000 \
  --bond-dims 64 \
  --n-shots 8192 \
  --out "${OUT}/hcgqe_qsci_${SLURM_JOB_ID}.json" | tee "${OUT}/hcgqe_qsci_${SLURM_JOB_ID}.log"

echo "Job finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
