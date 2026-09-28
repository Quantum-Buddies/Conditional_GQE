#!/bin/bash
# Gate 1 QSCI controls on identical SnO 14q H (skip classical_baseline; GATE 1 already OK).
# Columns: HF | MatGen-Q smoke best | frozen 12-char H-cGQE | 14q-pool sequences
#SBATCH --job-name=tin_qsci_ctrl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.err

set -euo pipefail
source "${SCRATCH_ROOT:-/scratch/kcwp264}/.aire_scratch_env.sh"
module load cuda/12.6.2
module load miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_ROOT:-/scratch/kcwp264}/.conda_envs/cudaq-env"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=${SCRATCH_ROOT:-/scratch/kcwp264}/Conditional-GQE_materials
BASE=${SCRATCH_ROOT:-/scratch/kcwp264}/baselines/gqe-qsci-euv-photoresists/code
OUT=${SCRATCH_ROOT:-/scratch/kcwp264}/Conditional-GQE_materials/results/tin_ab
HAM="${OUT}/hamiltonians_tin/hamiltonians.json"
POOL="${OUT}/pool_sno14q.json"
MATGENQ_RES="${BASE}/results_demo.json"
FROZEN_INFER="${OUT}/hcgqe_infer_sno14q_7386634.json"
CTRL_OUT="${OUT}/qsci_controls_${SLURM_JOB_ID}.json"

mkdir -p "${OUT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
source "${ROOT}/scripts/env_gpu.sh"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi

# Provenance copy of MatGen-Q smoke results (already GATE-1 validated)
cp -f "${MATGENQ_RES}" "${OUT}/matgenq_results_demo_smoke.json"

test -f "${HAM}" || { echo "Missing Hamiltonian ${HAM}"; exit 1; }
test -f "${POOL}" || { echo "Missing pool ${POOL}"; exit 1; }
test -f "${FROZEN_INFER}" || { echo "Missing frozen infer ${FROZEN_INFER}"; exit 1; }

python src/gqe/eval/tin_qsci_controls.py \
  --hamiltonians "${HAM}" \
  --pool "${POOL}" \
  --matgenq-results "${MATGENQ_RES}" \
  --frozen-infer "${FROZEN_INFER}" \
  --molecules sno_14q \
  --n-shots 8192 \
  --n-samples 49 \
  --n-random 8 \
  --seq-len 6 \
  --seed 42 \
  --max-union-dets 32 \
  --backend cudaq \
  --out "${CTRL_OUT}" | tee "${OUT}/qsci_controls_${SLURM_JOB_ID}.log"

echo "Job finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
