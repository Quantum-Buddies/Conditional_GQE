#!/bin/bash
# Fair-pair E0 QSCI on labelled-ours AVAS Hamiltonians. One L40S per ligand.
# Two GPUs = two independent 1-GPU array tasks (Me, CF3), not nvidia,mgpu.
# Primary sampler: pool-native UCCSD-like. Control: entangled-HF.
# 14-char H-cGQE is not loaded. Do not copy XYZ into tin_resist.yaml.
# Do NOT sbatch jobs/tin_dapo.sbatch from here.
#SBATCH --job-name=tin_inform_e0_qsci
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%A_%a.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%A_%a.err

set -euo pipefail
source /scratch/kcwp264/.aire_scratch_env.sh
module load cuda/12.6.2
module load miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/.conda_envs/cudaq-env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export CUDAQ_NVIDIA_OPTION="${CUDAQ_NVIDIA_OPTION:-fp64}"

ROOT=/scratch/kcwp264/Conditional-GQE_materials
OUT="${ROOT}/results/tin_ab/methyltin_ours"
LIGANDS=(me cf3)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
LIGAND="${TIN_E0_LIGAND:-${LIGANDS[${IDX}]}}"
CHARGE="${TIN_E0_CHARGE:-0}"
HAM="${OUT}/${LIGAND}/hamiltonian_c${CHARGE}.json"

mkdir -p "${OUT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-none} ligand=${LIGAND} charge=${CHARGE} on $(hostname) at $(date)"
nvidia-smi
source "${ROOT}/scripts/env_gpu.sh"

if [[ "${CUDAQ_NVIDIA_OPTION}" == *mgpu* ]]; then
  echo "Refusing mgpu for ≤24q fair E0. Use 1-GPU nvidia,fp64 array tasks."
  exit 1
fi

if [[ ! -f "${HAM}" ]]; then
  echo "Missing ${HAM}; CASCI export did not finish for this array index."
  exit 1
fi

python src/gqe/eval/tin_inform_e0.py \
  --ligand "${LIGAND}" \
  --charge "${CHARGE}" \
  --out-dir "${OUT}" \
  --qsci \
  --qsci-sampler both \
  --max-pool-ops 64 \
  --n-shots 16384 \
  --n-samples 100 500 1000

echo "QSCI finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
