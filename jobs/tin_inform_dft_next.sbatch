#!/bin/bash
# Vinyl SCF retry + CF3 (chemical next-ligand pick). Labelled-ours only.
# Do not copy XYZ into configs/tin_resist.yaml. Do NOT sbatch tin_dapo.sbatch.
#SBATCH --job-name=tin_inform_next
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%A_%a.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%A_%a.err

set -euo pipefail
source "${SCRATCH_ROOT:-/scratch/kcwp264}/.aire_scratch_env.sh"
module load miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_ROOT:-/scratch/kcwp264}/.conda_envs/cudaq-env"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

ROOT=${SCRATCH_ROOT:-/scratch/kcwp264}/Conditional-GQE_materials
OUT="${ROOT}/results/tin_ab/methyltin_ours"
LIGANDS=(vinyl cf3)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
LIGAND="${LIGANDS[${IDX}]}"

mkdir -p "${OUT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} ligand=${LIGAND} on $(hostname) at $(date)"
python -c "import pyscf, geometric, berny, tqdm; print('pyscf', pyscf.__version__, 'geometric ok')"

python src/gqe/eval/tin_inform_dft.py \
  --ligands "${LIGAND}" \
  --out-dir "${OUT}" \
  --solver geometric \
  --n-workers 1 \
  --fail-fast

python src/gqe/eval/tin_inform_rank.py --out-dir "${OUT}" || echo "rank.py deferred"

echo "Job finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
