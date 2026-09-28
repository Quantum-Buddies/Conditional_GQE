#!/bin/bash
# Parallel remaining-ligand DFT on `nodes` (one ligand per array task).
# Override the ligand list with TIN_REMAINING="ipr npr" if pick.json is stale.
# TIN_INFORM_CHAIN=0 so a finished task does not double-submit the next pick.
# Do NOT sbatch jobs/tin_dapo.sbatch. Do not copy XYZ into tin_resist.yaml.
#SBATCH --job-name=tin_inform_dft_par
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%A_%a.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%A_%a.err

set -euo pipefail
source /scratch/kcwp264/.aire_scratch_env.sh
module load miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/.conda_envs/cudaq-env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

ROOT=/scratch/kcwp264/Conditional-GQE_materials
OUT="${ROOT}/results/tin_ab/methyltin_ours"
mkdir -p "${OUT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

if [[ -n "${TIN_REMAINING:-}" ]]; then
  # shellcheck disable=SC2206
  LIGANDS=(${TIN_REMAINING})
else
  mapfile -t LIGANDS < <(python src/gqe/eval/tin_inform_loop.py --out-dir "${OUT}" --remaining-ids)
fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${#LIGANDS[@]}" -eq 0 || "${IDX}" -ge "${#LIGANDS[@]}" ]]; then
  echo "Array index ${IDX}: no remaining ligand (n=${#LIGANDS[@]})"
  exit 0
fi
LIGAND="${LIGANDS[${IDX}]}"

echo "Parallel DFT ligand=${LIGAND} job=${SLURM_JOB_ID} host=$(hostname) $(date)"
python src/gqe/eval/tin_inform_dft.py \
  --ligands "${LIGAND}" \
  --out-dir "${OUT}" \
  --solver geometric \
  --n-workers 1 \
  || echo "DFT returned non-zero for ${LIGAND}"

python src/gqe/eval/tin_inform_rank.py --out-dir "${OUT}"
python src/gqe/eval/tin_inform_loop.py --out-dir "${OUT}" --record-step "${LIGAND}" --job-id "${SLURM_JOB_ID:-}"
python src/gqe/eval/tin_inform_loop.py --out-dir "${OUT}" --status

echo "Parallel DFT finished ${LIGAND} at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
