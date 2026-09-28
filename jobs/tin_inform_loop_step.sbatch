#!/bin/bash
# One closed-loop step: score the ligand in pick.json, re-rank, pick next.
# Labelled-ours DFT only. GQE does not propose molecules.
# Do not copy XYZ into configs/tin_resist.yaml. Do NOT sbatch tin_dapo.sbatch.
#SBATCH --job-name=tin_inform_loop
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.err

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
SBATCH_SELF="${ROOT}/jobs/tin_inform_loop_step.sbatch"

mkdir -p "${OUT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

LIGAND="${TIN_LIGAND:-$(python src/gqe/eval/tin_inform_loop.py --out-dir "${OUT}" --next-id)}"
if [[ -z "${LIGAND}" ]]; then
  echo "Closed loop: no remaining ligand in pick.json"
  python src/gqe/eval/tin_inform_loop.py --out-dir "${OUT}" --status
  exit 0
fi

echo "Loop step ligand=${LIGAND} job=${SLURM_JOB_ID} host=$(hostname) $(date)"
python src/gqe/eval/tin_inform_dft.py \
  --ligands "${LIGAND}" \
  --out-dir "${OUT}" \
  --solver geometric \
  --n-workers 1 \
  || echo "DFT returned non-zero for ${LIGAND} (failed score recorded if present)"

python src/gqe/eval/tin_inform_rank.py --out-dir "${OUT}"
python src/gqe/eval/tin_inform_loop.py --out-dir "${OUT}" --record-step "${LIGAND}" --job-id "${SLURM_JOB_ID:-}"
python src/gqe/eval/tin_inform_loop.py --out-dir "${OUT}" --status

NEXT="$(python src/gqe/eval/tin_inform_loop.py --out-dir "${OUT}" --next-id || true)"
if [[ -n "${NEXT}" && "${TIN_INFORM_CHAIN:-1}" == "1" ]]; then
  echo "Chaining next loop step: ${NEXT}"
  sbatch "${SBATCH_SELF}" || echo "sbatch chain failed; score ${NEXT} later"
fi

echo "Loop step finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
