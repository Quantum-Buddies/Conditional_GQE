#!/bin/bash
# Fair labelled-ours AVAS + CASCI + JW export for Me / CF3 (charge 0).
# CPU on `nodes`. Same AO labels, same threshold. Does not change ΔBDE rank.
# Does not write tin_resist.yaml. Do NOT sbatch jobs/tin_dapo.sbatch from here.
#SBATCH --job-name=tin_inform_e0
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
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
LIGANDS=(me cf3)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
LIGAND="${TIN_E0_LIGAND:-${LIGANDS[${IDX}]}}"
CHARGE="${TIN_E0_CHARGE:-0}"

mkdir -p "${OUT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-none} ligand=${LIGAND} charge=${CHARGE} on $(hostname) at $(date)"
echo "partition=nodes cpus=${SLURM_CPUS_PER_TASK} mem=256G fair_pair=me_cf3 threshold_start=0.2"
python -c "import pyscf,openfermion,tqdm; print('pyscf', pyscf.__version__, 'openfermion', openfermion.__version__)"

python src/gqe/eval/tin_inform_e0.py \
  --ligand "${LIGAND}" \
  --charge "${CHARGE}" \
  --out-dir "${OUT}" \
  --max-qubits 24 \
  --threshold 0.2 \
  --fair-pair me_cf3 \
  --pyscf-verbose 3 \
  --skip-jw-check

echo "CASCI export finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
