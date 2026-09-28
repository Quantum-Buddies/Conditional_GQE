#!/bin/bash
# Labelled-ours R–Sn(OH)3 B3LYP opt + MP2 single-point (NOT MatGen-Q Table 1).
# One ligand per array task. Do not copy XYZ into configs/tin_resist.yaml.
# Do NOT submit jobs/tin_dapo.sbatch from this script.
#SBATCH --job-name=tin_inform_dft
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
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
LIGANDS=(me et nbu vinyl ph)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
LIGAND="${LIGANDS[${IDX}]}"

mkdir -p "${OUT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} ligand=${LIGAND} on $(hostname) at $(date)"
echo "partition=nodes cpus=${SLURM_CPUS_PER_TASK} mem=256G"
command -v python
python -c "import pyscf, geometric, berny, tqdm; print('pyscf', pyscf.__version__, 'geometric ok')"

python src/gqe/eval/tin_inform_dft.py \
  --ligands "${LIGAND}" \
  --out-dir "${OUT}" \
  --solver geometric \
  --n-workers 1 \
  --fail-fast

python src/gqe/eval/tin_inform_rank.py --out-dir "${OUT}" || echo "rank.py deferred (incomplete scores ok)"

echo "Job finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
