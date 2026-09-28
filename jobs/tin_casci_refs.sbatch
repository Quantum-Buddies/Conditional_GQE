#!/bin/bash
# AIRE CPU classical ladder for the MatGen-Q SnO 14q instance, then later
# H-cGQE tin-resist Hamiltonians. Partition is `nodes` (there is no `cpu`).
#SBATCH --job-name=tin_casci_refs
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.err

set -euo pipefail
source /scratch/kcwp264/.aire_scratch_env.sh
module load miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/.conda_envs/cudaq-env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

ROOT=/scratch/kcwp264/Conditional-GQE_materials
BASE=/scratch/kcwp264/baselines/gqe-qsci-euv-photoresists/code
OUT=/scratch/kcwp264/Conditional-GQE_materials/results/tin_ab
mkdir -p "${OUT}" "${ROOT}/logs"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "partition=nodes cpus=${SLURM_CPUS_PER_TASK} mem=256G"

cd "${BASE}"
python classical_baseline.py | tee "${OUT}/matgenq_classical_baseline_${SLURM_JOB_ID}.log"

# Export THEIR SnO Hamiltonian (def2-SVP + ECP). Do not use generate_hamiltonians.py.
if [[ -f "${ROOT}/src/gqe/data/export_matgenq_hamiltonian.py" ]]; then
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
  python src/gqe/data/export_matgenq_hamiltonian.py \
    --matgenq-code "${BASE}" \
    --out "${OUT}/hamiltonians_tin"
fi

echo "Job finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
