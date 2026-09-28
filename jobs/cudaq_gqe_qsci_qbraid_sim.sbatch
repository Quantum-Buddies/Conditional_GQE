#!/bin/bash
# Sample the GQE SnO 14q circuit on qBraid QIR state-vector (free, ≤2000 shots).
# Uses cudaq.set_target("qbraid"), not the qBraid Python SDK.
# Needs QBRAID_API_KEY in the environment (never in this file).
#SBATCH --job-name=cudaq_gqe_qirsv
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.err

set -euo pipefail
source "${SCRATCH_ROOT:-/scratch/kcwp264}/.aire_scratch_env.sh"
# CUDA-Q 0.15.1 venv: native qbraid target. Do not conda activate cudaq-env (0.14 + solvers).

ROOT=${SCRATCH_ROOT:-/scratch/kcwp264}/Conditional-GQE_materials
PYTHON=${SCRATCH_ROOT:-/scratch/kcwp264}/.conda_envs/cudaq15-qbraid/bin/python
HAM="${ROOT}/results/tin_ab/hamiltonians_tin/hamiltonians.json"
EXP="${ROOT}/experiments/cudaq-gqe-cepheus-sno14"
OUT_DIR="${EXP}/outputs"
GQE_JSON="${GQE_JSON:-${OUT_DIR}/cudaq_gqe_sno14_latest.json}"
OUT="${OUT_DIR}/qsci_qirsv_${SLURM_JOB_ID}.json"
mkdir -p "${OUT_DIR}" "${ROOT}/logs"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing CUDA-Q 0.15 qBraid venv at ${PYTHON}"
  exit 1
fi

if [[ ! -f "${GQE_JSON}" ]]; then
  echo "Missing GQE JSON ${GQE_JSON}; run jobs/cudaq_gqe_sno14.sbatch first."
  exit 1
fi
if [[ -z "${QBRAID_API_KEY:-}" ]]; then
  echo "QBRAID_API_KEY is unset. Export it in the submitting shell; do not put it in sbatch."
  exit 1
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDAQ_QBRAID_MACHINE="${CUDAQ_QBRAID_MACHINE:-qbraid:qbraid:sim:qir-sv}"
# Simulator: do not set CUDAQ_ALLOW_QPU.

cd "${ROOT}"
echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date) machine=${CUDAQ_QBRAID_MACHINE}"

set +e
"${PYTHON}" src/gqe/eval/gqe_qsci_sample.py \
  --gqe-json "${GQE_JSON}" \
  --hamiltonians "${HAM}" \
  --molecule sno_14q \
  --backend qbraid-qir \
  --shots 2000 \
  --n-samples 32 \
  --seed 42 \
  --async-submit \
  --out "${OUT}"
RC=$?
set -e
if [[ -f "${OUT}" ]]; then
  ln -sfn "${OUT}" "${OUT_DIR}/qsci_qirsv_latest.json"
fi
echo "QIR-SV exit ${RC} (2=miss 1.6 mHa, 3=sample error). Cepheus stays gated unless pass_chemical_accuracy."
echo "Job finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
exit "${RC}"
