#!/bin/bash
# Cepheus-1-108Q sample of the GQE SnO 14q circuit via cudaq.set_target("qbraid").
# HARD GATES: CUDAQ_ALLOW_QPU=1 in the submitting shell, GPU QSCI pass JSON,
# qBraid QIR-SV pass JSON. Never put QBRAID_API_KEY in this file.
# Prefer aws:rigetti:qpu:cepheus-1-108q (per-shot). Direct rigetti:rigetti is refused.
# Do NOT sbatch jobs/tin_dapo.sbatch. Do not set DEMOB_REAL.
#SBATCH --job-name=cudaq_gqe_cepheus
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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
GPU_JSON="${GPU_JSON:-${OUT_DIR}/qsci_gpu_latest.json}"
SIM_JSON="${SIM_JSON:-${OUT_DIR}/qsci_qirsv_latest.json}"
OUT="${OUT_DIR}/qsci_cepheus_${SLURM_JOB_ID}.json"
FUTURE="${OUT_DIR}/qsci_cepheus_${SLURM_JOB_ID}.future.json"
mkdir -p "${OUT_DIR}" "${ROOT}/logs"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing CUDA-Q 0.15 qBraid venv at ${PYTHON}"
  exit 1
fi
if [[ "${CUDAQ_ALLOW_QPU:-}" != "1" ]]; then
  echo "Refusing Cepheus: export CUDAQ_ALLOW_QPU=1 in the submitting shell after GPU+QIR-SV pass."
  exit 1
fi
if [[ -z "${QBRAID_API_KEY:-}" ]]; then
  echo "QBRAID_API_KEY is unset. Export it in the submitting shell; never write it here."
  exit 1
fi
if [[ ! -f "${GQE_JSON}" || ! -f "${GPU_JSON}" ]]; then
  echo "Missing GQE or GPU QSCI JSON. Run cudaq_gqe_sno14.sbatch first."
  exit 1
fi

"${PYTHON}" - "${GPU_JSON}" "${SIM_JSON}" <<'PY'
import json, sys
from pathlib import Path

def require_pass(path: str, label: str) -> None:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"Missing {label} JSON {path}")
    doc = json.loads(p.read_text())
    if not doc.get("pass_chemical_accuracy"):
        raise SystemExit(
            f"{label} did not pass ≤1.6 mHa (pass_chemical_accuracy=false). No Cepheus."
        )

require_pass(sys.argv[1], "GPU QSCI")
require_pass(sys.argv[2], "qBraid QIR-SV")
print("GPU and QIR-SV chemical-accuracy gates passed.")
PY

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDAQ_QBRAID_MACHINE="${CUDAQ_QBRAID_MACHINE:-aws:rigetti:qpu:cepheus-1-108q}"
export CUDAQ_ALLOW_QPU=1

cd "${ROOT}"
echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date) machine=${CUDAQ_QBRAID_MACHINE}"

"${PYTHON}" src/gqe/eval/gqe_qsci_sample.py \
  --gqe-json "${GQE_JSON}" \
  --hamiltonians "${HAM}" \
  --molecule sno_14q \
  --backend qbraid-cepheus \
  --shots 2000 \
  --n-samples 32 \
  --seed 42 \
  --recover \
  --async-submit \
  --future-path "${FUTURE}" \
  --max-two-qubit-gates 400 \
  --out "${OUT}"

ln -sfn "${OUT}" "${OUT_DIR}/qsci_cepheus_latest.json"
echo "Job finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
