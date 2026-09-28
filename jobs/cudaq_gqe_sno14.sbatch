#!/bin/bash
# CUDA-Q Solvers GQE on SnO 14q (GATE 1). 1x L40S. Then GPU QSCI sample.
# Do NOT sbatch jobs/tin_dapo.sbatch. Do not set DEMOB_REAL.
#SBATCH --job-name=cudaq_gqe_sno14
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/logs/%x_%j.err

set -euo pipefail
source "${SCRATCH_ROOT:-/scratch/kcwp264}/.aire_scratch_env.sh"
module load cuda/12.6.2
module load miniforge/24.7.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_ROOT:-/scratch/kcwp264}/.conda_envs/cudaq-env"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDAQ_NVIDIA_OPTION="${CUDAQ_NVIDIA_OPTION:-fp64}"

ROOT=${SCRATCH_ROOT:-/scratch/kcwp264}/Conditional-GQE_materials
HAM="${ROOT}/results/tin_ab/hamiltonians_tin/hamiltonians.json"
EXP="${ROOT}/experiments/cudaq-gqe-cepheus-sno14"
OUT_DIR="${EXP}/outputs"
GQE_JSON="${OUT_DIR}/cudaq_gqe_sno14_${SLURM_JOB_ID}.json"
QSCI_JSON="${OUT_DIR}/qsci_gpu_${SLURM_JOB_ID}.json"
mkdir -p "${OUT_DIR}" "${ROOT}/logs"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi
source "${ROOT}/scripts/env_gpu.sh"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python - <<'PY'
import json
from pathlib import Path
import cudaq
try:
    import cudaq_solvers
    solvers_v = getattr(cudaq_solvers, "__version__", "present")
except Exception as exc:
    solvers_v = f"unavailable:{exc}"
names = []
try:
    names = [getattr(t, "name", str(t)) for t in cudaq.get_targets()]
except Exception as exc:
    names = [f"get_targets_error:{exc}"]
payload = {
    "cudaq": getattr(cudaq, "__version__", "unknown"),
    "cudaq_solvers": solvers_v,
    "has_qbraid": any("qbraid" in str(n).lower() for n in names),
    "qbraid_targets": [n for n in names if "qbraid" in str(n).lower()],
    "hostname": __import__("socket").gethostname(),
}
print(json.dumps(payload, indent=2))
Path("experiments/cudaq-gqe-cepheus-sno14/outputs/cudaq_versions.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
if not payload["has_qbraid"]:
    print(
        "NOTE: cudaq-env 0.14.2 has no qbraid target (expected). "
        "QIR-SV/Cepheus use ${SCRATCH_ROOT:-/scratch/kcwp264}/.conda_envs/cudaq15-qbraid (0.15.1)."
    )
PY

python src/gqe/baselines/run_cudaq_gqe.py \
  --ham "${HAM}" \
  --molecule sno_14q \
  --max-qubits 16 \
  --ngates 6 \
  --max-iters 25 \
  --target nvidia \
  --target-option fp64 \
  --out "${GQE_JSON}"

# GPU QSCI of the emitted circuit (exit 2 = miss chemical accuracy; do not fail the GQE artefact).
set +e
python src/gqe/eval/gqe_qsci_sample.py \
  --gqe-json "${GQE_JSON}" \
  --hamiltonians "${HAM}" \
  --molecule sno_14q \
  --backend nvidia \
  --shots 8192 \
  --n-samples 32 \
  --seed 42 \
  --out "${QSCI_JSON}"
QSCI_RC=$?
set -e
echo "GPU QSCI exit ${QSCI_RC} (2 means miss 1.6 mHa; Cepheus stays gated)"
ln -sfn "${GQE_JSON}" "${OUT_DIR}/cudaq_gqe_sno14_latest.json"
if [[ -f "${QSCI_JSON}" ]]; then
  ln -sfn "${QSCI_JSON}" "${OUT_DIR}/qsci_gpu_latest.json"
fi

echo "Job finished at $(date)"
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
