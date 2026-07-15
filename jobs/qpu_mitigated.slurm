#!/bin/bash
#SBATCH --job-name=gqe-qpu-mitigated
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/qpu_mitigated_%j.out
#SBATCH --error=logs/qpu_mitigated_%j.err

# QPU Submission with Error Mitigation (REM + ZNE)
# Submits H-cGQE circuit to qBraid QPU devices with noise mitigation

set -euo pipefail

PYTHON=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python
PROJECT_ROOT=/scratch/kcwp264/Conditional-GQE_materials
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
mkdir -p logs results/phase3_final/qpu

echo "=== QPU Submission with Error Mitigation ==="
echo "Started: $(date)"
echo ""

BENCHMARK="results/phase3_final/benchmark_ch3i_consolidated.json"
CONFIG="configs/phase3_final/qpu_validation.yaml"
SHOTS=4096

# Device list (try in order)
DEVICES=(
    "aws:rigetti:qpu:cepheus-1-108q"
    "aws:iqm:qpu:garnet-20"
    "aws:ionq:qpu:forte-1-29"
    "qbraid:qbraid:sim:qir-sv"
)

for DEVICE in "${DEVICES[@]}"; do
    echo "--- Submitting to ${DEVICE} ---"
    OUT_FILE="results/phase3_final/qpu/qpu_${DEVICE//[:\/]/_}_mitigated.json"

    ${PYTHON} src/gqe/eval/submit_qpu.py \
        --benchmark "${BENCHMARK}" \
        --config "${CONFIG}" \
        --device "${DEVICE}" \
        --shots ${SHOTS} \
        --mitigate rem,zne \
        --zne-scales 1,2,3 \
        --zne-method richardson \
        --out "${OUT_FILE}" \
        --submit-only 2>&1 || {
        echo "  Failed on ${DEVICE}, trying next..."
        continue
    }

    echo "  Submitted! Output: ${OUT_FILE}"
    break
done

echo ""
echo "=== QPU Submission Complete ==="
echo "Finished: $(date)"
