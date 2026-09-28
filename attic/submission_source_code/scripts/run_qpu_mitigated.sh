#!/bin/bash
# QPU Submission with Error Mitigation (REM + ZNE) — portable, no Slurm
# Submits H-cGQE circuit to qBraid QPU devices with noise mitigation
#
# Usage: bash scripts/run_qpu_mitigated.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p logs results/phase3_final/qpu

echo "=== QPU Submission with Error Mitigation ==="
echo "Started: $(date)"
echo ""

BENCHMARK="results/phase3_final/benchmark_ch3i_consolidated.json"
CONFIG="configs/phase3_final/qpu_validation.yaml"
SHOTS=4096

DEVICES=(
    "aws:rigetti:qpu:cepheus-1-108q"
    "aws:iqm:qpu:garnet-20"
    "aws:ionq:qpu:forte-1-29"
    "qbraid:qbraid:sim:qir-sv"
)

for DEVICE in "${DEVICES[@]}"; do
    echo "--- Submitting to ${DEVICE} ---"
    OUT_FILE="results/phase3_final/qpu/qpu_${DEVICE//[:\/]/_}_mitigated.json"

    "${PYTHON}" src/gqe/eval/submit_qpu.py \
        --benchmark "${BENCHMARK}" --config "${CONFIG}" \
        --device "${DEVICE}" --shots ${SHOTS} \
        --mitigate rem,zne --zne-scales 1,2,3 --zne-method richardson \
        --out "${OUT_FILE}" --submit-only 2>&1 || {
        echo "  Failed on ${DEVICE}, trying next..."
        continue
    }
    echo "  Submitted! Output: ${OUT_FILE}"
    break
done

echo ""
echo "=== QPU Submission Complete ==="
echo "Finished: $(date)"
