#!/bin/bash
# GQE-QSCI Scaling Experiment: 4q -> 40q — portable, no Slurm
# Uses CUDA-Q tensornet-mps backend for >24q molecules
#
# Usage: bash scripts/run_qsci_scaling.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p logs results/phase3_final/qsci

echo "=== GQE-QSCI Scaling Experiment ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

HAM_FILE="results/data/hamiltonians_40plus.json/hamiltonians.json"
MOLECULES=(h2 lih beh2 n2 formaldehyde ethylene benzene_cas20)
N_SAMPLES=(100 500 1000 5000)
BOND_DIMS=(64 128 256)
N_SHOTS=8192
OUT_FILE="results/phase3_final/qsci/qsci_scaling_results.json"

"${PYTHON}" src/gqe/eval/qsci.py \
    --hamiltonians "${HAM_FILE}" \
    --molecules "${MOLECULES[@]}" \
    --n-samples "${N_SAMPLES[@]}" \
    --bond-dims "${BOND_DIMS[@]}" \
    --n-shots ${N_SHOTS} \
    --out "${OUT_FILE}"

echo ""
echo "=== QSCI Scaling Complete ==="
echo "Finished: $(date)"
echo "Results: ${OUT_FILE}"
