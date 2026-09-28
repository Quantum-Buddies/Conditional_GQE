#!/bin/bash
# RAFT Post-Training Alignment — portable, no Slurm
# Usage: bash scripts/run_post_alignment.sh [CHECKPOINT]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export CUDAQ_MPS_MAX_BOND=64
export NCCL_P2P_DISABLE=1 NCCL_NET=Socket

CHECKPOINT="${1:-results/train/h_cgqe_rl_warmstart.pt}"
HAMILTONIANS="results/data/hamiltonians_scaling.json/hamiltonians.json"
OUTPUT="results/train/h_cgqe_raft_aligned.pt"

echo "=== Starting Post-Training Alignment ==="
echo "Start: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT}"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

"${PYTHON}" scripts/train_post_alignment.py \
    --checkpoint "${CHECKPOINT}" --hamiltonians "${HAMILTONIANS}" \
    --out "${OUTPUT}" --epochs 50 --batch-size 4 --lr 5e-5 \
    --n-samples 50 --top-k 5 --use-cuda --target nvidia --target-option mqpu 2>&1

echo ""
echo "=== Alignment Complete ==="
echo "End: $(date)"
echo "Output: ${OUTPUT}"
