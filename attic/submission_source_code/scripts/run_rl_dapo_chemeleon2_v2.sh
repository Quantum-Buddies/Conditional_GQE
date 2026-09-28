#!/bin/bash
# Chemeleon2-mode RL Training v2 (optimized hyperparams, 3 GPU) — portable, no Slurm
# Key changes from v1:
#   - n-samples 32 -> 64, n-iters 2 -> 5, reuse-iters 1 -> 3
#   - buffer-batch-size 0 -> 64, buffer-size 1000 -> 2000
#   - adaptive-theta enabled, epochs 200 -> 500
# Usage: bash scripts/run_rl_dapo_chemeleon2_v2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export CUDAQ_MPS_MAX_BOND=64
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET=Socket

CHECKPOINT="results/train/h_cgqe_uccsd_model.pt"
HAMILTONIANS="results/data/hamiltonians_merged.json"
OUTPUT="results/train/h_cgqe_rl_chemeleon2_v2.pt"

echo "=== Chemeleon2-mode RL Training v2 (optimized hyperparams) ==="
echo "Start: $(date)"
echo "Output: ${OUTPUT}"
echo ""

"${PYTHON}" -u src/gqe/models/train_rl_dapo.py \
    --checkpoint "${CHECKPOINT}" --hamiltonians "${HAMILTONIANS}" \
    --molecules h2 lih beh2 n2 --out "${OUTPUT}" \
    --epochs 500 --n-samples 64 --n-iters 5 --reuse-iters 3 \
    --buffer-size 2000 --buffer-batch-size 64 --lr 1e-5 \
    --temperature 1.0 --top-p 0.9 --target-entropy 1.5 \
    --explore-eps 0.3 --adaptive-eps --force-entanglement --max-repeat 4 \
    --max-qubits 24 --target nvidia --target-option mqpu \
    --use-cuda --multi-gpu --use-bf16 --dynamic-sampling --token-level-loss \
    --entropy-coef 1e-5 --w-energy 1.0 --w-entangle 0.1 --w-depth 0.05 \
    --w-commute 0.05 --w-diversity 0.2 --target-len 10 --freq-penalty 1.0 \
    --curriculum --curriculum-warmup 50 --curriculum-steps 3 \
    --chemeleon2-mode --msun-threshold 0.1 \
    --adaptive-theta --adaptive-theta-iters 10 2>&1

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
echo "Output: ${OUTPUT}"
