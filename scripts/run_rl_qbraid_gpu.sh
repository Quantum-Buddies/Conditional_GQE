#!/bin/bash
# RL training launch script for qBraid Lab GPU instances.
# Optimized for H200 (141GB), H100 (80GB), or B200 (180GB).
#
# Key advantages over AIRE L40S:
#   - H200 141GB: handles up to ~30q single-GPU cuStateVec (vs 24q on L40S)
#   - H100/H200 SXM: NVLink enables multi-GPU cuStateVec (no PCIe IPC segfault)
#   - Much higher bandwidth → faster CUDA-Q energy evaluation → more samples/epoch
#
# Usage:
#   bash scripts/run_rl_qbraid_gpu.sh [GPU_TYPE]
#
#   GPU_TYPE: h200 (default), h100, b200, a100, l40s
#
# qBraid GPU pricing (credits/min, 100 credits = $1):
#   H200:   9.15 cr/min  (141GB, ~30q max)
#   H100:   8.95 cr/min  (80GB,  ~26q max)
#   B200:  14.57 cr/min  (180GB, ~32q max)
#   A100:   4.15 cr/min  (80GB,  ~26q max)
#   L40S:   3.80 cr/min  (48GB,  ~24q max)

set -euo pipefail

GPU_TYPE="${1:-h200}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="python"

# GPU-specific settings
case "$GPU_TYPE" in
    h200)
        MAX_QUBITS=30
        EXTRA_FLAGS=""
        ;;
    h100)
        MAX_QUBITS=26
        EXTRA_FLAGS=""
        ;;
    b200)
        MAX_QUBITS=32
        EXTRA_FLAGS=""
        ;;
    a100)
        MAX_QUBITS=26
        EXTRA_FLAGS=""
        ;;
    l40s)
        MAX_QUBITS=24
        EXTRA_FLAGS="--single-gpu"
        ;;
    *)
        echo "Unknown GPU type: $GPU_TYPE"
        echo "Supported: h200, h100, b200, a100, l40s"
        exit 1
        ;;
esac

CHECKPOINT="${PROJECT_ROOT}/results/train/h_cgqe_uccsd_model.pt"
HAMILTONIANS="${PROJECT_ROOT}/results/data/hamiltonians_merged.json"
OUTPUT="${PROJECT_ROOT}/results/train/h_cgqe_rl_qbraid_${GPU_TYPE}.pt"

echo "=== RL Training on qBraid GPU (${GPU_TYPE}) ==="
echo "Start: $(date)"
echo "Project: ${PROJECT_ROOT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Hamiltonians: ${HAMILTONIANS}"
echo "Output: ${OUTPUT}"
echo "Max qubits: ${MAX_QUBITS}"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo ""

cd "${PROJECT_ROOT}"

# Optimized hyperparameters for qBraid GPU training:
#   --n-samples 64: 2x more samples per molecule for better gradient estimates
#   --n-iters 5: GPT-QE paper's N_iter=5 for more gradient updates per epoch
#   --reuse-iters 3: off-policy GRPO reuse (arXiv:2505.22257), 3x simulation cost reduction
#   --buffer-batch-size 64: replay buffer off-policy training (now implemented)
#   --adaptive-theta: L-BFGS-B optimization on best circuit for better energy signal
#   --epochs 500: longer training to converge on H2 and N2
#   --max-qubits: GPU-specific (30 for H200, 26 for H100, 24 for L40S)
"${PYTHON_BIN}" -u src/gqe/models/train_rl_dapo.py \
    --checkpoint "${CHECKPOINT}" \
    --hamiltonians "${HAMILTONIANS}" \
    --molecules h2 lih beh2 n2 \
    --out "${OUTPUT}" \
    --epochs 500 \
    --n-samples 64 \
    --n-iters 5 \
    --reuse-iters 3 \
    --buffer-size 2000 \
    --buffer-batch-size 64 \
    --lr 1e-5 \
    --temperature 1.0 \
    --top-p 0.9 \
    --target-entropy 1.5 \
    --explore-eps 0.3 \
    --adaptive-eps \
    --force-entanglement \
    --max-repeat 4 \
    --max-qubits "${MAX_QUBITS}" \
    --target nvidia \
    --target-option mqpu \
    --use-cuda \
    --use-bf16 \
    --dynamic-sampling \
    --token-level-loss \
    --entropy-coef 1e-5 \
    --w-energy 1.0 \
    --w-entangle 0.1 \
    --w-depth 0.05 \
    --w-commute 0.05 \
    --w-diversity 0.2 \
    --target-len 10 \
    --freq-penalty 1.0 \
    --curriculum \
    --curriculum-warmup 50 \
    --curriculum-steps 3 \
    --chemeleon2-mode \
    --msun-threshold 0.1 \
    --adaptive-theta \
    --adaptive-theta-iters 10 \
    ${EXTRA_FLAGS} \
    2>&1

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
echo "Output saved to: ${OUTPUT}"

# Print cost summary
ELAPSED_MIN=$(($(date +%s) - START_TIME) / 60)
case "$GPU_TYPE" in
    h200)  CR_RATE=9.15 ;;
    h100)  CR_RATE=8.95 ;;
    b200)  CR_RATE=14.57 ;;
    a100)  CR_RATE=4.15 ;;
    l40s)  CR_RATE=3.80 ;;
esac
TOTAL_CR=$(echo "${ELAPSED_MIN} * ${CR_RATE}" | bc -l 2>/dev/null || echo "N/A")
echo "Elapsed: ${ELAPSED_MIN} min"
echo "Estimated cost: ${TOTAL_CR} credits (\$$(echo "${TOTAL_CR} / 100" | bc -l 2>/dev/null || echo "N/A"))"
