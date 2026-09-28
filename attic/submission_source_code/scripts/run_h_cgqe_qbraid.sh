#!/usr/bin/env bash
# End-to-end H-cGQE two-stage pipeline for qBraid / Phase 3 execution.
#
# Usage:
#   bash scripts/run_h_cgqe_qbraid.sh [MOLECULES...]
#
# Defaults to a small benchmark set: h2 lih beh2 n2
#
# This script:
#   1. Trains the H-cGQE Transformer (or skips if a checkpoint exists)
#   2. Generates operator sequences with constrained decoding (force_entanglement)
#   3. Optimizes rotation coefficients with L-BFGS-B on CUDA-Q GPU target
#   4. Plots scaling results
#
# Environment assumptions:
#   - Running inside qBraid Lab GPU instance (or any CUDA-Q + NVIDIA GPU node)
#   - Hamiltonian data at results/data/hamiltonians.json
#   - GQE baseline results at results/baselines/cudaq_gqe.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"

HAMILTONIANS="${RESULTS_DIR}/data/hamiltonians.json"
GQE_BASELINE="${RESULTS_DIR}/baselines/cudaq_gqe.json"
DATASET="${RESULTS_DIR}/train/gqe_supervised_dataset.pt"
CHECKPOINT="${RESULTS_DIR}/train/h_cgqe_model.pt"
GENERATED="${RESULTS_DIR}/inference/h_cgqe_generated.json"
OPTIMIZED="${RESULTS_DIR}/eval/h_cgqe_optimized.json"
EVALUATED="${RESULTS_DIR}/eval/h_cgqe_evaluation.json"
PLOTS="${RESULTS_DIR}/plots"

PYTHON="${CONDA_PREFIX:-/usr}/bin/python"

MOLECULES="${*:-h2 lih beh2 n2}"

echo "=== H-cGQE Phase 3 / qBraid Execution Pipeline ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Target molecules: ${MOLECULES}"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Verify prerequisites
# ---------------------------------------------------------------------------
if [ ! -f "${HAMILTONIANS}" ]; then
    echo "ERROR: Hamiltonian data not found at ${HAMILTONIANS}"
    echo "Run the Hamiltonian generation step first."
    exit 1
fi

if [ ! -f "${GQE_BASELINE}" ]; then
    echo "WARNING: GQE baseline not found at ${GQE_BASELINE}"
    echo "Some evaluation plots will be incomplete."
fi

# ---------------------------------------------------------------------------
# Step 1: Prepare supervised dataset (if missing)
# ---------------------------------------------------------------------------
if [ ! -f "${DATASET}" ]; then
    if [ ! -f "${GQE_BASELINE}" ]; then
        echo "ERROR: Cannot prepare dataset without GQE baseline results."
        exit 1
    fi
    echo "[Step 1/4] Preparing supervised dataset from GQE baseline..."
    ${PYTHON} "${PROJECT_ROOT}/src/gqe/data/prepare_gqe_dataset.py" \
        --ham "${HAMILTONIANS}" \
        --gqe-results "${GQE_BASELINE}" \
        --out-dir "${RESULTS_DIR}/train" \
        --augment-multiplier 5 \
        --coeff-noise 0.05
else
    echo "[Step 1/4] Using existing dataset: ${DATASET}"
fi

# ---------------------------------------------------------------------------
# Step 2: Train H-cGQE Transformer (if checkpoint missing)
# ---------------------------------------------------------------------------
if [ ! -f "${CHECKPOINT}" ]; then
    echo "[Step 2/4] Training H-cGQE Transformer..."
    ${PYTHON} "${PROJECT_ROOT}/src/gqe/models/train_h_cgqe.py" \
        --dataset "${DATASET}" \
        --out "${CHECKPOINT}" \
        --epochs 500 \
        --batch-size 4 \
        --lr 1e-4 \
        --use-cuda
else
    echo "[Step 2/4] Using existing checkpoint: ${CHECKPOINT}"
fi

# ---------------------------------------------------------------------------
# Step 3: Generate operator sequences with constrained decoding
# ---------------------------------------------------------------------------
echo "[Step 3/4] Generating operator sequences with force_entanglement and sampling..."
${PYTHON} "${PROJECT_ROOT}/src/gqe/models/infer_h_cgqe.py" \
    --checkpoint "${CHECKPOINT}" \
    --hamiltonians "${HAMILTONIANS}" \
    --molecules ${MOLECULES} \
    --out "${GENERATED}" \
    --n-samples 100 \
    --max-seq-len 64 \
    --force-entanglement \
    --sample \
    --max-repeat 4 \
    --use-cuda

# ---------------------------------------------------------------------------
# Step 4: Optimize coefficients on CUDA-Q GPU target
# ---------------------------------------------------------------------------
echo "[Step 4/4] Optimizing rotation coefficients on CUDA-Q nvidia-mqpu..."
${PYTHON} "${PROJECT_ROOT}/src/gqe/eval/optimize_h_cgqe_coefficients.py" \
    --generated "${GENERATED}" \
    --hamiltonians "${HAMILTONIANS}" \
    --out "${OPTIMIZED}" \
    --target nvidia \
    --target-option mqpu \
    --max-iter 100 \
    --top-k 10

# ---------------------------------------------------------------------------
# Optional: fixed-theta evaluation for comparison
# ---------------------------------------------------------------------------
echo "[Optional] Running fixed-theta evaluation..."
${PYTHON} "${PROJECT_ROOT}/src/gqe/eval/evaluate_h_cgqe.py" \
    --generated "${GENERATED}" \
    --baseline "${GQE_BASELINE}" \
    --hamiltonians "${HAMILTONIANS}" \
    --out "${EVALUATED}" \
    --target nvidia \
    --target-option mqpu

# ---------------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------------
echo "[Plotting] Generating scaling plots..."
mkdir -p "${PLOTS}"
${PYTHON} "${PROJECT_ROOT}/src/gqe/eval/plot_h_cgqe_scaling.py" \
    --eval "${EVALUATED}" \
    --optimized "${OPTIMIZED}" \
    --out-dir "${PLOTS}"

echo ""
echo "=== Pipeline Complete ==="
echo "Optimized results:   ${OPTIMIZED}"
echo "Evaluation results:  ${EVALUATED}"
echo "Plots:               ${PLOTS}"
