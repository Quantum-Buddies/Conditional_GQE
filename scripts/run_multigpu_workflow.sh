#!/usr/bin/env bash
# Multi-GPU workflow for chemistry-conditioned GQE on 2x NVIDIA L40S.
#
# Usage:
#   bash scripts/run_multigpu_workflow.sh
#
# This script:
#   1. Trains the chemistry encoder across both GPUs with PyTorch DDP
#   2. Exports conditioning vectors (priors) for each Hamiltonian record
#   3. Runs the conditioned CUDA-Q GQE with mqpu backend (Hamiltonian batching across GPUs)
#
# Environment assumptions:
#   - 2x NVIDIA L40S GPUs available on a single node
#   - cudaq-env conda environment active
#   - Hamiltonian data at results/data/hamiltonians.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"

HAMILTONIANS="${RESULTS_DIR}/data/hamiltonians.json"
ENCODER_OUT="${RESULTS_DIR}/train/ddp_chemistry_encoder.done"
EMBEDDINGS_OUT="${RESULTS_DIR}/train/ddp_graph_embeddings.json"
GQE_OUT="${RESULTS_DIR}/baselines/cudaq_gqe_mqpu.json"

PYTHON="${CONDA_PREFIX:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env}/bin/python"
TORCHRUN="${CONDA_PREFIX:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env}/bin/torchrun"

echo "=== Multi-GPU Chemistry-Conditioned GQE Workflow ==="
echo "Project root: ${PROJECT_ROOT}"
echo "GPUs available: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Train chemistry encoder with DDP across 2 GPUs
# ---------------------------------------------------------------------------
echo "[Step 1/3] Training chemistry encoder on 2 GPUs (PyTorch DDP)..."
${TORCHRUN} \
    --nproc_per_node=2 \
    "${PROJECT_ROOT}/src/gqe/models/train_chemistry_encoder_ddp.py" \
    --json "${HAMILTONIANS}" \
    --out "${ENCODER_OUT}" \
    --mode graph \
    --epochs 100 \
    --batch-size 4 \
    --hidden-dim 128 \
    --latent-dim 128 \
    --conditioning-dim 128 \
    --num-layers 3 \
    --learning-rate 1e-3 \
    --train-fraction 0.8 \
    --include-fragments

echo "Encoder training complete. Model saved to:"
ls -lh "${ENCODER_OUT%/*}"/ddp_graph_conditioning.pt 2>/dev/null || echo "(check ${ENCODER_OUT%/*})"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Export conditioning vectors (embeddings) from trained encoder
# ---------------------------------------------------------------------------
echo "[Step 2/3] Exporting conditioning vectors (single GPU is fine here)..."
${PYTHON} "${PROJECT_ROOT}/src/gqe/models/export_conditioning_vectors.py" \
    --json "${HAMILTONIANS}" \
    --checkpoint "${ENCODER_OUT%/*}/ddp_graph_conditioning.pt" \
    --out "${EMBEDDINGS_OUT}" \
    --include-fragments

echo "Conditioning vectors exported: ${EMBEDDINGS_OUT}"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Run CUDA-Q GQE with mqpu backend (Hamiltonian batching on 2 GPUs)
# ---------------------------------------------------------------------------
echo "[Step 3/3] Running CUDA-Q GQE with mqpu multi-GPU backend..."
# For mqpu with thread-based parallel execution across 2 GPUs on a single node:
${PYTHON} "${PROJECT_ROOT}/src/gqe/baselines/run_cudaq_gqe_mqpu.py" \
    --ham "${HAMILTONIANS}" \
    --out "${GQE_OUT}" \
    --target nvidia \
    --target-option mqpu \
    --max-iters 25 \
    --ngates 10 \
    --max-terms 32

echo "GQE complete. Results:"
ls -lh "${GQE_OUT}"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Workflow Summary ==="
echo "Encoder model:   ${ENCODER_OUT%/*}/ddp_graph_conditioning.pt"
echo "GQE results:     ${GQE_OUT}"
echo ""
echo "Next steps:"
echo "  - Compare results: python src/gqe/eval/compare_conditioning_ablation.py ..."
echo "  - Plot curves:     python src/gqe/eval/plot_conditioning_ablation.py ..."
