#!/bin/bash
# Phase 3 Final — Step 3: Run H-cGQE inference + coefficient optimization
# Usage: bash scripts/phase3/03_run_hcgqe.sh
set -e

export PY=${PY:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}
export LD_LIBRARY_PATH=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:$LD_LIBRARY_PATH
cd /scratch/kcwp264/Conditional-GQE_materials

HAM=results/data/hamiltonians_phase3.json/hamiltonians.json
CKPT=results/train/h_cgqe_model_rlqf_phase3.pt
OUT_DIR=results/phase3_final/hcgqe
mkdir -p "$OUT_DIR"

echo "=== Step 3: H-cGQE inference + optimization ==="

# Inference: generate circuits
echo "[3a] Generating circuits..."
$PY src/gqe/models/infer_h_cgqe.py \
    --checkpoint "$CKPT" \
    --hamiltonians "$HAM" \
    --out "$OUT_DIR/benchmark_ch3i_inference.json" \
    --molecules methyl_iodide \
    --n-samples 100 \
    --sample \
    --use-cuda \
    --max-pauli-len 24 \
    --max-seq-len 64 \
    --temperature 1.0 \
    --force-entanglement \
    --freq-penalty 1.0 \
    --max-repeat 4

# Coefficient optimization
echo ""
echo "[3b] Optimizing coefficients (L-BFGS-B)..."
$PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated "$OUT_DIR/benchmark_ch3i_inference.json" \
    --hamiltonians "$HAM" \
    --out "$OUT_DIR/benchmark_ch3i_optimized.json" \
    --top-k 10 \
    --max-iter 200 \
    --target nvidia --target-option mqpu

# Evaluation
echo ""
echo "[3c] Evaluating..."
$PY src/gqe/eval/evaluate_h_cgqe.py \
    --generated "$OUT_DIR/benchmark_ch3i_inference.json" \
    --baseline results/phase3_final/baselines/benchmark_ch3i_gqe.json \
    --hamiltonians "$HAM" \
    --out "$OUT_DIR/benchmark_ch3i_evaluation.json" \
    --target nvidia --target-option mqpu

echo ""
echo "=== H-cGQE complete ==="
echo "Results: $OUT_DIR/"
