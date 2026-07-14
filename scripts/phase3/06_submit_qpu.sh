#!/bin/bash
# Phase 3 Final — Step 6: Submit QPU job via qBraid
# Prerequisite: Run qpu_preflight.py first to check availability and cost
# Usage: bash scripts/phase3/06_submit_qpu.sh
set -e

export PY=${PY:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}
export LD_LIBRARY_PATH=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:$LD_LIBRARY_PATH
cd /scratch/kcwp264/Conditional-GQE_materials

OUT_DIR=results/phase3_final/qpu
mkdir -p "$OUT_DIR"

echo "=== Step 6: QPU submission ==="

# Step 6a: Preflight check
echo "[6a] Running QPU preflight..."
$PY scripts/qpu_preflight.py --dry-run --out "$OUT_DIR/preflight.json"

# Step 6b: Submit to QPU
echo ""
echo "[6b] Submitting best circuit to QPU..."
$PY src/gqe/eval/submit_qpu.py \
    --benchmark results/phase3_final/hcgqe/benchmark_ch3i_optimized.json \
    --config configs/phase3_final/qpu_validation.yaml \
    --out "$OUT_DIR/qpu_submission.json"

echo ""
echo "=== QPU submission complete ==="
echo "Monitor job with: $PY src/gqe/eval/collect_qpu.py --job-id <ID>"
echo "Results: $OUT_DIR/"
