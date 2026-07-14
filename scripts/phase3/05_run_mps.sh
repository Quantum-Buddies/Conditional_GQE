#!/bin/bash
# Phase 3 Final — Step 5: MPS scaling curve
# Usage: bash scripts/phase3/05_run_mps.sh
set -e

export PY=${PY:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}
export LD_LIBRARY_PATH=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:$LD_LIBRARY_PATH
cd /scratch/kcwp264/Conditional-GQE_materials

OUT_DIR=results/phase3_final/mps
mkdir -p "$OUT_DIR"

echo "=== Step 5: MPS scaling curve ==="

# Run MPS benchmark with bond dimension sweep
$PY src/gqe/eval/run_mps_scaling.py \
    --config configs/phase3_final/mps_scaling.yaml \
    --out "$OUT_DIR/mps_scaling_results.json"

echo ""
echo "=== MPS scaling complete ==="
echo "Results: $OUT_DIR/"
