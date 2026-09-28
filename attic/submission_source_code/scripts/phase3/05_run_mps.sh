#!/bin/bash
# Phase 3 Final — Step 5: MPS scaling curve
# Usage: bash scripts/phase3/05_run_mps.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export PY="${PYTHON}"

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
