#!/bin/bash
# Phase 3 Final — Step 2: Run baseline ansatz comparisons
# Runs hardware-efficient VQE and CUDA-Q GQE baseline on CH3I
# Usage: bash scripts/phase3/02_run_baselines.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export PY="${PYTHON}"

HAM=results/data/hamiltonians_phase3.json/hamiltonians.json
OUT_DIR=results/phase3_final/baselines
mkdir -p "$OUT_DIR"

echo "=== Step 2: Run baselines on CH3I ==="

# Hardware-efficient VQE baseline
echo "[2a] Hardware-efficient VQE..."
$PY src/gqe/baselines/run_cudaq_vqe.py \
    --ham "$HAM" \
    --out "$OUT_DIR/benchmark_ch3i_he_vqe.json" \
    --target nvidia \
    --molecule methyl_iodide \
    --maxiter 200 \
    --ansatz-reps 3

# CUDA-Q GQE baseline (UCCSD pool)
echo ""
echo "[2b] CUDA-Q GQE baseline (UCCSD pool)..."
$PY src/gqe/baselines/run_cudaq_gqe.py \
    --ham "$HAM" \
    --out "$OUT_DIR/benchmark_ch3i_gqe.json" \
    --target nvidia \
    --max-qubits 24 \
    --max-iters 25 \
    --ngates 10 \
    --molecule methyl_iodide

echo ""
echo "=== Baselines complete ==="
echo "Results: $OUT_DIR/"
