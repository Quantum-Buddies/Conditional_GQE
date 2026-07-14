#!/bin/bash
# Phase 3 Final — Step 8: Build 5-page PDF report
# Generates report only from version-controlled result JSONs
# Usage: bash scripts/phase3/08_build_report.sh
set -e

export PY=${PY:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}
export LD_LIBRARY_PATH=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:$LD_LIBRARY_PATH
cd /scratch/kcwp264/Conditional-GQE_materials

OUT=proposals/Ryoushi_Quantum_Buddies__Phase3_Version1.pdf

echo "=== Step 8: Build Phase 3 report ==="

$PY scripts/generate_phase3_report.py \
    --benchmark results/phase3_final/baselines/ \
    --hcgqe results/phase3_final/hcgqe/ \
    --fmo results/phase3_final/fmo/ \
    --mps results/phase3_final/mps/ \
    --qpu results/phase3_final/qpu/ \
    --out "$OUT"

echo ""
echo "=== Report generated: $OUT ==="
