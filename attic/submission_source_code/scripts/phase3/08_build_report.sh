#!/bin/bash
# Phase 3 Final — Step 8: Build 5-page PDF report
# Generates report only from version-controlled result JSONs
# Usage: bash scripts/phase3/08_build_report.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export PY="${PYTHON}"

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
