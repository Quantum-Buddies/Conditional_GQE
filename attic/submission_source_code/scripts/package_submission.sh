#!/usr/bin/env bash
# Package the GIC Phase 3 submission zip.
# Format: TeamName_Challenge_Phase3.zip
#
# Contents:
#   Write-Up.pdf          — 5-page PDF report
#   README.md             — Setup and reproducibility instructions
#   src/                  — Source code
#   scripts/              — Pipeline scripts
#   configs/              — Experiment configs
#   results/              — Key result JSONs (no large checkpoints)
#   jobs/                 — Slurm job scripts
#   environment-dgx-spark-cudaq.yml — Environment manifest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGING="${ROOT}/submission"
ZIP_NAME="Quantum-Buddies_Challenge_Phase3.zip"
ZIP_PATH="${ROOT}/${ZIP_NAME}"

echo "=== Packaging GIC Phase 3 Submission ==="

# Clean previous zip
rm -f "${ZIP_PATH}"

# Create staging directory structure
STAGE="${ROOT}/_submission_stage"
rm -rf "${STAGE}"
mkdir -p "${STAGE}"

# Copy write-up
cp "${STAGING}/Write-Up.pdf" "${STAGE}/"

# Copy README
cp "${STAGING}/README.md" "${STAGE}/"

# Copy source code
cp -r "${ROOT}/src" "${STAGE}/src"

# Copy scripts
mkdir -p "${STAGE}/scripts"
for f in \
    generate_submission_pdf.py \
    generate_fmo2_fragments.py \
    rl_optimize_and_submit.py \
    submit_sqd_to_cepheus.py \
    retrieve_and_sqd.py \
    consolidate_benchmark.py \
    build_gic_benchmark.py \
    generate_gic_submission.py \
    generate_phase3_pdf.py \
    generate_phase3_report.py \
    generate_report.py \
    generate_report_v2.py \
    plot_qpu_vs_gpu.py \
    extract_best_circuits.py \
; do
    if [ -f "${ROOT}/scripts/${f}" ]; then
        cp "${ROOT}/scripts/${f}" "${STAGE}/scripts/"
    fi
done

# Copy configs
if [ -d "${ROOT}/configs" ]; then
    cp -r "${ROOT}/configs" "${STAGE}/configs"
fi

# Copy jobs
if [ -d "${ROOT}/jobs" ]; then
    cp -r "${ROOT}/jobs" "${STAGE}/jobs"
fi

# Copy environment manifests
for f in environment-dgx-spark.yml environment-dgx-spark-cudaq.yml; do
    if [ -f "${ROOT}/${f}" ]; then
        cp "${ROOT}/${f}" "${STAGE}/"
    fi
done

# Copy key results (no large checkpoints or model files)
mkdir -p "${STAGE}/results/phase3_final/fmo"
mkdir -p "${STAGE}/results/phase3_final/qpu"
mkdir -p "${STAGE}/results/phase3_final/figures"
mkdir -p "${STAGE}/results/eval/benchmark"
mkdir -p "${STAGE}/results/qpu"
mkdir -p "${STAGE}/results/data/fragments"

# Phase 3 consolidated results
cp "${ROOT}/results/phase3_final/consolidated_phase3_results.json" "${STAGE}/results/phase3_final/" 2>/dev/null || true
cp "${ROOT}/results/phase3_final/benchmark_ch3i_consolidated.json" "${STAGE}/results/phase3_final/" 2>/dev/null || true

# FMO2 results
cp "${ROOT}/results/phase3_final/fmo/"*.json "${STAGE}/results/phase3_final/fmo/" 2>/dev/null || true

# QPU results
cp "${ROOT}/results/phase3_final/qpu/qpu_validation_consolidated.json" "${STAGE}/results/phase3_final/qpu/" 2>/dev/null || true
cp "${ROOT}/results/phase3_final/qpu/preflight.json" "${STAGE}/results/phase3_final/qpu/" 2>/dev/null || true

# RL optimized + QPU SQD
cp "${ROOT}/results/eval/h_cgqe_rl_optimized.json" "${STAGE}/results/eval/" 2>/dev/null || true
cp "${ROOT}/results/qpu/cepheus_rl_sqd_results.json" "${STAGE}/results/qpu/" 2>/dev/null || true

# Consolidated benchmark
cp "${ROOT}/results/eval/benchmark/gic2026_consolidated_benchmark.json" "${STAGE}/results/eval/benchmark/" 2>/dev/null || true

# Figures
cp "${ROOT}/results/phase3_final/figures/"*.png "${STAGE}/results/phase3_final/figures/" 2>/dev/null || true
cp "${ROOT}/results/eval/benchmark/"*.png "${STAGE}/results/eval/benchmark/" 2>/dev/null || true

# Fragment data
cp "${ROOT}/results/data/fragments/"*.json "${STAGE}/results/data/fragments/" 2>/dev/null || true

# Remove __pycache__ and .pyc files
find "${STAGE}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${STAGE}" -name "*.pyc" -delete 2>/dev/null || true

# Create zip
cd "${ROOT}"
zip -r "${ZIP_PATH}" "$(basename "${STAGE}")" -x "*/__pycache__/*" "*.pyc"

# Cleanup staging
rm -rf "${STAGE}"

echo ""
echo "=== Submission packaged ==="
echo "  File: ${ZIP_PATH}"
echo "  Size: $(du -h "${ZIP_PATH}" | cut -f1)"
echo ""
echo "  Contents:"
unzip -l "${ZIP_PATH}" | head -40
echo ""
echo "Upload ${ZIP_NAME} to Aqora."
