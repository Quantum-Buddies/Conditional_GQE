#!/bin/bash
# Post-Training Evaluation Pipeline (portable, no Slurm required)
#
# 1. H-cGQE inference + L-BFGS-B optimization on all molecules
# 2. FMO2 reconstruction (exact + GQE)
# 3. QPU manifest generation (H2, LiH)
# 4. QSCI for 32-40q systems
#
# Usage: bash scripts/run_eval_pipeline.sh [CHECKPOINT]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export CUDAQ_MQPU_NGPUS="${CUDAQ_MQPU_NGPUS:-2}"

CHECKPOINT="${1:-results/train/h_cgqe_model_rl_offline.pt}"
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT="results/train/h_cgqe_model_b200_sft.pt"
    echo "WARNING: RL checkpoint not found, using SFT checkpoint: ${CHECKPOINT}"
fi

HAMILTONIANS="results/data/hamiltonians_merged.json"
FRAGMENTS="results/data/fragments/fmo_hamiltonians.json"
ARCHIVE_DIR="results/train/h_cgqe_model_rl_offline_map_elites"
OUT_DIR="results/phase3_final"
mkdir -p "$OUT_DIR/fmo" "$OUT_DIR/eval" "$OUT_DIR/qpu" logs

echo "=== GQE Evaluation Pipeline ==="
echo "Start: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo ""

# Step 1: H-cGQE Inference + Optimization
echo "[1/4] H-cGQE inference + L-BFGS-B optimization..."
"${PYTHON}" -u src/gqe/eval/evaluate_h_cgqe.py \
    --checkpoint "${CHECKPOINT}" \
    --hamiltonians "${HAMILTONIANS}" \
    --target nvidia --target-option mqpu \
    --n-samples 200 --max-qubits 24 \
    --out "${OUT_DIR}/eval/h_cgqe_evaluation_final.json" \
    2>&1 || echo "  WARNING: eval step had issues, continuing..."

# Step 2: FMO2 Reconstruction
echo ""
echo "[2/4] FMO2 reconstruction..."
if [ -f "$FRAGMENTS" ]; then
    "${PYTHON}" -u src/gqe/eval/run_fmo2.py \
        --fragments "${FRAGMENTS}" --method exact \
        --out "${OUT_DIR}/fmo/fmo2_exact_final.json" 2>&1
    "${PYTHON}" -u src/gqe/eval/run_fmo2.py \
        --fragments "${FRAGMENTS}" --method hcgqe \
        --checkpoint "${CHECKPOINT}" --archive-dir "${ARCHIVE_DIR}" \
        --target nvidia --target-option mqpu --n-samples 100 \
        --out "${OUT_DIR}/fmo/fmo2_gqe_final.json" \
        2>&1 || echo "  WARNING: FMO2 GQE step had issues, continuing..."
else
    echo "  SKIP: Fragment Hamiltonians not found at ${FRAGMENTS}"
fi

# Step 3: QPU Manifest Generation
echo ""
echo "[3/4] QPU manifest generation..."
OPTIMIZED="results/eval/h_cgqe_uccsd_optimized.json"
if [ -f "$HAMILTONIANS" ] && [ -f "$OPTIMIZED" ]; then
    "${PYTHON}" -u scripts/phase3/generate_qpu_manifests.py \
        --molecules h2_0.74 lih_1.6_full \
        --hamiltonians "${HAMILTONIANS}" --optimized "${OPTIMIZED}" \
        --out-dir "${OUT_DIR}/qpu/manifests" --shots 4096 \
        2>&1 || echo "  WARNING: QPU manifest step had issues, continuing..."
else
    echo "  SKIP: Need hamiltonians + optimized results"
fi

# Step 4: QSCI
echo ""
echo "[4/4] QSCI for large systems..."
if [ -f "src/gqe/eval/qsci.py" ]; then
    "${PYTHON}" -u src/gqe/eval/qsci.py \
        --checkpoint "${CHECKPOINT}" \
        --hamiltonians results/data/hamiltonians_40plus/hamiltonians.json \
        --target nvidia \
        --out "${OUT_DIR}/eval/qsci_results.json" \
        2>&1 || echo "  WARNING: QSCI step had issues, continuing..."
else
    echo "  SKIP: QSCI script not found"
fi

echo ""
echo "=== Evaluation Pipeline Complete ==="
echo "End: $(date)"
echo "Results: ${OUT_DIR}/"
