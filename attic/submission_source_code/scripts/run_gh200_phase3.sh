#!/bin/bash
# All-in-one Phase 3 execution script for qBraid GH200 instance.
# Runs: RL training → H-cGQE evaluation → MPS scaling → QSCI 40q
#
# Usage (on qBraid Lab GH200 instance after setup):
#   bash scripts/run_gh200_phase3.sh
#
# Estimated time: ~9h total
# Estimated cost: ~2,483 credits (at 4.78 cr/min GH200 rate)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

START_TIME=$(date +%s)
CR_RATE=4.78  # GH200 credits/min

CHECKPOINT="${PROJECT_ROOT}/results/train/h_cgqe_uccsd_model.pt"
HAMILTONIANS="${PROJECT_ROOT}/results/data/hamiltonians_merged.json"
RESULTS_DIR="${PROJECT_ROOT}/results/phase3_final/gh200"
RL_OUTPUT="${RESULTS_DIR}/rl_model.pt"

mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "  Phase 3 GH200 Execution Pipeline"
echo "  Start: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Results: ${RESULTS_DIR}"
echo "============================================================"
echo ""

# --- Stage 1: RL Training (500 epochs, ~7.5h) ---
echo ">>> [1/4] RL Training (500 epochs, 4 molecules, ~7.5h)"
echo "    Start: $(date)"

python -u src/gqe/models/train_rl_dapo.py \
    --checkpoint "${CHECKPOINT}" \
    --hamiltonians "${HAMILTONIANS}" \
    --molecules h2 lih beh2 n2 \
    --out "${RL_OUTPUT}" \
    --epochs 500 \
    --n-samples 64 \
    --n-iters 5 \
    --reuse-iters 3 \
    --buffer-size 2000 \
    --buffer-batch-size 64 \
    --lr 1e-5 \
    --temperature 1.0 \
    --top-p 0.9 \
    --target-entropy 1.5 \
    --explore-eps 0.3 \
    --adaptive-eps \
    --force-entanglement \
    --max-repeat 4 \
    --max-qubits 28 \
    --mps-threshold 24 \
    --target nvidia \
    --target-option mqpu \
    --use-cuda \
    --use-bf16 \
    --dynamic-sampling \
    --token-level-loss \
    --entropy-coef 1e-5 \
    --w-energy 1.0 \
    --w-entangle 0.1 \
    --w-depth 0.05 \
    --w-commute 0.05 \
    --w-diversity 0.2 \
    --target-len 10 \
    --freq-penalty 1.0 \
    --curriculum \
    --curriculum-warmup 50 \
    --curriculum-steps 3 \
    --chemeleon2-mode \
    --msun-threshold 0.1 \
    --adaptive-theta \
    --adaptive-theta-iters 10 \
    2>&1 | tee "${RESULTS_DIR}/rl_training.log"

RL_END=$(date +%s)
RL_MIN=$(( (RL_END - START_TIME) / 60 ))
RL_CR=$(echo "${RL_MIN} * ${CR_RATE}" | bc -l)
echo "    End: $(date)"
echo "    Elapsed: ${RL_MIN} min (${RL_CR} credits)"
echo ""

# --- Stage 2: H-cGQE Evaluation (formaldehyde 24q + ethylene 28q, ~30min) ---
echo ">>> [2/4] H-cGQE Evaluation (24q + 28q via MPS, ~30min)"
echo "    Start: $(date)"

python -u src/gqe/eval/evaluate_h_cgqe.py \
    --checkpoint "${RL_OUTPUT}" \
    --hamiltonians "${HAMILTONIANS}" \
    --molecules formaldehyde ethylene \
    --max-qubits 28 \
    --target nvidia \
    --use-cuda \
    --output "${RESULTS_DIR}/eval_results.json" \
    2>&1 | tee "${RESULTS_DIR}/eval.log" || echo "    WARNING: Evaluation failed, continuing..."

EVAL_END=$(date +%s)
EVAL_MIN=$(( (EVAL_END - RL_END) / 60 ))
EVAL_CR=$(echo "${EVAL_MIN} * ${CR_RATE}" | bc -l)
echo "    End: $(date)"
echo "    Elapsed: ${EVAL_MIN} min (${EVAL_CR} credits)"
echo ""

# --- Stage 3: MPS Scaling Benchmark (24-40q × 4 bond dims, ~8min) ---
echo ">>> [3/4] MPS Scaling Benchmark (24q, 28q, 32q, 40q × D=32,64,128,256)"
echo "    Start: $(date)"

python -u src/gqe/eval/run_mps_scaling.py \
    --hamiltonians "${HAMILTONIANS}" \
    --molecules formaldehyde ethylene n2_ccpvdz benzene_cas20 \
    --bond-dims 32 64 128 256 \
    --target tensornet-mps \
    --output "${RESULTS_DIR}/mps_scaling_results.json" \
    2>&1 | tee "${RESULTS_DIR}/mps_scaling.log" || echo "    WARNING: MPS scaling failed, continuing..."

MPS_END=$(date +%s)
MPS_MIN=$(( (MPS_END - EVAL_END) / 60 ))
MPS_CR=$(echo "${MPS_MIN} * ${CR_RATE}" | bc -l)
echo "    End: $(date)"
echo "    Elapsed: ${MPS_MIN} min (${MPS_CR} credits)"
echo ""

# --- Stage 4: QSCI at 40q (benzene, ~1min) ---
echo ">>> [4/4] QSCI 40q (benzene CAS(20e,20o))"
echo "    Start: $(date)"

python -u src/gqe/eval/run_qsci.py \
    --hamiltonians "${HAMILTONIANS}" \
    --molecule benzene_cas20 \
    --target tensornet-mps \
    --bond-dim 256 \
    --n-samples 10000 \
    --output "${RESULTS_DIR}/qsci_40q.json" \
    2>&1 | tee "${RESULTS_DIR}/qsci.log" || echo "    WARNING: QSCI failed, continuing..."

QSCI_END=$(date +%s)
QSCI_MIN=$(( (QSCI_END - MPS_END) / 60 ))
QSCI_CR=$(echo "${QSCI_MIN} * ${CR_RATE}" | bc -l)
echo "    End: $(date)"
echo "    Elapsed: ${QSCI_MIN} min (${QSCI_CR} credits)"
echo ""

# --- Summary ---
TOTAL_MIN=$(( (QSCI_END - START_TIME) / 60 ))
TOTAL_CR=$(echo "${TOTAL_MIN} * ${CR_RATE}" | bc -l)

echo "============================================================"
echo "  Phase 3 GH200 Pipeline Complete"
echo "  End: $(date)"
echo ""
echo "  Stage breakdown:"
echo "    1. RL Training:     ${RL_MIN} min  (${RL_CR} credits)"
echo "    2. Evaluation:      ${EVAL_MIN} min  (${EVAL_CR} credits)"
echo "    3. MPS Scaling:     ${MPS_MIN} min  (${MPS_CR} credits)"
echo "    4. QSCI 40q:        ${QSCI_MIN} min  (${QSCI_CR} credits)"
echo ""
echo "  Total: ${TOTAL_MIN} min (${TOTAL_CR} credits)"
echo "  Remaining budget: $(echo "9644.91 - ${TOTAL_CR}" | bc -l) credits"
echo ""
echo "  Results saved to: ${RESULTS_DIR}/"
echo "    rl_model.pt              — RL-tuned model checkpoint"
echo "    rl_training.log          — Training log"
echo "    eval_results.json        — H-cGQE evaluation (24q, 28q)"
echo "    mps_scaling_results.json — MPS vs SV energy comparison"
echo "    qsci_40q.json            — QSCI benzene 40q result"
echo "============================================================"
