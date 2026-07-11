#!/bin/bash
# BeH2-focused training: higher creativity weight + more epochs
# BeH2 showed lowest diversity (0.19) in Chemeleon2 results
set -eo pipefail

PROJECT_ROOT="/scratch/kcwp264/Conditional-GQE_materials"
PYTHON="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"
export CUDAQ_MPS_MAX_BOND=64

cd "${PROJECT_ROOT}"

CHECKPOINT="results/train/h_cgqe_uccsd_model.pt"
HAMILTONIANS="results/data/hamiltonians_merged.json"
OUTPUT="results/train/h_cgqe_rl_beh2_boosted.pt"

echo "============================================"
echo "BeH2-Boosted Chemeleon2 Training"
echo "  Higher creativity weight (3.0 vs 1.0)"
echo "  Higher MMD diversity weight (2.0 vs 1.0)"
echo "  More epochs (400 vs 200)"
echo "  Longer curriculum warmup (50 vs 30)"
echo "============================================"

${PYTHON} src/gqe/models/train_rl_dapo.py \
    --checkpoint "${CHECKPOINT}" \
    --hamiltonians "${HAMILTONIANS}" \
    --molecules h2 lih beh2 n2 \
    --out "${OUTPUT}" \
    --use-cuda \
    --target nvidia \
    --target-option mqpu \
    --epochs 400 \
    --n-samples 32 \
    --max-qubits 24 \
    --use-bf16 \
    --force-entanglement \
    --curriculum \
    --curriculum-warmup 50 \
    --kl-coef 1.0 \
    --w-creativity 3.0 \
    --w-mmd-diversity 2.0 \
    --clip-low 0.001 \
    --clip-high 0.001 \
    --entropy-coef 1e-5 \
    --seed 42 \
    2>&1 | tee results/logs/beh2_boosted.out

echo ""
echo "BeH2-boosted training complete: ${OUTPUT}"

# Generate inference + analysis
echo "Generating inference samples..."

${PYTHON} src/gqe/models/infer_h_cgqe.py \
    --checkpoint "${OUTPUT}" \
    --hamiltonians "${HAMILTONIANS}" \
    --molecules h2 lih beh2 n2 \
    --n-samples 100 \
    --out results/inference/beh2_boosted_sampled.json \
    --use-cuda \
    --sample \
    --temperature 1.0 \
    --force-entanglement \
    --freq-penalty 1.0

echo "Analyzing diversity..."
${PYTHON} scripts/analyze_generated_circuits.py \
    --supervised results/inference/supervised_sampled.json \
    --chemeleon2 results/inference/beh2_boosted_sampled.json

echo "Done. Check results above for BeH2 diversity improvement."
