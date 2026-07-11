#!/bin/bash
# =============================================================================
# GIC 2026 Scaling Pipeline — H-cGQE for Next-Generation Materials Design
# =============================================================================
# Full pipeline: Generate Hamiltonians → Retrain RL on all molecules →
#   Inference on all molecules → Stage 2 optimization → Scaling plots
#
# Targets Mitsubishi Chemical/AIST use case:
#   EUV photoresist materials (IMePh, iodobenzene, methyl_iodide, phenol)
#   General materials informatics (water, ammonia, ethylene, etc.)
#
# Usage: bash scripts/run_gic2026_scaling.sh [--skip-hamiltonians]
# =============================================================================
set -eo pipefail

PROJECT_ROOT="/scratch/kcwp264/Conditional-GQE_materials"
PYTHON="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"
export CUDAQ_MPS_MAX_BOND=64

cd "${PROJECT_ROOT}"

# Config
CONFIG="configs/experiment_scaling_gic2026.yaml"
HAM_DIR="results/data/hamiltonians_gic2026"
HAM="${HAM_DIR}/hamiltonians.json"
RL_OUTPUT="results/train/h_cgqe_rl_gic2026.pt"
RL_METRICS="results/train/h_cgqe_rl_gic2026_rl_metrics.json"
SUPERVISED_CKPT="results/train/h_cgqe_uccsd_model.pt"

# All molecules from the GIC2026 config
ALL_MOLECULES="h2 lih beh2 n2 h2o nh3 ch4 ethylene formaldehyde acetylene hf co \
h2_0.5 h2_1.0 h2_1.5 h2_2.0 lih_1.2 lih_2.0 lih_3.0 n2_1.8 n2_2.5 beh2_1.0 beh2_1.6 \
iodobenzene_cas12 methyl_iodide_cas12 imeph_cas12 phenol_cas12 ocresol_cas12 anisole_cas12 \
benzene_cas12 toluene_cas12 \
lih_1.6_631g n2_1.1_631g_cas8 h2o_1.0_631g_cas8"

# Molecules <= 24 qubits (can run on nvidia-mqpu)
SIMULATABLE_MOLECULES="h2 lih beh2 n2 h2o nh3 ch4 ethylene formaldehyde acetylene hf co \
h2_0.5 h2_1.0 h2_1.5 h2_2.0 lih_1.2 lih_2.0 lih_3.0 n2_1.8 n2_2.5 beh2_1.0 beh2_1.6 \
iodobenzene_cas12 methyl_iodide_cas12 imeph_cas12 phenol_cas12 ocresol_cas12 anisole_cas12 \
benzene_cas12 toluene_cas12 \
lih_1.6_631g n2_1.1_631g_cas8 h2o_1.0_631g_cas8"

SKIP_HAM=false
if [ "$1" == "--skip-hamiltonians" ]; then
    SKIP_HAM=true
fi

echo "================================================================"
echo "GIC 2026 SCALING PIPELINE"
echo "  H-cGQE + Chemeleon2 RL for Materials Design"
echo "  Target: Mitsubishi Chemical/AIST EUV photoresist use case"
echo "================================================================"
echo "  Total molecules: $(echo ${ALL_MOLECULES} | wc -w)"
echo "  Simulatable (≤24q): $(echo ${SIMULATABLE_MOLECULES} | wc -w)"

# ================================================================
# STEP 1: Generate Hamiltonians for all molecules
# ================================================================
echo ""
echo "================================================================"
echo "STEP 1: Generate Hamiltonians"
echo "================================================================"
if [ "$SKIP_HAM" == true ] && [ -f "$HAM" ]; then
    echo "Skipping (Hamiltonians already exist at ${HAM})"
else
    mkdir -p "${HAM_DIR}"
    ${PYTHON} src/gqe/data/generate_hamiltonians.py \
        --config "${CONFIG}" \
        --out "${HAM_DIR}"
    echo "Hamiltonians saved to ${HAM}"
fi

# Show summary
${PYTHON} -c "
import json
with open('${HAM}') as f:
    data = json.load(f)
records = data['records'] if isinstance(data, dict) else data
print(f'Total molecules: {len(records)}')
for r in records:
    print(f\"  {r['name']:30s}  qubits={r['n_qubits']:3d}  terms={r['n_pauli_terms']:6d}  split={r.get('split','?')}\")
"

# ================================================================
# STEP 2: Retrain Chemeleon2 RL on ALL molecules
# ================================================================
echo ""
echo "================================================================"
echo "STEP 2: Chemeleon2 RL Training (all molecules)"
echo "  500 epochs, curriculum warmup=80, group_size=32"
echo "================================================================"
${PYTHON} src/gqe/models/train_rl_dapo.py \
    --checkpoint "${SUPERVISED_CKPT}" \
    --hamiltonians "${HAM}" \
    --molecules ${SIMULATABLE_MOLECULES} \
    --out "${RL_OUTPUT}" \
    --use-cuda \
    --target nvidia \
    --epochs 500 \
    --n-samples 32 \
    --max-qubits 24 \
    --use-bf16 \
    --force-entanglement \
    --curriculum \
    --curriculum-warmup 80 \
    --chemeleon2-mode \
    --seed 42 \
    2>&1 | tee results/logs/gic2026_rl_training.out

echo "RL training complete: ${RL_OUTPUT}"

# ================================================================
# STEP 3: Inference — generate circuits for all molecules
# ================================================================
echo ""
echo "================================================================"
echo "STEP 3: Inference (stochastic sampling, 100 circuits per molecule)"
echo "================================================================"

# 3a. Supervised baseline
INFER_SUP="results/inference/gic2026_supervised_sampled.json"
echo "  Generating supervised baseline..."
${PYTHON} src/gqe/models/infer_h_cgqe.py \
    --checkpoint "${SUPERVISED_CKPT}" \
    --hamiltonians "${HAM}" \
    --molecules ${SIMULATABLE_MOLECULES} \
    --n-samples 100 \
    --out "${INFER_SUP}" \
    --use-cuda \
    --sample \
    --temperature 1.0 \
    --force-entanglement \
    --freq-penalty 1.0

# 3b. Chemeleon2 RL model
INFER_RL="results/inference/gic2026_chemeleon2_sampled.json"
echo "  Generating Chemeleon2 RL circuits..."
${PYTHON} src/gqe/models/infer_h_cgqe.py \
    --checkpoint "${RL_OUTPUT}" \
    --hamiltonians "${HAM}" \
    --molecules ${SIMULATABLE_MOLECULES} \
    --n-samples 100 \
    --out "${INFER_RL}" \
    --use-cuda \
    --sample \
    --temperature 1.0 \
    --force-entanglement \
    --freq-penalty 1.0

# ================================================================
# STEP 4: Stage 2 coefficient optimization
# ================================================================
echo ""
echo "================================================================"
echo "STEP 4: Stage 2 Coefficient Optimization (L-BFGS-B, top-20 per molecule)"
echo "================================================================"

OPT_SUP="results/eval/gic2026_supervised_stage2.json"
OPT_RL="results/eval/gic2026_chemeleon2_stage2.json"

echo "  Optimizing supervised circuits..."
${PYTHON} src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated "${INFER_SUP}" \
    --hamiltonians "${HAM}" \
    --out "${OPT_SUP}" \
    --target nvidia \
    --max-iter 200 \
    --top-k 20 \
    --max-qubits 24

echo "  Optimizing Chemeleon2 circuits..."
${PYTHON} src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated "${INFER_RL}" \
    --hamiltonians "${HAM}" \
    --out "${OPT_RL}" \
    --target nvidia \
    --max-iter 200 \
    --top-k 20 \
    --max-qubits 24

# ================================================================
# STEP 5: GQE Baseline (CUDA-Q solvers.gqe)
# ================================================================
echo ""
echo "================================================================"
echo "STEP 5: CUDA-Q GQE Baseline"
echo "================================================================"
GQE_BASELINE="results/baselines/gic2026_cudaq_gqe.json"
${PYTHON} src/gqe/baselines/run_cudaq_gqe.py \
    --ham "${HAM}" \
    --out "${GQE_BASELINE}" \
    --target nvidia \
    --max-qubits 24 \
    2>&1 | tee results/logs/gic2026_gqe_baseline.out || echo "GQE baseline failed (non-fatal)"

# ================================================================
# STEP 6: Generate comprehensive scaling plots
# ================================================================
echo ""
echo "================================================================"
echo "STEP 6: Scaling Plots & Analysis"
echo "================================================================"
${PYTHON} scripts/plot_gic2026_scaling.py \
    --supervised-opt "${OPT_SUP}" \
    --chemeleon2-opt "${OPT_RL}" \
    --gqe-baseline "${GQE_BASELINE}" \
    --rl-metrics "${RL_METRICS}" \
    --supervised-infer "${INFER_SUP}" \
    --chemeleon2-infer "${INFER_RL}" \
    --hamiltonians "${HAM}" \
    --output-dir results/plots/gic2026

echo ""
echo "================================================================"
echo "GIC 2026 SCALING PIPELINE COMPLETE"
echo "  Results: results/eval/gic2026_*.json"
echo "  Plots:   results/plots/gic2026/"
echo "================================================================"
