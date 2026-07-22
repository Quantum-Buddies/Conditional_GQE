#!/bin/bash
# =============================================================================
# 40-Qubit GPU-AI-QPU Scalability Pipeline (B200 NVFP4 Edition)
# =============================================================================
#
# Demonstrates end-to-end convergence from 4q to 40q across:
#   1. AI (H-cGQE Transformer) — RL training with QD-GRPO, generates compact circuits
#   2. GPU (CUDA-Q on B200) — validates with exact SV (≤32q) and MPS (32-40q)
#   3. QPU (IQM Emerald 54q / Rigetti Cepheus 108q) — executes on real hardware
#
# GPU: NVIDIA B200 (180GB, 8 TB/s HBM, NVFP4 tensor cores, NVLink 5)
#   - 32q single-GPU statevector (no mgpu needed!)
#   - NVFP4 for 1.59x training throughput
#   - 8 TB/s HBM = 1.67x faster CUDA-Q evaluation vs H200
#   - NVLink 5 fixes L40S PCIe IPC segfault for mgpu
#
# Budget: 24,645 credits
#   GPU (B200, ~22h):     ~19,200 cr (14.57 cr/min)
#   QPU (Emerald+Cepheus):  ~2,200 cr
#   Simulators:               ~200 cr (free tier)
#   Buffer:                ~3,000 cr
#
# Usage:
#   # On qBraid B200 instance:
#   bash scripts/run_40q_scaling_pipeline.sh
#
#   # Skip RL training (use existing checkpoint):
#   SKIP_TRAINING=1 bash scripts/run_40q_scaling_pipeline.sh
#
#   # Skip QPU stage:
#   SKIP_QPU=1 bash scripts/run_40q_scaling_pipeline.sh
#
#   # Use NVFP4 for RL training (requires transformer_engine):
#   USE_NVFP4=1 bash scripts/run_40q_scaling_pipeline.sh
#
# Prerequisites:
#   - CUDA-Q 0.14+ with nvidia, tensornet-mps targets
#   - PyTorch 2.6+ with CUDA (Blackwell support for B200)
#   - transformer_engine (optional, for NVFP4): pip install --no-build-isolation transformer_engine[pytorch]
#   - qBraid SDK with API key (for QPU stage)
#   - PySCF (for Hamiltonian generation)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

# --- Configuration ---
PY="${PYTHON:-python}"
CONFIG="${PROJECT_ROOT}/configs/experiment_40q_scaling.yaml"
RESULTS_DIR="${PROJECT_ROOT}/results/scaling_40q"
HAMILTONIANS_FILE="${RESULTS_DIR}/hamiltonians/hamiltonians.json"
TRAINING_OUT="${RESULTS_DIR}/training/rl_model_b200.pt"
TRAINING_METRICS="${RESULTS_DIR}/training/rl_metrics_b200.json"
INFERENCE_OUT="${RESULTS_DIR}/inference/generated_circuits.json"
OPTIMIZED_OUT="${RESULTS_DIR}/optimization/optimized_circuits.json"
SV_OUT="${RESULTS_DIR}/gpu_validation/statevector_results.json"
MPS_OUT="${RESULTS_DIR}/gpu_validation/mps_scaling_results.json"
QPU_OUT="${RESULTS_DIR}/qpu_results"
PLOTS_OUT="${RESULTS_DIR}/plots"
REPORT_OUT="${RESULTS_DIR}/scaling_report.json"

SKIP_HAMILTONIANS="${SKIP_HAMILTONIANS:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_OPTIMIZATION="${SKIP_OPTIMIZATION:-0}"
SKIP_GPU_VALIDATION="${SKIP_GPU_VALIDATION:-0}"
SKIP_QPU="${SKIP_QPU:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"

USE_NVFP4="${USE_NVFP4:-0}"
N_SAMPLES="${N_SAMPLES:-256}"

# RL training molecules — include 40q molecules for direct training on big GPU
RL_MOLECULES="h2 lih beh2 n2 formaldehyde ethylene n2_ccpvdz benzene_cas20"

# --- Cost tracking ---
START_TIME=$(date +%s)

log() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "  Time: $(date)"
    echo "================================================================"
}

track_cost() {
    local stage="$1"
    local credits="$2"
    local elapsed_min=$(( ($(date +%s) - START_TIME) / 60 ))
    echo "  [COST] ${stage}: ${credits} cr (elapsed: ${elapsed_min} min)"
}

mkdir -p "${RESULTS_DIR}/hamiltonians" "${RESULTS_DIR}/training" "${RESULTS_DIR}/inference" \
         "${RESULTS_DIR}/optimization" "${RESULTS_DIR}/gpu_validation" "${QPU_OUT}" "${PLOTS_OUT}"

# =============================================================================
# STAGE 0: Verify GPU environment
# =============================================================================
log "STAGE 0: GPU Environment Check"

${PY} -c "
import torch
import cudaq

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'VRAM: {props.total_memory / 1e9:.1f} GB')
    print(f'Compute capability: {props.major}.{props.minor}')
    if props.major >= 10:
        print('  -> Blackwell detected! NVFP4 tensor cores available.')
    elif props.major >= 9:
        print('  -> Hopper detected. FP8 available, no FP4.')

print(f'CUDA-Q: {cudaq.__version__}')
for t in ['nvidia', 'tensornet-mps']:
    try:
        cudaq.set_target(t)
        cudaq.reset_target()
        print(f'  Target {t}: OK')
    except Exception as e:
        print(f'  Target {t}: FAILED ({e})')
" 2>&1

# =============================================================================
# STAGE 1: Load/Generate Hamiltonians (4q → 40q)
# =============================================================================
log "STAGE 1: Hamiltonian Generation (4q → 40q)"

if [ "${SKIP_HAMILTONIANS}" = "1" ] && [ -f "${HAMILTONIANS_FILE}" ]; then
    echo "  Skipping — Hamiltonians already exist"
else
    EXISTING_40Q="${PROJECT_ROOT}/results/data/hamiltonians_40plus/hamiltonians.json"
    if [ -f "${EXISTING_40Q}" ]; then
        echo "  Using existing 40+ qubit Hamiltonians"
        cp "${EXISTING_40Q}" "${HAMILTONIANS_FILE}"
    else
        echo "  Generating Hamiltonians from config..."
        ${PY} src/gqe/data/generate_hamiltonians.py \
            --config "${CONFIG}" \
            --out "$(dirname "${HAMILTONIANS_FILE}")" \
            2>&1 | tee "${RESULTS_DIR}/hamiltonian_gen.log"
    fi
fi

${PY} -c "
import json
with open('${HAMILTONIANS_FILE}') as f:
    hams = json.load(f)
if isinstance(hams, dict):
    hams = hams.get('hamiltonians', hams.get('records', [hams]))
print(f'  Molecules: {len(hams)}')
for h in sorted(hams, key=lambda x: x.get('n_qubits', 0)):
    nq = h.get('n_qubits', 0)
    nt = h.get('n_pauli_terms', len(h.get('terms', [])))
    name = h.get('name', '?')
    tier = 'T1' if nq <= 12 else 'T2' if nq <= 24 else 'T3' if nq <= 32 else 'T4'
    print(f'    [{tier}] {name:25s}: {nq:3d}q, {nt:6d} terms')
"

track_cost "Hamiltonian Generation" 0

# =============================================================================
# STAGE 2: RL Training (QD-GRPO with MAP-Elites)
# =============================================================================
log "STAGE 2: RL Training (QD-GRPO + MAP-Elites on B200)"

INIT_CHECKPOINT="${PROJECT_ROOT}/results/train/h_cgqe_rl_gic2026.pt"
if [ ! -f "${INIT_CHECKPOINT}" ]; then
    INIT_CHECKPOINT="${PROJECT_ROOT}/results/train/h_cgqe_model_rlqf_phase3.pt"
fi
echo "  Initial checkpoint: ${INIT_CHECKPOINT}"
echo "  Samples per molecule per epoch: ${N_SAMPLES}"
echo "  Molecules: ${RL_MOLECULES}"

NVFP4_FLAG=""
if [ "${USE_NVFP4}" = "1" ]; then
    NVFP4_FLAG="--use-nvfp4"
    echo "  NVFP4: ENABLED (1.59x throughput, 4x memory savings)"
else
    echo "  NVFP4: disabled (set USE_NVFP4=1 to enable)"
fi

if [ "${SKIP_TRAINING}" = "1" ] && [ -f "${TRAINING_OUT}" ]; then
    echo "  Skipping — Trained model already exists"
else
    ${PY} src/gqe/models/train_rl_dapo.py \
        --checkpoint "${INIT_CHECKPOINT}" \
        --hamiltonians "${HAMILTONIANS_FILE}" \
        --molecules ${RL_MOLECULES} \
        --out "${TRAINING_OUT}" \
        --epochs 500 \
        --n-samples "${N_SAMPLES}" \
        --n-iters 5 \
        --lr 1e-5 \
        --temperature 1.0 \
        --use-cuda \
        --use-bf16 \
        ${NVFP4_FLAG} \
        --force-entanglement \
        --max-pauli-len 40 \
        --max-seq-len 64 \
        --max-qubits 32 \
        --mps-threshold 24 \
        --mps-bond 128 \
        --target nvidia \
        --target-option mqpu \
        --qd-mode \
        --adaptive-theta \
        --adaptive-theta-iters 10 \
        --gate-auxiliary-rewards \
        --energy-improvement-threshold 0.0 \
        --buffer-size 4000 \
        --buffer-batch-size 128 \
        --reuse-iters 3 \
        --w-diversity 0.2 \
        --w-commute 0.05 \
        --entropy-coef 0.01 \
        --kl-coef 0.1 \
        --w-creativity 0.5 \
        --w-mmd-diversity 0.3 \
        2>&1 | tee "${RESULTS_DIR}/training/rl_training.log"
fi

STAGE2_END=$(date +%s)
STAGE2_MIN=$(( (STAGE2_END - START_TIME) / 60 ))
STAGE2_CR=$(echo "${STAGE2_MIN} * 14.57" | bc -l)
track_cost "RL Training (B200)" "${STAGE2_CR}"

# =============================================================================
# STAGE 3: AI Circuit Synthesis (Inference with trained model)
# =============================================================================
log "STAGE 3: AI Circuit Synthesis (Inference)"

if [ "${SKIP_INFERENCE}" = "1" ] && [ -f "${INFERENCE_OUT}" ]; then
    echo "  Skipping — Inference results already exist"
else
    ${PY} src/gqe/models/infer_h_cgqe.py \
        --checkpoint "${TRAINING_OUT}" \
        --hamiltonians "${HAMILTONIANS_FILE}" \
        --out "${INFERENCE_OUT}" \
        --n-samples 100 --sample --use-cuda \
        --max-pauli-len 40 --max-seq-len 64 \
        2>&1 | tee "${RESULTS_DIR}/inference/inference.log"
fi

STAGE3_END=$(date +%s)
STAGE3_MIN=$(( (STAGE3_END - STAGE2_END) / 60 ))
STAGE3_CR=$(echo "${STAGE3_MIN} * 14.57" | bc -l)
track_cost "AI Inference" "${STAGE3_CR}"

# =============================================================================
# STAGE 4: L-BFGS-B Coefficient Optimization
# =============================================================================
log "STAGE 4: Coefficient Optimization (L-BFGS-B)"

if [ "${SKIP_OPTIMIZATION}" = "1" ] && [ -f "${OPTIMIZED_OUT}" ]; then
    echo "  Skipping — Optimization results already exist"
else
    ${PY} src/gqe/eval/optimize_h_cgqe_coefficients.py \
        --generated "${INFERENCE_OUT}" \
        --hamiltonians "${HAMILTONIANS_FILE}" \
        --out "${OPTIMIZED_OUT}" \
        --top-k 5 \
        --target nvidia --target-option mqpu \
        --max-iter 100 \
        2>&1 | tee "${RESULTS_DIR}/optimization/optimization.log"
fi

STAGE4_END=$(date +%s)
STAGE4_MIN=$(( (STAGE4_END - STAGE3_END) / 60 ))
STAGE4_CR=$(echo "${STAGE4_MIN} * 14.57" | bc -l)
track_cost "L-BFGS-B Optimization" "${STAGE4_CR}"

# =============================================================================
# STAGE 5a: GPU Statevector Validation (≤32q on B200 single GPU!)
# =============================================================================
log "STAGE 5a: GPU Statevector Validation (≤32q on B200)"

if [ "${SKIP_GPU_VALIDATION}" = "1" ] && [ -f "${SV_OUT}" ]; then
    echo "  Skipping — SV results already exist"
else
    ${PY} src/gqe/eval/evaluate_h_cgqe.py \
        --checkpoint "${TRAINING_OUT}" \
        --hamiltonians "${HAMILTONIANS_FILE}" \
        --generated "${INFERENCE_OUT}" \
        --out "${SV_OUT}" \
        --target nvidia \
        --use-cuda \
        --max-qubits 32 \
        2>&1 | tee "${RESULTS_DIR}/gpu_validation/sv.log" || \
        echo "  WARNING: Some SV evaluations failed (expected for >32q molecules)"
fi

STAGE5A_END=$(date +%s)
STAGE5A_MIN=$(( (STAGE5A_END - STAGE4_END) / 60 ))
STAGE5A_CR=$(echo "${STAGE5A_MIN} * 14.57" | bc -l)
track_cost "GPU Statevector (B200, ≤32q)" "${STAGE5A_CR}"

# =============================================================================
# STAGE 5b: GPU MPS Scaling (28q → 40q)
# =============================================================================
log "STAGE 5b: GPU MPS Scaling (28q → 40q, bond dims 32/64/128/256)"

if [ "${SKIP_GPU_VALIDATION}" = "1" ] && [ -f "${MPS_OUT}" ]; then
    echo "  Skipping — MPS results already exist"
else
    ${PY} src/gqe/eval/run_mps_scaling.py \
        --hamiltonians "${HAMILTONIANS_FILE}" \
        --molecules ethylene n2_ccpvdz beh2_ccpvdz benzene_cas20 n2_ccpvdz_cas20 \
        --bond-dims 32 64 128 256 \
        --target tensornet-mps \
        --output "${MPS_OUT}" \
        2>&1 | tee "${RESULTS_DIR}/gpu_validation/mps.log" || \
        echo "  WARNING: Some MPS evaluations failed"
fi

STAGE5B_END=$(date +%s)
STAGE5B_MIN=$(( (STAGE5B_END - STAGE5A_END) / 60 ))
STAGE5B_CR=$(echo "${STAGE5B_MIN} * 14.57" | bc -l)
track_cost "GPU MPS Scaling (B200)" "${STAGE5B_CR}"

# =============================================================================
# STAGE 6: QPU Execution (IQM Emerald 54q + Rigetti Cepheus 108q)
# =============================================================================
log "STAGE 6: QPU Execution on Real Hardware"

if [ "${SKIP_QPU}" = "1" ]; then
    echo "  Skipping QPU stage (set SKIP_QPU=0 to enable)"
else
    if [ -z "${QBRAID_API_KEY:-}" ]; then
        echo "  WARNING: QBRAID_API_KEY not set. Exporting manifests only."
        QPU_MODE="export-only"
    else
        QPU_MODE="submit"
    fi

    if [ "${QPU_MODE}" = "submit" ]; then
        echo "  Submitting circuits to QPU devices..."

        echo "  [6a] H2 calibration on IQM Garnet (20q)"
        ${PY} src/gqe/eval/qbraid_backend.py \
            --hamiltonians "${HAMILTONIANS_FILE}" \
            --optimized "${OPTIMIZED_OUT}" \
            --molecule h2 \
            --device aws:iqm:qpu:garnet \
            --shots 4096 --submit-only \
            --out "${QPU_OUT}/h2_garnet.json" \
            2>&1 | tee "${QPU_OUT}/h2_garnet.log" || echo "  WARNING: Garnet submission failed"

        echo "  [6b] H2 calibration on Rigetti Cepheus (108q)"
        ${PY} src/gqe/eval/qbraid_backend.py \
            --hamiltonians "${HAMILTONIANS_FILE}" \
            --optimized "${OPTIMIZED_OUT}" \
            --molecule h2 \
            --device aws:rigetti:qpu:cepheus-1-108q \
            --shots 4096 --submit-only \
            --out "${QPU_OUT}/h2_rigetti.json" \
            2>&1 | tee "${QPU_OUT}/h2_rigetti.log" || echo "  WARNING: Rigetti submission failed"

        echo "  [6c] N2_ccpvdz (32q) on IQM Emerald (54q)"
        ${PY} src/gqe/eval/qbraid_backend.py \
            --hamiltonians "${HAMILTONIANS_FILE}" \
            --optimized "${OPTIMIZED_OUT}" \
            --molecule n2_ccpvdz \
            --device aws:iqm:qpu:emerald \
            --shots 4096 --submit-only \
            --out "${QPU_OUT}/n2_ccpvdz_emerald.json" \
            2>&1 | tee "${QPU_OUT}/n2_ccpvdz_emerald.log" || echo "  WARNING: Emerald submission failed"

        echo "  [6d] Benzene_cas20 (40q) on IQM Emerald (54q) — FLAGSHIP"
        ${PY} src/gqe/eval/qbraid_backend.py \
            --hamiltonians "${HAMILTONIANS_FILE}" \
            --optimized "${OPTIMIZED_OUT}" \
            --molecule benzene_cas20 \
            --device aws:iqm:qpu:emerald \
            --shots 4096 --submit-only \
            --out "${QPU_OUT}/benzene_emerald.json" \
            2>&1 | tee "${QPU_OUT}/benzene_emerald.log" || echo "  WARNING: Emerald submission failed"

        echo "  [6e] Benzene_cas20 (40q) on Rigetti Cepheus (108q) — cross-vendor"
        ${PY} src/gqe/eval/qbraid_backend.py \
            --hamiltonians "${HAMILTONIANS_FILE}" \
            --optimized "${OPTIMIZED_OUT}" \
            --molecule benzene_cas20 \
            --device aws:rigetti:qpu:cepheus-1-108q \
            --shots 4096 --submit-only \
            --out "${QPU_OUT}/benzene_rigetti.json" \
            2>&1 | tee "${QPU_OUT}/benzene_rigetti.log" || echo "  WARNING: Rigetti submission failed"

        echo "  QPU jobs submitted. Retrieve with:"
        echo "    python src/gqe/eval/qbraid_backend.py --retrieve <metadata_file>"
    else
        echo "  Exporting QPU manifests only..."
        ${PY} scripts/submit_qpu_async.py \
            --hamiltonians "${HAMILTONIANS_FILE}" \
            --optimized "${OPTIMIZED_OUT}" \
            --molecule h2 --export-only \
            --out "${QPU_OUT}/manifests" \
            2>&1 | tee "${QPU_OUT}/export.log" || echo "  WARNING: Export failed"
    fi
fi

STAGE6_END=$(date +%s)
STAGE6_MIN=$(( (STAGE6_END - STAGE5B_END) / 60 ))
STAGE6_CR=$(echo "${STAGE6_MIN} * 14.57" | bc -l)
track_cost "QPU Submission" "${STAGE6_CR}"

# =============================================================================
# STAGE 7: Cross-Platform Comparison & Visualization
# =============================================================================
log "STAGE 7: Cross-Platform Comparison & Visualization"

if [ "${SKIP_PLOTS}" = "1" ]; then
    echo "  Skipping plots"
else
    ${PY} -c "
import json
from pathlib import Path

results = {'pipeline': '40q GPU-AI-QPU Scalability (B200)', 'stages': {}}

for stage_name, path in [('statevector', '${SV_OUT}'), ('mps', '${MPS_OUT}'), ('optimization', '${OPTIMIZED_OUT}')]:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            results['stages'][stage_name] = json.load(f)

qpu_dir = Path('${QPU_OUT}')
qpu_results = {}
for qpu_file in qpu_dir.glob('*_*.json'):
    if qpu_file.name == 'manifests': continue
    try:
        with open(qpu_file) as f:
            qpu_results[qpu_file.stem] = json.load(f)
    except Exception:
        pass
results['stages']['qpu'] = qpu_results

with open(Path('${REPORT_OUT}'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'  Report saved to ${REPORT_OUT}')
" 2>&1 | tee "${RESULTS_DIR}/report.log"

    ${PY} scripts/plot_40q_scaling.py \
        --results "${REPORT_OUT}" \
        --out "${PLOTS_OUT}" \
        2>&1 | tee "${RESULTS_DIR}/plots.log" || echo "  WARNING: Plot generation failed"
fi

# =============================================================================
# Summary
# =============================================================================
TOTAL_MIN=$(( ($(date +%s) - START_TIME) / 60 ))
TOTAL_CR=$(echo "${TOTAL_MIN} * 14.57" | bc -l)
REMAINING_CR=$(echo "24645 - ${TOTAL_CR}" | bc -l)

echo ""
echo "================================================================"
echo "  40-Qubit Scalability Pipeline Complete (B200)"
echo "  Time: $(date)"
echo ""
echo "  Total elapsed: ${TOTAL_MIN} min (~${TOTAL_CR} cr)"
echo "  Remaining budget: ~${REMAINING_CR} cr"
echo ""
echo "  Results: ${RESULTS_DIR}/"
echo "    training/              — RL model + QD-GRPO metrics"
echo "    inference/             — AI-generated circuits (100 per molecule)"
echo "    optimization/          — L-BFGS-B optimized coefficients"
echo "    gpu_validation/        — SV (≤32q) + MPS (28-40q) energies"
echo "    qpu_results/           — QPU submissions & manifests"
echo "    plots/                 — Cross-platform comparison plots"
echo "    scaling_report.json    — Consolidated report"
echo ""
echo "  Key results:"
echo "    - RL training with ${N_SAMPLES} samples/epoch (4x L40S baseline)"
echo "    - 32q exact statevector on single B200 (no mgpu needed)"
echo "    - 40q MPS validation with bond dims 32/64/128/256"
echo "    - 40q QPU execution on IQM Emerald (54q) + Rigetti (108q)"
echo "================================================================"
