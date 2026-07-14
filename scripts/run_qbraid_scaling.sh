#!/usr/bin/env bash
# qBraid Cloud GPU Scaling Pipeline — B200 / H200 / B200x4
#
# This script orchestrates the full H-cGQE scaling pipeline on qBraid's
# high-VRAM GPU instances, enabling qubit counts (26-40) that were impossible
# on AIRE L40S (capped at 24 due to PCIe IPC segfault).
#
# Pipeline stages:
#   1. Generate Hamiltonians for B200 scaling config (26-40 qubit molecules)
#   2. RL training with off-policy GRPO (reuse_iters=3, 3x cheaper simulation)
#   3. Iterative RAFT post-training (3 rounds STaR loop + model soup)
#   4. Inference + coefficient optimization on large molecules
#   5. Validate on free qBraid simulator (zero credits)
#
# Usage:
#   bash scripts/run_qbraid_scaling.sh --stage all --instance gpu-h200
#   bash scripts/run_qbraid_scaling.sh --stage train --instance gpu-b200
#   bash scripts/run_qbraid_scaling.sh --stage raft --instance gpu-h200
#   bash scripts/run_qbraid_scaling.sh --stage validate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"

PYTHON="${CONDA_PREFIX:-/usr}/bin/python"
if [ -f "/scratch/kcwp264/.conda_envs/cudaq-env/bin/python" ]; then
    PYTHON="/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"
fi

# Defaults
STAGE="all"
INSTANCE="gpu-h200"
CONFIG="${PROJECT_ROOT}/configs/experiment_scaling_b200.yaml"
ROUNDS=3
REUSE_ITERS=3
MAX_QUBITS=40
MPS_THRESHOLD=30
MPS_BOND=128

# qBraid instance pricing (credits per hour)
declare -A INSTANCE_PRICING
INSTANCE_PRICING[gpu-l40s]=228
INSTANCE_PRICING[gpu-gh200]=287
INSTANCE_PRICING[gpu-h200]=549
INSTANCE_PRICING[gpu-b200]=874
INSTANCE_PRICING[gpu-b200-4x]=3395

# qBraid instance VRAM
declare -A INSTANCE_VRAM
INSTANCE_VRAM[gpu-l40s]=48
INSTANCE_VRAM[gpu-gh200]=96
INSTANCE_VRAM[gpu-h200]=141
INSTANCE_VRAM[gpu-b200]=192
INSTANCE_VRAM[gpu-b200-4x]=768

show_help() {
    echo "qBraid Cloud GPU Scaling Pipeline"
    echo "=================================="
    echo "Usage: bash scripts/run_qbraid_scaling.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --stage STAGE       Pipeline stage: all, hamiltonian, train, raft, validate (default: all)"
    echo "  --instance NAME     qBraid instance: gpu-h200, gpu-b200, gpu-b200-4x (default: gpu-h200)"
    echo "  --config PATH       Config file (default: configs/experiment_scaling_b200.yaml)"
    echo "  --rounds N          RAFT rounds (default: 3)"
    echo "  --reuse-iters N     Off-policy GRPO reuse iterations (default: 3)"
    echo "  --max-qubits N      Max qubits (default: 40)"
    echo "  --help              Show this help"
    echo ""
    echo "Instance pricing (credits/hour):"
    echo "  gpu-l40s:    228 cr/hr  (48GB VRAM,  baseline)"
    echo "  gpu-gh200:   287 cr/hr  (96GB VRAM,  Grace Hopper)"
    echo "  gpu-h200:    549 cr/hr  (141GB VRAM, HBM3e)"
    echo "  gpu-b200:    874 cr/hr  (192GB VRAM, Blackwell)"
    echo "  gpu-b200-4x: 3395 cr/hr (768GB VRAM, 4x B200)"
    echo ""
    echo "Estimated credit costs:"
    echo "  H200 full pipeline:  ~8-12 hrs = ~4,400-6,600 cr"
    echo "  B200 full pipeline:  ~6-8 hrs  = ~5,240-6,990 cr"
    echo "  B200x4 (36q eval):   ~2-3 hrs  = ~6,790-10,185 cr"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --instance) INSTANCE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --reuse-iters) REUSE_ITERS="$2"; shift 2 ;;
        --max-qubits) MAX_QUBITS="$2"; shift 2 ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

VRAM=${INSTANCE_VRAM[$INSTANCE]:-unknown}
PRICE=${INSTANCE_PRICING[$INSTANCE]:-unknown}

echo "============================================================"
echo "qBraid Cloud GPU Scaling Pipeline"
echo "============================================================"
echo "  Instance:     ${INSTANCE} (${VRAM}GB VRAM, ${PRICE} cr/hr)"
echo "  Config:       ${CONFIG}"
echo "  Stage:        ${STAGE}"
echo "  Max qubits:   ${MAX_QUBITS}"
echo "  Reuse iters:  ${REUSE_ITERS} (off-policy GRPO)"
echo "  RAFT rounds:  ${ROUNDS}"
echo "============================================================"

# --- Stage 1: Generate Hamiltonians ---
if [[ "${STAGE}" == "all" || "${STAGE}" == "hamiltonian" ]]; then
    echo ""
    echo "=== [Stage 1] Generating Hamiltonians for B200 scaling config ==="
    HAM_OUT="${RESULTS_DIR}/data/hamiltonians_b200"
    mkdir -p "${HAM_OUT}"

    "${PYTHON}" "${PROJECT_ROOT}/src/gqe/data/generate_hamiltonians.py" \
        --config "${CONFIG}" \
        --out "${HAM_OUT}/hamiltonians.json"

    echo "Hamiltonians saved to: ${HAM_OUT}/hamiltonians.json"
    "${PYTHON}" -c "
import json
with open('${HAM_OUT}/hamiltonians.json') as f:
    data = json.load(f)
print(f'Total molecules: {len(data)}')
for r in data:
    nq = r.get('n_qubits', '?')
    print(f'  {r[\"name\"]:30s}: {nq} qubits')
"
fi

HAM_FILE="${RESULTS_DIR}/data/hamiltonians_b200/hamiltonians.json"
if [ ! -f "${HAM_FILE}" ]; then
    # Fallback to existing Hamiltonians
    for path in \
        "${RESULTS_DIR}/data/hamiltonians_scaling.json/hamiltonians.json" \
        "${RESULTS_DIR}/data/hamiltonians_gic2026/hamiltonians.json" \
        "${RESULTS_DIR}/data/hamiltonians.json"; do
        if [ -f "${path}" ]; then
            HAM_FILE="${path}"
            break
        fi
    done
fi
echo "Using Hamiltonians: ${HAM_FILE}"

# --- Stage 2: RL Training with off-policy GRPO ---
if [[ "${STAGE}" == "all" || "${STAGE}" == "train" ]]; then
    echo ""
    echo "=== [Stage 2] RL Training with off-policy GRPO (reuse_iters=${REUSE_ITERS}) ==="

    RL_CKPT="${RESULTS_DIR}/train/h_cgqe_rl_b200.pt"

    "${PYTHON}" "${PROJECT_ROOT}/src/gqe/models/train_rl_dapo.py" \
        --hamiltonians "${HAM_FILE}" \
        --config "${CONFIG}" \
        --out "${RL_CKPT}" \
        --epochs 500 \
        --n-samples 64 \
        --batch-size 8 \
        --lr 1e-5 \
        --reuse-iters "${REUSE_ITERS}" \
        --target nvidia \
        --target-option mqpu \
        --max-qubits "${MAX_QUBITS}" \
        --mps-threshold "${MPS_THRESHOLD}" \
        --mps-bond "${MPS_BOND}" \
        --use-cuda \
        --use-bf16 \
        --force-entanglement \
        --curriculum \
        --curriculum-warmup 50 \
        --curriculum-steps 4 \
        --dynamic-sampling \
        --repo-beta 0.05

    echo "RL training complete: ${RL_CKPT}"
fi

# --- Stage 3: Iterative RAFT Post-Training ---
if [[ "${STAGE}" == "all" || "${STAGE}" == "raft" ]]; then
    echo ""
    echo "=== [Stage 3] Iterative RAFT (STaR loop, ${ROUNDS} rounds) ==="

    RL_CKPT="${RESULTS_DIR}/train/h_cgqe_rl_b200.pt"
    if [ ! -f "${RL_CKPT}" ]; then
        # Fallback to any available RL checkpoint
        for path in \
            "${RESULTS_DIR}/train/h_cgqe_rl_from_scratch.pt" \
            "${RESULTS_DIR}/train/h_cgqe_uccsd_rl.pt" \
            "${RESULTS_DIR}/train/h_cgqe_rl.pt"; do
            if [ -f "${path}" ]; then
                RL_CKPT="${path}"
                break
            fi
        done
    fi

    bash "${PROJECT_ROOT}/scripts/run_iterative_raft.sh" \
        --rounds "${ROUNDS}" \
        --checkpoint "${RL_CKPT}" \
        --hamiltonians "${HAM_FILE}" \
        --n-samples 100 \
        --top-k 10 \
        --epochs 50 \
        --adaptive-n \
        --use-cuda \
        --max-qubits "${MAX_QUBITS}"
fi

# --- Stage 4: Inference + Coefficient Optimization ---
if [[ "${STAGE}" == "all" || "${STAGE}" == "inference" ]]; then
    echo ""
    echo "=== [Stage 4] Inference + L-BFGS-B Coefficient Optimization ==="

    SOUP_CKPT="${RESULTS_DIR}/train/h_cgqe_star_soup.pt"
    if [ ! -f "${SOUP_CKPT}" ]; then
        # Use last RAFT round
        SOUP_CKPT="${RESULTS_DIR}/train/h_cgqe_raft_round_${ROUNDS}.pt"
    fi

    INF_OUT="${RESULTS_DIR}/inference/h_cgqe_b200_inference.json"
    EVAL_OUT="${RESULTS_DIR}/eval/h_cgqe_b200_optimized.json"
    mkdir -p "${RESULTS_DIR}/inference" "${RESULTS_DIR}/eval"

    "${PYTHON}" "${PROJECT_ROOT}/src/gqe/eval/evaluate_h_cgqe.py" \
        --checkpoint "${SOUP_CKPT}" \
        --hamiltonians "${HAM_FILE}" \
        --out "${INF_OUT}" \
        --max-qubits "${MAX_QUBITS}" \
        --target nvidia \
        --target-option mqpu \
        --mps-threshold "${MPS_THRESHOLD}" \
        --mps-bond "${MPS_BOND}" \
        --n-samples 200

    "${PYTHON}" "${PROJECT_ROOT}/src/gqe/eval/optimize_h_cgqe_coefficients.py" \
        --inference "${INF_OUT}" \
        --hamiltonians "${HAM_FILE}" \
        --out "${EVAL_OUT}" \
        --max-qubits "${MAX_QUBITS}" \
        --target nvidia \
        --target-option mqpu

    echo "Optimized results: ${EVAL_OUT}"
fi

# --- Stage 5: Validate on free qBraid simulator ---
if [[ "${STAGE}" == "all" || "${STAGE}" == "validate" ]]; then
    echo ""
    echo "=== [Stage 5] Validation on free qBraid simulator (zero credits) ==="

    EVAL_OUT="${RESULTS_DIR}/eval/h_cgqe_b200_optimized.json"
    if [ ! -f "${EVAL_OUT}" ]; then
        EVAL_OUT="${RESULTS_DIR}/eval/h_cgqe_uccsd_optimized.json"
    fi

    VAL_OUT="${RESULTS_DIR}/eval/qbraid_b200_validation_report.json"

    "${PYTHON}" "${PROJECT_ROOT}/scripts/validate_on_qbraid.py" \
        --hamiltonians "${HAM_FILE}" \
        --optimized "${EVAL_OUT}" \
        --out "${VAL_OUT}"

    echo "Validation report: ${VAL_OUT}"
fi

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "  Instance: ${INSTANCE} (${PRICE} cr/hr)"
echo "  Credits used: see qBraid dashboard"
echo "============================================================"
