#!/usr/bin/env bash
# Iterative RAFT (STaR-style) Post-Training Loop for H-cGQE
#
# Implements the Self-Taught Reasoner (STaR) / DeepSeek-R1 Stage 2 approach:
#   Round 0: RL checkpoint → sample N circuits → optimize → top-k → SFT → ckpt_1
#   Round 1: ckpt_1 → sample N circuits → optimize → top-k → SFT → ckpt_2
#   ...
#   Final:   model_soup(ckpt_1, ckpt_2, ..., ckpt_R) → h_cgqe_star_soup.pt
#
# Each round produces a better model that generates higher-quality candidates
# for the next round, creating a self-improvement loop.
#
# Usage:
#   bash scripts/run_iterative_raft.sh --rounds 3 --checkpoint results/train/h_cgqe_rl.pt
#   bash scripts/run_iterative_raft.sh --rounds 5 --checkpoint results/train/h_cgqe_rl.pt --adaptive-n

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"

PYTHON="/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"
if [ ! -f "${PYTHON}" ]; then
    PYTHON="${CONDA_PREFIX:-/usr}/bin/python"
fi

# Defaults
ROUNDS=3
CHECKPOINT=""
HAMILTONIANS=""
OUT_DIR="${RESULTS_DIR}/train"
N_SAMPLES=50
TOP_K=5
EPOCHS=50
LR=5e-5
TEMPERATURE=1.0
TOP_P=0.9
MAX_SEQ_LEN=64
MAX_QUBITS=48
ADAPTIVE_N=false
USE_CUDA=false
SOUP=true

show_help() {
    echo "Iterative RAFT (STaR) Post-Training Loop"
    echo "=========================================="
    echo "Usage: bash scripts/run_iterative_raft.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --rounds N          Number of RAFT rounds (default: 3)"
    echo "  --checkpoint PATH   Starting checkpoint (required)"
    echo "  --hamiltonians PATH Hamiltonians JSON (auto-detected if omitted)"
    echo "  --out-dir PATH      Output directory (default: results/train)"
    echo "  --n-samples N       Candidates per molecule per round (default: 50)"
    echo "  --top-k N           Top-k sequences to keep per molecule (default: 5)"
    echo "  --epochs N          SFT epochs per round (default: 50)"
    echo "  --lr LR             Learning rate (default: 5e-5)"
    echo "  --temperature T     Sampling temperature (default: 1.0)"
    echo "  --top-p P           Nucleus sampling cutoff (default: 0.9)"
    echo "  --max-seq-len N     Max sequence length (default: 64)"
    echo "  --max-qubits N      Max qubits for molecules (default: 48)"
    echo "  --adaptive-n        Scale n_samples by qubit count (test-time compute scaling)"
    echo "  --use-cuda          Use GPU for SFT training"
    echo "  --no-soup           Skip model soup averaging at the end"
    echo "  --help              Show this help"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds) ROUNDS="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --hamiltonians) HAMILTONIANS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
        --max-qubits) MAX_QUBITS="$2"; shift 2 ;;
        --adaptive-n) ADAPTIVE_N=true; shift ;;
        --use-cuda) USE_CUDA=true; shift ;;
        --no-soup) SOUP=false; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: --checkpoint is required"
    show_help
    exit 1
fi

# Auto-detect Hamiltonians
if [ -z "${HAMILTONIANS}" ]; then
    for path in \
        "${RESULTS_DIR}/data/hamiltonians_scaling.json/hamiltonians.json" \
        "${RESULTS_DIR}/data/hamiltonians_gic2026/hamiltonians.json" \
        "${RESULTS_DIR}/data/hamiltonians.json"; do
        if [ -f "${path}" ]; then
            HAMILTONIANS="${path}"
            break
        fi
    done
fi
if [ ! -f "${HAMILTONIANS}" ]; then
    echo "ERROR: No Hamiltonians file found. Specify with --hamiltonians"
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Iterative RAFT (STaR) Post-Training Loop"
echo "============================================================"
echo "  Starting checkpoint: ${CHECKPOINT}"
echo "  Hamiltonians:        ${HAMILTONIANS}"
echo "  Rounds:              ${ROUNDS}"
echo "  Samples/molecule:    ${N_SAMPLES} $(if ${ADAPTIVE_N}; then echo "(adaptive)"; fi)"
echo "  Top-k:               ${TOP_K}"
echo "  SFT epochs/round:    ${EPOCHS}"
echo "  Learning rate:       ${LR}"
echo "  Temperature:         ${TEMPERATURE}"
echo "  Model soup:          ${SOUP}"
echo "============================================================"

CURRENT_CKPT="${CHECKPOINT}"

for round in $(seq 1 "${ROUNDS}"); do
    echo ""
    echo "============================================================"
    echo "RAFT ROUND ${round}/${ROUNDS}"
    echo "============================================================"

    OUT_CKPT="${OUT_DIR}/h_cgqe_raft_round_${round}.pt"

    EXTRA_ARGS=""
    if ${ADAPTIVE_N}; then
        EXTRA_ARGS="${EXTRA_ARGS} --adaptive-n-samples"
    fi
    if ${USE_CUDA}; then
        EXTRA_ARGS="${EXTRA_ARGS} --use-cuda"
    fi

    "${PYTHON}" "${PROJECT_ROOT}/scripts/train_post_alignment.py" \
        --checkpoint "${CURRENT_CKPT}" \
        --hamiltonians "${HAMILTONIANS}" \
        --out "${OUT_CKPT}" \
        --epochs "${EPOCHS}" \
        --batch-size 4 \
        --lr "${LR}" \
        --n-samples "${N_SAMPLES}" \
        --top-k "${TOP_K}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --max-qubits "${MAX_QUBITS}" \
        ${EXTRA_ARGS}

    if [ ! -f "${OUT_CKPT}" ]; then
        echo "ERROR: Round ${round} failed - no checkpoint produced"
        exit 1
    fi

    echo "Round ${round} complete: ${OUT_CKPT}"
    CURRENT_CKPT="${OUT_CKPT}"

    # Decrease temperature slightly each round for exploitation
    TEMPERATURE=$(echo "${TEMPERATURE} * 0.9" | bc -l)
    echo "  Next round temperature: ${TEMPERATURE}"
done

# --- Model Soup: average all round checkpoints ---
if ${SOUP}; then
    echo ""
    echo "============================================================"
    echo "MODEL SOUP: Averaging ${ROUNDS} checkpoints"
    echo "============================================================"

    SOUP_OUT="${OUT_DIR}/h_cgqe_star_soup.pt"

    "${PYTHON}" "${PROJECT_ROOT}/src/gqe/models/model_soup.py" \
        --checkpoints $(for r in $(seq 1 "${ROUNDS}"); do echo "${OUT_DIR}/h_cgqe_raft_round_${r}.pt"; done) \
        --out "${SOUP_OUT}"

    echo "Model soup saved to: ${SOUP_OUT}"
    echo ""
    echo "============================================================"
    echo "STaR loop complete!"
    echo "  Best round checkpoint: ${CURRENT_CKPT}"
    echo "  Model soup:            ${SOUP_OUT}"
    echo "============================================================"
else
    echo ""
    echo "STaR loop complete! Final checkpoint: ${CURRENT_CKPT}"
fi
