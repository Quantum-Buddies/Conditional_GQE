#!/usr/bin/env bash
# =============================================================================
# Portable Evaluation Launcher for qBraid GPU Instances
# =============================================================================
# Three-stage evaluation pipeline:
#   1. infer     : Generate circuits from RL checkpoint (checkpoint → JSON)
#   2. eval      : Evaluate energies via CUDA-Q (generated JSON → eval JSON)
#   3. optimize  : L-BFGS-B coefficient optimization (generated JSON → optimized JSON)
#   4. report    : Generate Phase 3 PDF report
#
# Usage:
#   bash scripts/evaluate_rl.sh [infer|eval|optimize|report|all]
#   Default: all (infer → eval → optimize → report)
#
# All paths are relative to the repo root (auto-detected).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="${1:-all}"

# --- Source GPU-specific environment ---
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env_gpu.sh"

# --- Paths (all relative to ROOT) ---
RL_CKPT="${RL_CKPT:-$ROOT/results/train/h_cgqe_model_qbraid_rl.pt}"
HAMILTONIANS="$ROOT/results/data/hamiltonians_gic2026/hamiltonians.json"
BASELINE="$ROOT/results/baselines/cudaq_gqe_baseline.json"
RESULTS_DIR="$ROOT/results/phase3_final"
GENERATED_JSON="$RESULTS_DIR/h_cgqe_generated.json"
EVAL_JSON="$RESULTS_DIR/h_cgqe_evaluation.json"
OPTIMIZED_JSON="$RESULTS_DIR/h_cgqe_optimized.json"
REPORT_PDF="$RESULTS_DIR/gic2026_phase3_report.pdf"

mkdir -p "$RESULTS_DIR"

# --- Auto-detect GPU-specific limits ---
MAX_QUBITS=28
case "$GPU_CC" in
    9.0*)  [ "$GPU_VRAM_GB" -ge 140 ] 2>/dev/null && MAX_QUBITS=30 || MAX_QUBITS=26 ;;
    10.0*) MAX_QUBITS=32 ;;
    8.0*)  MAX_QUBITS=26 ;;
    *)     MAX_QUBITS=24 ;;
esac

# --- Auto-generate molecule list ---
MOLECULES="$(python3 -c "
import json
d = json.load(open('$HAMILTONIANS'))
recs = sorted(
    [r for r in d['records'] if r.get('n_qubits', 99) <= $MAX_QUBITS],
    key=lambda r: (r.get('n_qubits', 99), r.get('name', '')),
)
print(' '.join(r['name'] for r in recs))
" 2>/dev/null || echo 'h2 lih beh2 n2')"

# --- Credit rate ---
get_credit_rate() {
    case "$GPU_NAME" in
        *H200*)  echo 9.15 ;;
        *H100*)  echo 8.95 ;;
        *B200*)  echo 14.57 ;;
        *A100*)  echo 4.15 ;;
        *)       echo 0.0 ;;
    esac
}

# =====================================================================
# Stage 1: Inference — generate circuits from RL checkpoint
# =====================================================================
run_infer() {
    echo "============================================================"
    echo "  [1/4] Inference — Generating circuits from RL checkpoint"
    echo "  Checkpoint : $RL_CKPT"
    echo "  Molecules  : $(echo $MOLECULES | wc -w)"
    echo "  Max qubits : $MAX_QUBITS"
    echo "============================================================"

    if [ ! -f "$RL_CKPT" ]; then
        echo "  ERROR: RL checkpoint not found: $RL_CKPT"
        echo "  Run 'bash scripts/train_rl.sh full' first."
        exit 1
    fi

    python3 -u "$ROOT/src/gqe/models/infer_h_cgqe.py" \
        --checkpoint "$RL_CKPT" \
        --hamiltonians "$HAMILTONIANS" \
        --molecules $MOLECULES \
        --out "$GENERATED_JSON" \
        --n-samples 100 \
        --max-terms 128 \
        --max-pauli-len 24 \
        --max-seq-len 64 \
        --temperature 1.0 \
        --force-entanglement \
        --max-repeat 4 \
        --freq-penalty 1.0 \
        --use-cuda \
        2>&1 | tee "$RESULTS_DIR/infer.log"

    echo "  Generated circuits → $GENERATED_JSON"
}

# =====================================================================
# Stage 2: Evaluation — compute energies via CUDA-Q
# =====================================================================
run_eval() {
    echo "============================================================"
    echo "  [2/4] Evaluation — CUDA-Q energy computation"
    echo "  Generated  : $GENERATED_JSON"
    echo "  Baseline   : $BASELINE"
    echo "============================================================"

    if [ ! -f "$GENERATED_JSON" ]; then
        echo "  Generated JSON not found — running inference first..."
        run_infer
    fi

    python3 -u "$ROOT/src/gqe/eval/evaluate_h_cgqe.py" \
        --generated "$GENERATED_JSON" \
        --baseline "$BASELINE" \
        --hamiltonians "$HAMILTONIANS" \
        --out "$EVAL_JSON" \
        --target nvidia \
        --max-qubits "$MAX_QUBITS" \
        2>&1 | tee "$RESULTS_DIR/eval.log"

    echo "  Evaluation results → $EVAL_JSON"
}

# =====================================================================
# Stage 3: Optimization — L-BFGS-B coefficient optimization
# =====================================================================
run_optimize() {
    echo "============================================================"
    echo "  [3/4] Optimization — L-BFGS-B coefficient tuning"
    echo "  Generated  : $GENERATED_JSON"
    echo "============================================================"

    if [ ! -f "$GENERATED_JSON" ]; then
        echo "  Generated JSON not found — running inference first..."
        run_infer
    fi

    python3 -u "$ROOT/src/gqe/eval/optimize_h_cgqe_coefficients.py" \
        --generated "$GENERATED_JSON" \
        --hamiltonians "$HAMILTONIANS" \
        --out "$OPTIMIZED_JSON" \
        --target nvidia \
        --max-iter 100 \
        --top-k 10 \
        --max-qubits "$MAX_QUBITS" \
        2>&1 | tee "$RESULTS_DIR/optimize.log"

    echo "  Optimized energies → $OPTIMIZED_JSON"
}

# =====================================================================
# Stage 4: Report — generate Phase 3 PDF
# =====================================================================
run_report() {
    echo "============================================================"
    echo "  [4/4] Report — Generating Phase 3 PDF"
    echo "============================================================"

    python3 -u "$ROOT/scripts/generate_phase3_pdf.py" \
        --benchmark-dir "$ROOT/results" \
        --hcgqe-dir "$RESULTS_DIR" \
        --fmo-dir "$ROOT/results" \
        --mps-dir "$ROOT/results" \
        --qpu-dir "$ROOT/results" \
        --out "$REPORT_PDF" \
        2>&1 | tee "$RESULTS_DIR/report.log" || \
        echo "  WARNING: PDF generation failed (non-critical). Results are in JSON."

    if [ -f "$REPORT_PDF" ]; then
        echo "  Report → $REPORT_PDF"
    fi
}

# =====================================================================
# Mode: all — full evaluation pipeline
# =====================================================================
run_all() {
    local START=$(date +%s)

    run_infer
    echo ""
    run_eval
    echo ""
    run_optimize
    echo ""
    run_report

    local ELAPSED=$(( ($(date +%s) - START) / 60 ))
    local RATE; RATE=$(get_credit_rate)
    local COST; COST=$(python3 -c "print(f'{$ELAPSED * $RATE:.1f}')" 2>/dev/null || echo "N/A")
    echo ""
    echo "============================================================"
    echo "  EVALUATION COMPLETE"
    echo "  Total time  : ${ELAPSED} min"
    echo "  Est. cost   : ${COST} credits"
    echo ""
    echo "  Outputs:"
    echo "    Generated  : $GENERATED_JSON"
    echo "    Evaluation : $EVAL_JSON"
    echo "    Optimized  : $OPTIMIZED_JSON"
    echo "    Report     : $REPORT_PDF"
    echo "============================================================"
}

# =====================================================================
# Dispatch
# =====================================================================
cd "$ROOT"

case "$MODE" in
    infer)    run_infer ;;
    eval)     run_eval ;;
    optimize) run_optimize ;;
    report)   run_report ;;
    all)      run_all ;;
    *)
        echo "Usage: bash scripts/evaluate_rl.sh [infer|eval|optimize|report|all]"
        echo "  infer    : Generate circuits from RL checkpoint"
        echo "  eval     : Evaluate energies via CUDA-Q"
        echo "  optimize : L-BFGS-B coefficient optimization"
        echo "  report   : Generate Phase 3 PDF report"
        echo "  all      : Full pipeline (infer → eval → optimize → report)"
        exit 1
        ;;
esac
