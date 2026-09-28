#!/usr/bin/env bash
# =============================================================================
# run_all_reproducible.sh — End-to-end reproducibility runner for GIC 2026
# =============================================================================
# Runs the full pipeline from Hamiltonian generation through report figures.
# Designed for qBraid or any CUDA-Q + PyTorch environment.
#
# Usage:
#   bash scripts/run_all_reproducible.sh [stage]
#
# Stages:
#   all       — Run everything (default)
#   setup     — Check environment + download checkpoints
#   chem      — Generate Hamiltonians
#   train     — SFT + RL training
#   eval      — Evaluation + L-BFGS-B optimization
#   baselines — Run classical + VQE baselines
#   qpu       — QPU submission + SQD (requires qBraid credits)
#   figures   — Generate all report figures + docx
#   report    — Generate final .docx report
#
# Credit budget: ~1,925 qBraid credits remaining (QPU stage only)
# GPU stages are free on qBraid or local hardware.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

STAGE="${1:-all}"

log() { echo -e "\n\033[1;34m[$(date +%H:%M:%S)]\033[0m $1"; }

# ---------------------------------------------------------------------------
# Stage: setup
# ---------------------------------------------------------------------------
run_setup() {
    log "Checking environment..."
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
    python -c "import cudaq; print(f'CUDA-Q {cudaq.__version__}')" 2>/dev/null || echo "WARNING: cudaq not found (install with: pip install cudaq)"
    python -c "import openfermion; print(f'OpenFermion {openfermion.__version__}')"
    python -c "import pyscf; print(f'PySCF {pyscf.__version__}')"

    log "Installing dependencies..."
    pip install -r requirements.txt

    log "Downloading model checkpoints from Hugging Face..."
    python scripts/download_models.py --only essential
}

# ---------------------------------------------------------------------------
# Stage: chem — Generate molecular Hamiltonians
# ---------------------------------------------------------------------------
run_chem() {
    log "Generating GIC 2026 molecular Hamiltonians..."
    python src/gqe/data/generate_hamiltonians.py \
        --config configs/gic2026_molecules.yaml \
        --out results/data/hamiltonians_gic2026/

    log "Verifying Hamiltonian file..."
    python -c "
import json
with open('results/data/hamiltonians_gic2026/hamiltonians.json') as f:
    data = json.load(f)
print(f'Molecules: {len(data)}')
for m in data:
    print(f'  {m[\"name\"]:20s}  {m[\"n_qubits\"]:2d}q  {len(m[\"pauli_terms\"])} terms')
"
}

# ---------------------------------------------------------------------------
# Stage: train — SFT + RL
# ---------------------------------------------------------------------------
run_train() {
    log "Stage 1: Supervised fine-tuning..."
    python src/gqe/models/train_supervised.py \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --config configs/gic2026_molecules.yaml \
        --out results/train/h_cgqe_model_sft.pt \
        --use-cuda

    log "Stage 2: DAPO reinforcement learning..."
    python src/gqe/models/train_rl_dapo.py \
        --checkpoint results/train/h_cgqe_model_sft.pt \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --out results/train/h_cgqe_model_rl.pt \
        --n-epochs 50 \
        --target nvidia \
        --max-qubits 24

    log "Training complete. Checkpoints in results/train/"
}

# ---------------------------------------------------------------------------
# Stage: eval — Generate + optimize circuits
# ---------------------------------------------------------------------------
run_eval() {
    log "Evaluating H-cGQE circuits..."
    python src/gqe/eval/evaluate_h_cgqe.py \
        --checkpoint results/train/h_cgqe_model_rl.pt \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --out results/eval/h_cgqe_evaluation.json

    log "L-BFGS-B coefficient optimization..."
    python src/gqe/eval/optimize_h_cgqe_coefficients.py \
        --evaluation results/eval/h_cgqe_evaluation.json \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --target nvidia-mqpu \
        --out results/eval/h_cgqe_optimized.json

    log "Consolidating benchmark results..."
    python scripts/consolidate_results.py \
        --eval results/eval/h_cgqe_evaluation.json \
        --optimized results/eval/h_cgqe_optimized.json \
        --out results/phase3_final/consolidated_results_gic2026.json
}

# ---------------------------------------------------------------------------
# Stage: baselines — Classical + VQE comparisons
# ---------------------------------------------------------------------------
run_baselines() {
    log "Running CUDA-Q GQE baseline..."
    python src/gqe/baselines/run_cudaq_gqe.py \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --out results/baselines/cudaq_gqe.json

    log "Running HEA-VQE baseline..."
    python src/gqe/baselines/run_cudaq_vqe.py \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --mode cudaq_hwe_vqe \
        --out results/baselines/cudaq_vqe.json

    log "Running ADAPT-VQE baseline (small molecules only)..."
    python src/gqe/baselines/run_cudaq_vqe.py \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --mode adapt_vqe \
        --max-qubits 4 \
        --out results/baselines/adapt_vqe_h2.json

    log "Running exact diagonalization (FCI reference)..."
    python -c "
from src.gqe.baselines.run_exact_diag import run_exact_diagonalization
run_exact_diagonalization(
    'results/data/hamiltonians_gic2026/hamiltonians.json',
    'results/baselines/exact_diagonalization.json'
)
" 2>/dev/null || echo "  (exact_diag module not available, using existing FCI refs)"

    log "Building classical baseline comparison..."
    python scripts/build_gic_benchmark.py \
        --consolidated results/phase3_final/consolidated_results_gic2026.json \
        --vqe results/baselines/cudaq_vqe.json \
        --adapt results/baselines/adapt_vqe_h2.json \
        --gqe results/baselines/cudaq_gqe.json \
        --out results/phase3_final/classical_baseline_comparison.json
}

# ---------------------------------------------------------------------------
# Stage: qpu — QPU submission + SQD (uses qBraid credits)
# ---------------------------------------------------------------------------
run_qpu() {
    log "Exporting SQD manifests (no credits needed)..."
    python scripts/rl_optimize_and_submit.py \
        --checkpoint results/train/h_cgqe_model_rl.pt \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --molecules h2 lih beh2 methyl_iodide iodobenzene \
        --export-only

    log "Submitting to Rigetti Cepheus (AWS Braket: 30 credits/task + 0.0425 credits/shot)..."
    echo "  At 8192 shots: ~378 credits/molecule (~5 molecules possible)"
    echo "  At 4096 shots: ~204 credits/molecule (~9 molecules possible)"
    read -p "  Proceed with QPU submission? [y/N] " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        python scripts/submit_sqd_to_cepheus.py \
            --manifests results/qpu/sqd_manifests/ \
            --device aws:rigetti:qpu:cepheus-1-108q \
            --shots 4096

        log "Retrieving results and running SQD post-processing..."
        python scripts/retrieve_and_sqd.py \
            --ledger results/qpu/qpu_ledger.sqlite \
            --out results/qpu/cepheus_rl_sqd_results.json
    else
        echo "  Skipped QPU submission."
    fi
}

# ---------------------------------------------------------------------------
# Stage: figures — Generate all report PNGs
# ---------------------------------------------------------------------------
run_figures() {
    log "Generating report figures..."
    python scripts/plot_phase3_report_figures.py
    log "Figures saved to results/phase3_final/figures/"
}

# ---------------------------------------------------------------------------
# Stage: report — Generate .docx
# ---------------------------------------------------------------------------
run_report() {
    log "Generating Word report..."
    python scripts/generate_phase3_report_docx.py
    log "Report saved to proposals/Ryoushi_Quantum_Buddies_Phase3_Report.docx"
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
case "$STAGE" in
    all)
        run_setup
        run_chem
        run_train
        run_eval
        run_baselines
        run_figures
        run_report
        log "QPU stage skipped. Run 'bash scripts/run_all_reproducible.sh qpu' to submit to Rigetti."
        ;;
    setup)     run_setup ;;
    chem)      run_chem ;;
    train)     run_train ;;
    eval)      run_eval ;;
    baselines) run_baselines ;;
    qpu)       run_qpu ;;
    figures)   run_figures ;;
    report)    run_report ;;
    *) echo "Unknown stage: $STAGE"; echo "Usage: bash scripts/run_all_reproducible.sh [all|setup|chem|train|eval|baselines|qpu|figures|report]"; exit 1 ;;
esac

log "Done."
