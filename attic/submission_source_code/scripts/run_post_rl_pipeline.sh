#!/bin/bash
# =============================================================================
# Post-RL Pipeline Orchestrator
# Runs all phases: extract -> benchmark -> plot -> PDF
# (QPU submission phases are optional and require qBraid credentials)
#
# Usage:
#   bash scripts/run_post_rl_pipeline.sh [phases]
#
# Phases:
#   all       - Run all phases (default)
#   extract   - Extract best circuits from RL checkpoint
#   benchmark - Build consolidated benchmark table
#   plot      - Generate figures
#   pdf       - Generate GIC submission PDF
#   qpu       - Submit to QPU (requires qBraid)
#   help      - Show this help
#
# Prerequisites:
#   - RL checkpoint at results/train/h_cgqe_model_qbraid_rl.pt
#   - Hamiltonians at results/data/hamiltonians_gic2026/hamiltonians.json
#   - Python with fpdf, matplotlib, torch, cudaq
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- Config ---
CKPT="${CKPT:-results/train/h_cgqe_model_qbraid_rl.pt}"
HAMS="${HAMS:-results/data/hamiltonians_gic2026/hamiltonians.json}"
BEST_CIRCUITS="${BEST_CIRCUITS:-results/train/h_cgqe_model_qbraid_rl_best_circuits.json}"
RL_METRICS="${RL_METRICS:-results/train/h_cgqe_model_qbraid_rl_rl_metrics.json}"
ARCHIVE_DIR="${ARCHIVE_DIR:-results/train/h_cgqe_model_qbraid_rl_map_elites}"
ENERGY_CACHE="${ENERGY_CACHE:-results/train/rl_energy_cache.sqlite}"
GQE_BASELINE="${GQE_BASELINE:-results/baselines/cudaq_gqe_uccsd_3gpu.json}"
BENCHMARK_OUT="${BENCHMARK_OUT:-results/eval/gic_benchmark_consolidated.json}"
FIGURES_DIR="${FIGURES_DIR:-results/eval/figures}"
PDF_OUT="${PDF_OUT:-proposals/GIC2026_Submission.pdf}"
PY="${PY:-python3}"

# --- CUDA-Q Performance Tuning ---
export CUDAQ_ENABLE_MEMPOOL="${CUDAQ_ENABLE_MEMPOOL:-1}"
export CUDAQ_FUSION_MAX_QUBITS="${CUDAQ_FUSION_MAX_QUBITS:-6}"
export CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS="${CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS:--1}"
export CUDAQ_MGPU_FUSE="${CUDAQ_MGPU_FUSE:-4}"
export CUDAQ_MAX_GPU_MEMORY_GB="${CUDAQ_MAX_GPU_MEMORY_GB:-NONE}"

# --- NISQ Pipeline Config ---
OPTIMIZED_OUT="${OPTIMIZED_OUT:-results/eval/h_cgqe_rl_optimized.json}"
SQD_PILOT_DIR="${SQD_PILOT_DIR:-results/eval/sqd_pilot}"
QPU_LEDGER_DB="${QPU_LEDGER_DB:-results/qpu/qpu_ledger.sqlite}"
QPU_DEVICE="${QPU_DEVICE:-aws:rigetti:qpu:cepheus-1-108q}"
QPU_SHOTS="${QPU_SHOTS:-4096}"
QPU_BUDGET="${QPU_BUDGET:-13403}"
SQD_MOLECULES="${SQD_MOLECULES:-h2 lih}"

# --- Helpers ---
log() { echo -e "\033[1;34m[$(date +%H:%M:%S)]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

check_file() {
    if [ ! -f "$1" ]; then
        err "Missing required file: $1"
        exit 1
    fi
}

# --- Phase functions ---
run_optimize() {
    log "Phase 0: Deterministic L-BFGS-B coefficient optimization"
    check_file "$HAMS"

    GENERATED="${GENERATED:-results/inference/h_cgqe_generated.json}"
    if [ ! -f "$GENERATED" ]; then
        log "  No generated sequences found at $GENERATED, skipping optimization"
        return 0
    fi

    if [ -f "$OPTIMIZED_OUT" ]; then
        log "  Optimized results already exist: $OPTIMIZED_OUT"
        log "  (Delete to re-optimize)"
        return 0
    fi

    $PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
        --generated "$GENERATED" \
        --hamiltonians "$HAMS" \
        --out "$OPTIMIZED_OUT" \
        --target nvidia --target-option fp32 \
        --max-iter 100 \
        --n-starts 4 \
        --seed 42
}

run_sqd_pilot() {
    log "Phase 1: SQD pilot control suite (local, no QPU credits)"
    check_file "$HAMS"

    OPT_ARG=""
    if [ -f "$OPTIMIZED_OUT" ]; then
        OPT_ARG="--optimized $OPTIMIZED_OUT"
        log "  Using optimized results: $OPTIMIZED_OUT"
    elif [ -f "results/eval/h_cgqe_optimized.json" ]; then
        OPT_ARG="--optimized results/eval/h_cgqe_optimized.json"
        log "  Using legacy optimized results: results/eval/h_cgqe_optimized.json"
    else
        log "  WARNING: No optimized results found, SQD will use HF-only state"
    fi

    $PY scripts/run_sqd_pilot.py \
        --hamiltonians "$HAMS" \
        --molecules $SQD_MOLECULES \
        $OPT_ARG \
        --shots 4096 \
        --out "$SQD_PILOT_DIR/"
}

run_sim_bench() {
    log "Phase: Simulator benchmark on free qBraid simulators"
    check_file "$HAMS"

    local sim_devices="${SIM_DEVICES:-ionq:ionq:sim:simulator}"
    local sim_shots="${SIM_SHOTS:-4096}"
    local sim_molecules="${SIM_MOLECULES:-h2 lih}"
    local sim_out="${SIM_BENCH_OUT:-results/eval/simulator_benchmark.json}"
    local sim_max_qwc="${SIM_MAX_QWC:-20}"

    local opt_arg=""
    if [ -f "$OPTIMIZED_OUT" ]; then
        opt_arg="--optimized $OPTIMIZED_OUT"
    elif [ -f "results/eval/h_cgqe_optimized.json" ]; then
        opt_arg="--optimized results/eval/h_cgqe_optimized.json"
    fi

    log "  Devices: $sim_devices"
    log "  Molecules: $sim_molecules"
    log "  Shots: $sim_shots"
    log "  Max QWC circuits (auto-sqd-only threshold): $sim_max_qwc"

    $PY scripts/run_simulator_benchmark.py \
        --hamiltonians "$HAMS" \
        $opt_arg \
        --molecules $sim_molecules \
        --devices $sim_devices \
        --shots $sim_shots \
        --max-qwc-circuits $sim_max_qwc \
        --out "$sim_out"
}

run_qpu() {
    # Unified QPU function — handles both manifest export and job submission.
    # Mode is controlled by QPU_MODE env var:
    #   export  — Export manifests only (no credits spent)
    #   submit  — Submit jobs to QPU (spends credits, requires explicit opt-in)
    #   both    — Export then submit (default when phase is 'qpu')
    local mode="${QPU_MODE:-both}"
    local export_type="${QPU_EXPORT_TYPE:-qwc}"   # qwc | sqd | both
    local device="${QPU_DEVICE:-aws:rigetti:qpu:cepheus-1-108q}"
    local shots="${QPU_SHOTS:-4096}"
    local molecules="${QPU_MOLECULES:-h2}"
    local budget="${QPU_BUDGET:-13403}"
    local ledger_db="${QPU_LEDGER_DB:-results/qpu/qpu_ledger.sqlite}"
    local qpu_dir="results/qpu"
    local opt_arg=""
    local generated_arg=""

    log "Phase 2: QPU pipeline (mode=$mode, type=$export_type, device=$device)"

    check_file "$HAMS"
    mkdir -p "$qpu_dir"

    # Build optional args
    if [ -f "$OPTIMIZED_OUT" ]; then
        opt_arg="--optimized $OPTIMIZED_OUT"
    elif [ -f "results/eval/h_cgqe_optimized.json" ]; then
        opt_arg="--optimized results/eval/h_cgqe_optimized.json"
    fi
    if [ -f "results/inference/h_cgqe_generated.json" ]; then
        generated_arg="--generated results/inference/h_cgqe_generated.json"
    fi

    # --- Export phase ---
    if [ "$mode" = "export" ] || [ "$mode" = "both" ]; then
        log "  Exporting manifests ($export_type) for: $molecules"

        for mol in $molecules; do
            local out_sqd="$qpu_dir/sqd_manifest_${mol}.json"
            local out_qwc="$qpu_dir/qwc_manifest_${mol}.json"

            if [ "$export_type" = "sqd" ] || [ "$export_type" = "both" ]; then
                log "  Exporting SQD sampling manifest for $mol ..."
                $PY src/gqe/eval/qbraid_backend.py \
                    --hamiltonians "$HAMS" \
                    --molecule "$mol" \
                    $opt_arg \
                    --device "$device" \
                    --shots "$shots" \
                    --export-sqd \
                    --out "$out_sqd" \
                    --ledger-db "$ledger_db" \
                    --budget "$budget" 2>&1 || \
                    err "  SQD export for $mol failed (check qiskit/qbraid)"
            fi

            if [ "$export_type" = "qwc" ] || [ "$export_type" = "both" ]; then
                log "  Exporting QWC diagnostic manifest for $mol ..."
                $PY src/gqe/eval/qbraid_backend.py \
                    --hamiltonians "$HAMS" \
                    --molecule "$mol" \
                    $opt_arg \
                    --device "$device" \
                    --shots "$shots" \
                    --export-qwc \
                    --out "$out_qwc" \
                    --ledger-db "$ledger_db" \
                    --budget "$budget" 2>&1 || \
                    err "  QWC export for $mol failed (check qiskit/qbraid)"
            fi
        done

        log "  Manifests exported to $qpu_dir/ (no credits spent)"
    fi

    # --- Submit phase ---
    if [ "$mode" = "submit" ] || [ "$mode" = "both" ]; then
        log "  WARNING: This will spend qBraid credits!"
        log "  Device: $device | Shots: $shots | Budget: $budget credits"
        log "  Press Ctrl+C within 5 seconds to abort..."
        sleep 5

        for mol in $molecules; do
            local out_result="$qpu_dir/qpu_result_${mol}.json"

            log "  Submitting $mol to $device ($shots shots) ..."

            # Try batched QWC submission via qbraid_backend.py
            if [ -n "$generated_arg" ] && [ -n "$opt_arg" ]; then
                $PY src/gqe/eval/qbraid_backend.py \
                    --hamiltonians "$HAMS" \
                    $generated_arg \
                    $opt_arg \
                    --molecule "$mol" \
                    --device "$device" \
                    --shots "$shots" \
                    --submit-only \
                    --out "$out_result" \
                    --ledger-db "$ledger_db" \
                    --budget "$budget" 2>&1 || \
                    err "  QPU submission for $mol failed (check credentials)"
            else
                # Fallback: submit via submit_qpu.py using benchmark
                if [ -f "$BENCHMARK_OUT" ]; then
                    $PY src/gqe/eval/submit_qpu.py \
                        --benchmark "$BENCHMARK_OUT" \
                        --device "$device" \
                        --shots "$shots" \
                        --out "$out_result" \
                        --submit-only 2>&1 || \
                        err "  QPU submission for $mol failed (check credentials)"
                else
                    err "  Cannot submit $mol: missing generated/optimized files and no benchmark"
                fi
            fi

            log "  Job submitted for $mol. Result will be at: $out_result"
        done

        log "  QPU jobs submitted. Retrieve with:"
        log "    python -m gqe.eval.qpu_ledger poll --db $ledger_db"
        log "    python -m gqe.eval.qpu_ledger retrieve --db $ledger_db"
    fi
}

run_extract() {
    log "Phase 1A: Extract best circuits from RL checkpoint"
    check_file "$CKPT"
    check_file "$HAMS"

    if [ -f "$BEST_CIRCUITS" ]; then
        log "  Best circuits already exist: $BEST_CIRCUITS"
        log "  (Delete to re-extract, or use --force)"
        return 0
    fi

    $PY scripts/extract_best_circuits.py \
        --checkpoint "$CKPT" \
        --hamiltonians "$HAMS" \
        --energy-cache "$ENERGY_CACHE" \
        --archive-dir "$ARCHIVE_DIR" \
        --out "$BEST_CIRCUITS" \
        --n-samples 64 \
        --target nvidia --target-option fp32
}

run_benchmark() {
    log "Phase 3A: Build consolidated benchmark table"
    check_file "$BEST_CIRCUITS"
    check_file "$HAMS"

    QPU_ARG=""
    if [ -f "results/eval/simulator_validation.json" ]; then
        QPU_ARG="--qpu-results results/eval/simulator_validation.json"
        log "  Including QPU results: results/eval/simulator_validation.json"
    fi

    GQE_ARG=""
    if [ -f "$GQE_BASELINE" ]; then
        GQE_ARG="--gqe-baseline $GQE_BASELINE"
        log "  Including GQE baseline: $GQE_BASELINE"
    fi

    RL_ARG=""
    if [ -f "$RL_METRICS" ]; then
        RL_ARG="--rl-metrics $RL_METRICS"
    fi

    $PY scripts/build_gic_benchmark.py \
        --best-circuits "$BEST_CIRCUITS" \
        --hamiltonians "$HAMS" \
        $RL_ARG $GQE_ARG $QPU_ARG \
        --out "$BENCHMARK_OUT"

    log "  CSV: ${BENCHMARK_OUT%.json}.csv"
}

run_plot() {
    log "Phase 3B-C: Generate figures"
    check_file "$BENCHMARK_OUT"

    $PY scripts/plot_qpu_vs_gpu.py \
        --benchmark "$BENCHMARK_OUT" \
        --out-dir "$FIGURES_DIR"
}

run_pdf() {
    log "Phase 4: Generate GIC submission PDF"
    check_file "$BENCHMARK_OUT"

    ARCHIVE_ARG=""
    if [ -d "$ARCHIVE_DIR" ]; then
        ARCHIVE_ARG="--archive-dir $ARCHIVE_DIR"
    fi

    RL_ARG=""
    if [ -f "$RL_METRICS" ]; then
        RL_ARG="--rl-metrics $RL_METRICS"
    fi

    SCALING_ARG=""
    if [ -f "$FIGURES_DIR/scaling_error.png" ]; then
        SCALING_ARG="--scaling-plot $FIGURES_DIR/scaling_error.png"
    fi

    QPU_PLOT_ARG=""
    if [ -f "$FIGURES_DIR/qpu_vs_gpu.png" ]; then
        QPU_PLOT_ARG="--qpu-plot $FIGURES_DIR/qpu_vs_gpu.png"
    fi

    $PY scripts/generate_gic_submission.py \
        --benchmark "$BENCHMARK_OUT" \
        $RL_ARG $ARCHIVE_ARG $SCALING_ARG $QPU_PLOT_ARG \
        --out "$PDF_OUT"

    log "  PDF: $PDF_OUT"
}

run_sqd_qpu() {
    # Submit SQD-specific circuits to QPU using the unified run_qpu infrastructure.
    # This exports a computational-basis (Z-basis) sampling manifest and optionally
    # submits it, producing counts that feed into the SQD pipeline.
    # Uses QPU_MODE=export by default (safe — no credits spent unless explicitly overridden).
    local mode="${SQD_QPU_MODE:-export}"
    local device="${SQD_QPU_DEVICE:-aws:rigetti:qpu:cepheus-1-108q}"
    local shots="${SQD_QPU_SHOTS:-4096}"
    local molecules="${SQD_MOLECULES:-h2 lih}"
    local qpu_dir="results/qpu"
    local opt_arg=""

    log "Phase 2S: SQD-QPU — Z-basis sampling circuits for SQD (mode=$mode)"

    check_file "$HAMS"
    mkdir -p "$qpu_dir"

    if [ -f "$OPTIMIZED_OUT" ]; then
        opt_arg="--optimized $OPTIMIZED_OUT"
    elif [ -f "results/eval/h_cgqe_optimized.json" ]; then
        opt_arg="--optimized results/eval/h_cgqe_optimized.json"
    fi

    # Export SQD sampling manifests
    if [ "$mode" = "export" ] || [ "$mode" = "both" ]; then
        log "  Exporting SQD Z-basis sampling manifests for: $molecules"
        for mol in $molecules; do
            local out_sqd="$qpu_dir/sqd_sampling_${mol}.json"
            log "  Exporting SQD sampling circuit for $mol ($shots shots, $device)..."
            $PY src/gqe/eval/qbraid_backend.py \
                --hamiltonians "$HAMS" \
                --molecule "$mol" \
                $opt_arg \
                --device "$device" \
                --shots "$shots" \
                --export-sqd \
                --out "$out_sqd" 2>&1 || \
                err "  SQD export for $mol failed"
        done
        log "  SQD manifests exported to $qpu_dir/"
        log "  Feed counts from QPU into: python scripts/run_sqd_pilot.py --hardware-counts <counts.json>"
    fi

    # Submit SQD circuits to QPU
    if [ "$mode" = "submit" ] || [ "$mode" = "both" ]; then
        log "  WARNING: This will spend qBraid credits!"
        log "  Device: $device | Shots: $shots"
        log "  Press Ctrl+C within 5 seconds to abort..."
        sleep 5

        for mol in $molecules; do
            local out_result="$qpu_dir/sqd_qpu_result_${mol}.json"
            local manifest="$qpu_dir/sqd_sampling_${mol}.json"

            if [ ! -f "$manifest" ]; then
                err "  No SQD manifest found for $mol at $manifest — run export first"
                continue
            fi

            log "  Submitting SQD sampling circuit for $mol to $device ..."

            # Submit the SQD manifest's circuit to the QPU
            # The manifest contains a single Z-basis measurement circuit
            $PY -c "
import json, sys, time
from pathlib import Path

manifest_path = '$manifest'
out_path = '$out_result'
device = '$device'
shots = $shots

with open(manifest_path) as f:
    manifest = json.load(f)

from qbraid import QbraidProvider
from qiskit import QuantumCircuit

# Reconstruct circuit from QASM
qasm_str = manifest.get('circuit_qasm', '')
if not qasm_str:
    print('ERROR: No QASM in manifest', file=sys.stderr)
    sys.exit(1)

try:
    from qiskit.qasm2 import loads as qasm2_loads
    circuit = qasm2_loads(qasm_str)
except Exception:
    circuit = QuantumCircuit.from_qasm_str(qasm_str)

provider = QbraidProvider()
devices = provider.get_devices()
qdevice = next((d for d in devices if d.id == device), None)
if qdevice is None:
    qdevice = next((d for d in devices if device in d.id), None)
if qdevice is None:
    print(f'Device {device} not found', file=sys.stderr)
    sys.exit(1)

print(f'  Device: {qdevice.id}')
print(f'  Submitting {circuit.num_qubits}q SQD sampling circuit, {shots} shots...')
t0 = time.time()
job = qdevice.run(circuit, shots=shots)
runtime = time.time() - t0
job_id = job.id if hasattr(job, 'id') else str(job)
print(f'  Job ID: {job_id}')
print(f'  Submit time: {runtime:.2f}s')

result = {
    'job_id': job_id,
    'device_id': qdevice.id,
    'shots': shots,
    'molecule': manifest.get('molecule', '$mol'),
    'pipeline_stage': 'sqd_qpu',
    'manifest_path': manifest_path,
    'submit_time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'submit_runtime_seconds': runtime,
    'circuit_qubits': circuit.num_qubits,
    'circuit_depth': circuit.depth(),
}
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'  Saved: {out_path}')
" 2>&1 || err "  SQD QPU submission for $mol failed"
        done

        log "  SQD QPU jobs submitted. Retrieve counts with:"
        log "    python -m gqe.eval.qpu_ledger poll --db $QPU_LEDGER_DB"
        log "    python -m gqe.eval.qpu_ledger retrieve --db $QPU_LEDGER_DB"
        log "  Then feed counts to SQD pipeline:"
        log "    python scripts/run_sqd_pilot.py --hardware-counts <counts.json> --hamiltonians $HAMS"
    fi
}

show_help() {
    cat <<'HELP'
Post-RL Pipeline Orchestrator

Usage: bash scripts/run_post_rl_pipeline.sh [phase ...]

Phases:
  all        Run all phases (default)
  optimize   Deterministic L-BFGS-B coefficient optimization (GPU)
  sqd-pilot  Local SQD control suite (no QPU credits)
  sim-bench  Simulator benchmark on free qBraid simulators (IonQ/QIR)
  sqd-qpu    SQD Z-basis sampling circuits to QPU (export by default)
  qpu        Unified QPU pipeline: export manifests + submit jobs
             (QPU_MODE=export for manifests only, =submit to spend credits,
              =both for both. QPU_EXPORT_TYPE=qwc|sqd|both)
  extract    Extract best circuits from RL checkpoint
  benchmark  Build consolidated benchmark table
  plot       Generate figures (scaling, QPU vs GPU, error distribution)
  pdf        Generate 3-page GIC submission PDF

Environment variables:
  CKPT           RL checkpoint path (default: results/train/h_cgqe_model_qbraid_rl.pt)
  HAMS           Hamiltonians JSON path
  BEST_CIRCUITS  Best circuits JSON path
  GQE_BASELINE   GQE baseline JSON path
  BENCHMARK_OUT  Benchmark output path
  FIGURES_DIR    Figures output directory
  PDF_OUT        PDF output path
  PY             Python executable (default: python3)
  OPTIMIZED_OUT  Optimized coefficients output path
  SQD_PILOT_DIR  SQD pilot output directory
  SQD_MOLECULES  Molecules for SQD pilot (default: "h2 lih")
  SIM_DEVICES    Simulator device IDs (default: ionq:ionq:sim:simulator)
  SIM_SHOTS      Simulator shots (default: 4096)
  SIM_MOLECULES  Molecules for sim benchmark (default: "h2 lih")
  SIM_MAX_QWC    Auto-sqd-only threshold (default: 20 QWC circuits)
  QPU_DEVICE     qBraid QPU device ID (default: aws:rigetti:qpu:cepheus-1-108q)
  QPU_SHOTS      QPU shots (default: 4096)
  QPU_BUDGET     Credit budget (default: 13403)
  QPU_MODE       QPU pipeline mode: export|submit|both (default: both)
  QPU_EXPORT_TYPE  Manifest type: qwc|sqd|both (default: qwc)
  QPU_MOLECULES  Molecules for QPU submission (default: h2)
  QPU_LEDGER_DB  SQLite ledger path (default: results/qpu/qpu_ledger.sqlite)
  SQD_QPU_MODE   SQD-QPU mode: export|submit|both (default: export)
  SQD_QPU_DEVICE SQD-QPU device (default: same as QPU_DEVICE)
  SQD_QPU_SHOTS  SQD-QPU shots (default: same as QPU_SHOTS)

Examples:
  bash scripts/run_post_rl_pipeline.sh                    # Run all phases
  bash scripts/run_post_rl_pipeline.sh optimize sqd-pilot # Optimize + SQD pilot
  bash scripts/run_post_rl_pipeline.sh sim-bench       # Free simulator benchmark
  bash scripts/run_post_rl_pipeline.sh benchmark plot pdf # Skip extraction + QPU
  QPU_MODE=export bash scripts/run_post_rl_pipeline.sh qpu  # Export manifests only
  QPU_MODE=submit bash scripts/run_post_rl_pipeline.sh qpu  # Submit to QPU (spends credits!)
  bash scripts/run_post_rl_pipeline.sh sqd-qpu            # Export SQD sampling circuits
  SQD_QPU_MODE=submit bash scripts/run_post_rl_pipeline.sh sqd-qpu  # Submit SQD circuits
HELP
}

# --- Main ---
PHASES="${*:-all}"

for phase in $PHASES; do
    case "$phase" in
        all)       run_extract; run_benchmark; run_plot; run_pdf ;;
        optimize)  run_optimize ;;
        sqd-pilot) run_sqd_pilot ;;
        sim-bench) run_sim_bench ;;
        sqd-qpu)   run_sqd_qpu ;;
        qpu)       run_qpu ;;
        extract)   run_extract ;;
        benchmark) run_benchmark ;;
        plot)      run_plot ;;
        pdf)       run_pdf ;;
        help|-h|--help) show_help; exit 0 ;;
        *) err "Unknown phase: $phase"; show_help; exit 1 ;;
    esac
done

log "Pipeline complete."
