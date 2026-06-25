#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}"

mkdir -p "${ROOT}/results/data" "${ROOT}/results/baselines" "${ROOT}/results/train" "${ROOT}/results/tables" "${ROOT}/results/plots"

HAM_FILE="${ROOT}/results/data/hamiltonians.json"
ADAPT_FILE="${ROOT}/results/baselines/adapt_vqe.json"
CUDAQ_FILE="${ROOT}/results/baselines/cudaq_vqe.json"
GQE_FILE="${ROOT}/results/baselines/cudaq_gqe.json"
EXACT_FILE="${ROOT}/results/baselines/exact_diagonalization.json"

echo "[1/8] Generating Hamiltonian dataset..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/data/generate_hamiltonians.py" \
  --config "${ROOT}/configs/experiment.yaml" \
  --out "${ROOT}/results/data"

echo "[2/8] Running exact diagonalization references (where tractable)..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/baselines/run_exact_diagonalization.py" \
  --ham "${HAM_FILE}" \
  --out "${EXACT_FILE}"

echo "[3/8] Running ADAPT-VQE baseline..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/baselines/run_adapt_vqe.py" \
  --ham "${HAM_FILE}" \
  --out "${ADAPT_FILE}"

echo "[4/8] Running CUDA-Q VQE baseline..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/baselines/run_cudaq_vqe.py" \
  --ham "${HAM_FILE}" \
  --out "${CUDAQ_FILE}"

if [[ "${RUN_CUDAQ_GQE:-0}" == "1" ]]; then
  echo "[5/8] Running CUDA-Q GQE baseline (optional)..."
  "${PYTHON_BIN}" "${ROOT}/src/gqe/baselines/run_cudaq_gqe.py" \
    --ham "${HAM_FILE}" \
    --out "${GQE_FILE}"
else
  echo "[5/8] Skipping CUDA-Q GQE baseline (set RUN_CUDAQ_GQE=1 to enable)."
fi

echo "[6/8] Training strict supervised model..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/models/train_supervised.py" \
  --config "${ROOT}/configs/experiment.yaml" \
  --out "${ROOT}/results/train/supervised_train.done"

echo "[7/8] Aggregating metrics..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/eval/aggregate_metrics.py" \
  --ham "${HAM_FILE}" \
  --baseline "${ADAPT_FILE}" \
  --cudaq-baseline "${CUDAQ_FILE}" \
  --gqe-baseline "${GQE_FILE}" \
  --reference "${EXACT_FILE}" \
  --train "${ROOT}/results/train/train_metrics.json" \
  --out "${ROOT}/results/tables/benchmark_summary.csv"

echo "[8/8] Generating benchmark plots..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/eval/plot_benchmark_results.py" \
  --summary-csv "${ROOT}/results/tables/benchmark_summary.csv" \
  --train-json "${ROOT}/results/train/train_metrics.json" \
  --out-dir "${ROOT}/results/plots" \
  --manifest "${ROOT}/results/plots/benchmark_plot_manifest.json"

echo "Experiment suite complete. See ${ROOT}/results."

