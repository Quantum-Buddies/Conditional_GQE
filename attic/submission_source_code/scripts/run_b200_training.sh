#!/usr/bin/env bash
# Portable single-B200 H-cGQE training launcher.
# Runs supervised warm-start followed by DAPO RL fine-tuning.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="$PYTHON"
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then PYTHON="$(command -v python)"; fi
HAM_FILE="$HAM_FILE"; [ -n "$HAM_FILE" ] || HAM_FILE="results/data/hamiltonians_merged.json"
DATASET="$DATASET"; [ -n "$DATASET" ] || DATASET="results/train/gqe_supervised_dataset.pt"
SUPERVISED="$SUPERVISED"; [ -n "$SUPERVISED" ] || SUPERVISED="results/train/h_cgqe_b200_supervised.pt"
RL_OUT="$RL_OUT"; [ -n "$RL_OUT" ] || RL_OUT="results/train/h_cgqe_b200_rl.pt"
MOLECULES="$MOLECULES"; [ -n "$MOLECULES" ] || MOLECULES="h2_0.74 lih_1.6_full n2_1.1_full beh2_1.3_full"
RL_EPOCHS="$RL_EPOCHS"; [ -n "$RL_EPOCHS" ] || RL_EPOCHS=500
RL_SAMPLES="$RL_SAMPLES"; [ -n "$RL_SAMPLES" ] || RL_SAMPLES=64
RL_ITERS="$RL_ITERS"; [ -n "$RL_ITERS" ] || RL_ITERS=5
MAX_QUBITS="$MAX_QUBITS"; [ -n "$MAX_QUBITS" ] || MAX_QUBITS=30
MAX_TERMS="$MAX_TERMS"; [ -n "$MAX_TERMS" ] || MAX_TERMS=256
SKIP_SUPERVISED="$SKIP_SUPERVISED"
[ -n "$SKIP_SUPERVISED" ] || SKIP_SUPERVISED=0

if [ ! -f "$DATASET" ] || [ ! -f "$HAM_FILE" ]; then
    echo "Missing dataset or Hamiltonian file." >&2; exit 1
fi
mkdir -p "$(dirname "$SUPERVISED")" "$(dirname "$RL_OUT")"

if [ "$SKIP_SUPERVISED" != "1" ] || [ ! -f "$SUPERVISED" ]; then
    "$PYTHON" -u src/gqe/models/train_h_cgqe.py --dataset "$DATASET" --out "$SUPERVISED" --use-cuda
fi

"$PYTHON" -u src/gqe/models/train_rl_dapo.py \
    --checkpoint "$SUPERVISED" --hamiltonians "$HAM_FILE" --molecules $MOLECULES \
    --out "$RL_OUT" --epochs "$RL_EPOCHS" --n-samples "$RL_SAMPLES" --n-iters "$RL_ITERS" \
    --use-cuda --use-bf16 --target nvidia --target-option mqpu \
    --max-qubits "$MAX_QUBITS" --max-terms "$MAX_TERMS" --adaptive-theta \
    --gate-auxiliary-rewards --dynamic-sampling --curriculum --force-entanglement
echo "RL checkpoint written to: $RL_OUT"
