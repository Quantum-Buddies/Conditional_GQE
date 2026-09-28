#!/bin/bash
# Phase 3 Final — Step 7: Collect QPU results
# Usage: bash scripts/phase3/07_collect_qpu.sh [job_id]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export PY="${PYTHON}"

OUT_DIR=results/phase3_final/qpu

echo "=== Step 7: Collect QPU results ==="

JOB_ID=${1:-}
if [ -z "$JOB_ID" ]; then
    # Read job ID from submission file
    JOB_ID=$($PY -c "import json; print(json.load(open('$OUT_DIR/qpu_submission.json'))['job_id'])" 2>/dev/null || echo "")
fi

if [ -z "$JOB_ID" ]; then
    echo "ERROR: No job ID provided and no submission file found."
    echo "Usage: bash scripts/phase3/07_collect_qpu.sh <job_id>"
    exit 1
fi

echo "Collecting results for job: $JOB_ID"
$PY src/gqe/eval/collect_qpu.py \
    --job-id "$JOB_ID" \
    --out "$OUT_DIR/qpu_validation_result.json"

echo ""
echo "=== QPU results collected ==="
echo "Results: $OUT_DIR/qpu_validation_result.json"
