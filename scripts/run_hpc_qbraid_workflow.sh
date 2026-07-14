#!/usr/bin/env bash
# HPC-Quantum-AI hybrid orchestrator script.
#
# Provides a tight-knit workflow that:
#   1. Submits model training/simulation tasks to Slurm (HPC stage)
#   2. Submits generated quantum circuits in a batch asynchronously to qBraid QPUs (Quantum QPU stage)
#   3. Polls/retrieves QPU results and compiles final metrics (AI evaluation stage)
#
# Usage:
#   bash scripts/run_hpc_qbraid_workflow.sh --hpc-submit        # Submit training to Slurm
#   bash scripts/run_hpc_qbraid_workflow.sh --qpu-submit        # Batch-submit circuits to qBraid QPU
#   bash scripts/run_hpc_qbraid_workflow.sh --qpu-retrieve      # Retrieve and process QPU results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"
EVAL_DIR="${RESULTS_DIR}/eval"

PYTHON="${CONDA_PREFIX:-/usr}/bin/python"

# Targets and paths
HAMILTONIANS="${RESULTS_DIR}/data/hamiltonians_scaling.json/hamiltonians.json"
GENERATED="${RESULTS_DIR}/inference/h_cgqe_uccsd_inference.json"
OPTIMIZED="${RESULTS_DIR}/eval/h_cgqe_uccsd_optimized.json"

DEVICE="aws:rigetti:qpu:cepheus-1-108q"
SHOTS=1024

# Ensure output directory exists
mkdir -p "${EVAL_DIR}"

show_help() {
    echo "H-cGQE HPC-Quantum-AI Workflow Orchestrator"
    echo "------------------------------------------"
    echo "Usage: bash scripts/run_hpc_qbraid_workflow.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  --hpc-submit      Submit Stage 1 RL / scaling training to the Slurm partition"
    echo "  --hpc-status      Show active Slurm job queue status"
    echo "  --qpu-submit      Submit Stage 2 optimized circuits as asynchronous batches to qBraid"
    echo "  --qpu-status      Poll qBraid QPU queue and job status for submitted tasks"
    echo "  --qpu-retrieve    Retrieve results from completed QPU jobs and calculate final energy"
    echo "  --help            Show this help menu"
}

hpc_submit() {
    echo "=== [HPC Stage] Submitting GIC 2026 scaling pipeline to Slurm ==="
    SLURM_JOB="${PROJECT_ROOT}/jobs/gic2026_scaling.slurm"
    if [ ! -f "${SLURM_JOB}" ]; then
        echo "ERROR: Slurm file not found at ${SLURM_JOB}"
        exit 1
    fi
    JOB_ID=$(sbatch --parsable "${SLURM_JOB}")
    echo "Job submitted successfully! Slurm Job ID: ${JOB_ID}"
    echo "You can check status using: squeue -j ${JOB_ID} or 'bash scripts/run_hpc_qbraid_workflow.sh --hpc-status'"
}

hpc_status() {
    echo "=== [HPC Stage] Current Slurm queue status ==="
    squeue -u "$(whoami)"
}

qpu_submit() {
    echo "=== [Quantum QPU Stage] Submitting optimized circuits to qBraid QPU ==="
    if [ ! -f "${HAMILTONIANS}" ] || [ ! -f "${GENERATED}" ] || [ ! -f "${OPTIMIZED}" ]; then
        echo "ERROR: Pre-requisite results (Hamiltonians, generated, optimized) not found in results/."
        echo "Ensure the HPC training and local optimizations are complete."
        exit 1
    fi

    # Retrieve molecules evaluated in optimized file
    MOLECULES=$(python -c "
import json
with open('${OPTIMIZED}') as f:
    data = json.load(f)
mols = [x['molecule'] for x in data]
print(' '.join(mols))
")

    echo "Found optimized molecules: ${MOLECULES}"
    for mol in ${MOLECULES}; do
        echo "  Submitting batch circuit job for molecule: ${mol} to device: ${DEVICE}..."
        OUT_FILE="${EVAL_DIR}/qbraid_qpu_${mol}.json"
        
        # Submit asynchronously via our upgraded backend
        python "${PROJECT_ROOT}/src/gqe/eval/qbraid_backend.py" \
            --hamiltonians "${HAMILTONIANS}" \
            --generated "${GENERATED}" \
            --optimized "${OPTIMIZED}" \
            --molecule "${mol}" \
            --device "${DEVICE}" \
            --shots "${SHOTS}" \
            --submit-only \
            --out "${OUT_FILE}"
    done
    echo "All batch jobs successfully submitted to qBraid!"
    echo "Metadata files are stored in ${EVAL_DIR}/"
}

qpu_status_or_retrieve() {
    MODE=$1 # 'status' or 'retrieve'
    METADATA_FILES=$(find "${EVAL_DIR}" -name "qbraid_job_metadata_*.json")
    
    if [ -z "${METADATA_FILES}" ]; then
        echo "No active qBraid job metadata files found in ${EVAL_DIR}."
        exit 0
    fi

    echo "=== [Quantum QPU Stage] Querying qBraid QPU Jobs ==="
    for meta in ${METADATA_FILES}; do
        filename=$(basename "${meta}")
        mol=$(echo "${filename}" | cut -d'_' -f4)
        out_name="qbraid_qpu_${mol}.json"
        out_path="${EVAL_DIR}/${out_name}"

        if [ "${MODE}" = "status" ]; then
            python -c "
import json, sys
from qbraid.runtime import load_job
with open('${meta}') as f:
    data = json.load(f)
for jid in data['job_ids']:
    job = load_job(jid)
    print(f'Molecule: ${mol} | JobID: {jid} | Status: {job.status()}')
"
        else:
            echo "Attempting retrieval for ${mol}..."
            python "${PROJECT_ROOT}/src/gqe/eval/qbraid_backend.py" \
                --retrieve "${meta}" \
                --out "${out_path}"
        fi
    done
}

# Parse command line argument
if [ $# -lt 1 ]; then
    show_help
    exit 0
fi

case "$1" in
    --hpc-submit)
        hpc_submit
        ;;
    --hpc-status)
        hpc_status
        ;;
    --qpu-submit)
        qpu_submit
        ;;
    --qpu-status)
        qpu_status_or_retrieve "status"
        ;;
    --qpu-retrieve)
        qpu_status_or_retrieve "retrieve"
        ;;
    --help|-h)
        show_help
        ;;
    *)
        echo "Invalid option: $1"
        show_help
        exit 1
        ;;
esac
