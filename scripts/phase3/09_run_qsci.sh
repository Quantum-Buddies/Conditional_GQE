#!/bin/bash
# QSCI scaling experiment: 12q -> 40q
# Uses CUDA-Q tensornet-mps backend for >24q, nvidia for <=24q
# Run on AIRE HPC with 1x L40S GPU

set -euo pipefail

PYTHON=${PYTHON:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}
PROJECT_ROOT=${PROJECT_ROOT:-/scratch/kcwp264/Conditional-GQE_materials}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

# Hamiltonian data with 40+ qubit molecules
HAM_FILE="results/data/hamiltonians_40plus.json/hamiltonians.json"

# Molecules to run (increasing qubit count)
MOLECULES=(
    "n2"                  # 20 qubits
    "ethylene"            # 28 qubits
    "formaldehyde"        # 24 qubits
    "benzene_cas20"       # 40 qubits -- BONUS POINT target
)

# Subspace sizes to sweep
N_SAMPLES=(100 500 1000 5000)

# MPS bond dimensions to sweep
BOND_DIMS=(64 128 256)

# Sampling shots
N_SHOTS=8192

# Output
OUT_DIR="results/phase3_final/qsci"
OUT_FILE="${OUT_DIR}/qsci_scaling_results.json"

mkdir -p "${OUT_DIR}"

echo "=== GQE-QSCI Scaling Experiment ==="
echo "Molecules: ${MOLECULES[*]}"
echo "Subspace sizes: ${N_SAMPLES[*]}"
echo "Bond dimensions: ${BOND_DIMS[*]}"
echo "Shots: ${N_SHOTS}"
echo "Output: ${OUT_FILE}"
echo ""

${PYTHON} src/gqe/eval/qsci.py \
    --hamiltonians "${HAM_FILE}" \
    --molecules "${MOLECULES[@]}" \
    --n-samples "${N_SAMPLES[@]}" \
    --bond-dims "${BOND_DIMS[@]}" \
    --n-shots ${N_SHOTS} \
    --out "${OUT_FILE}"

echo ""
echo "=== QSCI Scaling Complete ==="
echo "Results saved to: ${OUT_FILE}"
