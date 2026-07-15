#!/bin/bash
#SBATCH --job-name=gqe-qsci-scaling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/qsci_scaling_%j.out
#SBATCH --error=logs/qsci_scaling_%j.err

# GQE-QSCI Scaling Experiment: 4q -> 40q
# Uses CUDA-Q tensornet-mps backend for >24q molecules
# Run on AIRE HPC with 1x L40S GPU

set -euo pipefail

PYTHON=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python
PROJECT_ROOT=/scratch/kcwp264/Conditional-GQE_materials
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
mkdir -p logs results/phase3_final/qsci

echo "=== GQE-QSCI Scaling Experiment ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Hamiltonian data with 40+ qubit molecules
HAM_FILE="results/data/hamiltonians_40plus.json/hamiltonians.json"

# Molecules to run (increasing qubit count: 4, 12, 14, 20, 24, 28, 40)
MOLECULES=(h2 lih beh2 n2 formaldehyde ethylene benzene_cas20)

# Subspace sizes to sweep
N_SAMPLES=(100 500 1000 5000)

# MPS bond dimensions to sweep
BOND_DIMS=(64 128 256)

# Sampling shots
N_SHOTS=8192

# Output
OUT_FILE="results/phase3_final/qsci/qsci_scaling_results.json"

${PYTHON} src/gqe/eval/qsci.py \
    --hamiltonians "${HAM_FILE}" \
    --molecules "${MOLECULES[@]}" \
    --n-samples "${N_SAMPLES[@]}" \
    --bond-dims "${BOND_DIMS[@]}" \
    --n-shots ${N_SHOTS} \
    --out "${OUT_FILE}"

echo ""
echo "=== QSCI Scaling Complete ==="
echo "Finished: $(date)"
echo "Results: ${OUT_FILE}"
