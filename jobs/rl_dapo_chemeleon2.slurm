#!/bin/bash
#SBATCH --job-name=rl-chemeleon2
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/results/logs/rl-chemeleon2-%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/results/logs/rl-chemeleon2-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:3

set -euo pipefail

PROJECT_ROOT="/scratch/kcwp264/Conditional-GQE_materials"
PYTHON_BIN="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"
CHECKPOINT="${PROJECT_ROOT}/results/train/h_cgqe_uccsd_model.pt"
HAMILTONIANS="${PROJECT_ROOT}/results/data/hamiltonians_merged.json"
OUTPUT="${PROJECT_ROOT}/results/train/h_cgqe_rl_chemeleon2.pt"

export CUDAQ_MPS_MAX_BOND=64

echo "=== Chemeleon2-mode RL Training ==="
echo "Start: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Hamiltonians: ${HAMILTONIANS}"
echo "Output: ${OUTPUT}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -u src/gqe/models/train_rl_dapo.py \
    --checkpoint "${CHECKPOINT}" \
    --hamiltonians "${HAMILTONIANS}" \
    --molecules h2 lih beh2 n2 \
    --out "${OUTPUT}" \
    --epochs 200 \
    --n-samples 32 \
    --n-iters 2 \
    --lr 1e-5 \
    --temperature 1.0 \
    --top-p 0.9 \
    --target-entropy 1.5 \
    --explore-eps 0.3 \
    --adaptive-eps \
    --force-entanglement \
    --max-repeat 4 \
    --max-qubits 24 \
    --target nvidia \
    --target-option mqpu \
    --use-cuda \
    --multi-gpu \
    --use-bf16 \
    --dynamic-sampling \
    --token-level-loss \
    --entropy-coef 1e-5 \
    --w-energy 1.0 \
    --w-entangle 0.1 \
    --w-depth 0.05 \
    --w-commute 0.05 \
    --w-diversity 0.2 \
    --target-len 10 \
    --freq-penalty 1.0 \
    --buffer-size 1000 \
    --curriculum \
    --curriculum-warmup 30 \
    --curriculum-steps 3 \
    --chemeleon2-mode \
    --msun-threshold 0.1 \
    2>&1

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
echo "Output saved to: ${OUTPUT}"
