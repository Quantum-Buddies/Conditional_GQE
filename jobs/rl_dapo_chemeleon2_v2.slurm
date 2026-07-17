#!/bin/bash
#SBATCH --job-name=rl-chemeleon2-v2
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/results/logs/rl-chemeleon2-v2-%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/results/logs/rl-chemeleon2-v2-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:3

set -euo pipefail

PROJECT_ROOT="/scratch/kcwp264/Conditional-GQE_materials"
PYTHON_BIN="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"
CHECKPOINT="${PROJECT_ROOT}/results/train/h_cgqe_uccsd_model.pt"
HAMILTONIANS="${PROJECT_ROOT}/results/data/hamiltonians_merged.json"
OUTPUT="${PROJECT_ROOT}/results/train/h_cgqe_rl_chemeleon2_v2.pt"

export CUDAQ_MPS_MAX_BOND=64

echo "=== Chemeleon2-mode RL Training v2 (optimized hyperparams) ==="
echo "Start: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Hamiltonians: ${HAMILTONIANS}"
echo "Output: ${OUTPUT}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""
echo "Key changes from v1:"
echo "  - n-samples 32 -> 64 (2x better gradient estimates)"
echo "  - n-iters 2 -> 5 (GPT-QE paper N_iter=5)"
echo "  - reuse-iters 1 -> 3 (3x simulation cost reduction)"
echo "  - buffer-batch-size 0 -> 64 (replay buffer now implemented)"
echo "  - buffer-size 1000 -> 2000 (larger replay pool)"
echo "  - adaptive-theta enabled (better energy signal)"
echo "  - epochs 200 -> 500 (longer training for H2/N2 convergence)"
echo "  - curriculum-warmup 30 -> 50 (longer warmup for larger samples)"
echo ""

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -u src/gqe/models/train_rl_dapo.py \
    --checkpoint "${CHECKPOINT}" \
    --hamiltonians "${HAMILTONIANS}" \
    --molecules h2 lih beh2 n2 \
    --out "${OUTPUT}" \
    --epochs 500 \
    --n-samples 64 \
    --n-iters 5 \
    --reuse-iters 3 \
    --buffer-size 2000 \
    --buffer-batch-size 64 \
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
    --curriculum \
    --curriculum-warmup 50 \
    --curriculum-steps 3 \
    --chemeleon2-mode \
    --msun-threshold 0.1 \
    --adaptive-theta \
    --adaptive-theta-iters 10 \
    2>&1

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
echo "Output saved to: ${OUTPUT}"
