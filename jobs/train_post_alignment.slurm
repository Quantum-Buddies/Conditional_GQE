#!/usr/bin/env bash
#SBATCH --job-name=gqe-alignment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/results/logs/gqe-alignment-%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/results/logs/gqe-alignment-%j.err

# RAFT Post-Training Alignment on GPU
# Expected runtime: ~1-2 hours depending on target molecule count

PROJECT_ROOT="/scratch/kcwp264/Conditional-GQE_materials"
PYTHON_BIN="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"

HAMILTONIANS="${PROJECT_ROOT}/results/data/hamiltonians_scaling.json/hamiltonians.json"
CHECKPOINT="${PROJECT_ROOT}/results/train/h_cgqe_rl_warmstart.pt"
OUTPUT="${PROJECT_ROOT}/results/train/h_cgqe_raft_aligned.pt"

export CUDAQ_MPS_MAX_BOND=64
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket

echo "=== Starting Post-Training Alignment ==="
echo "Start: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Hamiltonians: ${HAMILTONIANS}"
echo "Output: ${OUTPUT}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/train_post_alignment.py \
    --checkpoint "${CHECKPOINT}" \
    --hamiltonians "${HAMILTONIANS}" \
    --out "${OUTPUT}" \
    --epochs 50 \
    --batch-size 4 \
    --lr 5e-5 \
    --n-samples 50 \
    --top-k 5 \
    --use-cuda \
    --target nvidia \
    --target-option mqpu

echo ""
echo "=== Alignment Complete ==="
echo "End: $(date)"
echo "Output saved to: ${OUTPUT}"
