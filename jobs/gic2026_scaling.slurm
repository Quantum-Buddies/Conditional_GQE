#!/bin/bash
#SBATCH --job-name=gic2026-scale
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/results/logs/gic2026-scale-%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/results/logs/gic2026-scale-%j.err

# GIC 2026 Full Scaling Pipeline on 3x L40S GPUs
# Expected runtime: ~4-6 hours
#   - Hamiltonian generation: ~30 min
#   - RL training (500 epochs, all molecules): ~2-3 hours
#   - Inference (100 samples × 30 molecules): ~15 min
#   - Stage 2 optimization (top-20 × 30 molecules): ~1 hour
#   - GQE baseline: ~15 min
#   - Plots: ~1 min

export CUDAQ_MPS_MAX_BOND=64
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket

bash /scratch/kcwp264/Conditional-GQE_materials/scripts/run_gic2026_scaling.sh
