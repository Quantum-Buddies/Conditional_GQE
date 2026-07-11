#!/bin/bash
#SBATCH --job-name=scaling-bench
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/results/scaling_benchmark/slurm-%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/results/scaling_benchmark/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:l40s:3

set -euo pipefail

echo "=== Scalability Benchmark Slurm Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Start: $(date)"
echo ""

cd /scratch/kcwp264/Conditional-GQE_materials
bash scripts/run_scalability_benchmark.sh results/train/h_cgqe_rl_warmstart.pt

echo ""
echo "End: $(date)"
echo "=== Done ==="
