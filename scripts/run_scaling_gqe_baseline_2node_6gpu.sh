#!/bin/bash -l
# Run CUDA-Q GQE baseline step on 6 GPUs across 2 AIRE L40S nodes (3 per node).
# This is an embarrassingly-parallel single-GPU-per-task script; it does NOT use
# MPI or pooled memory, so it avoids the CUDA-aware MPI and power-of-2 issues.
# Each task processes a disjoint chunk of molecules from the scaling config.
# The results are merged into a single JSON at the end.
#
# Usage:
#   sbatch scripts/run_scaling_gqe_baseline_2node_6gpu.sh
#
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --gres=gpu:l40s:3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 2:00:00

set -e

module load miniforge
conda activate /mnt/scratch/kcwp264/.conda_envs/cudaq-env
module purge

export PY=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python
cd /scratch/kcwp264/Conditional-GQE_materials

HAM=results/data/hamiltonians_scaling.json/hamiltonians.json
OUT_DIR=results/baselines
FINAL_OUT=$OUT_DIR/cudaq_gqe_6gpu_scaling.json

mkdir -p $OUT_DIR

# Wrapper: each Slurm task sees only its local GPU. SLURM_LOCALID is 0/1/2
# on each node, matching the 3 GPUs allocated per node via --gres.
cat > /tmp/wrapper_mqpu_6gpu.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
EOF
chmod +x /tmp/wrapper_mqpu_6gpu.sh

echo "=================================================="
echo "STEP 1: Run GQE baseline on 6 GPUs (2 nodes, 3 GPUs/node)"
echo "=================================================="
srun -n 6 --ntasks-per-node=3 /tmp/wrapper_mqpu_6gpu.sh \
  $PY src/gqe/baselines/run_cudaq_gqe_chunk.py \
    --config configs/experiment_scaling.yaml \
    --task-id $SLURM_PROCID \
    --num-tasks 6 \
    --ham $HAM \
    --out $OUT_DIR/chunk_${SLURM_PROCID}.json \
    --target nvidia --target-option mqpu \
    --max-qubits 25

echo "=================================================="
echo "STEP 2: Merge chunk results"
echo "=================================================="
$PY - <<PYEOF
import json, glob, sys
from pathlib import Path

out_dir = Path("$OUT_DIR")
results = []
for chunk in sorted(out_dir.glob("chunk_*.json")):
    with chunk.open("r", encoding="utf-8") as f:
        results.extend(json.load(f).get("results", []))

final = Path("$FINAL_OUT")
with final.open("w", encoding="utf-8") as f:
    json.dump({"results": results}, f, indent=2)
print(f"Merged {len(results)} molecule results into {final}")
PYEOF

echo "=================================================="
echo "GQE baseline on 6 GPUs complete."
echo "  Output: $FINAL_OUT"
echo "=================================================="
