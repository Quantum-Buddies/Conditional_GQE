#!/bin/bash
# Offline RL Training (cache-only, 2 GPU) — portable, no Slurm
# Uses precomputed SQLite energy cache. No CUDA-Q required for training.
#
# Prerequisites:
#   - results/train/rl_energy_cache.sqlite (git lfs pull)
#   - results/train/h_cgqe_model_b200_sft.pt (git lfs pull)
#
# Usage: bash scripts/run_rl_offline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET=Socket

ENERGY_CACHE="results/train/rl_energy_cache.sqlite"
SFT_CHECKPOINT="results/train/h_cgqe_model_b200_sft.pt"
HAMILTONIANS="results/data/hamiltonians_gic2026/hamiltonians.json"
OUTPUT="results/train/h_cgqe_model_rl_offline.pt"

echo "=== Offline RL Training (cache-only mode) ==="
echo "Start: $(date)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"

if [ ! -f "$ENERGY_CACHE" ]; then
    echo "ERROR: Energy cache not found. Run: git lfs pull"
    exit 1
fi
if [ ! -f "$SFT_CHECKPOINT" ]; then
    echo "ERROR: SFT checkpoint not found. Run: git lfs pull"
    exit 1
fi

"${PYTHON}" -u src/gqe/models/train_rl_dapo.py \
    --checkpoint "${SFT_CHECKPOINT}" --hamiltonians "${HAMILTONIANS}" \
    --molecules h2 lih beh2 n2 h2o nh3 ch4 ethylene formaldehyde acetylene hf co \
                imeph_cas12 iodobenzene_cas12 methyl_iodide_cas12 phenol_cas12 \
                benzene_cas12 toluene_cas12 anisole_cas12 ocresol_cas12 \
                diarylethene_frag_cas12 \
                h2_0.5 h2_1.0 h2_1.5 h2_2.0 lih_1.2 lih_2.0 lih_3.0 \
                n2_1.8 n2_2.5 beh2_1.0 beh2_1.6 lih_1.6_631g \
                n2_1.1_631g_cas8 h2o_1.0_631g_cas8 \
    --out "${OUTPUT}" --epochs 200 --n-samples 48 --n-iters 4 --lr 1e-5 \
    --temperature 1.0 --top-p 0.9 --target-entropy 1.5 \
    --explore-eps 0.3 --adaptive-eps --force-entanglement --max-repeat 4 \
    --max-qubits 28 --use-cuda --multi-gpu --use-bf16 --dynamic-sampling \
    --token-level-loss --entropy-coef 1e-5 --w-energy 1.0 --w-entangle 0.1 \
    --w-depth 0.05 --w-commute 0.05 --w-diversity 0.2 --target-len 10 \
    --freq-penalty 1.0 --buffer-size 1000 --curriculum --curriculum-warmup 30 \
    --curriculum-steps 3 --qd-mode --qd-novelty-weight 0.3 \
    --energy-cache "${ENERGY_CACHE}" --cache-only 2>&1

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
echo "Output: ${OUTPUT}"
