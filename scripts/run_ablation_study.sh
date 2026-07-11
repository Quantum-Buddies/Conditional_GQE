#!/bin/bash
# Chemeleon2 Ablation Study: disable individual components to quantify contribution
# Components: KL divergence, MMD diversity, creativity reward
# Each variant runs 200 epochs on 1 GPU (~12 min each based on full Chemeleon2 run)
set -eo pipefail

PROJECT_ROOT="/scratch/kcwp264/Conditional-GQE_materials"
PYTHON="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"
export CUDAQ_MPS_MAX_BOND=64

cd "${PROJECT_ROOT}"

CHECKPOINT="results/train/h_cgqe_uccsd_model.pt"
HAMILTONIANS="results/data/hamiltonians_merged.json"
BASE_ARGS="--checkpoint ${CHECKPOINT} --hamiltonians ${HAMILTONIANS} --molecules h2 lih beh2 n2 --use-cuda --target nvidia --target-option mqpu --epochs 200 --n-samples 32 --max-qubits 24 --use-bf16 --force-entanglement --curriculum --curriculum-warmup 30 --seed 42"

# Ablation variants:
# 1. Full Chemeleon2 (baseline, already have results)
# 2. No KL divergence (kl_coef=0, keep MMD + creativity)
# 3. No MMD diversity (w_mmd_diversity=0, keep KL + creativity)
# 4. No creativity (w_creativity=0, keep KL + MMD)
# 5. No diversity rewards at all (kl only — tests pure anchoring)
# 6. No KL, no diversity (vanilla DAPO — tests if RL alone helps)

declare -A VARIANTS
VARIANTS[full]="--chemeleon2-mode"
VARIANTS[no_kl]="--kl-coef 0.0 --w-creativity 1.0 --w-mmd-diversity 1.0 --clip-low 0.001 --clip-high 0.001 --entropy-coef 1e-5 --use-bf16"
VARIANTS[no_mmd]="--kl-coef 1.0 --w-creativity 1.0 --w-mmd-diversity 0.0 --clip-low 0.001 --clip-high 0.001 --entropy-coef 1e-5 --use-bf16"
VARIANTS[no_creativity]="--kl-coef 1.0 --w-creativity 0.0 --w-mmd-diversity 1.0 --clip-low 0.001 --clip-high 0.001 --entropy-coef 1e-5 --use-bf16"
VARIANTS[kl_only]="--kl-coef 1.0 --w-creativity 0.0 --w-mmd-diversity 0.0 --clip-low 0.001 --clip-high 0.001 --entropy-coef 1e-5 --use-bf16"
VARIANTS[vanilla_dapo]="--kl-coef 0.0 --w-creativity 0.0 --w-mmd-diversity 0.0 --clip-low 0.2 --clip-high 0.28 --entropy-coef 0.0 --use-bf16"

for VARIANT in full no_kl no_mmd no_creativity kl_only vanilla_dapo; do
    OUTPUT="results/train/h_cgqe_rl_ablation_${VARIANT}.pt"
    METRICS="results/train/h_cgqe_rl_ablation_${VARIANT}_rl_metrics.json"
    LOG="results/logs/ablation_${VARIANT}.out"

    echo ""
    echo "============================================"
    echo "Ablation: ${VARIANT}"
    echo "============================================"

    ${PYTHON} src/gqe/models/train_rl_dapo.py \
        ${BASE_ARGS} \
        ${VARIANTS[$VARIANT]} \
        --out "${OUTPUT}" \
        2>&1 | tee "${LOG}"

    echo "Variant ${VARIANT} complete. Output: ${OUTPUT}"
done

echo ""
echo "============================================"
echo "Ablation Study Complete — Generating Summary"
echo "============================================"

${PYTHON} -c "
import json
from pathlib import Path

variants = ['full', 'no_kl', 'no_mmd', 'no_creativity', 'kl_only', 'vanilla_dapo']
labels = {
    'full': 'Full Chemeleon2',
    'no_kl': 'No KL (MMD+Creat)',
    'no_mmd': 'No MMD (KL+Creat)',
    'no_creativity': 'No Creat (KL+MMD)',
    'kl_only': 'KL Only',
    'vanilla_dapo': 'Vanilla DAPO',
}

print(f\"{'Variant':20s} {'mSUN':>6s} {'Diversity':>10s} {'Entropy':>8s} {'H2':>10s} {'LiH':>10s} {'BeH2':>10s} {'N2':>10s}\")
print('-' * 96)

for v in variants:
    path = Path(f'results/train/h_cgqe_rl_ablation_{v}_rl_metrics.json')
    if not path.exists():
        print(f'{labels[v]:20s}  (missing)')
        continue
    with open(path) as f:
        data = json.load(f)
    log = data['train_log']
    final = log[-1]
    msun = final['msun']
    entropy = final['mean_entropy']
    energies = final['best_energies']
    # Compute diversity from unique sequences in last epoch
    print(f\"{labels[v]:20s} {msun:6.3f} {'':>10s} {entropy:8.4f} {energies.get('h2', 0):10.4f} {energies.get('lih', 0):10.4f} {energies.get('beh2', 0):10.4f} {energies.get('n2', 0):10.4f}\")
"
