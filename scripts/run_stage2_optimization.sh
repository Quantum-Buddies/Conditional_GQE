#!/bin/bash
set -eo pipefail

PROJECT_ROOT="/scratch/kcwp264/Conditional-GQE_materials"
PYTHON="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"
export CUDAQ_MPS_MAX_BOND=64

cd "${PROJECT_ROOT}"

echo "============================================"
echo "Stage 2: Coefficient Optimization"
echo "============================================"

# Run on both supervised and Chemeleon2 generated circuits
for MODEL in supervised chemeleon2; do
    INPUT="results/inference/${MODEL}_sampled.json"
    OUTPUT="results/eval/${MODEL}_stage2_optimized.json"

    echo ""
    echo "--- Optimizing ${MODEL} circuits ---"
    echo "    Input:  ${INPUT}"
    echo "    Output: ${OUTPUT}"

    ${PYTHON} src/gqe/eval/optimize_h_cgqe_coefficients.py \
        --generated "${INPUT}" \
        --hamiltonians results/data/hamiltonians_merged.json \
        --out "${OUTPUT}" \
        --target nvidia \
        --target-option mqpu \
        --max-iter 200 \
        --top-k 20 \
        --max-qubits 24
done

echo ""
echo "============================================"
echo "Stage 2 Complete — Comparing Results"
echo "============================================"

${PYTHON} -c "
import json

models = {
    'Supervised': 'results/eval/supervised_stage2_optimized.json',
    'Chemeleon2': 'results/eval/chemeleon2_stage2_optimized.json',
}

results = {}
for label, path in models.items():
    with open(path) as f:
        data = json.load(f)
    results[label] = {r['molecule']: r['best_energy'] for r in data if r.get('best_energy') is not None}

molecules = sorted(set().union(*[set(r.keys()) for r in results.values()]))
print(f\"{'Molecule':12s} {'Supervised':>14s} {'Chemeleon2':>14s} {'Delta':>10s}\")
print('-' * 52)
for mol in molecules:
    sup = results['Supervised'].get(mol, float('inf'))
    chm = results['Chemeleon2'].get(mol, float('inf'))
    delta = chm - sup
    print(f'{mol:12s} {sup:14.6f} {chm:14.6f} {delta:+10.6f}')
print()
print('Negative delta = Chemeleon2 found lower energy (better)')
"
