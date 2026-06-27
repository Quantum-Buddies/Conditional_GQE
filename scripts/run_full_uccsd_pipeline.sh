#!/bin/bash
# Full end-to-end pipeline with FIXED UCCSD operator pool.
#
# Steps:
#   1. GQE baseline with UCCSD pool (3 GPUs, nvidia-mqpu) — generates training data
#   2. Prepare supervised dataset from GQE output
#   3. Train H-cGQE Transformer on corrected UCCSD data (GPU)
#   4. Inference: generate circuits with retrained model
#   5. Optimize coefficients with L-BFGS-B (3 GPUs)
#   6. Evaluate: compare H-cGQE vs GQE baseline vs FCI reference
#
# Usage: on a GPU node with 3 L40S GPUs:
#   bash scripts/run_full_uccsd_pipeline.sh
set -e

export PY=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python
export LD_LIBRARY_PATH=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:$LD_LIBRARY_PATH

cd /scratch/kcwp264/Conditional-GQE_materials

HAM=results/data/hamiltonians_scaling.json/hamiltonians.json
GQE_OUT=results/baselines/cudaq_gqe_uccsd_3gpu.json
DATASET_OUT=results/train/uccsd_dataset
MODEL_OUT=results/train/h_cgqe_uccsd_model.pt
INFER_OUT=results/inference/h_cgqe_uccsd_inference.json
OPT_OUT=results/eval/h_cgqe_uccsd_optimized.json
EVAL_OUT=results/eval/h_cgqe_uccsd_evaluation.json

# Scale factors from GPT-QE paper: T = {±2^k/320} for k=0..5
SCALE_FACTORS="0.003125 -0.003125 0.00625 -0.00625 0.0125 -0.0125 0.025 -0.025 0.05 -0.05 0.1 -0.1"

# Molecules to evaluate
MOLECULES="h2_0.74 lih_1.6_full n2_1.1_full beh2_1.3_full iodobenzene_cas12 methyl_iodide_cas12 imeph_cas12 phenol_cas12 lih_1.6_631g n2_1.1_631g_cas8 h2o_1.0_631g_cas8"

echo "=================================================="
echo "STEP 1: GQE baseline with UCCSD pool (3 GPUs)"
echo "=================================================="
if [ -f "$GQE_OUT" ]; then
    echo "GQE baseline already exists at $GQE_OUT, skipping."
else
    $PY src/gqe/baselines/run_cudaq_gqe_mqpu.py \
        --ham $HAM \
        --out $GQE_OUT \
        --target nvidia --target-option mqpu \
        --max-qubits 24 \
        --max-iters 25 \
        --ngates 10 \
        --pool-scale $SCALE_FACTORS
fi

echo ""
echo "=================================================="
echo "STEP 2: SKIPPED (pure RL from scratch — no supervised dataset needed)"
echo "=================================================="

echo ""
echo "=================================================="
echo "STEP 3: SKIPPED (pure RL from scratch — no supervised pretraining)"
echo "  arXiv:2502.19402: RL from scratch outperforms SFT-then-RL"
echo "  SFT memorizes patterns; RL discovers general strategies"
echo "=================================================="

RL_MODEL_OUT=results/train/h_cgqe_rl_from_scratch.pt

echo ""
echo "=================================================="
echo "STEP 3b: Pure RL from Scratch with DAPO (3 GPUs)"
echo "  300 epochs, BF16, clip-higher, dynamic sampling, entropy bonus,"
echo "  top-p, adaptive eps, REPO advantages, curriculum learning"
echo "  NO supervised pretraining — model learns from energy rewards only"
echo "=================================================="
$PY src/gqe/models/train_rl_dapo.py \
    --from-scratch \
    --hamiltonians $HAM \
    --molecules $MOLECULES \
    --out $RL_MODEL_OUT \
    --epochs 300 \
    --n-samples 50 \
    --lr 3e-4 \
    --temperature 1.0 \
    --d-model 256 \
    --nhead 8 \
    --encoder-layers 4 \
    --decoder-layers 6 \
    --dim-feedforward 1024 \
    --dropout 0.1 \
    --clip-low 0.2 \
    --clip-high 0.28 \
    --dynamic-sampling \
    --top-p 0.9 \
    --entropy-coef 0.01 \
    --adaptive-temp \
    --min-temp 0.7 \
    --max-temp 2.0 \
    --target-entropy 1.5 \
    --explore-eps 0.3 \
    --adaptive-eps \
    --repo-beta 0.05 \
    --curriculum \
    --curriculum-warmup 30 \
    --curriculum-steps 3 \
    --use-bf16 \
    --w-energy 1.0 \
    --w-entangle 0.1 \
    --w-depth 0.05 \
    --w-commute 0.05 \
    --w-diversity 0.2 \
    --target-len 10 \
    --freq-penalty 1.0 \
    --buffer-size 1000 \
    --target nvidia \
    --target-option mqpu \
    --theta 0.01 \
    --max-qubits 24 \
    --use-cuda \
    --multi-gpu \
    --force-entanglement \
    --max-repeat 4

# Use RL-tuned model for inference
if [ -f "$RL_MODEL_OUT" ]; then
    INFER_MODEL=$RL_MODEL_OUT
    echo "Using RL from-scratch model for inference: $RL_MODEL_OUT"
else
    echo "RL model not found at $RL_MODEL_OUT! Exiting."
    exit 1
fi

echo ""
echo "=================================================="
echo "STEP 4: Inference — generate circuits with retrained model"
echo "=================================================="
$PY src/gqe/models/infer_h_cgqe.py \
    --checkpoint $INFER_MODEL \
    --hamiltonians $HAM \
    --out $INFER_OUT \
    --molecules $MOLECULES \
    --n-samples 50 \
    --sample \
    --use-cuda \
    --max-pauli-len 22 \
    --max-seq-len 64 \
    --temperature 1.0 \
    --force-entanglement \
    --freq-penalty 1.0 \
    --max-repeat 4

echo ""
echo "=================================================="
echo "STEP 5: Optimize coefficients with L-BFGS-B (3 GPUs)"
echo "=================================================="
$PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated $INFER_OUT \
    --hamiltonians $HAM \
    --out $OPT_OUT \
    --top-k 5 \
    --target nvidia --target-option mqpu

echo ""
echo "=================================================="
echo "STEP 6: Evaluate H-cGQE vs UCCSD GQE baseline"
echo "=================================================="
$PY src/gqe/eval/evaluate_h_cgqe.py \
    --generated $INFER_OUT \
    --baseline $GQE_OUT \
    --hamiltonians $HAM \
    --out $EVAL_OUT \
    --target nvidia --target-option mqpu

echo ""
echo "=================================================="
echo "PIPELINE COMPLETE"
echo "=================================================="
echo "Output files:"
echo "  GQE baseline:  $GQE_OUT"
echo "  Dataset:       $DATASET_OUT/gqe_supervised_dataset.pt"
echo "  Model:         $MODEL_OUT"
echo "  RL Model:      $RL_MODEL_OUT"
echo "  Inference:     $INFER_OUT"
echo "  Optimized:     $OPT_OUT"
echo "  Evaluation:    $EVAL_OUT"
echo ""
echo "Quick comparison:"
$PY -c "
import json
with open('$EVAL_OUT') as f:
    data = json.load(f)
if isinstance(data, list):
    results = data
else:
    results = data.get('results', [])
print(f'{\"Molecule\":<28s} {\"Ref\":>12s} {\"GQE\":>12s} {\"H-cGQE\":>12s} {\"Err(mHa)\":>10s} {\"Imprv\":>10s}')
print('-'*90)
for r in results:
    name = r.get('molecule', r.get('system', '?'))
    ref = r.get('reference_energy')
    gqe = r.get('baseline_energy')
    hcgqe = r.get('best_generated_energy')
    err = r.get('error_vs_reference')
    imprv = r.get('improvement_over_baseline')
    ref_s = f'{ref:.4f}' if ref else 'N/A'
    gqe_s = f'{gqe:.4f}' if gqe else 'N/A'
    hcgqe_s = f'{hcgqe:.4f}' if hcgqe else 'N/A'
    err_s = f'{err*1000:.2f}' if err else 'N/A'
    imprv_s = f'{imprv*1000:.2f}' if imprv else 'N/A'
    print(f'{name:<28s} {ref_s:>12s} {gqe_s:>12s} {hcgqe_s:>12s} {err_s:>10s} {imprv_s:>10s}')
print()
print('Chemical accuracy threshold: 1.6 mHa')
"
