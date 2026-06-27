#!/bin/bash
# Run scaling experiments with tensornet backend on single L40S GPU.
# Usage: on a GPU node, run:  bash scripts/run_scaling.sh
set -e

export PY=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python
cd /scratch/kcwp264/Conditional-GQE_materials

HAM=results/data/hamiltonians_scaling.json/hamiltonians.json
GQE_OUT=results/baselines/cudaq_gqe_scaling.json
INFER_OUT=results/inference/h_cgqe_generated_scaling.json
OPT_OUT=results/eval/h_cgqe_optimized_scaling.json
EVAL_OUT=results/eval/h_cgqe_evaluation_scaling.json
RLQF_CKPT=results/train/h_cgqe_model_rlqf_phase3.pt

echo "=================================================="
echo "STEP 1: Generate Hamiltonians"
echo "=================================================="
$PY src/gqe/data/generate_hamiltonians.py \
    --config configs/experiment_scaling.yaml \
    --out results/data/hamiltonians_scaling.json

echo "=================================================="
echo "STEP 2: GQE baseline (tensornet)"
echo "=================================================="
$PY src/gqe/baselines/run_cudaq_gqe.py \
    --ham $HAM \
    --out $GQE_OUT \
    --target tensornet \
    --max-qubits 25

echo "=================================================="
echo "STEP 3: H-cGQE inference (RLQF model)"
echo "=================================================="
$PY src/gqe/models/infer_h_cgqe.py \
    --checkpoint $RLQF_CKPT \
    --hamiltonians $HAM \
    --out $INFER_OUT \
    --n-samples 50 --sample --use-cuda \
    --max-pauli-len 22 --max-seq-len 64 \
    --molecules h2_0.74 lih_1.6_full n2_1.1_full beh2_1.3_full \
    iodobenzene_cas12 methyl_iodide_cas12 \
    imeph_cas12 phenol_cas12 \
    lih_1.6_631g n2_1.1_631g_cas8 h2o_1.0_631g_cas8

echo "=================================================="
echo "STEP 4: Optimize coefficients (tensornet)"
echo "=================================================="
$PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated $INFER_OUT \
    --hamiltonians $HAM \
    --out $OPT_OUT \
    --n-sequences 5 \
    --target tensornet

echo "=================================================="
echo "STEP 5: Evaluate H-cGQE (tensornet)"
echo "=================================================="
$PY src/gqe/eval/evaluate_h_cgqe.py \
    --generated $INFER_OUT \
    --hamiltonians $HAM \
    --out $EVAL_OUT \
    --target tensornet

echo "=================================================="
echo "DONE. Output files:"
echo "  $GQE_OUT"
echo "  $INFER_OUT"
echo "  $OPT_OUT"
echo "  $EVAL_OUT"
echo "=================================================="
