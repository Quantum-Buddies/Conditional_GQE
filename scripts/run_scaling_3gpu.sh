#!/bin/bash
# Run scaling experiments on 3x L40S GPUs.
# Uses nvidia-mqpu for parallel evaluation across all 3 GPUs. This is the
# recommended way to utilize all 3 GPUs on the AIRE L40S nodes because the
# mgpu backend (pooled memory) and tensornet multi-GPU are not practical or
# not supported on the pip-installed CUDA-Q environment.
# Usage: on a GPU node with 3 GPUs, run:  bash scripts/run_scaling_3gpu.sh
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
echo "STEP 1: Generate Hamiltonians (already done if exists)"
echo "=================================================="
if [ ! -f "$HAM" ]; then
    $PY src/gqe/data/generate_hamiltonians.py \
        --config configs/experiment_scaling.yaml \
        --out results/data/hamiltonians_scaling.json
else
    echo "Hamiltonians already exist at $HAM, skipping."
fi

echo "=================================================="
echo "STEP 2: GQE baseline (nvidia-mqpu, 3 parallel GPUs)"
echo "=================================================="
$PY src/gqe/baselines/run_cudaq_gqe.py \
    --ham $HAM \
    --out $GQE_OUT \
    --target nvidia --target-option mqpu \
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
echo "STEP 4: Optimize coefficients (nvidia-mqpu, 3 GPUs)"
echo "=================================================="
$PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated $INFER_OUT \
    --hamiltonians $HAM \
    --out $OPT_OUT \
    --top-k 5 \
    --target nvidia --target-option mqpu

echo "=================================================="
echo "STEP 5: Evaluate H-cGQE (nvidia-mqpu, 3 GPUs)"
echo "=================================================="
$PY src/gqe/eval/evaluate_h_cgqe.py \
    --generated $INFER_OUT \
    --baseline $GQE_OUT \
    --hamiltonians $HAM \
    --out $EVAL_OUT \
    --target nvidia --target-option mqpu

echo "=================================================="
echo "ALL DONE. Output files:"
echo "  $GQE_OUT"
echo "  $INFER_OUT"
echo "  $OPT_OUT"
echo "  $EVAL_OUT"
echo "=================================================="
