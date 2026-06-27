#!/bin/bash
# Run GQE baseline with the FIXED UCCSD operator pool on 3x L40S GPUs.
#
# This script replaces the old broken baseline that used Hamiltonian Pauli terms
# as the operator pool (causing diagonal sequence collapse). The new pool is
# built from UCCSD fermionic excitations mapped through Jordan-Wigner — every
# operator contains X/Y, so Z-only collapse is impossible.
#
# Usage: on a GPU node with 3 L40S GPUs:
#   bash scripts/run_uccsd_gqe_3gpu.sh
#
# Or submit via Slurm:
#   sbatch jobs/gqe-suite.slurm
set -e

export PY=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python
export LD_LIBRARY_PATH=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:$LD_LIBRARY_PATH

cd /scratch/kcwp264/Conditional-GQE_materials

HAM=results/data/hamiltonians_scaling.json/hamiltonians.json
GQE_OUT=results/baselines/cudaq_gqe_uccsd_3gpu.json
INFER_OUT=results/inference/h_cgqe_generated_uccsd.json
OPT_OUT=results/eval/h_cgqe_optimized_uccsd.json
EVAL_OUT=results/eval/h_cgqe_evaluation_uccsd.json
RLQF_CKPT=results/train/h_cgqe_model_rlqf_phase3.pt

echo "=================================================="
echo "STEP 1: Verify Hamiltonians exist"
echo "=================================================="
if [ ! -f "$HAM" ]; then
    $PY src/gqe/data/generate_hamiltonians.py \
        --config configs/experiment_scaling.yaml \
        --out results/data/hamiltonians_scaling.json
else
    echo "Hamiltonians already exist at $HAM"
fi

echo "=================================================="
echo "STEP 2: GQE baseline with UCCSD pool (nvidia-mqpu, 3 GPUs)"
echo "  Every operator contains X/Y — no diagonal collapse possible"
echo "=================================================="
$PY src/gqe/baselines/run_cudaq_gqe_mqpu.py \
    --ham $HAM \
    --out $GQE_OUT \
    --target nvidia --target-option mqpu \
    --max-qubits 24 \
    --max-iters 25 \
    --ngates 10 \
    --pool-scale 0.003125 -0.003125 0.00625 -0.00625 0.0125 -0.0125 0.025 -0.025 0.05 -0.05 0.1 -0.1

echo "=================================================="
echo "STEP 3: H-cGQE inference (RLQF model)"
echo "=================================================="
if [ -f "$RLQF_CKPT" ]; then
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
else
    echo "WARNING: RLQF checkpoint not found at $RLQF_CKPT"
    echo "Skipping inference. Run supervised training first, then RLQF."
    echo "To train: $PY src/gqe/models/train_supervised.py --use-cuda"
fi

echo "=================================================="
echo "STEP 4: Optimize coefficients (nvidia-mqpu, 3 GPUs)"
echo "=================================================="
if [ -f "$INFER_OUT" ]; then
    $PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
        --generated $INFER_OUT \
        --hamiltonians $HAM \
        --out $OPT_OUT \
        --top-k 5 \
        --target nvidia --target-option mqpu
else
    echo "Skipping coefficient optimization (no inference output)."
fi

echo "=================================================="
echo "STEP 5: Evaluate H-cGQE vs UCCSD baseline (3 GPUs)"
echo "=================================================="
if [ -f "$INFER_OUT" ]; then
    $PY src/gqe/eval/evaluate_h_cgqe.py \
        --generated $INFER_OUT \
        --baseline $GQE_OUT \
        --hamiltonians $HAM \
        --out $EVAL_OUT \
        --target nvidia --target-option mqpu
else
    echo "Skipping evaluation (no inference output)."
fi

echo "=================================================="
echo "DONE. Output files:"
echo "  GQE baseline:  $GQE_OUT"
echo "  Inference:     $INFER_OUT"
echo "  Optimized:     $OPT_OUT"
echo "  Evaluation:    $EVAL_OUT"
echo "=================================================="
echo ""
echo "Quick comparison:"
$PY -c "
import json
with open('$GQE_OUT') as f:
    data = json.load(f)
for r in data.get('results', []):
    name = r.get('system', '?')
    be = r.get('baseline_energy')
    ref = r.get('reference_energy')
    delta = r.get('delta_energy')
    status = r.get('status', 'ok')
    if be is not None and ref is not None:
        print(f'  {name:30s}  energy={be:.6f}  ref={ref:.6f}  error={delta*1000:.2f} mHa')
    else:
        print(f'  {name:30s}  status={status}')
"
