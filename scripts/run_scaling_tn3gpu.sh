#!/bin/bash
# Run scaling experiments with tensornet backend across 3x L40S GPUs.
# WARNING: The multi-GPU (MPI) mode of the tensornet backend is NOT supported for
# pip-installed CUDA-Q. It only works with CUDA-Q built from source against a
# CUDA-aware MPI. On this conda/pip installation, using -np > 1 with tensornet will
# segfault. This script is kept as a reference but should NOT be used for 3-GPU runs.
# To utilize all 3 GPUs, use scripts/run_scaling_3gpu.sh (nvidia-mqpu).
# To run tensornet on a single GPU, use -np 1 or run it without mpiexec.
#
# IMPORTANT: The conda env's Open MPI 5.0.10 is built with --with-cuda (CUDA-aware).
# Make sure no system MPI module is loaded (module purge) to avoid library conflicts.
set -e

export PY=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python
cd /scratch/kcwp264/Conditional-GQE_materials

# CUDA-Q MPI plugin (rebuilt via activate_custom_mpi.sh against conda's CUDA-aware Open MPI)
export CUDAQ_MPI_COMM_LIB=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib/python3.11/site-packages/distributed_interfaces/libcudaq_distributed_interface_mpi.so
# Enable logging for debugging
export CUDAQ_LOG_LEVEL=info

# Conda env's Open MPI 5.0.10 is built with --with-cuda (CUDA-aware)
# Use it directly — no system MPI or LD_PRELOAD needed
MPI="mpiexec --oversubscribe -np 3"

HAM=results/data/hamiltonians_scaling.json/hamiltonians.json
GQE_OUT=results/baselines/cudaq_gqe_tn3gpu_scaling.json
INFER_OUT=results/inference/h_cgqe_generated_scaling.json
OPT_OUT=results/eval/h_cgqe_optimized_tn3gpu_scaling.json
EVAL_OUT=results/eval/h_cgqe_evaluation_tn3gpu_scaling.json
RLQF_CKPT=results/train/h_cgqe_model_rlqf_phase3.pt

echo "=================================================="
echo "STEP 1: Verify tensornet 3-GPU backend works"
echo "=================================================="
cat > /tmp/test_tn_3gpu.py << 'PYEOF'
import cudaq
from cudaq import spin

if cudaq.mpi.is_initialized():
    print("MPI already initialized")
else:
    cudaq.mpi.initialize()
    print(f"MPI initialized: rank={cudaq.mpi.rank()}, num_ranks={cudaq.mpi.num_ranks()}")

cudaq.set_target('tensornet')
print(f'tensornet target set successfully on rank {cudaq.mpi.rank()}')

@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])

# Test with 30 qubits — beyond single-GPU statevector limit
ham = spin.z(0)
for i in range(1, 30):
    ham += spin.z(i)

energy = cudaq.observe(ghz, ham, 30)
if cudaq.mpi.rank() == 0:
    print(f'30-qubit GHZ observe succeeded on 3x L40S (tensornet): E={energy.expectation()}')
    print('tensornet 3-GPU backend verified OK')

cudaq.mpi.finalize()
PYEOF
$MPI $PY /tmp/test_tn_3gpu.py

echo "=================================================="
echo "STEP 2: GQE baseline (tensornet, 3 GPUs)"
echo "=================================================="
$MPI $PY src/gqe/baselines/run_cudaq_gqe.py \
    --ham $HAM \
    --out $GQE_OUT \
    --target tensornet \
    --max-qubits 50

echo "=================================================="
echo "STEP 3: H-cGQE inference (RLQF model — single GPU, no MPI needed)"
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
echo "STEP 4: Optimize coefficients (tensornet, 3 GPUs)"
echo "=================================================="
$MPI $PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated $INFER_OUT \
    --hamiltonians $HAM \
    --out $OPT_OUT \
    --top-k 5 \
    --target tensornet

echo "=================================================="
echo "STEP 5: Evaluate H-cGQE (tensornet, 3 GPUs)"
echo "=================================================="
$MPI $PY src/gqe/eval/evaluate_h_cgqe.py \
    --generated $INFER_OUT \
    --baseline $GQE_OUT \
    --hamiltonians $HAM \
    --out $EVAL_OUT \
    --target tensornet

echo "=================================================="
echo "ALL DONE (tensornet 3-GPU). Output files:"
echo "  $GQE_OUT"
echo "  $INFER_OUT"
echo "  $OPT_OUT"
echo "  $EVAL_OUT"
echo "=================================================="
