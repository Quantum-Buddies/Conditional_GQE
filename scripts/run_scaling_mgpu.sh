#!/bin/bash
# Run scaling experiments with nvidia-mgpu backend (2x L40S pooled memory).
# mgpu requires power-of-2 MPI ranks, so we use 2 of the 3 available GPUs.
# WARNING: On the AIRE L40S nodes, the 3 GPUs are connected via PCIe only (no NVLink
# and no direct P2P). CUDA-Q/cuStateVec's distributed statevector mode relies on
# Open MPI's smcuda BTL for GPU-buffer communication, and on this hardware it
# segfaults once the qubit count crosses the distribution threshold (default 25).
# Therefore this script is kept as a working 2-GPU verification only. To actually
# utilize all 3 GPUs, use scripts/run_scaling_3gpu.sh (nvidia-mqpu, 3 independent GPUs).
# Usage: on a GPU node with 3 GPUs, run:  bash scripts/run_scaling_mgpu.sh
set -e

export PY=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python
cd /scratch/kcwp264/Conditional-GQE_materials

# CUDA-Q MPI plugin (rebuilt via activate_custom_mpi.sh against conda's CUDA-aware Open MPI 5.0.10)
export CUDAQ_MPI_COMM_LIB=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib/python3.11/site-packages/distributed_interfaces/libcudaq_distributed_interface_mpi.so
# Disable fabric-based memory sharing (PCIe-only L40S systems)
export UBACKEND_USE_FABRIC_HANDLE=0
# Point cuStateVec to the exact libmpi.so (CUDA-aware Open MPI 5.0.10 in conda env)
export CUDAQ_MGPU_LIB_MPI=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib/libmpi.so
export CUDAQ_MGPU_COMM_PLUGIN_TYPE=OpenMPI
# L40S are PCIe-only (no NVLink). Force OB1 PML + smcuda BTL and disable CUDA IPC
# to force CPU-staged copies and avoid the MPI_ERR_OTHER crash in MPI_Isend.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=smcuda,self,tcp
export OMPI_MCA_btl_smcuda_use_cuda_ipc=0
export OMPI_MCA_btl_smcuda_use_cuda_ipc_same_gpu=0
export OMPI_MCA_pml_ucx_priority=0
export UCX_TLS=^cuda

HAM=results/data/hamiltonians_scaling.json/hamiltonians.json
GQE_OUT=results/baselines/cudaq_gqe_mgpu_scaling.json
INFER_OUT=results/inference/h_cgqe_generated_scaling.json
OPT_OUT=results/eval/h_cgqe_optimized_mgpu_scaling.json
EVAL_OUT=results/eval/h_cgqe_evaluation_mgpu_scaling.json
RLQF_CKPT=results/train/h_cgqe_model_rlqf_phase3.pt
# mgpu requires power-of-2 ranks. Use 2 GPUs (96 GB pooled, ~33 qubit limit)
# Limit CUDA_VISIBLE_DEVICES to 2 GPUs to avoid MPI mapping issues with 3 GPUs
export CUDA_VISIBLE_DEVICES=0,1
MPI="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/mpiexec --oversubscribe --map-by ppr:2:node -np 2"

echo "=================================================="
echo "STEP 1: Verify mgpu backend works"
echo "=================================================="
cat > /tmp/test_mgpu.py << 'PYEOF'
import os, ctypes
# Each MPI rank needs a CUDA context on its assigned GPU BEFORE MPI_Init,
# otherwise Open MPI's smcuda BTL fails to initialize and falls back to TCP
# (which cannot handle GPU buffers).
local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
libcudart = ctypes.CDLL('/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib/libcudart.so')
libcudart.cudaSetDevice(local_rank)
d = ctypes.c_void_p()
libcudart.cudaMalloc(ctypes.byref(d), 4)
libcudart.cudaFree(d)

from mpi4py import MPI  # this calls MPI_Init()
mpi_comm = MPI.COMM_WORLD
print(f"mpi4py initialized: rank={mpi_comm.Get_rank()}, size={mpi_comm.Get_size()}")

import cudaq
from cudaq import spin

if cudaq.mpi.is_initialized():
    print("MPI already initialized")
else:
    cudaq.mpi.initialize()
    print(f"cudaq MPI initialized: rank={cudaq.mpi.rank()}, num_ranks={cudaq.mpi.num_ranks()}")

cudaq.set_target('nvidia', option='mgpu,fp32')
print(f'mgpu target set successfully on rank {cudaq.mpi.rank()}')

@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])

# Use observe (not sample) — sample segfaults on PCIe-only systems.
# Keep n <= 24 so the statevector is NOT distributed across MPI ranks (the
# cuStateVec distribution threshold is 25 by default). Distributed mgpu fails
# on this PCIe-only L40S node due to broken CUDA IPC in Open MPI's smcuda BTL.
N_QUBITS = 24
ham = spin.z(0)
for i in range(1, N_QUBITS):
    ham += spin.z(i)

energy = cudaq.observe(ghz, ham, N_QUBITS)
if cudaq.mpi.rank() == 0:
    print(f'{N_QUBITS}-qubit GHZ observe succeeded on 2x L40S (mgpu): E={energy.expectation()}')
    print('mgpu backend verified OK')

cudaq.mpi.finalize()
PYEOF
$MPI $PY /tmp/test_mgpu.py

echo "=================================================="
echo "STEP 2: GQE baseline (nvidia-mgpu, 2 GPUs, non-distributed)"
echo "=================================================="
# max-qubits <= 24 prevents cuStateVec from entering distributed statevector mode
# (threshold=25), which segfaults on this PCIe-only L40S node.
$MPI $PY src/gqe/baselines/run_cudaq_gqe.py \
    --ham $HAM \
    --out $GQE_OUT \
    --target nvidia --target-option mgpu,fp32 \
    --max-qubits 24

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
echo "STEP 4: Optimize coefficients (nvidia-mgpu, 2 GPUs, non-distributed)"
echo "=================================================="
$MPI $PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated $INFER_OUT \
    --hamiltonians $HAM \
    --out $OPT_OUT \
    --top-k 5 \
    --max-qubits 24 \
    --target nvidia --target-option mgpu,fp32

echo "=================================================="
echo "STEP 5: Evaluate H-cGQE (nvidia-mgpu, 2 GPUs, non-distributed)"
echo "=================================================="
$MPI $PY src/gqe/eval/evaluate_h_cgqe.py \
    --generated $INFER_OUT \
    --baseline $GQE_OUT \
    --hamiltonians $HAM \
    --out $EVAL_OUT \
    --max-qubits 24 \
    --target nvidia --target-option mgpu,fp32

echo "=================================================="
echo "ALL DONE (mgpu). Output files:"
echo "  $GQE_OUT"
echo "  $INFER_OUT"
echo "  $OPT_OUT"
echo "  $EVAL_OUT"
echo "=================================================="
