#!/bin/bash -l
# Run scaling experiments with nvidia-mgpu on 4 GPUs across 2 nodes (2 per node).
# WARNING: This is the closest multi-node setup that respects the mgpu power-of-2
# requirement. The AIRE L40S nodes have 3 GPUs each, so we use 2 per node and
# leave 2 idle. Multi-node GPU communication also requires a CUDA-aware inter-
# node MPI transport (e.g., InfiniBand, UCX, or Cray MPICH with GPU support).
# The conda-forge Open MPI may NOT provide such a transport, so this script
# will likely fail for circuits that require inter-node statevector exchange.
# It is provided as a reference.
# Usage: sbatch scripts/run_scaling_mgpu_2node_4gpu.sh

#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --gres=gpu:l40s:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 2:00:00

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
export PY="${PYTHON}"

# CUDA-Q MPI plugin path (auto-detect)
_CUDAQ_MPI_PLUGIN=$(python -c "import os, cudaq; print(os.path.join(os.path.dirname(cudaq.__file__), '..', 'distributed_interfaces', 'libcudaq_distributed_interface_mpi.so'))" 2>/dev/null || echo "")
if [ -n "${_CUDAQ_MPI_PLUGIN}" ] && [ -f "${_CUDAQ_MPI_PLUGIN}" ]; then
    export CUDAQ_MPI_COMM_LIB="${_CUDAQ_MPI_PLUGIN}"
fi
export UBACKEND_USE_FABRIC_HANDLE=0
_LIBMPI=$(python -c "import os, mpi4py; print(os.path.join(os.path.dirname(mpi4py.__file__), '..', '..', 'lib', 'libmpi.so'))" 2>/dev/null || echo "")
if [ -n "${_LIBMPI}" ] && [ -f "${_LIBMPI}" ]; then
    export CUDAQ_MGPU_LIB_MPI="${_LIBMPI}"
fi
export CUDAQ_MGPU_COMM_PLUGIN_TYPE=OpenMPI

# Open MPI setup for multi-node. The conda TCP BTL is not CUDA-aware, so
# inter-node GPU transfers will fail unless the cluster provides a GPU-aware
# network transport (InfiniBand/UCX). On a pure-Ethernet AIRE node this is
# expected to crash for distributed statevectors.
export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_btl=smcuda,self,tcp
export OMPI_MCA_btl_smcuda_use_cuda_ipc=0
export OMPI_MCA_btl_smcuda_use_cuda_ipc_same_gpu=0
export OMPI_MCA_btl_openib_allow_ib=true
export UCX_TLS=^cuda

HAM=results/data/hamiltonians_scaling.json/hamiltonians.json
GQE_OUT=results/baselines/cudaq_gqe_mgpu_2node_4gpu_scaling.json
INFER_OUT=results/inference/h_cgqe_generated_2node_4gpu_scaling.json
OPT_OUT=results/eval/h_cgqe_optimized_2node_4gpu_scaling.json
EVAL_OUT=results/eval/h_cgqe_evaluation_2node_4gpu_scaling.json
RLQF_CKPT=results/train/h_cgqe_model_rlqf_phase3.pt

# Wrapper: each MPI rank only sees its local GPU.
# SLURM_LOCALID is 0/1 on each node; each node has 2 GPUs allocated via --gres.
cat > /tmp/wrapper_mgpu_4gpu.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
EOF
chmod +x /tmp/wrapper_mgpu_4gpu.sh

SRUN="srun --mpi=pmi2 -n 4 --ntasks-per-node=2 /tmp/wrapper_mgpu_4gpu.sh"

echo "=================================================="
echo "STEP 1: Verify mgpu backend works (4 ranks, 2 nodes)"
echo "=================================================="
cat > /tmp/test_mgpu_4gpu.py << 'PYEOF'
import os, ctypes
local_rank = int(os.environ.get('SLURM_LOCALID', 0))
import cudaq
import ctypes.util
_cudart = ctypes.util.find_library('cudart')
if _cudart is None:
    import os
    for d in ['/usr/local/cuda/lib64', os.environ.get('CONDA_PREFIX', '')]:
        p = os.path.join(d, 'libcudart.so')
        if os.path.exists(p):
            _cudart = p
            break
libcudart = ctypes.CDLL(_cudart)
libcudart.cudaSetDevice(local_rank)
d = ctypes.c_void_p(); libcudart.cudaMalloc(ctypes.byref(d), 4); libcudart.cudaFree(d)

if not cudaq.mpi.is_initialized():
    cudaq.mpi.initialize()
print(f"rank={cudaq.mpi.rank()}, num_ranks={cudaq.mpi.num_ranks()}, local_gpu={local_rank}")

cudaq.set_target('nvidia', option='mgpu,fp32')
print(f'mgpu target set on rank {cudaq.mpi.rank()}')

@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n); h(q[0])
    for i in range(n - 1): x.ctrl(q[i], q[i + 1])

# Keep below the cuStateVec distribution threshold (25) to avoid inter-node
# GPU communication, which is not supported by the conda Open MPI.
N_QUBITS = 24
ham = cudaq.spin.z(0)
for i in range(1, N_QUBITS): ham += cudaq.spin.z(i)
energy = cudaq.observe(ghz, ham, N_QUBITS)
if cudaq.mpi.rank() == 0:
    print(f'{N_QUBITS}-qubit GHZ observe succeeded on 4 GPUs (2 nodes): E={energy.expectation()}')
PYEOF
$SRUN $PY /tmp/test_mgpu_4gpu.py

echo "=================================================="
echo "STEP 2: GQE baseline (nvidia-mgpu, 4 GPUs, 2 nodes)"
echo "=================================================="
$SRUN $PY src/gqe/baselines/run_cudaq_gqe.py \
    --ham $HAM \
    --out $GQE_OUT \
    --target nvidia --target-option mgpu,fp32 \
    --max-qubits 24

echo "=================================================="
echo "STEP 3: H-cGQE inference (single GPU, no MPI)"
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
echo "STEP 4: Optimize coefficients (4 GPUs, 2 nodes)"
echo "=================================================="
$SRUN $PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --generated $INFER_OUT \
    --hamiltonians $HAM \
    --out $OPT_OUT \
    --top-k 5 \
    --max-qubits 24 \
    --target nvidia --target-option mgpu,fp32

echo "=================================================="
echo "STEP 5: Evaluate H-cGQE (4 GPUs, 2 nodes)"
echo "=================================================="
$SRUN $PY src/gqe/eval/evaluate_h_cgqe.py \
    --generated $INFER_OUT \
    --baseline $GQE_OUT \
    --hamiltonians $HAM \
    --out $EVAL_OUT \
    --max-qubits 24 \
    --target nvidia --target-option mgpu,fp32

echo "=================================================="
echo "ALL DONE (4-GPU 2-node mgpu). Output files:"
echo "  $GQE_OUT"
echo "  $INFER_OUT"
echo "  $OPT_OUT"
echo "  $EVAL_OUT"
echo "=================================================="
