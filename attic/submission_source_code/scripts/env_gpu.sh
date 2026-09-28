#!/usr/bin/env bash
# =============================================================================
# GPU-Specific CUDA-Q Environment Resolver
# =============================================================================
# Sourced (not executed) by train_rl.sh and evaluate_rl.sh.
# Auto-detects GPU compute capability and sets optimal CUDA-Q env vars.
#
# Usage (inside another script):
#   source "$(dirname "$0")/env_gpu.sh"
# =============================================================================

# --- Resolve script root for fallback use ---
_ENV_GPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_ENV_GPU_ROOT="$(cd "$_ENV_GPU_DIR/.." && pwd)"

# --- Detect GPU Compute Capability ---
_GPU_CC="$(python3 -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'{cap[0]}.{cap[1]}')
else:
    print('0.0')
" 2>/dev/null || echo '0.0')"

_GPU_NAME="$(python3 -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print('unknown')
" 2>/dev/null || echo 'unknown')"

_GPU_VRAM_GB="$(python3 -c "
import torch
if torch.cuda.is_available():
    print(int(torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    print(0)
" 2>/dev/null || echo 0)"

echo "  [env_gpu] GPU: $_GPU_NAME  CC: $_GPU_CC  VRAM: ${_GPU_VRAM_GB}GB"

# --- CUDA-Q Gate Fusion Level (CC-dependent) ---
# CC 9.0  (H100/H200):  fusion 5 (fp32) / 6 (fp64)
# CC 10.0 (B200):       fusion 5 (fp32) / 4 (fp64) + FP32 emulation
# CC 8.0  (A100):       fusion 4 (fp32) / 5 (fp64)
# Others (L40S, etc.):  fusion 4
case "$_GPU_CC" in
    9.0*)
        export CUDAQ_FUSION_MAX_QUBITS="${CUDAQ_FUSION_MAX_QUBITS:-5}"
        export CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS="${CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS:--1}"
        ;;
    10.0*)
        export CUDAQ_FUSION_MAX_QUBITS="${CUDAQ_FUSION_MAX_QUBITS:-5}"
        export CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS="${CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS:--1}"
        export CUDAQ_ALLOW_FP32_EMULATED="${CUDAQ_ALLOW_FP32_EMULATED:-1}"
        export CUBLAS_EMULATE_SINGLE_PRECISION="${CUBLAS_EMULATE_SINGLE_PRECISION:-1}"
        export CUBLAS_EMULATION_STRATEGY="${CUBLAS_EMULATION_STRATEGY:-performant}"
        ;;
    8.0*)
        export CUDAQ_FUSION_MAX_QUBITS="${CUDAQ_FUSION_MAX_QUBITS:-4}"
        export CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS="${CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS:--1}"
        ;;
    *)
        export CUDAQ_FUSION_MAX_QUBITS="${CUDAQ_FUSION_MAX_QUBITS:-4}"
        export CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS="${CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS:--1}"
        ;;
esac

# --- Universal CUDA-Q Settings ---
export CUDAQ_ENABLE_MEMPOOL="${CUDAQ_ENABLE_MEMPOOL:-1}"
export CUDAQ_MAX_CPU_MEMORY_GB="${CUDAQ_MAX_CPU_MEMORY_GB:-0}"
export CUDAQ_MAX_GPU_MEMORY_GB="${CUDAQ_MAX_GPU_MEMORY_GB:-NONE}"
export CUDAQ_FUSION_NUM_HOST_THREADS="${CUDAQ_FUSION_NUM_HOST_THREADS:-$(nproc 2>/dev/null || echo 8)}"
export CUDAQ_MGPU_FUSE="${CUDAQ_MGPU_FUSE:-6}"

# --- PyTorch TF32 ---
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"

# Avoid Dynamo graph breaks on Transformer causal-mask .item() / scalar ops.
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS="${TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS:-1}"

# --- Portable LD_LIBRARY_PATH from pip-installed nvidia packages ---
_NVIDIA_SITE="$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")"
if [ -n "$_NVIDIA_SITE" ] && [ -d "$_NVIDIA_SITE/nvidia" ]; then
    _CUDA_LIBS=""
    for _subdir in cu13 cublas cudnn cufft curand cusolver cusparse nccl cuda_runtime; do
        _libdir="$_NVIDIA_SITE/nvidia/$_subdir/lib"
        [ -d "$_libdir" ] && _CUDA_LIBS="$_libdir:$_CUDA_LIBS"
    done
    export LD_LIBRARY_PATH="${_CUDA_LIBS}${_NVIDIA_SITE}:${LD_LIBRARY_PATH:-}"
fi

# --- Export GPU info for downstream scripts ---
export GPU_CC="$_GPU_CC"
export GPU_NAME="$_GPU_NAME"
export GPU_VRAM_GB="$_GPU_VRAM_GB"
