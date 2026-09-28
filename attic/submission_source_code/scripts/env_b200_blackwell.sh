#!/usr/bin/env bash
# =============================================================================
# Blackwell (B200) acceleration env — MUST be sourced before `import cudaq`
# and before launching PyTorch training.
#
# Covers:
#   1) cuBLAS BF16x9 FP32 emulation (Tensor Core path for FP32 GEMMs)
#   2) CUDA-Q / cuStateVec Blackwell FP32→BF16 emulation + gate fusion + mempool
#   3) PyTorch TF32 / cuBLAS overrides
# =============================================================================

# --- cuBLAS: BF16x9 emulated FP32 (CUDA 12.9+ / CC 10.0+) ---
# Docs: https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation
export CUBLAS_EMULATE_SINGLE_PRECISION="${CUBLAS_EMULATE_SINGLE_PRECISION:-1}"
export CUBLAS_EMULATION_STRATEGY="${CUBLAS_EMULATION_STRATEGY:-performant}"

# --- CUDA-Q nvidia backend (cuStateVec) Blackwell knobs ---
# Docs: https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html
# IMPORTANT: these are read at cudaq import / first set_target time.
export CUDAQ_ALLOW_FP32_EMULATED="${CUDAQ_ALLOW_FP32_EMULATED:-1}"
export CUDAQ_ENABLE_MEMPOOL="${CUDAQ_ENABLE_MEMPOOL:-1}"
# B200 default fusion for fp32 is 5; keep explicit and allow override.
export CUDAQ_FUSION_MAX_QUBITS="${CUDAQ_FUSION_MAX_QUBITS:-5}"
export CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS="${CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS--1}"
# Circuit preprocessing threads (machine has many CPU cores)
export CUDAQ_FUSION_NUM_HOST_THREADS="${CUDAQ_FUSION_NUM_HOST_THREADS:-16}"
# Use full B200 HBM for statevectors unless caller caps it
export CUDAQ_MAX_GPU_MEMORY_GB="${CUDAQ_MAX_GPU_MEMORY_GB:-NONE}"
export CUDAQ_MAX_CPU_MEMORY_GB="${CUDAQ_MAX_CPU_MEMORY_GB:-0}"

# Multi-GPU fusion knob (harmless on single GPU; used if mgpu ever enabled)
export CUDAQ_MGPU_FUSE="${CUDAQ_MGPU_FUSE:-5}"

# --- PyTorch / cuBLAS TF32 override ---
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}"

# Prefer NVIDIA libs from the torch wheel tree (already set by launcher LD_LIBRARY_PATH)
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
