"""GPU-accelerated primitives for the GQE pipeline.

Modules:
  cudaq_tuning   — Environment variable setup for optimal CUDA-Q performance
  fast_pauli     — Vectorized Pauli word operations using NumPy bit manipulation
  fast_qwc       — Vectorized QWC grouping (replaces O(n²) Python loop)
  gpu_parity     — GPU/Triton kernel for parity computation in QWC result parsing
  batched_optimizer — Multi-GPU batched L-BFGS-B coefficient optimization
"""
