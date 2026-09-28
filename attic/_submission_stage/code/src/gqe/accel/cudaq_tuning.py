"""CUDA-Q environment tuning for maximum GPU performance.

Sets environment variables BEFORE cudaq import to control:
  - Gate fusion level (CUDAQ_MGPU_FUSE, CUDAQ_FUSION_MAX_QUBITS)
  - Memory pool (CUDAQ_ENABLE_MEMPOOL)
  - Multi-GPU QPU count (CUDAQ_MQPU_NGPUS)
  - GPU memory limits (CUDAQ_MAX_GPU_MEMORY_GB)
  - Diagonal gate fusion (CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS)

Must be called BEFORE `import cudaq` for env vars to take effect.
"""
from __future__ import annotations

import os
import functools


_DEFAULTS = {
    "CUDAQ_ENABLE_MEMPOOL": "1",
    "CUDAQ_FUSION_MAX_QUBITS": "6",
    "CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS": "-1",
    "CUDAQ_MAX_GPU_MEMORY_GB": "NONE",
}

_MGPU_DEFAULTS = {
    "CUDAQ_MGPU_FUSE": "4",
    "CUDAQ_MGPU_NQUBITS_THRESH": "25",
}

_applied = False


def apply_cudaq_env(
    fusion_level: int = 4,
    max_fusion_qubits: int = 6,
    enable_mempool: bool = True,
    mgpu_fuse: int | None = None,
    n_gpus: int | None = None,
    max_gpu_memory_gb: str = "NONE",
    diagonal_gate_max: int = -1,
) -> dict[str, str]:
    """Set CUDA-Q environment variables for optimal performance.

    Args:
        fusion_level: Gate fusion level for mgpu backend (default 4).
            Higher = more gates fused = fewer matrix ops but more memory.
            Tune per-application: 4-6 is good for VQE-style circuits.
        max_fusion_qubits: Max qubits for gate fusion (default 6).
            L40S has 48GB — can handle 6-qubit fusion windows.
        enable_mempool: Enable CUDA memory pool for fast alloc/dealloc.
        mgpu_fuse: Override for CUDAQ_MGPU_FUSE (default = fusion_level).
        n_gpus: Number of GPUs for CUDAQ_MQPU_NGPUS. None = auto-detect.
        max_gpu_memory_gb: GPU memory limit. "NONE" = unlimited.
        diagonal_gate_max: Max qubits for diagonal gate fusion. -1 = auto.

    Returns:
        Dict of env vars that were set.
    """
    global _applied
    env: dict[str, str] = {}

    env["CUDAQ_ENABLE_MEMPOOL"] = "1" if enable_mempool else "0"
    env["CUDAQ_FUSION_MAX_QUBITS"] = str(max_fusion_qubits)
    env["CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS"] = str(diagonal_gate_max)
    env["CUDAQ_MAX_GPU_MEMORY_GB"] = max_gpu_memory_gb

    if mgpu_fuse is not None:
        env["CUDAQ_MGPU_FUSE"] = str(mgpu_fuse)
    else:
        env["CUDAQ_MGPU_FUSE"] = str(fusion_level)

    if n_gpus is not None:
        env["CUDAQ_MQPU_NGPUS"] = str(n_gpus)

    for key, val in env.items():
        os.environ[key] = val

    _applied = True
    return env


def apply_for_l40s(n_gpus: int = 3) -> dict[str, str]:
    """Apply tuned defaults for AIRE L40S GPUs (48GB, PCIe, no NVLink).

    Key tuning choices:
      - fusion_level=4: Good balance for VQE circuits (2-4 qubit gates)
      - max_fusion_qubits=6: L40S has enough VRAM for 6-qubit fusion windows
      - mempool=ON: Reduces allocation overhead for repeated observe() calls
      - mgpu_fuse=4: Matches fusion_level for multi-GPU mode
      - CUDAQ_MQPU_NGPUS=3: One virtual QPU per L40S
    """
    return apply_cudaq_env(
        fusion_level=4,
        max_fusion_qubits=6,
        enable_mempool=True,
        mgpu_fuse=4,
        n_gpus=n_gpus,
        max_gpu_memory_gb="NONE",
        diagonal_gate_max=-1,
    )


def apply_for_b200(n_gpus: int = 1) -> dict[str, str]:
    """Apply tuned defaults for NVIDIA B200 (192GB, NVLink).

    B200 has massive VRAM and NVLink — can use aggressive fusion.
    """
    return apply_cudaq_env(
        fusion_level=6,
        max_fusion_qubits=8,
        enable_mempool=True,
        mgpu_fuse=6,
        n_gpus=n_gpus,
        max_gpu_memory_gb="NONE",
        diagonal_gate_max=-1,
    )


def is_applied() -> bool:
    """Check if env vars have been applied."""
    return _applied


def ensure_applied(n_gpus: int | None = None) -> dict[str, str]:
    """Apply env vars once (idempotent). Auto-detects GPU count."""
    if _applied:
        return {}
    if n_gpus is None:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            n_gpus = int(result.stdout.strip()) if result.returncode == 0 else 1
        except Exception:
            n_gpus = 1
    return apply_for_l40s(n_gpus=n_gpus)
