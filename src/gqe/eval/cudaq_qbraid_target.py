"""CUDA-Q qBraid target helper.

Sets ``cudaq.set_target("qbraid", machine=...)`` for qBraid-brokered
simulators and Rigetti QPUs. Never passes ``api_key`` (use ``QBRAID_API_KEY``
in the environment). Hardware machines require ``CUDAQ_ALLOW_QPU=1``.

The qBraid CUDA-Q target (NVIDIA/cuda-quantum#4328) does not support
``emulate=True``; dry runs use ``qbraid:qbraid:sim:qir-sv``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

QIR_SV_MACHINE = "qbraid:qbraid:sim:qir-sv"
CEPHEUS_BRAKET_MACHINE = "aws:rigetti:qpu:cepheus-1-108q"
CEPHEUS_DIRECT_MACHINE = "rigetti:rigetti:qpu:cepheus-1-108q"

DEFAULT_MACHINE = QIR_SV_MACHINE

MACHINE_ALIASES = {
    "qir-sv": QIR_SV_MACHINE,
    "qbraid-sim": QIR_SV_MACHINE,
    "cepheus": CEPHEUS_BRAKET_MACHINE,
    "cepheus-braket": CEPHEUS_BRAKET_MACHINE,
    "cepheus-direct": CEPHEUS_DIRECT_MACHINE,
}


def resolve_machine(machine: str | None = None) -> str:
    """Resolve alias or ``CUDAQ_QBRAID_MACHINE`` to a qBraid device QRN."""
    raw = (machine or os.environ.get("CUDAQ_QBRAID_MACHINE") or DEFAULT_MACHINE).strip()
    return MACHINE_ALIASES.get(raw, raw)


def is_simulator_machine(machine: str) -> bool:
    m = machine.lower()
    return ":sim:" in m or "simulator" in m or "qir-sv" in m


def is_hardware_qbraid_machine(machine: str) -> bool:
    m = machine.lower()
    if is_simulator_machine(m):
        return False
    return ":qpu:" in m or m.startswith("rigetti:") or m.startswith("aws:rigetti")


def is_direct_rigetti(machine: str) -> bool:
    return machine.startswith("rigetti:rigetti:")


def qpu_allowed() -> bool:
    return os.environ.get("CUDAQ_ALLOW_QPU", "").strip() == "1"


def direct_rigetti_allowed() -> bool:
    return os.environ.get("CUDAQ_ALLOW_DIRECT_RIGETTI", "").strip() == "1"


def set_qbraid_machine(
    machine: str | None = None,
    *,
    allow_qpu: bool | None = None,
    cudaq_module: Any | None = None,
) -> str:
    """Set the CUDA-Q target to qBraid.

    Args:
        machine: Device QRN or alias (``qir-sv``, ``cepheus``).
        allow_qpu: Override ``CUDAQ_ALLOW_QPU`` for tests.
        cudaq_module: Injected ``cudaq`` module (defaults to ``import cudaq``).

    Returns:
        Resolved machine id that was set.

    Raises:
        RuntimeError: hardware without ``CUDAQ_ALLOW_QPU=1``, direct Rigetti
            without ``CUDAQ_ALLOW_DIRECT_RIGETTI=1``, or missing CUDA-Q.
    """
    resolved = resolve_machine(machine)
    hardware = is_hardware_qbraid_machine(resolved)
    allowed = qpu_allowed() if allow_qpu is None else bool(allow_qpu)
    if hardware and not allowed:
        raise RuntimeError(
            f"Refusing hardware qBraid machine {resolved!r} without "
            "CUDAQ_ALLOW_QPU=1 (set the env var, never pass api_key in code)."
        )
    if is_direct_rigetti(resolved) and not direct_rigetti_allowed():
        raise RuntimeError(
            f"Refusing per-minute direct Rigetti {resolved!r}. "
            f"Use {CEPHEUS_BRAKET_MACHINE} or set CUDAQ_ALLOW_DIRECT_RIGETTI=1."
        )
    if os.environ.get("CUDAQ_QBRAID_API_KEY_INLINE"):
        raise RuntimeError(
            "Refusing inline API key. Export QBRAID_API_KEY in the environment."
        )

    mod = cudaq_module
    if mod is None:
        try:
            import cudaq as mod
        except ImportError as exc:
            raise RuntimeError("CUDA-Q is required for the qBraid target") from exc

    # Never pass api_key= ; the target reads QBRAID_API_KEY.
    mod.set_target("qbraid", machine=resolved)
    return resolved


def persist_async_result(result: Any, path: Path) -> Path:
    """Write a CUDA-Q async sample handle so a later process can ``.get()``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = str(result)
    path.write_text(payload, encoding="utf-8")
    meta = path.with_suffix(path.suffix + ".meta.json")
    meta.write_text(
        json.dumps({"path": str(path), "kind": "cudaq_async_sample"}, indent=2),
        encoding="utf-8",
    )
    return path


def load_async_result(path: Path, cudaq_module: Any | None = None) -> Any:
    """Rehydrate ``cudaq.AsyncSampleResult`` from a persisted future file."""
    text = Path(path).read_text(encoding="utf-8")
    mod = cudaq_module
    if mod is None:
        import cudaq as mod
    for name in ("AsyncSampleResult", "async_result"):
        cls = getattr(mod, name, None)
        if cls is not None:
            try:
                return cls(text)
            except TypeError:
                continue
    raise RuntimeError(
        f"Could not restore CUDA-Q async sample result from {path}. "
        "Need cudaq.AsyncSampleResult(json_text)."
    )
