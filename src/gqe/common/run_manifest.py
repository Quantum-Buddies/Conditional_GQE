"""Run manifest utilities for reproducible experiment tracking.

Every experiment produces an immutable JSON manifest with:
- git commit / dirty state
- Python / CUDA-Q / qBraid / Qiskit / PySCF versions
- pip freeze snapshot
- GPU topology (nvidia-smi)
- command line, seed, UTC timestamps
- input file hashes

No API keys are ever stored in manifests.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _git_commit() -> str:
    return _run(["git", "rev-parse", "HEAD"])


def _git_dirty() -> bool:
    out = _run(["git", "status", "--porcelain"])
    return len(out) > 0


def _git_diff_stat() -> str:
    return _run(["git", "diff", "--stat"])


def _python_version() -> str:
    return sys.version


def _import_version(mod: str) -> str:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except ImportError:
        return "not-installed"


def _pip_freeze() -> str:
    return _run([sys.executable, "-m", "pip", "freeze"], timeout=60)


def _nvidia_smi() -> str:
    return _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], timeout=10)


def _file_hash(path: str | Path) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run_manifest(
    *,
    command: str,
    seed: int | None = None,
    input_files: list[str | Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a run manifest dictionary.

    Args:
        command: The command line that produced this run.
        seed: Random seed used, if any.
        input_files: Files whose hashes should be recorded.
        extra: Additional metadata to merge in.

    Returns:
        Manifest dict ready for JSON serialization.
    """
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": uuid.uuid4().hex[:12],
        "timestamp_utc": _utc_now(),
        "command": command,
        "seed": seed,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "git_diff_stat": _git_diff_stat(),
        "python_version": _python_version(),
        "platform": platform.platform(),
        "versions": {
            "cudaq": _import_version("cudaq"),
            "qbraid": _import_version("qbraid"),
            "qiskit": _import_version("qiskit"),
            "pyscf": _import_version("pyscf"),
            "torch": _import_version("torch"),
            "numpy": _import_version("numpy"),
            "openfermion": _import_version("openfermion"),
        },
        "gpu_info": _nvidia_smi(),
        "pip_freeze": _pip_freeze(),
        "input_hashes": {},
    }

    if input_files:
        for f in input_files:
            fp = Path(f)
            manifest["input_hashes"][str(fp)] = _file_hash(fp)

    if extra:
        manifest.update(extra)

    return manifest


def save_run_manifest(
    manifest: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Save manifest as immutable JSON.

    If the target file already exists, a numeric suffix is appended
    to prevent overwriting.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        stem = p.stem
        suffix = p.suffix
        i = 1
        while p.exists():
            p = p.parent / f"{stem}_{i}{suffix}"
            i += 1

    with p.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return p


def attach_manifest_to_result(
    result: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Attach a run manifest to a result dict under the 'run_manifest' key."""
    result["run_manifest"] = manifest
    return result
