#!/usr/bin/env python3
"""qBraid Skill wrapper: install H-cGQE dependencies."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Install the qBraid-compatible requirement set."""
    req_file = ROOT / "requirements-qbraid.txt"
    print(f"Installing dependencies from {req_file}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    print("Verifying core imports...")
    subprocess.check_call([sys.executable, "-c", "import torch, cudaq, qiskit; print('Core imports OK')"])
    print("Installation complete.")


if __name__ == "__main__":
    main()
