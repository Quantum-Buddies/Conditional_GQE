#!/usr/bin/env bash
# Install dependencies for the H-cGQE qBraid Skill.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Installing H-cGQE qBraid Skill dependencies ==="

pip install -r "${PROJECT_ROOT}/requirements-qbraid.txt"

echo ""
echo "Verifying imports..."
python -c "import torch, cudaq, qiskit, qbraid; print('All core imports OK')"

echo ""
echo "Installation complete."
