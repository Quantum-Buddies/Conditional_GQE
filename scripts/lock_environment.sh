#!/usr/bin/env bash
# Lock the computational environment for GIC Phase 3 reproducibility.
#
# Usage:
#   bash scripts/lock_environment.sh [OUTPUT_DIR]
#
# Default output: results/gic2026/manifests/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${1:-${PROJECT_ROOT}/results/gic2026/manifests}"

# Use cudaq-env Python by default; allow override via PYTHON env var
PYTHON="${PYTHON:-/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}"

mkdir -p "${OUT_DIR}"

TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
PREFIX="${OUT_DIR}/env_${TIMESTAMP}"

echo "=== Locking environment for GIC Phase 3 ==="
echo "Output prefix: ${PREFIX}"
echo ""

# Git
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${PREFIX}_git_commit.txt"
git -C "${PROJECT_ROOT}" status --porcelain > "${PREFIX}_git_status.txt"
git -C "${PROJECT_ROOT}" diff --stat > "${PREFIX}_git_diff.txt"
echo "  Git commit: $(cat "${PREFIX}_git_commit.txt")"
echo "  Git dirty: $([ -s "${PREFIX}_git_status.txt" ] && echo 'yes' || echo 'no')"

# Python
${PYTHON} --version > "${PREFIX}_python_version.txt" 2>&1
echo "  Python: $(cat "${PREFIX}_python_version.txt")"

# pip freeze
${PYTHON} -m pip freeze > "${PREFIX}_pip_freeze.txt" 2>&1
echo "  pip freeze: $(wc -l < "${PREFIX}_pip_freeze.txt") packages"

# GPU
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${PREFIX}_gpu_info.txt" 2>&1
    echo "  GPU: $(cat "${PREFIX}_gpu_info.txt" | tail -n +2 | wc -l) device(s)"
else
    echo "no GPU detected" > "${PREFIX}_gpu_info.txt"
    echo "  GPU: not available"
fi

# CUDA-Q version
${PYTHON} -c "import cudaq; print(cudaq.__version__)" > "${PREFIX}_cudaq_version.txt" 2>&1 || \
    echo "cudaq not installed" > "${PREFIX}_cudaq_version.txt"
echo "  CUDA-Q: $(cat "${PREFIX}_cudaq_version.txt")"

# qBraid version
${PYTHON} -c "import qbraid; print(qbraid.__version__)" > "${PREFIX}_qbraid_version.txt" 2>&1 || \
    echo "qbraid not installed" > "${PREFIX}_qbraid_version.txt"
echo "  qBraid: $(cat "${PREFIX}_qbraid_version.txt")"

# Qiskit version
${PYTHON} -c "import qiskit; print(qiskit.__version__)" > "${PREFIX}_qiskit_version.txt" 2>&1 || \
    echo "qiskit not installed" > "${PREFIX}_qiskit_version.txt"
echo "  Qiskit: $(cat "${PREFIX}_qiskit_version.txt")"

# PySCF version
${PYTHON} -c "import pyscf; print(pyscf.__version__)" > "${PREFIX}_pyscf_version.txt" 2>&1 || \
    echo "pyscf not installed" > "${PREFIX}_pyscf_version.txt"
echo "  PySCF: $(cat "${PREFIX}_pyscf_version.txt")"

# qBraid API key check (presence only, never the value)
if [ -n "${QBRAID_API_KEY:-}" ]; then
    echo "QBRAID_API_KEY is set (value not recorded)" > "${PREFIX}_qbraid_api_key.txt"
else
    echo "QBRAID_API_KEY is NOT set" > "${PREFIX}_qbraid_api_key.txt"
fi
echo "  qBraid API key: $(cat "${PREFIX}_qbraid_api_key.txt")"

# Available QPUs (if qBraid is installed and key is set)
if ${PYTHON} -c "import qbraid" 2>/dev/null && [ -n "${QBRAID_API_KEY:-}" ]; then
    echo "  Querying available QPUs..."
    ${PYTHON} -c "
from qbraid import QbraidProvider
provider = QbraidProvider()
try:
    devices = provider.get_devices()
    for d in devices:
        print(f'{d.id}|{d.status()}|{d.num_qubits}|{d.provider}')
except Exception as e:
    print(f'Error listing devices: {e}')
" > "${PREFIX}_available_qpus.txt" 2>&1 || echo "Failed to list QPUs" > "${PREFIX}_available_qpus.txt"
    echo "  QPUs listed: $(wc -l < "${PREFIX}_available_qpus.txt") device(s)"
else
    echo "qBraid not available or no API key — skipping QPU listing" > "${PREFIX}_available_qpus.txt"
    echo "  QPUs: skipped (no qBraid or no API key)"
fi

echo ""
echo "=== Environment lock complete ==="
echo "Manifest files in: ${OUT_DIR}/"
echo "Timestamp: ${TIMESTAMP}"
