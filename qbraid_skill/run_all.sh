#!/usr/bin/env bash
# Run the full H-cGQE Phase 3 pipeline on qBraid.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

bash "${PROJECT_ROOT}/scripts/run_h_cgqe_qbraid.sh" "$@"
