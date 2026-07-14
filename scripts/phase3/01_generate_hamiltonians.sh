#!/bin/bash
# Phase 3 Final — Step 1: Generate molecular Hamiltonians
# Usage: bash scripts/phase3/01_generate_hamiltonians.sh
set -e

export PY=${PY:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}
export LD_LIBRARY_PATH=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:$LD_LIBRARY_PATH
cd /scratch/kcwp264/Conditional-GQE_materials

echo "=== Step 1: Generate Hamiltonians ==="

# Phase 3 molecule set (already generated, verify existence)
HAM_PHASE3=results/data/hamiltonians_phase3.json/hamiltonians.json
HAM_SCALING=results/data/hamiltonians_scaling.json/hamiltonians.json
HAM_40PLUS=results/data/hamiltonians_40plus.json/hamiltonians.json

for f in "$HAM_PHASE3" "$HAM_SCALING" "$HAM_40PLUS"; do
    if [ -f "$f" ]; then
        echo "  EXISTS: $f"
    else
        echo "  MISSING: $f — run src/gqe/data/generate_hamiltonians.py to generate"
        exit 1
    fi
done

echo ""
echo "All Hamiltonian data available."
