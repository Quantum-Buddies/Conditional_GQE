#!/bin/bash
# Phase 3 Final — Step 4: FMO2 reconstruction
# Usage: bash scripts/phase3/04_run_fmo.sh
set -e

export PY=${PY:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}
export LD_LIBRARY_PATH=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:$LD_LIBRARY_PATH
cd /scratch/kcwp264/Conditional-GQE_materials

OUT_DIR=results/phase3_final/fmo
mkdir -p "$OUT_DIR"

echo "=== Step 4: FMO2 reconstruction of IMePh ==="

# Step 4a: Generate fragment Hamiltonians (if not already present)
FRAG_HAM=results/data/fragments/fmo_hamiltonians.json
if [ ! -f "$FRAG_HAM" ]; then
    echo "[4a] Generating fragment Hamiltonians..."
    $PY src/gqe/data/fragment_molecule.py \
        --molecule imeph \
        --config configs/phase3_final/fmo_imeph.yaml \
        --out "$FRAG_HAM"
else
    echo "[4a] Fragment Hamiltonians already exist: $FRAG_HAM"
fi

# Step 4b: Exact-fragment FMO2 (all fragments via exact diagonalization)
echo ""
echo "[4b] Exact-fragment FMO2..."
$PY src/gqe/eval/run_fmo2.py \
    --fragments "$FRAG_HAM" \
    --method exact \
    --out "$OUT_DIR/fmo2_exact.json" \
    --target nvidia

# Step 4c: H-cGQE simulator FMO2
echo ""
echo "[4c] H-cGQE simulator FMO2..."
$PY src/gqe/eval/run_fmo2.py \
    --fragments "$FRAG_HAM" \
    --method hcgqe \
    --checkpoint results/train/h_cgqe_model_rlqf_phase3.pt \
    --out "$OUT_DIR/fmo2_hcgqe.json" \
    --target nvidia --target-option mqpu

# Step 4d: Error decomposition
echo ""
echo "[4d] Computing error decomposition..."
$PY src/gqe/eval/fmo2_error_decomposition.py \
    --exact "$OUT_DIR/fmo2_exact.json" \
    --hcgqe "$OUT_DIR/fmo2_hcgqe.json" \
    --out "$OUT_DIR/fmo2_error_decomposition.json"

echo ""
echo "=== FMO2 complete ==="
echo "Results: $OUT_DIR/"
