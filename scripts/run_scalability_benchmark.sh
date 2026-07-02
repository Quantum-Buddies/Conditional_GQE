#!/bin/bash
# Scalability benchmark for H-cGQE pipeline (GIC Mitsubishi Challenge).
#
# Sweeps molecule sizes from 4 to 40+ qubits, runs the full pipeline
# (inference + L-BFGS-B optimization + evaluation) for each.
#
# Strategy:
#   - Small molecules (≤24q): nvidia-mqpu, sequential, uses all 3 GPUs
#   - Large molecules (>24q): tensornet-mps, run 2 in parallel (1 GPU each)
#     40q molecules run sequentially with reduced bond dimension
#   - Skip already-completed molecules
#
# Usage: on a GPU node with 3 L40S GPUs:
#   bash scripts/run_scalability_benchmark.sh [RL_MODEL]
#
# If RL_MODEL is not provided, defaults to results/train/h_cgqe_rl_warmstart.pt
set -e

export PY=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python
export LD_LIBRARY_PATH=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:$LD_LIBRARY_PATH

cd /scratch/kcwp264/Conditional-GQE_materials

HAM=results/data/hamiltonians_scaling.json/hamiltonians.json
GQE_OUT=results/baselines/cudaq_gqe_uccsd_3gpu.json
RL_MODEL=${1:-results/train/h_cgqe_rl_warmstart.pt}
SCALING_DIR=results/scaling_benchmark
REPORT_OUT=$SCALING_DIR/scalability_report.json
N_GPUS=${N_GPUS:-3}
MPS_PARALLEL=${MPS_PARALLEL:-2}  # 2 MPS processes in parallel (avoid OOM)

mkdir -p $SCALING_DIR

# --- Auto-generate missing 40+ qubit Hamiltonians ---
LARGE_MOLS="n2_ccpvdz beh2_ccpvdz ethylene n2_ccpvdz_cas20 benzene_cas20"
NEED_GEN=false
for MOL in $LARGE_MOLS; do
    if ! $PY -c "import json; data=json.load(open('$HAM')); records=data.get('records',[]); assert any(r.get('name')=='$MOL' for r in records)" 2>/dev/null; then
        NEED_GEN=true
        break
    fi
done

if [ "$NEED_GEN" = true ]; then
    echo "=================================================="
    echo "Generating missing 40+ qubit Hamiltonians..."
    echo "  (This uses PySCF — CPU only, may take 10-30 min)"
    echo "=================================================="
    $PY src/gqe/data/generate_hamiltonians.py \
        --config configs/scaling_40plus.yaml \
        --out results/data/hamiltonians_40plus
    # Merge with existing Hamiltonians
    $PY scripts/merge_hamiltonians.py \
        results/data/hamiltonians_merged.json \
        $HAM \
        results/data/hamiltonians_40plus/hamiltonians.json
    HAM=results/data/hamiltonians_merged.json
    echo "Using merged Hamiltonians: $HAM"
fi

# Molecule sweep ordered by qubit count (small → large)
# Format: "name:n_qubits:description:backend"
# Backend: mqpu = statevector nvidia-mqpu (≤24q), mps = tensornet-mps (>24q)
MOLECULES_SWEEP=(
    "h2_0.74:4:H2 at equilibrium:mqpu"
    "lih_1.6_full:12:LiH full STO-3G:mqpu"
    "beh2_1.3_full:14:BeH2 full STO-3G:mqpu"
    "n2_1.1_full:20:N2 full STO-3G:mqpu"
    "ethylene:28:Ethylene STO-3G:mps"
    "n2_ccpvdz:32:N2 cc-pVDZ CAS(10e,16o):mps"
    "beh2_ccpvdz:32:BeH2 cc-pVDZ CAS(6e,16o):mps"
    "benzene_cas20:40:Benzene CAS(12e,20o):mps"
    "n2_ccpvdz_cas20:40:N2 cc-pVDZ CAS(10e,20o):mps"
)

echo "=================================================="
echo "GIC Scalability Benchmark"
echo "  Model: $RL_MODEL"
echo "  Molecules: ${#MOLECULES_SWEEP[@]} (4 → 40+ qubits)"
echo "  Backends: nvidia-mqpu (≤24q) + tensornet-mps (>24q, ${MPS_PARALLEL} parallel)"
echo "  GPUs: ${N_GPUS}x L40S"
echo "=================================================="

# Helper: check if a molecule's results already exist
is_complete() {
    local mol=$1
    [ -f "$SCALING_DIR/opt_${mol}.json" ] && \
      $PY -c "import json; d=json.load(open('$SCALING_DIR/opt_${mol}.json')); r=d.get('results',d); assert r and (isinstance(r,list) and len(r)>0 or isinstance(r,dict) and r)" 2>/dev/null
}

# Helper: run inference + optimization for a single molecule on a specific GPU
run_single_molecule() {
    local MOL=$1 NQUBITS=$2 DESC=$3 BACKEND=$4 GPU_ID=$5
    local INFER_OUT=$SCALING_DIR/infer_${MOL}.json
    local OPT_OUT=$SCALING_DIR/opt_${MOL}.json

    if [ "$BACKEND" = "mps" ]; then
        local CUDAQ_TARGET="tensornet-mps"
        local CUDAQ_OPT="--target-option fp32"
        local TOPK=5
        local NSAMP=50
        local MAXITER=150
        # Set MPS bond dimension based on qubit count to control memory
        if [ $NQUBITS -ge 40 ]; then
            local MPS_BOND=32
        elif [ $NQUBITS -ge 32 ]; then
            local MPS_BOND=48
        else
            local MPS_BOND=64
        fi
        local MPS_ENV="CUDAQ_MPS_MAX_BOND=$MPS_BOND"
    else
        local CUDAQ_TARGET="nvidia"
        local CUDAQ_OPT="--target-option mqpu"
        local TOPK=10
        local NSAMP=100
        local MAXITER=200
    fi

    echo "[GPU $GPU_ID] $MOL ($NQUBITS qubits): $DESC [backend: $BACKEND]"

    local T0=$(date +%s.%N)

    # Step 1: Inference
    if [ "$BACKEND" = "mps" ]; then
        env CUDA_VISIBLE_DEVICES=$GPU_ID $MPS_ENV $PY src/gqe/models/infer_h_cgqe.py \
            --checkpoint $RL_MODEL \
            --hamiltonians $HAM \
            --out $INFER_OUT \
            --molecules $MOL \
            --n-samples $NSAMP --sample --use-cuda \
            --max-pauli-len 22 --max-seq-len 128 \
            --temperature 1.0 \
            --force-entanglement --freq-penalty 1.0 --max-repeat 4 2>&1 | sed "s/^/[GPU $GPU_ID] /"
    else
        CUDA_VISIBLE_DEVICES=$GPU_ID $PY src/gqe/models/infer_h_cgqe.py \
            --checkpoint $RL_MODEL \
            --hamiltonians $HAM \
            --out $INFER_OUT \
            --molecules $MOL \
            --n-samples $NSAMP --sample --use-cuda \
            --max-pauli-len 22 --max-seq-len 128 \
            --temperature 1.0 \
            --force-entanglement --freq-penalty 1.0 --max-repeat 4 2>&1 | sed "s/^/[GPU $GPU_ID] /"
    fi

    # Step 2: L-BFGS-B optimization
    if [ "$BACKEND" = "mps" ]; then
        env CUDA_VISIBLE_DEVICES=$GPU_ID $MPS_ENV $PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
            --generated $INFER_OUT \
            --hamiltonians $HAM \
            --out $OPT_OUT \
            --top-k $TOPK \
            --target $CUDAQ_TARGET $CUDAQ_OPT \
            --max-iter $MAXITER --max-qubits 60 2>&1 | sed "s/^/[GPU $GPU_ID] /"
    else
        CUDA_VISIBLE_DEVICES=$GPU_ID $PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
            --generated $INFER_OUT \
            --hamiltonians $HAM \
            --out $OPT_OUT \
            --top-k $TOPK \
            --target $CUDAQ_TARGET $CUDAQ_OPT \
            --max-iter $MAXITER --max-qubits 60 2>&1 | sed "s/^/[GPU $GPU_ID] /"
    fi

    local T1=$(date +%s.%N)
    local TOTAL_TIME=$(echo "$T1 - $T0" | bc)
    echo "[GPU $GPU_ID] $MOL done in ${TOTAL_TIME}s"
}

# =========================================================
# Phase 1: Small molecules (≤24q) — sequential on all GPUs
# =========================================================
echo ""
echo "=== Phase 1: Small molecules (nvidia-mqpu, all GPUs) ==="

for entry in "${MOLECULES_SWEEP[@]}"; do
    IFS=':' read -r MOL NQUBITS DESC BACKEND <<< "$entry"
    
    # Skip MPS molecules in phase 1
    if [ "$BACKEND" = "mps" ]; then
        continue
    fi

    # Skip completed
    if is_complete "$MOL"; then
        echo "  Skipping $MOL (already complete)"
        continue
    fi

    echo ""
    echo "--- $MOL ($NQUBITS qubits): $DESC [backend: $BACKEND] ---"
    run_single_molecule "$MOL" "$NQUBITS" "$DESC" "$BACKEND" "0,1,2"
done

# =========================================================
# Phase 2: Large molecules (>24q) — 3 in parallel (1 GPU each)
# =========================================================
echo ""
echo "=== Phase 2: Large molecules (tensornet-mps, ${MPS_PARALLEL} parallel) ==="

# Collect MPS molecules that still need to run
MPS_TODO=()
for entry in "${MOLECULES_SWEEP[@]}"; do
    IFS=':' read -r MOL NQUBITS DESC BACKEND <<< "$entry"
    if [ "$BACKEND" = "mps" ] && ! is_complete "$MOL"; then
        MPS_TODO+=("$entry")
    fi
done

if [ ${#MPS_TODO[@]} -eq 0 ]; then
    echo "  All MPS molecules already complete."
else
    echo "  ${#MPS_TODO[@]} molecules to run, ${MPS_PARALLEL} in parallel"
    
    # Run in batches of MPS_PARALLEL
    BATCH_IDX=0
    while [ $BATCH_IDX -lt ${#MPS_TODO[@]} ]; do
        echo ""
        echo "  --- Batch starting at index $BATCH_IDX ---"
        PIDS=()
        
        for GPU_ID in $(seq 0 $((MPS_PARALLEL - 1))); do
            IDX=$((BATCH_IDX + GPU_ID))
            if [ $IDX -ge ${#MPS_TODO[@]} ]; then
                break
            fi
            IFS=':' read -r MOL NQUBITS DESC BACKEND <<< "${MPS_TODO[$IDX]}"
            echo "  Launching $MOL on GPU $GPU_ID"
            run_single_molecule "$MOL" "$NQUBITS" "$DESC" "$BACKEND" "$GPU_ID" &
            PIDS+=($!)
        done

        # Wait for this batch to finish
        echo "  Waiting for batch to complete..."
        for PID in "${PIDS[@]}"; do
            wait $PID
        done
        echo "  Batch complete."
        
        BATCH_IDX=$((BATCH_IDX + MPS_PARALLEL))
    done
fi

# =========================================================
# Phase 3: Collect results and generate report
# =========================================================
echo ""
echo "=================================================="
echo "Generating scalability report..."
echo "=================================================="

$PY -c "
import json
from pathlib import Path

scaling_dir = Path('$SCALING_DIR')
sweep = [
    ('h2_0.74', 4, 'H2 at equilibrium'),
    ('lih_1.6_full', 12, 'LiH full STO-3G'),
    ('beh2_1.3_full', 14, 'BeH2 full STO-3G'),
    ('n2_1.1_full', 20, 'N2 full STO-3G'),
    ('ethylene', 28, 'Ethylene STO-3G'),
    ('n2_ccpvdz', 32, 'N2 cc-pVDZ CAS(10e,16o)'),
    ('beh2_ccpvdz', 32, 'BeH2 cc-pVDZ CAS(6e,16o)'),
    ('benzene_cas20', 40, 'Benzene CAS(12e,20o)'),
    ('n2_ccpvdz_cas20', 40, 'N2 cc-pVDZ CAS(10e,20o)'),
]

results = []
for mol, nq, desc in sweep:
    opt_file = scaling_dir / f'opt_{mol}.json'
    if not opt_file.exists():
        print(f'  {mol}: MISSING')
        continue
    with open(opt_file) as f:
        data = json.load(f)
    res = data.get('results', data) if isinstance(data, dict) else data
    if isinstance(res, list) and res:
        best = min(r.get('optimized_energy', r.get('best_energy', 0)) for r in res)
    elif isinstance(res, dict) and res:
        best = res.get('optimized_energy', res.get('best_energy', 0))
    else:
        print(f'  {mol}: EMPTY')
        continue
    results.append({
        'molecule': mol,
        'n_qubits': nq,
        'description': desc,
        'best_energy': best,
    })
    print(f'  {mol}: {nq}q, E={best:.6f} Ha')

# Load GQE baseline
gqe_map = {}
gqe_file = Path('$GQE_OUT')
if gqe_file.exists():
    with open(gqe_file) as f:
        gqe = json.load(f)
    gqe_map = {r['system']: r for r in gqe.get('results', [])}

# Add baseline and reference energies
for r in results:
    mol = r['molecule']
    if mol in gqe_map:
        r['gqe_baseline_energy'] = gqe_map[mol].get('baseline_energy')
        r['reference_energy'] = gqe_map[mol].get('reference_energy')
        r['gqe_delta'] = gqe_map[mol].get('delta_energy')
        if r['gqe_baseline_energy'] is not None:
            r['improvement_over_gqe'] = r['best_energy'] - r['gqe_baseline_energy']
        if r['reference_energy'] is not None:
            r['error_vs_ref_mHa'] = abs(r['best_energy'] - r['reference_energy']) * 1000

# Save report
with open('$REPORT_OUT', 'w') as f:
    json.dump({
        'model': '$RL_MODEL',
        'n_gpus': $N_GPUS,
        'gpu_type': 'L40S',
        'backends': 'nvidia-mqpu (<=24q) + tensornet-mps (>24q, parallel)',
        'results': results,
    }, f, indent=2)

# Print summary table
print()
print('=' * 100)
print('GIC SCALABILITY BENCHMARK REPORT')
print('=' * 100)
print(f'{\"Molecule\":<22s} {\"Qubits\":>6s} {\"H-cGQE (Ha)\":>14s} {\"GQE (Ha)\":>14s} {\"Ref (Ha)\":>14s} {\"Err(mHa)\":>10s} {\"Imprv(mHa)\":>12s}')
print('-' * 100)
for r in results:
    mol = r['molecule']
    nq = r['n_qubits']
    e = r['best_energy']
    gqe = r.get('gqe_baseline_energy')
    ref = r.get('reference_energy')
    err = r.get('error_vs_ref_mHa')
    imprv = r.get('improvement_over_gqe')
    gqe_s = f'{gqe:.6f}' if gqe else 'N/A'
    ref_s = f'{ref:.6f}' if ref else 'N/A'
    err_s = f'{err:.2f}' if err else 'N/A'
    imprv_s = f'{imprv*1000:.2f}' if imprv else 'N/A'
    print(f'{mol:<22s} {nq:>6d} {e:>14.6f} {gqe_s:>14s} {ref_s:>14s} {err_s:>10s} {imprv_s:>12s}')
print()
print('Chemical accuracy threshold: 1.6 mHa')
print(f'Report saved to: $REPORT_OUT')
"

echo ""
echo "=================================================="
echo "SCALABILITY BENCHMARK COMPLETE"
echo "=================================================="
