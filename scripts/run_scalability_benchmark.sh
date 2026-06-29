#!/bin/bash
# Scalability benchmark for H-cGQE pipeline (GIC Mitsubishi Challenge).
#
# Sweeps molecule sizes from 4 to 20 qubits, runs the full pipeline
# (inference + L-BFGS-B optimization + evaluation) for each, and
# generates a scalability report showing:
#   - Energy accuracy vs molecule size (qubits)
#   - Wall-clock time vs molecule size
#   - GPU parallelism efficiency
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

mkdir -p $SCALING_DIR

# Molecule sweep ordered by qubit count (small → large)
# Format: "name:n_qubits:description:backend"
# Backend: mqpu = statevector nvidia-mqpu (≤24q), mps = tensornet-mps (>24q)
MOLECULES_SWEEP=(
    "h2_0.74:4:H2 at equilibrium:mqpu"
    "lih_1.6_full:12:LiH full STO-3G:mqpu"
    "beh2_1.3_full:14:BeH2 full STO-3G:mqpu"
    "n2_1.1_full:20:N2 full STO-3G:mqpu"
    "n2_ccpvdz:32:N2 cc-pVDZ frozen core:mps"
    "beh2_ccpvdz:30:BeH2 cc-pVDZ:mps"
    "ethylene:28:Ethylene STO-3G:mps"
    "n2_ccpvdz_full:40:N2 cc-pVDZ full:mps"
    "benzene_cas20:40:Benzene CAS(12e,20o):mps"
)

echo "=================================================="
echo "GIC Scalability Benchmark"
echo "  Model: $RL_MODEL"
echo "  Molecules: ${#MOLECULES_SWEEP[@]} (4 → 40+ qubits)"
echo "  Backends: nvidia-mqpu (≤24q) + tensornet-mps (>24q)"
echo "  GPUs: 3x L40S"
echo "=================================================="

# Run inference + optimization for each molecule individually
# to measure per-molecule timing
RESULTS_JSON="[]"
for entry in "${MOLECULES_SWEEP[@]}"; do
    IFS=':' read -r MOL NQUBITS DESC BACKEND <<< "$entry"
    echo ""
    echo "--- $MOL ($NQUBITS qubits): $DESC [backend: $BACKEND] ---"

    INFER_OUT=$SCALING_DIR/infer_${MOL}.json
    OPT_OUT=$SCALING_DIR/opt_${MOL}.json

    # Select CUDA-Q backend based on qubit count
    if [ "$BACKEND" = "mps" ]; then
        CUDAQ_TARGET="tensornet-mps"
        CUDAQ_OPT=""
    else
        CUDAQ_TARGET="nvidia"
        CUDAQ_OPT="--target-option mqpu"
    fi

    # Step 1: Inference
    echo "  Inference ($CUDAQ_TARGET)..."
    T0=$(date +%s.%N)
    $PY src/gqe/models/infer_h_cgqe.py \
        --checkpoint $RL_MODEL \
        --hamiltonians $HAM \
        --out $INFER_OUT \
        --molecules $MOL \
        --n-samples 100 --sample --use-cuda \
        --max-pauli-len 22 --max-seq-len 128 \
        --temperature 1.0 \
        --force-entanglement --freq-penalty 1.0 --max-repeat 4
    T1=$(date +%s.%N)
    INFER_TIME=$(echo "$T1 - $T0" | bc)

    # Step 2: L-BFGS-B coefficient optimization
    echo "  L-BFGS-B optimization ($CUDAQ_TARGET)..."
    T0=$(date +%s.%N)
    $PY src/gqe/eval/optimize_h_cgqe_coefficients.py \
        --generated $INFER_OUT \
        --hamiltonians $HAM \
        --out $OPT_OUT \
        --top-k 10 \
        --target $CUDAQ_TARGET $CUDAQ_OPT \
        --max-iter 200 --max-qubits 48
    T1=$(date +%s.%N)
    OPT_TIME=$(echo "$T1 - $T0" | bc)

    # Extract best energy
    BEST_E=$($PY -c "
import json
with open('$OPT_OUT') as f:
    data = json.load(f)
results = data.get('results', data) if isinstance(data, dict) else data
if isinstance(results, list):
    best = min(r.get('optimized_energy', r.get('best_energy', 0)) for r in results)
else:
    best = results.get('optimized_energy', 0)
print(f'{best:.6f}')
")

    TOTAL_TIME=$(echo "$INFER_TIME + $OPT_TIME" | bc)
    echo "  Result: E=$BEST_E Ha  time=${TOTAL_TIME}s"

    # Append to results JSON
    RESULTS_JSON=$($PY -c "
import json
results = $RESULTS_JSON
results.append({
    'molecule': '$MOL',
    'n_qubits': $NQUBITS,
    'description': '$DESC',
    'best_energy': float('$BEST_E'),
    'infer_time_s': float('$INFER_TIME'),
    'optimize_time_s': float('$OPT_TIME'),
    'total_time_s': float('$TOTAL_TIME'),
})
print(json.dumps(results))
")
done

# Get GQE baseline energies for comparison
echo ""
echo "=================================================="
echo "Generating scalability report..."
echo "=================================================="

$PY -c "
import json

results = $RESULTS_JSON

# Load GQE baseline
with open('$GQE_OUT') as f:
    gqe = json.load(f)
gqe_map = {r['system']: r for r in gqe['results']}

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
        'n_gpus': 3,
        'gpu_type': 'L40S',
        'backend': 'nvidia-mqpu',
        'results': results,
    }, f, indent=2)

# Print summary table
print()
print('=' * 100)
print('GIC SCALABILITY BENCHMARK REPORT')
print('=' * 100)
print(f'{\"Molecule\":<20s} {\"Qubits\":>6s} {\"H-cGQE (Ha)\":>14s} {\"GQE (Ha)\":>14s} {\"Ref (Ha)\":>14s} {\"Err(mHa)\":>10s} {\"Imprv(mHa)\":>12s} {\"Time(s)\":>10s}')
print('-' * 100)
for r in results:
    mol = r['molecule']
    nq = r['n_qubits']
    e = r['best_energy']
    gqe = r.get('gqe_baseline_energy')
    ref = r.get('reference_energy')
    err = r.get('error_vs_ref_mHa')
    imprv = r.get('improvement_over_gqe')
    t = r['total_time_s']
    gqe_s = f'{gqe:.6f}' if gqe else 'N/A'
    ref_s = f'{ref:.6f}' if ref else 'N/A'
    err_s = f'{err:.2f}' if err else 'N/A'
    imprv_s = f'{imprv*1000:.2f}' if imprv else 'N/A'
    print(f'{mol:<20s} {nq:>6d} {e:>14.6f} {gqe_s:>14s} {ref_s:>14s} {err_s:>10s} {imprv_s:>12s} {t:>10.1f}')
print()
print('Chemical accuracy threshold: 1.6 mHa')
print(f'Report saved to: $REPORT_OUT')
"

echo ""
echo "=================================================="
echo "SCALABILITY BENCHMARK COMPLETE"
echo "=================================================="
