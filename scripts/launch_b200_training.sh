#!/usr/bin/env bash
# =============================================================================
# B200 Training Launcher — Conditional-GQE
# =============================================================================
# Strategy: Supervised warm-start → DAPO RL (main run)
#           Direct RL from scratch  (ablation, secondary)
#
# Hardware: 1× NVIDIA B200 (180 GB HBM3e, sm_100, CUDA 13.2)
#           2× Intel Xeon 6960P (288 threads), 3 TiB RAM
#
# Molecule inventory:
#   Stage 1 SFT  : hamiltonians_merged.json (21 mol, 4–40q)
#   Stage 2 RL   : hamiltonians_gic2026/hamiltonians.json (35 mol, 4–28q)
#   Stage 2 RL xl: hamiltonians_40plus/hamiltonians.json (10 mol, 4–40q)
#
# Usage:
#   bash scripts/launch_b200_training.sh [sft|rl|both|cache|ablation|ablation-smoke|cache+ablation]
#   Default: both (sft then rl)
#
# Fast RL path:
#   bash scripts/launch_b200_training.sh cache          # precompute energies (once)
#   bash scripts/launch_b200_training.sh ablation       # train with cache + reuse-iters=16
#   # or: bash scripts/launch_b200_training.sh cache+ablation
# =============================================================================

set -euo pipefail

STAGE="${1:-both}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$ROOT/results"
DATA="$ROOT/results/data"

# ------------------------------------------------------------------
# Blackwell / B200 acceleration (cuBLAS BF16x9 + CUDA-Q FP32 emulation)
# MUST be sourced before any Python that imports cudaq.
# ------------------------------------------------------------------
# shellcheck source=/dev/null
source "$ROOT/scripts/env_b200_blackwell.sh"
echo "=== Blackwell env ==="
echo "  CUBLAS_EMULATE_SINGLE_PRECISION=$CUBLAS_EMULATE_SINGLE_PRECISION  strategy=$CUBLAS_EMULATION_STRATEGY"
echo "  CUDAQ_ALLOW_FP32_EMULATED=$CUDAQ_ALLOW_FP32_EMULATED  FUSION_MAX_QUBITS=$CUDAQ_FUSION_MAX_QUBITS"
echo "  CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS=$CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS"
echo "  CUDAQ_ENABLE_MEMPOOL=$CUDAQ_ENABLE_MEMPOOL  FUSION_HOST_THREADS=$CUDAQ_FUSION_NUM_HOST_THREADS"
echo "  CUDAQ_MAX_GPU_MEMORY_GB=$CUDAQ_MAX_GPU_MEMORY_GB"
echo "  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=$TORCH_ALLOW_TF32_CUBLAS_OVERRIDE  NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"

# ------------------------------------------------------------------
# Portable CUDA library path resolution (qBraid / containerised env)
# ------------------------------------------------------------------
NVIDIA_SITE="$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)/nvidia"
CUDA_LIBS=""
for subdir in cu13 cublas cudnn cufft curand cusolver cusparse nccl cuda_runtime; do
    libdir="$NVIDIA_SITE/$subdir/lib"
    [ -d "$libdir" ] && CUDA_LIBS="$libdir:$CUDA_LIBS"
done
# Include stub dir for missing optional libs (libcufile, libcupti, libnvshmem)
STUB_DIR="$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)"
export LD_LIBRARY_PATH="${CUDA_LIBS}${STUB_DIR}:${LD_LIBRARY_PATH:-}"

# ------------------------------------------------------------------
# Verify GPU
# ------------------------------------------------------------------
echo "=== GPU check ==="
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check LD_LIBRARY_PATH'
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'  {name}  sm_{cap[0]}{cap[1]}  {vram:.0f} GB VRAM  torch={torch.__version__}')
"

# ------------------------------------------------------------------
# Stage 1: Supervised Fine-Tuning (warm-start)
# ------------------------------------------------------------------
run_sft() {
    echo ""
    echo "=== Stage 1: Supervised warm-start ==="
    echo "    Dataset  : $RESULTS/train/gqe_supervised_dataset.pt"
    echo "    Output   : $RESULTS/train/h_cgqe_model_b200_sft.pt"
    echo "    GPU      : B200 (BF16, batch 1024, val-every 2, fast commute, 500 epochs)"
    echo ""

    python3 "$ROOT/src/gqe/models/train_h_cgqe.py" \
        --dataset "$RESULTS/train/gqe_supervised_dataset.pt" \
        --out "$RESULTS/train/h_cgqe_model_b200_sft.pt" \
        --epochs 500 \
        --batch-size 1024 \
        --lr 5e-4 \
        --d-model 256 \
        --nhead 8 \
        --enc-layers 4 \
        --dec-layers 4 \
        --dim-ff 1024 \
        --dropout 0.1 \
        --train-split 0.8 \
        --use-cuda \
        --use-bf16 \
        --num-workers 0 \
        --grad-accum 1 \
        --val-every 2 \
        --commutator-weight 0.1 \
        --commutator-ramp-epochs 100 \
        --label-smoothing 0.1 \
        --patience 60 \
        --min-delta 1e-4 \
        2>&1 | tee "$RESULTS/train/sft_b200.log"

    echo "SFT complete → $RESULTS/train/h_cgqe_model_b200_sft.pt"
}

# ------------------------------------------------------------------
# Stage 2: DAPO RL fine-tuning (main run)
# ------------------------------------------------------------------
run_rl_main() {
    echo ""
    echo "=== Stage 2: DAPO RL (main — GIC 35-molecule set) ==="
    echo "    Checkpoint: $RESULTS/train/h_cgqe_model_b200_sft.pt"
    echo "    Molecules : 35 GIC2026 systems (4–28q)"
    echo "    Output    : $RESULTS/train/h_cgqe_model_b200_rl_main.pt"
    echo ""

    # Build molecule list from GIC2026 dataset
    MOLECULES=$(python3 -c "
import json
d = json.load(open('$DATA/hamiltonians_gic2026/hamiltonians.json'))
names = [r['name'] for r in d['records']]
print(' '.join(names))
")

    python3 "$ROOT/src/gqe/models/train_rl_dapo.py" \
        --checkpoint "$RESULTS/train/h_cgqe_model_b200_sft.pt" \
        --hamiltonians "$DATA/hamiltonians_gic2026/hamiltonians.json" \
        --molecules $MOLECULES \
        --out "$RESULTS/train/h_cgqe_model_b200_rl_main.pt" \
        --epochs 300 \
        --n-samples 64 \
        --n-iters 5 \
        --lr 1e-5 \
        --clip-low 0.2 \
        --clip-high 0.28 \
        --use-cuda \
        --use-bf16 \
        --single-gpu \
        --target nvidia \
        --target-option fp32 \
        --max-qubits 28 \
        --mps-threshold 24 \
        --mps-bond 64 \
        --curriculum \
        --curriculum-warmup 30 \
        --force-entanglement \
        --entropy-coef 0.01 \
        --adaptive-temp \
        --explore-eps 0.3 \
        --kl-coef 0.05 \
        --w-creativity 0.1 \
        2>&1 | tee "$RESULTS/train/rl_main_b200.log"

    echo "RL main run complete → $RESULTS/train/h_cgqe_model_b200_rl_main.pt"
}

# ------------------------------------------------------------------
# Stage 2 XL: DAPO RL on 40q scaling systems
# ------------------------------------------------------------------
run_rl_xl() {
    echo ""
    echo "=== Stage 2 XL: DAPO RL (40q scaling set) ==="
    echo "    Checkpoint: $RESULTS/train/h_cgqe_model_b200_rl_main.pt"
    echo "    Molecules : 10 records up to 40q"
    echo "    Output    : $RESULTS/train/h_cgqe_model_b200_rl_40q.pt"
    echo ""

    MOLECULES_40=$(python3 -c "
import json
d = json.load(open('$DATA/hamiltonians_40plus/hamiltonians.json'))
names = [r['name'] for r in d['records']]
print(' '.join(names))
")

    python3 "$ROOT/src/gqe/models/train_rl_dapo.py" \
        --checkpoint "$RESULTS/train/h_cgqe_model_b200_rl_main.pt" \
        --hamiltonians "$DATA/hamiltonians_40plus/hamiltonians.json" \
        --molecules $MOLECULES_40 \
        --out "$RESULTS/train/h_cgqe_model_b200_rl_40q.pt" \
        --epochs 150 \
        --n-samples 32 \
        --n-iters 3 \
        --lr 5e-6 \
        --use-cuda \
        --use-bf16 \
        --single-gpu \
        --target nvidia \
        --target-option fp32 \
        --max-qubits 40 \
        --mps-threshold 24 \
        --mps-bond 128 \
        --curriculum \
        --force-entanglement \
        2>&1 | tee "$RESULTS/train/rl_40q_b200.log"

    echo "RL 40q complete → $RESULTS/train/h_cgqe_model_b200_rl_40q.pt"
}

# ------------------------------------------------------------------
# Precompute persistent circuit→energy cache (CUDA-Q once, reuse in RL)
# ------------------------------------------------------------------
run_energy_cache() {
    echo ""
    echo "=== Precompute RL energy cache (CUDA-Q → SQLite) ==="
    echo "    This is the expensive one-time step; training then hits the cache."
    echo ""

    HAMILTONIANS="$DATA/hamiltonians_rl_b200/hamiltonians.json"
    if [ ! -f "$HAMILTONIANS" ]; then
        HAMILTONIANS="$DATA/hamiltonians_merged.json"
    fi
    CACHE_OUT="$RESULTS/train/rl_energy_cache.sqlite"
    N_PER_MOL="${CACHE_N_PER_MOL:-512}"

    echo "    Hamiltonians: $HAMILTONIANS"
    echo "    Output      : $CACHE_OUT"
    echo "    n-per-mol   : $N_PER_MOL"
    echo ""

    python3 "$ROOT/src/gqe/data/precompute_rl_energy_cache.py" \
        --hamiltonians "$HAMILTONIANS" \
        --out "$CACHE_OUT" \
        --n-per-mol "$N_PER_MOL" \
        --max-qubits 40 \
        --max-seq-len 64 \
        --theta 0.01 \
        --eval-async-chunk 24 \
        --target nvidia \
        --target-option fp32 \
        --mps-threshold 28 \
        2>&1 | tee "$RESULTS/train/rl_energy_cache_precompute.log"

    echo "Energy cache ready → $CACHE_OUT"
}

# ------------------------------------------------------------------
# Ablation: Direct RL from scratch (secondary comparison)
# ------------------------------------------------------------------
run_rl_ablation() {
    echo ""
    echo "=== Ablation: Direct RL from scratch (no SFT warm-start) ==="
    echo "    Compact policy d_model=256 (~8–10M, SFT-scale), BF16, energy cache + reuse-iters=16,"
    echo "    size curriculum, chunked async eval, max-qubits 40"
    echo "    Output    : $RESULTS/train/h_cgqe_model_b200_rl_scratch.pt"
    echo ""

    HAMILTONIANS="$DATA/hamiltonians_rl_b200/hamiltonians.json"
    if [ ! -f "$HAMILTONIANS" ]; then
        HAMILTONIANS="$DATA/hamiltonians_merged.json"
    fi
    CACHE_OUT="$RESULTS/train/rl_energy_cache.sqlite"
    echo "    Hamiltonians: $HAMILTONIANS"
    if [ -f "$CACHE_OUT" ]; then
        echo "    Energy cache: $CACHE_OUT (write-through)"
        ENERGY_CACHE_ARGS=(--energy-cache "$CACHE_OUT")
    else
        echo "    Energy cache: MISSING — run: bash scripts/launch_b200_training.sh cache"
        echo "                  Continuing without cache (slow CUDA-Q path)."
        ENERGY_CACHE_ARGS=()
    fi

    # Ascending qubit order so curriculum stages grow small → large
    MOLECULES=$(python3 -c "
import json
d = json.load(open('$HAMILTONIANS'))
recs = sorted(
    [r for r in d['records'] if r.get('n_qubits', 99) <= 40],
    key=lambda r: (r.get('n_qubits', 99), r.get('name', '')),
)
print(' '.join(r['name'] for r in recs))
")
    echo "    Molecules : $MOLECULES (sorted by n_qubits; SV <=28, MPS 29-40)"
    echo ""

    python3 "$ROOT/src/gqe/models/train_rl_dapo.py" \
        --from-scratch \
        --hamiltonians "$HAMILTONIANS" \
        --molecules $MOLECULES \
        --out "$RESULTS/train/h_cgqe_model_b200_rl_scratch.pt" \
        --epochs 200 \
        --n-samples 128 \
        --n-iters 8 \
        --reuse-iters 16 \
        --lr 1e-4 \
        --d-model 256 \
        --nhead 8 \
        --encoder-layers 4 \
        --decoder-layers 4 \
        --dim-feedforward 1024 \
        --dropout 0.1 \
        --use-cuda \
        --use-bf16 \
        --single-gpu \
        --target nvidia \
        --target-option fp32 \
        --max-qubits 40 \
        --mps-threshold 28 \
        --mps-bond 64 \
        --force-entanglement \
        --no-adaptive-theta \
        --curriculum \
        --curriculum-warmup 15 \
        --curriculum-steps 4 \
        --max-resample-attempts 1 \
        --no-dynamic-sampling \
        --eval-async \
        --eval-async-chunk 24 \
        "${ENERGY_CACHE_ARGS[@]}" \
        2>&1 | tee "$RESULTS/train/rl_scratch_b200.log"

    echo "Ablation (scratch RL) complete → $RESULTS/train/h_cgqe_model_b200_rl_scratch.pt"
}

run_rl_ablation_smoke() {
    echo ""
    echo "=== Ablation smoke: Direct RL from scratch (~2 min sanity check) ==="
    echo "    Compact policy d_model=256 (~8–10M), BF16, chunked async eval"
    echo "    Output    : $RESULTS/train/h_cgqe_model_b200_rl_scratch_smoke.pt"
    echo ""

    CACHE_OUT="$RESULTS/train/rl_energy_cache.sqlite"
    HAMILTONIANS="$DATA/hamiltonians_rl_b200/hamiltonians.json"
    if [ ! -f "$HAMILTONIANS" ]; then
        HAMILTONIANS="$DATA/hamiltonians_merged.json"
        MOLECULES="h2_0.74 lih_1.6_full"
    else
        MOLECULES=$(python3 -c "
import json
d = json.load(open('$HAMILTONIANS'))
recs = sorted(
    [r for r in d['records'] if r.get('n_qubits', 99) <= 40],
    key=lambda r: r.get('n_qubits', 99),
)
print(' '.join(r['name'] for r in recs[:2]))
")
    fi
    echo "    Hamiltonians: $HAMILTONIANS"
    echo "    Molecules : $MOLECULES (2 smallest, 2 epochs, 32 samples)"
    echo ""

    python3 "$ROOT/src/gqe/models/train_rl_dapo.py" \
        --from-scratch \
        --hamiltonians "$HAMILTONIANS" \
        --molecules $MOLECULES \
        --out "$RESULTS/train/h_cgqe_model_b200_rl_scratch_smoke.pt" \
        --epochs 2 \
        --n-samples 32 \
        --n-iters 8 \
        --reuse-iters 4 \
        --lr 1e-4 \
        --d-model 256 \
        --nhead 8 \
        --encoder-layers 4 \
        --decoder-layers 4 \
        --dim-feedforward 1024 \
        --dropout 0.1 \
        --use-cuda \
        --use-bf16 \
        --single-gpu \
        --target nvidia \
        --target-option fp32 \
        --max-qubits 40 \
        --mps-threshold 28 \
        --mps-bond 64 \
        --force-entanglement \
        --no-adaptive-theta \
        --no-curriculum \
        --max-resample-attempts 1 \
        --no-dynamic-sampling \
        --eval-async \
        --eval-async-chunk 16 \
        ${CACHE_OUT:+--energy-cache "$CACHE_OUT"} \
        2>&1 | tee "$RESULTS/train/rl_scratch_b200_smoke.log"

    echo "Ablation smoke complete → $RESULTS/train/h_cgqe_model_b200_rl_scratch_smoke.pt"
}

# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------
mkdir -p "$RESULTS/train"

case "$STAGE" in
    sft)
        run_sft
        ;;
    rl)
        run_rl_main
        run_rl_xl
        ;;
    both)
        run_sft
        run_rl_main
        run_rl_xl
        ;;
    cache)
        run_energy_cache
        ;;
    ablation)
        run_rl_ablation
        ;;
    ablation-smoke)
        run_rl_ablation_smoke
        ;;
    cache+ablation)
        run_energy_cache
        run_rl_ablation
        ;;
    *)
        echo "Usage: $0 [sft|rl|both|cache|ablation|ablation-smoke|cache+ablation]"
        exit 1
        ;;
esac

echo ""
echo "=== All requested stages complete ==="
