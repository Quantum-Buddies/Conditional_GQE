#!/bin/bash
# Setup script for qBraid Lab GPU instances (H200, H100, B200, etc.)
# Run this after launching a qBraid Lab on-demand GPU instance.
#
# Usage:
#   bash scripts/setup_qbraid_gpu.sh
#
# This installs CUDA-Q, clones the repo, and prepares the environment
# for RL training on the GPU instance.

set -euo pipefail

echo "=== qBraid GPU Environment Setup ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"
echo ""

# 1. Install CUDA-Q and dependencies
echo ">>> Installing CUDA-Q and Python dependencies..."
pip install --quiet \
    cudaq \
    cudaq-solvers \
    qiskit \
    scipy \
    numpy \
    tqdm \
    torch \
    transformers

echo ">>> Verifying CUDA-Q installation..."
python -c "
import cudaq
print(f'CUDA-Q version: {cudaq.__version__}')
cudaq.set_target('nvidia')
print('CUDA-Q nvidia target: OK')
# Check GPU memory
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}, {props.total_mem / 1e9:.1f} GB')
"

# 2. Clone or update the repo
REPO_URL="https://github.com/Quantum-Buddies/Conditional_GQE.git"
REPO_DIR="Conditional-GQE_materials"

if [ -d "$REPO_DIR" ]; then
    echo ">>> Updating existing repo..."
    cd "$REPO_DIR"
    git pull --rebase || echo "Warning: git pull failed, continuing with existing code"
    cd ..
else
    echo ">>> Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# 3. Download Hamiltonian data if not present
if [ ! -f "results/data/hamiltonians_merged.json" ]; then
    echo ">>> Hamiltonian data not found. Generating..."
    mkdir -p results/data
    python src/gqe/data/generate_hamiltonians.py \
        --molecules h2 lih beh2 n2 \
        --out results/data/hamiltonians_merged.json
else
    echo ">>> Hamiltonian data found."
fi

# 4. Check for supervised checkpoint
if [ ! -f "results/train/h_cgqe_uccsd_model.pt" ]; then
    echo ">>> WARNING: Supervised checkpoint not found at results/train/h_cgqe_uccsd_model.pt"
    echo "    You will need to either:"
    echo "    a) Train the supervised model first, or"
    echo "    b) Copy the checkpoint from AIRE HPC, or"
    echo "    c) Use --from-scratch for pure RL training"
else
    echo ">>> Supervised checkpoint found."
fi

echo ""
echo "=== Setup Complete ==="
echo "To start RL training, run:"
echo "  bash scripts/run_rl_qbraid_gpu.sh"
echo ""
echo "Or manually:"
echo "  python src/gqe/models/train_rl_dapo.py \\"
echo "    --checkpoint results/train/h_cgqe_uccsd_model.pt \\"
echo "    --hamiltonians results/data/hamiltonians_merged.json \\"
echo "    --molecules h2 lih beh2 n2 \\"
echo "    --out results/train/h_cgqe_rl_qbraid.pt \\"
echo "    --epochs 500 --n-samples 64 --n-iters 5 --reuse-iters 3 \\"
echo "    --buffer-batch-size 64 --adaptive-theta \\"
echo "    --max-qubits 30 --target nvidia --use-cuda --use-bf16"
