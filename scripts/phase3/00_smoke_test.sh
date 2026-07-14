#!/bin/bash
# Phase 3 Final — Smoke test: verify environment, imports, and data availability
# Usage: bash scripts/phase3/00_smoke_test.sh
set -e

export PY=${PY:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}
export LD_LIBRARY_PATH=/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:$LD_LIBRARY_PATH
cd /scratch/kcwp264/Conditional-GQE_materials

echo "=== Phase 3 Smoke Test ==="
echo ""

echo "[1/5] Python version:"
$PY --version

echo ""
echo "[2/5] Key imports:"
$PY -c "
import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
import cudaq; print(f'  CUDA-Q {cudaq.__version__}')
import pyscf; print(f'  PySCF {pyscf.__version__}')
import scipy; print(f'  SciPy {scipy.__version__}')
import numpy; print(f'  NumPy {numpy.__version__}')
print('  All imports OK')
"

echo ""
echo "[3/5] Hamiltonian data:"
$PY -c "
import json
from pathlib import Path
for p in [
    'results/data/hamiltonians_phase3.json/hamiltonians.json',
    'results/data/hamiltonians_scaling.json/hamiltonians.json',
    'results/data/hamiltonians_40plus.json/hamiltonians.json',
]:
    if Path(p).exists():
        with open(p) as f:
            data = json.load(f)
        n = len(data.get('records', []))
        print(f'  {p}: {n} records')
    else:
        print(f'  {p}: MISSING')
"

echo ""
echo "[4/5] Model checkpoint:"
$PY -c "
import torch
from pathlib import Path
for p in sorted(Path('results/train').glob('h_cgqe_*phase3*.pt')):
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    n = sum(v.numel() for v in ckpt.get('model_state', ckpt.get('model_state_dict', {})).values())
    print(f'  {p.name}: {n:,} params, enc={cfg.get(\"encoder_layers\",\"?\")}, dec={cfg.get(\"decoder_layers\",\"?\")}')
    break
else:
    print('  WARNING: No phase3 checkpoint found')
"

echo ""
echo "[5/5] Run manifest module:"
$PY -c "
from src.gqe.common.run_manifest import create_manifest, save_manifest
print('  run_manifest: OK')
"

echo ""
echo "=== Smoke test passed ==="
