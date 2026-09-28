#!/usr/bin/env bash
# =============================================================================
# Portable One-Shot Setup Script for qBraid GPU Instances
# =============================================================================
# Run this immediately after `git clone` on a new qBraid instance.
#
# No sudo required. No system conda required.
# git-lfs is installed as a prebuilt binary in $HOME/.local/bin.
# Python deps are installed via `python3 -m pip` (qBraid-safe).
#
# Usage:
#   bash scripts/setup_env.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "  Conditional-GQE: Portable Environment Setup"
echo "  Root : $ROOT"
echo "  Host : $(hostname)"
echo "  Date : $(date)"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# 1. Git LFS — install prebuilt binary to $HOME/.local/bin (no sudo)
# ------------------------------------------------------------------
echo ">>> [1/5] Git LFS..."

# Ensure $HOME/.local/bin exists and is on PATH
mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

if command -v git-lfs &>/dev/null; then
    echo "    git-lfs already available: $(git-lfs --version)"
else
    echo "    git-lfs not found — downloading prebuilt binary (no sudo needed)..."

    # Detect architecture
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64)  LFS_ARCH="linux-amd64" ;;
        aarch64) LFS_ARCH="linux-arm64" ;;
        *)       LFS_ARCH="linux-amd64" ;;  # fallback
    esac

    # Fetch latest release tag from GitHub API
    LFS_VERSION="$(python3 -c "
import urllib.request, json
url = 'https://api.github.com/repos/git-lfs/git-lfs/releases/latest'
req = urllib.request.Request(url, headers={'User-Agent': 'setup-script'})
data = json.loads(urllib.request.urlopen(req, timeout=15).read())
print(data['tag_name'].lstrip('v'))
" 2>/dev/null || echo '3.5.1')"

    LFS_TARBALL="git-lfs-${LFS_ARCH}-v${LFS_VERSION}.tar.gz"
    LFS_URL="https://github.com/git-lfs/git-lfs/releases/download/v${LFS_VERSION}/${LFS_TARBALL}"
    LFS_TMPDIR="$(mktemp -d)"

    echo "    Downloading: ${LFS_URL}"
    if python3 -c "
import urllib.request, sys, os
out = os.path.join('${LFS_TMPDIR}', '${LFS_TARBALL}')
urllib.request.urlretrieve('${LFS_URL}', out)
print(f'    Downloaded to {out} ({os.path.getsize(out) // 1024} KB)')
" 2>/dev/null; then
        cd "$LFS_TMPDIR"
        tar xzf "$LFS_TARBALL" 2>/dev/null
        # The tarball contains git-lfs binary directly
        if [ -f "git-lfs" ]; then
            cp git-lfs "$HOME/.local/bin/git-lfs"
            chmod +x "$HOME/.local/bin/git-lfs"
            echo "    Installed: $(git-lfs --version)"
        else
            # Some releases use install.sh
            if [ -f "install.sh" ]; then
                sed -i "s|^prefix=.*|prefix=\"$HOME/.local\"|" install.sh 2>/dev/null || true
                bash install.sh 2>/dev/null || true
                echo "    Installed via install.sh: $(git-lfs --version 2>/dev/null || echo 'check manually')"
            fi
        fi
        cd "$ROOT"
    else
        echo "    WARNING: Could not download git-lfs binary."
        echo "             Manual install: wget $LFS_URL && tar xzf && cp git-lfs ~/.local/bin/"
    fi
    rm -rf "$LFS_TMPDIR"
fi

# Pull LFS assets
if command -v git-lfs &>/dev/null; then
    echo "    Pulling LFS-tracked assets..."
    cd "$ROOT"
    git lfs install --skip-repo 2>/dev/null || true
    git lfs pull 2>/dev/null || echo "    WARNING: git lfs pull failed — check network or LFS quota."
else
    echo "    WARNING: git-lfs unavailable. LFS-tracked files (.pt, .sqlite) will be"
    echo "             pointer stubs. Install git-lfs manually and run 'git lfs pull'."
fi

# ------------------------------------------------------------------
# 2. Python Dependencies (qBraid-safe: use python3 -m pip)
# ------------------------------------------------------------------
echo ""
echo ">>> [2/5] Python Dependencies..."
cd "$ROOT"

# On qBraid, bare `pip` defaults to /opt/conda/bin/pip (non-persistent).
# Using `python3 -m pip` ensures we install into the active environment.
PIP_CMD="python3 -m pip"

if [ -f "requirements-qbraid.txt" ]; then
    $PIP_CMD install --quiet -r requirements-qbraid.txt
elif [ -f "requirements.txt" ]; then
    $PIP_CMD install --quiet -r requirements.txt
else
    $PIP_CMD install --quiet pyyaml numpy scipy matplotlib tqdm \
        openfermion openfermionpyscf pyscf \
        qiskit qiskit-algorithms qiskit-nature \
        cudaq cudaq-solvers fpdf2
fi

# ------------------------------------------------------------------
# 3. PyTorch with CUDA
# ------------------------------------------------------------------
echo ""
echo ">>> [3/5] PyTorch CUDA..."
python3 -c "
import torch, sys, subprocess
if not torch.cuda.is_available():
    print('    CUDA not available — reinstalling PyTorch (cu126)...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                    'torch', 'torchvision',
                    '--index-url', 'https://download.pytorch.org/whl/cu126',
                    '--force-reinstall'])
else:
    print(f'    PyTorch {torch.__version__} with CUDA ready.')
"

# ------------------------------------------------------------------
# 4. GPU & CUDA-Q Verification
# ------------------------------------------------------------------
echo ""
echo ">>> [4/5] GPU & CUDA-Q Verification..."
python3 -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'    GPU  : {name}')
    print(f'    CC   : sm_{cap[0]}{cap[1]}')
    print(f'    VRAM : {vram:.1f} GB')
else:
    print('    WARNING: No GPU detected by PyTorch.')
    sys.exit(1)

try:
    import cudaq
    cudaq.set_target('nvidia')
    print(f'    CUDA-Q: {cudaq.__version__} (target=nvidia OK)')
except Exception as e:
    print(f'    WARNING: CUDA-Q test failed: {e}')
"

# ------------------------------------------------------------------
# 5. Asset Audit
# ------------------------------------------------------------------
echo ""
echo ">>> [5/5] Asset Audit..."

check_asset() {
    if [ -f "$1" ]; then
        local sz; sz=$(du -h "$1" | cut -f1)
        echo "    [OK]      $2 ($sz)"
    else
        echo "    [MISSING] $2"
    fi
}

check_asset "$ROOT/results/train/h_cgqe_model_b200_sft.pt"       "SFT Warm-Start Checkpoint"
check_asset "$ROOT/results/train/rl_energy_cache.sqlite"          "SQLite Energy Cache (25K entries)"
check_asset "$ROOT/results/train/rl_pretrain_from_cache.json"     "Pretrain Bootstrap JSON (24K circuits)"
check_asset "$ROOT/results/data/hamiltonians_gic2026/hamiltonians.json" "GIC 2026 Hamiltonians (35 molecules)"
check_asset "$ROOT/results/baselines/cudaq_gqe_baseline.json"     "GQE Baseline Results"

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo ""
echo "  Next Steps:"
echo "    Smoke test : bash scripts/train_rl.sh smoke"
echo "    Full RL    : bash scripts/train_rl.sh full"
echo "    Evaluation : bash scripts/evaluate_rl.sh all"
echo "============================================================"
