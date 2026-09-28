#!/usr/bin/env bash
# =============================================================================
# Phase 3 GQE Smoke Test — Single-command verification for judges
#
# Verifies the complete pipeline:
#   1. DedupCache SQLite persistence (save/load)
#   2. Offline RL training with --cache-only mode
#   3. FMO2 energy reconstruction (exact baseline)
#   4. QPU manifest generation (QWC grouping)
#   5. Code import sanity check
#
# Usage:
#   bash scripts/phase3/00_smoke_test.sh
#
# Requirements:
#   - Python 3.10+ with numpy, scipy
#   - CUDA-Q (optional, for quantum evaluation steps)
#   - PyTorch (for H-cGQE model steps)
#   - Qiskit (for QPU manifest generation)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
PASS=0
FAIL=0
SKIP=0

section() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }
skip() { echo "  [SKIP] $1"; SKIP=$((SKIP + 1)); }

# -----------------------------------------------------------------------------
# 0. Environment check
# -----------------------------------------------------------------------------
section "0. Environment Check"
echo "  Python: $($PYTHON --version 2>&1)"
echo "  Root: ${ROOT}"

$PYTHON -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>/dev/null && pass "numpy available" || fail "numpy missing"
$PYTHON -c "import scipy; print(f'  scipy: {scipy.__version__}')" 2>/dev/null && pass "scipy available" || fail "scipy missing"
$PYTHON -c "import qiskit; print(f'  qiskit: {qiskit.__version__}')" 2>/dev/null && pass "qiskit available" || skip "qiskit not installed (needed for QPU manifests)"
$PYTHON -c "import torch; print(f'  torch: {torch.__version__}')" 2>/dev/null && pass "torch available" || skip "torch not installed (needed for RL training)"
$PYTHON -c "import cudaq; print(f'  cudaq: {cudaq.__version__}')" 2>/dev/null && pass "cudaq available" || skip "cudaq not installed (needed for quantum eval)"

# -----------------------------------------------------------------------------
# 1. DedupCache SQLite persistence
# -----------------------------------------------------------------------------
section "1. DedupCache SQLite Persistence"

$PYTHON -c "
import sys, os, tempfile
sys.path.insert(0, '.')
from src.gqe.rl.map_elites import DedupCache

with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
    db_path = f.name

try:
    cache = DedupCache(
        molecule_id='h2_test', n_qubits=4, n_electrons=2,
        optimizer_iters=5, initial_theta=0.01,
        sqlite_path=db_path,
    )
    cache.put(['YZYI'], -1.116743)
    cache.put(['XZXI'], -1.115624)
    assert len(cache) == 2, f'Expected 2 entries, got {len(cache)}'

    e = cache.get(['YZYI'])
    assert e is not None and abs(e - (-1.116743)) < 1e-6, f'Wrong energy: {e}'

    cache.close()
    cache2 = DedupCache.from_sqlite(
        db_path, molecule_id='h2_test', n_qubits=4, n_electrons=2,
        optimizer_iters=5, initial_theta=0.01,
    )
    assert len(cache2) == 2, f'Expected 2 entries after reload, got {len(cache2)}'
    e2 = cache2.get(['YZYI'])
    assert e2 is not None and abs(e2 - (-1.116743)) < 1e-6, f'Wrong energy after reload: {e2}'
    cache2.close()
    print('  SQLite save/load/reload: OK')
finally:
    os.unlink(db_path)
" && pass "DedupCache SQLite save/load" || fail "DedupCache SQLite save/load"

# -----------------------------------------------------------------------------
# 2. Offline RL training (--cache-only mode)
# -----------------------------------------------------------------------------
section "2. Offline RL Training (cache-only mode)"

$PYTHON -c "
import sys, os, tempfile
sys.path.insert(0, '.')
from src.gqe.rl.map_elites import DedupCache
from src.gqe.models.train_rl_dapo import evaluate_energies_qd

with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
    db_path = f.name

try:
    cache = DedupCache(
        molecule_id='h2_test', n_qubits=4, n_electrons=2,
        sqlite_path=db_path,
    )
    cache.put(['YZYI'], -1.116743)
    cache.put(['XZXI'], -1.115624)

    record = {'name': 'h2_test', 'n_qubits': 4, 'n_electrons': 2, 'terms': []}
    ops_batch = [['YZYI'], ['XZXI'], ['ZZZZ']]
    energies, stats = evaluate_energies_qd(
        ops_batch, record, cache,
        cache_only=True, hf_energy=-1.1167,
    )
    assert len(energies) == 3
    assert abs(energies[0] - (-1.116743)) < 1e-6, f'Cached energy wrong: {energies[0]}'
    assert abs(energies[2] - (-1.1167)) < 1e-6, f'Penalty energy wrong: {energies[2]}'
    assert stats['hits'] == 2 and stats['misses'] == 1
    print(f'  Cache-only eval: {stats[\"hits\"]} hits, {stats[\"misses\"]} misses')
    cache.close()
finally:
    os.unlink(db_path)
" && pass "Offline RL cache-only evaluation" || fail "Offline RL cache-only evaluation"

# -----------------------------------------------------------------------------
# 3. FMO2 energy reconstruction (exact baseline)
# -----------------------------------------------------------------------------
section "3. FMO2 Energy Reconstruction"

FRAGMENTS="${ROOT}/results/data/fragments/fmo_hamiltonians.json"
if [ -f "$FRAGMENTS" ]; then
    $PYTHON -c "
import sys
sys.path.insert(0, '.')
from src.gqe.eval.run_fmo2 import run_fmo2
result = run_fmo2('${FRAGMENTS}', method='exact')
assert 'fmo2_energy' in result
assert 'monomer_sum' in result
assert 'pair_correction' in result
print(f'  FMO2 energy: {result[\"fmo2_energy\"]:.6f} Ha')
print(f'  Monomers: {result[\"n_fragments\"]}, elapsed: {result[\"elapsed_seconds\"]:.1f}s')
" && pass "FMO2 exact reconstruction" || fail "FMO2 exact reconstruction"
else
    skip "FMO2 test (fragment Hamiltonians not found at $FRAGMENTS)"
fi

# -----------------------------------------------------------------------------
# 4. QPU manifest generation (QWC grouping)
# -----------------------------------------------------------------------------
section "4. QPU Manifest Generation"

HAMILTONIANS="${ROOT}/results/data/hamiltonians_merged.json"
OPTIMIZED="${ROOT}/results/eval/h_cgqe_uccsd_optimized.json"
if [ -f "$HAMILTONIANS" ] && [ -f "$OPTIMIZED" ]; then
    $PYTHON scripts/phase3/generate_qpu_manifests.py \
        --molecules h2_0.74 \
        --hamiltonians "$HAMILTONIANS" \
        --optimized "$OPTIMIZED" \
        --out-dir /tmp/phase3_smoke_manifests \
        --shots 1024 \
        2>&1 && pass "QPU manifest generation" || fail "QPU manifest generation"

    if [ -f /tmp/phase3_smoke_manifests/h2_0.74_cepheus_manifest.json ]; then
        $PYTHON -c "
import json
with open('/tmp/phase3_smoke_manifests/h2_0.74_cepheus_manifest.json') as f:
    m = json.load(f)
assert m['molecule'] == 'h2_0.74'
assert m['n_qubits'] == 4
assert m['n_groups'] > 0
assert len(m['groups']) == m['n_groups']
print(f'  H2: {m[\"n_hamiltonian_terms\"]} terms -> {m[\"n_groups\"]} QWC groups')
print(f'  Cost: {m[\"cost_estimate\"][\"total_cost\"]:.2f} credits')
" && pass "QPU manifest structure valid" || fail "QPU manifest structure valid"
        rm -rf /tmp/phase3_smoke_manifests
    else
        fail "QPU manifest file not created"
    fi
else
    skip "QPU manifest (requires hamiltonians + optimized results)"
fi

# -----------------------------------------------------------------------------
# 5. Code import sanity check
# -----------------------------------------------------------------------------
section "5. Code Import Sanity Check"

$PYTHON -c "
import sys
sys.path.insert(0, '.')

from src.gqe.rl.map_elites import DedupCache, MAPElitesArchive, PerMoleculeArchives
print('  map_elites: OK')

from src.gqe.common.operator_pool import build_uccsd_operator_pool, build_uccsd_pauli_words
print('  operator_pool: OK')

from src.gqe.common.hamiltonian_utils import load_hamiltonian_records, find_record_by_name
print('  hamiltonian_utils: OK')

from src.gqe.eval.run_fmo2 import run_fmo2, hcgqe_fragment_energy, exact_energy_from_hamiltonian
print('  run_fmo2: OK')

try:
    from src.gqe.models.train_rl_dapo import evaluate_energies_qd, compute_reward
    print('  train_rl_dapo: OK')
except ImportError as e:
    print(f'  train_rl_dapo: SKIP ({e})')

try:
    from src.gqe.models.h_cgqe_transformer import HcGQEModel
    print('  h_cgqe_transformer: OK')
except ImportError as e:
    print(f'  h_cgqe_transformer: SKIP ({e})')

try:
    from src.gqe.eval.qbraid_backend import _group_qwc_terms, _build_ansatz_circuit
    print('  qbraid_backend: OK')
except ImportError as e:
    print(f'  qbraid_backend: SKIP ({e})')
" && pass "All imports successful" || fail "Import errors detected"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
section "Smoke Test Summary"
echo "  PASS: ${PASS}"
echo "  FAIL: ${FAIL}"
echo "  SKIP: ${SKIP}"
echo ""

if [ ${FAIL} -eq 0 ]; then
    echo "  All critical tests passed"
    exit 0
else
    echo "  ${FAIL} test(s) failed"
    exit 1
fi
