"""Test building a CUDA-Q kernel with tiled operators for n2."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cudaq
cudaq.set_target("nvidia")
print(f"target: {cudaq.get_target().name}")

import numpy as np
from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    hamiltonian_to_spin_operator,
    get_active_electron_count,
)

records = load_hamiltonian_records(ROOT / "results" / "data" / "hamiltonians_40plus.json" / "hamiltonians.json")
rec = [r for r in records if r["name"] == "n2"][0]
nq = int(rec["n_qubits"])
ne = get_active_electron_count(rec)
print(f"n2: {nq}q, {ne}e")

ham = hamiltonian_to_spin_operator(rec)
print("hamiltonian OK")

# Load operators
import json
with open(ROOT / "results" / "eval" / "h_cgqe_operators_for_qsci.json") as f:
    op_data = json.load(f)
entry = [e for e in op_data if e["molecule"] == "n2"][0]
base_ops = entry["best_sequence"]["operators"]
base_thetas = entry["best_sequence"]["thetas"]
print(f"base: {len(base_ops)} ops")

# Tile to 4*20=80
target_depth = 4 * nq
n_base = len(base_ops)
n_repeats = max(1, target_depth // n_base + (1 if target_depth % n_base else 0))
tiled_ops = []
tiled_thetas = []
for rep in range(n_repeats):
    for i, op in enumerate(base_ops):
        tiled_ops.append(op)
        if rep == 0 and i < len(base_thetas):
            tiled_thetas.append(base_thetas[i])
        else:
            base_theta = base_thetas[i % len(base_thetas)] if base_thetas else 0.1
            perturbation = 0.02 * (rep + 1) * ((-1) ** i)
            tiled_thetas.append(base_theta + perturbation)
tiled_ops = tiled_ops[:target_depth]
tiled_thetas = tiled_thetas[:target_depth]
print(f"tiled: {len(tiled_ops)} ops, {len(tiled_thetas)} thetas")

# Pad pauli words
def _pad_pauli_word(word, n_qubits):
    if len(word) == n_qubits:
        return word
    if len(word) < n_qubits:
        return word + "I" * (n_qubits - len(word))
    return word[:n_qubits]

padded = [_pad_pauli_word(w, nq) for w in tiled_ops]
print(f"padded first: {padded[0]}")
print(f"padded last: {padded[-1]}")
pauli_words = [cudaq.pauli_word(w) for w in padded]
print("pauli words OK")

# Build kernel
@cudaq.kernel
def kernel(n_qubits_k: int, n_electrons_k: int,
           thetas: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(n_qubits_k)
    for i in range(n_electrons_k):
        x(q[i])
    for i in range(len(words)):
        exp_pauli(thetas[i], q, words[i])

print("kernel defined OK")

# Test observe
thetas_arr = tiled_thetas[:5]  # just first 5 for quick test
test_pauli = pauli_words[:5]
result = cudaq.observe(kernel, ham, nq, ne, thetas_arr, test_pauli)
print(f"observe OK with 5 ops: E={result.expectation():.6f}")

# Now test with all 80
result = cudaq.observe(kernel, ham, nq, ne, tiled_thetas, pauli_words)
print(f"observe OK with {len(tiled_thetas)} ops: E={result.expectation():.6f}")
