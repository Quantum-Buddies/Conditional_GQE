"""Minimal test to isolate 'Invalid target name (n2)' error."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cudaq
cudaq.set_target("nvidia")
print(f"CUDA-Q target: {cudaq.get_target().name}")

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    hamiltonian_to_spin_operator,
    get_active_electron_count,
)

records = load_hamiltonian_records(ROOT / "results" / "data" / "hamiltonians_40plus.json" / "hamiltonians.json")
for r in records:
    if r["name"] == "n2":
        nq = int(r["n_qubits"])
        ne = get_active_electron_count(r)
        print(f"n2: {nq}q, {ne}e")
        ham = hamiltonian_to_spin_operator(r)
        print("Hamiltonian built OK")

        @cudaq.kernel
        def hf_kern(n_q: int, n_e: int):
            q = cudaq.qvector(n_q)
            for i in range(n_e):
                x(q[i])

        result = cudaq.observe(hf_kern, ham, nq, ne)
        print(f"HF energy: {result.expectation():.6f}")
        break
