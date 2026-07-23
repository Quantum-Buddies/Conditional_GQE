"""Quick validation: run GQE with the new UCCSD operator pool on H2."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "gqe"))

from common.hamiltonian_utils import (
    load_hamiltonian_records,
    hamiltonian_to_spin_operator,
    get_active_electron_count,
    exact_diagonalize_hamiltonian,
)
from common.operator_pool import build_uccsd_operator_pool

import cudaq
import cudaq_solvers as solvers
from cudaq_solvers.gqe_algorithm.gqe import get_default_config


def test_uccsd_pool():
    """Test UCCSD operator pool generation and GQE solver execution on H2."""
    cudaq.set_target("qpp-cpu")

    # Load H2 record
    ham_path = Path("results/data/hamiltonians.json")
    if not ham_path.exists():
        ham_path = Path("results/data/hamiltonians_scaling.json/hamiltonians.json")
    records = load_hamiltonian_records(ham_path)
    h2 = records[0]
    n_qubits = int(h2["n_qubits"])
    n_electrons = get_active_electron_count(h2)
    spin_ham = hamiltonian_to_spin_operator(h2)

    print(f"Molecule: {h2['name']}, n_qubits={n_qubits}, n_electrons={n_electrons}")

    # Exact reference
    ref_energy, _ = exact_diagonalize_hamiltonian(h2)
    print(f"Exact FCI energy: {ref_energy:.10f}")

    # Build UCCSD pool
    pool = build_uccsd_operator_pool(h2)
    print(f"UCCSD pool size: {len(pool)} operators")

    # Verify no Z-only
    z_only = sum(1 for _, _, pw in pool if "X" not in pw and "Y" not in pw)
    print(f"Z-only operators: {z_only} (must be 0)")
    assert z_only == 0, "FAIL: Z-only operators found in pool!"

    @cudaq.kernel
    def ansatz_kernel(
        n_qubits: int, n_electrons: int, coeffs: list[float], words: list[cudaq.pauli_word]
    ):
        q = cudaq.qvector(n_qubits)
        for i in range(n_electrons):
            x(q[i])
        for i in range(len(coeffs)):
            exp_pauli(coeffs[i], q, words[i])

    # Build lookup: SpinOperator id -> (coefficient, pauli_string)
    _op_to_data: dict[int, tuple[complex, str]] = {}
    for _op, _c, _pstr in pool:
        _op_to_data[id(_op)] = (_c, _pstr)

    def cost(sampled_ops, **kwargs):
        full_coeffs: list[float] = []
        full_words: list[cudaq.pauli_word] = []
        for op in sampled_ops:
            data = _op_to_data.get(id(op))
            if data is None:
                raise RuntimeError("Sampled operator not found in pool lookup")
            _c, _pstr = data
            full_coeffs.append(float(_c.real))
            full_words.append(_pstr)
        result = cudaq.observe(
            ansatz_kernel, spin_ham, n_qubits, n_electrons, full_coeffs, full_words
        )
        return float(result.expectation())

    cfg = get_default_config()
    cfg.use_fabric_logging = False
    cfg.verbose = True
    cfg.save_trajectory = False

    operators_only = [op for op, _, _ in pool]
    print("\nRunning GQE with UCCSD pool (max_iters=25, ngates=10)...")
    minE, best_ops = solvers.gqe(
        cost, operators_only, max_iters=25, ngates=10, config=cfg
    )

    print(f"\n{'='*50}")
    print(f"GQE Ground Energy = {minE:.10f}")
    print(f"Reference (FCI)   = {ref_energy:.10f}")
    error_mha = abs(minE - ref_energy) * 1000
    print(f"Error (mHa)       = {error_mha:.2f}")
    print(f"Chemical accuracy = 1.6 mHa")
    print(f"Within accuracy   = {'YES' if error_mha < 1.6 else 'NO'}")
    print(f"{'='*50}")

    print(f"\nBest operators (indices): {best_ops}")
    for idx in best_ops:
        op, coeff, pw = pool[int(idx)]
        print(f"  [{idx:3d}] {pw}  coeff={coeff.real:.6f}")

    assert minE is not None
    assert isinstance(minE, float)


if __name__ == "__main__":
    test_uccsd_pool()
