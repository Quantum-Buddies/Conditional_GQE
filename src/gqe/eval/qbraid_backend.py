"""qBraid-backed execution backend for H-cGQE circuits.

Translates an H-cGQE operator sequence and a Hamiltonian record into a Qiskit
parameterized circuit, then executes the circuit on a qBraid-managed simulator
or QPU. This provides a Phase 3 execution path that does not require the
CUDA-Q `nvidia` target, enabling runs on IBM, IonQ, and other qBraid devices.

Usage:
    python src/gqe/eval/qbraid_backend.py \
        --hamiltonians results/data/hamiltonians.json \
        --generated results/inference/h_cgqe_generated.json \
        --optimized results/eval/h_cgqe_optimized.json \
        --molecule h2 \
        --device qbraid_qir_simulator \
        --out results/eval/qbraid_h2_energy.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
    get_active_electron_count,
)

try:
    from qbraid import QbraidProvider
except ImportError:
    QbraidProvider = None

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
except ImportError:
    QuantumCircuit = None
    Parameter = None
    SparsePauliOp = None
    transpile = None


def _needs_qiskit() -> None:
    if QuantumCircuit is None or Parameter is None or SparsePauliOp is None:
        raise ImportError("qBraid backend requires qiskit. Install it with: pip install qiskit")


def _build_ansatz_circuit(
    n_qubits: int,
    n_electrons: int,
    operators: list[str],
) -> tuple[Any, list[str], list[Any]]:
    """Build a Qiskit parameterized circuit from an H-cGQE operator sequence.

    The ansatz prepares the Hartree-Fock state (n_electrons qubits in |1>) and
    applies a sequence of Pauli rotations e^{-i theta_k / 2 P_k}.

    Returns:
        circuit: Qiskit QuantumCircuit with one Parameter per operator.
        pauli_words: List of Pauli words in operator order.
        thetas: List of circuit Parameter objects.
    """
    _needs_qiskit()

    circuit = QuantumCircuit(n_qubits)

    # Hartree-Fock state: occupy the first n_electrons qubits
    for i in range(n_electrons):
        circuit.x(i)

    thetas = [Parameter(f"theta_{i}") for i in range(len(operators))]
    pauli_words: list[str] = []

    for i, word in enumerate(operators):
        # Pad word to n_qubits if it is compact (e.g. 'ZIZI' for 4 qubits)
        if len(word) < n_qubits:
            word = word + "I" * (n_qubits - len(word))
        pauli_words.append(word)

        theta = thetas[i]
        qubits_with_pauli = [q for q, op in enumerate(word) if op != "I"]
        if not qubits_with_pauli:
            continue

        # Basis change: rotate each non-Z Pauli into the Z basis
        # X -> H; Y -> H S (because H S Y S^\u2020 H = Z)
        for q, op in enumerate(word):
            if op == "X":
                circuit.h(q)
            elif op == "Y":
                circuit.h(q)
                circuit.s(q)

        # CNOT ladder to reduce multi-qubit Pauli Z chain to a single qubit
        for q in range(len(qubits_with_pauli) - 1):
            circuit.cx(qubits_with_pauli[q], qubits_with_pauli[q + 1])

        target = qubits_with_pauli[-1]
        circuit.rz(theta, target)

        # Undo CNOT ladder
        for q in range(len(qubits_with_pauli) - 2, -1, -1):
            circuit.cx(qubits_with_pauli[q], qubits_with_pauli[q + 1])

        # Undo basis change
        for q, op in enumerate(word):
            if op == "X":
                circuit.h(q)
            elif op == "Y":
                circuit.sdg(q)
                circuit.h(q)

    return circuit, pauli_words, thetas


def _measure_pauli_term(
    circuit: Any,
    term: str,
    theta_values: np.ndarray,
    thetas: list[Any],
    device: str,
    shots: int = 1024,
) -> float:
    """Measure the expectation value of a single Pauli term on the qBraid device.

    Args:
        circuit: Parameterized ansatz circuit.
        term: Pauli string (e.g. 'IZIZ') for the Hamiltonian term.
        theta_values: Bound values for the circuit parameters.
        thetas: List of circuit Parameter objects.
        device: qBraid device name or ID.
        shots: Number of shots per measurement.

    Returns:
        Estimated expectation value (real float).
    """
    _needs_qiskit()

    if QbraidProvider is None:
        raise ImportError("qBraid SDK not installed. Install it with: pip install qbraid")

    bound_circuit = circuit.bind_parameters({t: float(v) for t, v in zip(thetas, theta_values)})

    # Add measurement basis rotations for the Pauli term
    meas = QuantumCircuit(bound_circuit.num_qubits)
    meas.compose(bound_circuit, inplace=True)
    for q, op in enumerate(term):
        # Rotate into the measurement basis of the Pauli operator
        # X -> H; Y -> H S (because (H S)^\u2020 Z (H S) = Y)
        if op == "X":
            meas.h(q)
        elif op == "Y":
            meas.h(q)
            meas.s(q)
    meas.measure_all()

    provider = QbraidProvider()
    qdevice = provider.get_device(device)
    job = qdevice.run(meas, shots=shots)
    result = job.result()

    counts = result.measurement_counts()
    n_shots = sum(counts.values())
    if n_shots == 0:
        return 0.0

    exp = 0.0
    for bitstring, count in counts.items():
        parity = sum(int(bitstring[q]) for q, op in enumerate(term) if op != "I") % 2
        sign = -1 if parity == 1 else 1
        exp += sign * count / n_shots
    return exp


def evaluate_energy_qbraid(
    molecule_record: dict[str, Any],
    operators: list[str],
    theta_values: np.ndarray | None = None,
    device: str = "qbraid_qir_simulator",
    shots: int = 1024,
) -> dict[str, Any]:
    """Evaluate the energy of an H-cGQE circuit using a qBraid backend.

    Args:
        molecule_record: Hamiltonian record with terms and metadata.
        operators: H-cGQE operator sequence (Pauli words).
        theta_values: Optional rotation parameters. If None, uses zeros.
        device: qBraid device name or ID.
        shots: Number of shots per Pauli term measurement.

    Returns:
        Dict with keys: energy, device, shots, runtime_seconds, term_expectations.
    """
    _needs_qiskit()

    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    circuit, _, thetas = _build_ansatz_circuit(n_qubits, n_electrons, operators)

    if theta_values is None:
        theta_values = np.zeros(len(operators))
    elif len(theta_values) != len(operators):
        raise ValueError(
            f"theta_values length ({len(theta_values)}) does not match operators ({len(operators)})"
        )

    start = time.perf_counter()
    energy = 0.0
    term_expectations = {}

    for term in molecule_record.get("terms", []):
        word = term["term"]
        coeff = complex(float(term.get("real", 0.0)), float(term.get("imag", 0.0)))
        if abs(coeff) < 1e-14:
            continue
        exp = _measure_pauli_term(
            circuit, word, theta_values, thetas, device=device, shots=shots
        )
        term_expectations[word] = {"coeff_real": coeff.real, "coeff_imag": coeff.imag, "expectation": exp}
        energy += coeff.real * exp

    runtime = time.perf_counter() - start

    return {
        "energy": float(energy),
        "device": device,
        "shots": shots,
        "runtime_seconds": runtime,
        "term_expectations": term_expectations,
        "metadata": {
            "n_qubits": n_qubits,
            "n_electrons": n_electrons,
            "n_operators": len(operators),
            "n_hamiltonian_terms": len(molecule_record.get("terms", [])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H-cGQE circuits on qBraid backends")
    parser.add_argument("--hamiltonians", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--molecule", type=str, required=True)
    parser.add_argument("--device", type=str, default="qbraid_qir_simulator")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = load_hamiltonian_records(args.hamiltonians)
    record = find_record_by_name(records, args.molecule)

    with args.generated.open("r", encoding="utf-8") as f:
        generated = json.load(f)
    with args.optimized.open("r", encoding="utf-8") as f:
        optimized = json.load(f)

    # Find the best optimized sequence for this molecule
    mol_opt = None
    for entry in optimized.get("results", []):
        if entry.get("molecule") == args.molecule:
            mol_opt = entry
            break
    if mol_opt is None:
        raise ValueError(f"No optimized data for molecule {args.molecule}")

    best_seq = mol_opt.get("best_sequence", {})
    operators = best_seq.get("operators", [])
    thetas = best_seq.get("thetas", [])

    result = evaluate_energy_qbraid(
        record, operators, theta_values=np.asarray(thetas), device=args.device, shots=args.shots
    )
    result["molecule"] = args.molecule

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"qBraid energy for {args.molecule}: {result['energy']:.6f} Ha")
    print(f"Result saved to: {args.out}")


if __name__ == "__main__":
    main()
