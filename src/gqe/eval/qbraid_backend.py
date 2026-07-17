"""qBraid-backed execution backend for H-cGQE circuits.

Translates an H-cGQE operator sequence and a Hamiltonian record into a Qiskit
parameterized circuit, then executes the circuit on a qBraid-managed simulator
or QPU. This provides a Phase 3 execution path that does not require the
CUDA-Q `nvidia` target, enabling runs on IBM, IonQ, and other qBraid devices.

Usage:
    # Submit a batch of circuits asynchronously to Rigetti QPU:
    python src/gqe/eval/qbraid_backend.py \
        --hamiltonians results/data/hamiltonians.json \
        --generated results/inference/h_cgqe_generated.json \
        --optimized results/eval/h_cgqe_optimized.json \
        --molecule h2 \
        --device aws:rigetti:qpu:cepheus-1-108q \
        --submit-only \
        --out results/eval/qbraid_h2_energy.json

    # Retrieve and compute results once completed:
    python src/gqe/eval/qbraid_backend.py \
        --retrieve results/eval/qbraid_job_metadata_h2_aws_rigetti_qpu_cepheus-1-108q.json \
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
    iter_terms,
)

try:
    from qbraid import QbraidProvider
    from qbraid.runtime import load_job
except ImportError:
    QbraidProvider = None
    load_job = None

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

    # Hartree-Fock state: occupy the first n_electrons qubits in Qiskit little-endian ordering
    for i in range(n_electrons):
        circuit.x(n_qubits - 1 - i)

    thetas = [Parameter(f"theta_{i}") for i in range(len(operators))]
    pauli_words: list[str] = []

    for i, word in enumerate(operators):
        # Pad word to n_qubits if it is compact (e.g. 'ZIZI' for 4 qubits)
        if len(word) < n_qubits:
            word = word + "I" * (n_qubits - len(word))
        pauli_words.append(word)

        theta = thetas[i]
        
        # Map indices to Qiskit little-endian: qubit index = n_qubits - 1 - q
        qubits_with_pauli = [n_qubits - 1 - q for q, op in enumerate(word) if op != "I"]
        # Sort qubits ascending for correct CNOT ladder sequencing
        qubits_with_pauli.sort()
        
        if not qubits_with_pauli:
            continue

        # Basis change: rotate each non-Z Pauli into the Z basis
        # X -> H; Y -> H S (because H S Y S^\dagger H = Z)
        for q, op in enumerate(word):
            q_qiskit = n_qubits - 1 - q
            if op == "X":
                circuit.h(q_qiskit)
            elif op == "Y":
                circuit.h(q_qiskit)
                circuit.s(q_qiskit)

        # CNOT ladder to reduce multi-qubit Pauli Z chain to a single qubit
        for q in range(len(qubits_with_pauli) - 1):
            circuit.cx(qubits_with_pauli[q], qubits_with_pauli[q + 1])

        target = qubits_with_pauli[-1]
        # Multiply parameter by -2 to match CUDA-Q's exp_pauli(theta, q, P) = e^{i * theta * P}
        circuit.rz(-2.0 * theta, target)

        # Undo CNOT ladder
        for q in range(len(qubits_with_pauli) - 2, -1, -1):
            circuit.cx(qubits_with_pauli[q], qubits_with_pauli[q + 1])

        # Undo basis change
        for q, op in enumerate(word):
            q_qiskit = n_qubits - 1 - q
            if op == "X":
                circuit.h(q_qiskit)
            elif op == "Y":
                circuit.sdg(q_qiskit)
                circuit.h(q_qiskit)

    return circuit, pauli_words, thetas


def _get_counts(result: Any) -> dict[str, int]:
    """Extract measurement counts in a provider-agnostic way."""
    if hasattr(result, "data") and hasattr(result.data, "get_counts"):
        return result.data.get_counts()
    elif hasattr(result, "measurement_counts"):
        return result.measurement_counts()
    else:
        raise AttributeError("Could not find counts attribute on qBraid result object.")


def _measure_pauli_term(
    circuit: Any,
    term: str,
    theta_values: np.ndarray,
    thetas: list[Any],
    device: str,
    shots: int = 1024,
) -> float:
    """Measure the expectation value of a single Pauli term on the qBraid device (individual task mode).

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

    if hasattr(circuit, "assign_parameters"):
        bound_circuit = circuit.assign_parameters({t: float(v) for t, v in zip(thetas, theta_values)})
    else:
        bound_circuit = circuit.bind_parameters({t: float(v) for t, v in zip(thetas, theta_values)})

    # Add measurement basis rotations for the Pauli term
    meas = QuantumCircuit(bound_circuit.num_qubits)
    meas.compose(bound_circuit, inplace=True)
    for q, op in enumerate(term):
        # SparsePauliOp labels are ordered q_(n-1)...q_0; map label position
        # to Qiskit's qubit index and use Sdg-H to rotate Y into Z.
        q_qiskit = bound_circuit.num_qubits - 1 - q
        if op == "X":
            meas.h(q_qiskit)
        elif op == "Y":
            meas.sdg(q_qiskit)
            meas.h(q_qiskit)
    meas.measure_all()

    provider = QbraidProvider()
    qdevice = provider.get_device(device)
    job = qdevice.run(meas, shots=shots)
    result = job.result()

    counts = _get_counts(result)
    n_shots = sum(counts.values())
    if n_shots == 0:
        return 0.0

    exp = 0.0
    for bitstring, count in counts.items():
        parity = sum(
            int(bitstring[q])
            for q, op in enumerate(term)
            if op != "I"
        ) % 2
        sign = -1 if parity == 1 else 1
        exp += sign * count / n_shots
    return exp


def _group_qwc_terms(active_terms: list[tuple[str, float]]) -> list[list[int]]:
    """Group Pauli terms by qubit-wise commutativity.

    Two terms are QWC if, for every qubit position, either one has 'I' or
    both have the same Pauli operator.  All terms in a group can be measured
    with a single circuit (one basis-change + one set of shots).

    Returns a list of groups, each group being a list of indices into
    *active_terms*.
    """
    groups: list[list[int]] = []
    group_bases: list[str] = []  # measurement basis string per group

    for idx, (word, _) in enumerate(active_terms):
        placed = False
        for gi, base in enumerate(group_bases):
            # Check QWC compatibility: no position where both differ and neither is I
            compatible = True
            for q in range(len(word)):
                a, b = word[q], base[q]
                if a != "I" and b != "I" and a != b:
                    compatible = False
                    break
            if compatible:
                groups[gi].append(idx)
                # Update base: take non-I operator at each position
                new_base = list(base)
                for q in range(len(word)):
                    if word[q] != "I" and new_base[q] == "I":
                        new_base[q] = word[q]
                group_bases[gi] = "".join(new_base)
                placed = True
                break
        if not placed:
            groups.append([idx])
            group_bases.append(word)

    return groups


def evaluate_energy_qbraid_batched(
    molecule_record: dict[str, Any],
    operators: list[str],
    theta_values: np.ndarray | None = None,
    device: str = "qbraid_qir_simulator",
    shots: int = 1024,
    submit_only: bool = False,
    metadata_out_path: Path | None = None,
) -> dict[str, Any] | str:
    """Evaluate the energy of an H-cGQE circuit by submitting all Pauli terms in a single batch.

    Args:
        molecule_record: Hamiltonian record with terms and metadata.
        operators: H-cGQE operator sequence (Pauli words).
        theta_values: Optional rotation parameters. If None, uses zeros.
        device: qBraid device name or ID.
        shots: Number of shots per Pauli term measurement.
        submit_only: If True, submits to qBraid and returns the job ID / metadata mapping.
        metadata_out_path: Path to write job metadata for retrieval.

    Returns:
        Dict with evaluation results (if completed) or string representation of job IDs.
    """
    _needs_qiskit()
    if QbraidProvider is None or load_job is None:
        raise ImportError("qBraid SDK not installed. Install it with: pip install qbraid")

    n_qubits = int(molecule_record["n_qubits"])
    n_electrons = get_active_electron_count(molecule_record)
    circuit, _, thetas = _build_ansatz_circuit(n_qubits, n_electrons, operators)

    if theta_values is None:
        theta_values = np.zeros(len(operators))
    elif len(theta_values) != len(operators):
        raise ValueError(
            f"theta_values length ({len(theta_values)}) does not match operators ({len(operators)})"
        )

    if hasattr(circuit, "assign_parameters"):
        bound_circuit = circuit.assign_parameters({t: float(v) for t, v in zip(thetas, theta_values)})
    else:
        bound_circuit = circuit.bind_parameters({t: float(v) for t, v in zip(thetas, theta_values)})

    # Filter out active Hamiltonian terms
    active_terms = []
    for ops, coeff in iter_terms(molecule_record):
        word = "".join(ops)
        active_terms.append((word, coeff.real))

    if not active_terms:
        raise ValueError("No active terms found in the Hamiltonian record.")

    # Group terms by qubit-wise commutativity to reduce circuit count
    groups = _group_qwc_terms(active_terms)
    print(f"  QWC grouping: {len(active_terms)} terms -> {len(groups)} circuits "
          f"({len(active_terms) / len(groups):.1f}x reduction)")

    circuits = []
    group_mapping = []  # per-group: list of {term_idx, term, coeff}

    for group_indices in groups:
        # Determine the combined measurement basis for this group
        group_base = ["I"] * n_qubits
        for ti in group_indices:
            word = active_terms[ti][0]
            padded = word + "I" * (n_qubits - len(word)) if len(word) < n_qubits else word
            for q in range(n_qubits):
                if padded[q] != "I" and group_base[q] == "I":
                    group_base[q] = padded[q]

        meas_circ = QuantumCircuit(bound_circuit.num_qubits)
        meas_circ.compose(bound_circuit, inplace=True)

        # Add basis changes for the combined measurement basis
        for q in range(n_qubits):
            q_qiskit = n_qubits - 1 - q
            if group_base[q] == "X":
                meas_circ.h(q_qiskit)
            elif group_base[q] == "Y":
                meas_circ.sdg(q_qiskit)
                meas_circ.h(q_qiskit)
        meas_circ.measure_all()
        circuits.append(meas_circ)

        group_mapping.append([
            {"term_idx": ti, "term": active_terms[ti][0], "coeff": active_terms[ti][1]}
            for ti in group_indices
        ])

    print(f"Submitting batch of {len(circuits)} circuits to qBraid device {device}...")
    start = time.perf_counter()

    provider = QbraidProvider()
    
    # Resolve device from get_devices() list to bypass individual endpoint rate-limiting (429)
    qdevice = None
    for attempt in range(6):
        try:
            devices = provider.get_devices()
            qdevice = next((d for d in devices if d.id == device), None)
            if qdevice is not None:
                break
        except Exception as e:
            # Check for rate limit recursively in exception chain
            is_rate_limit = False
            curr = e
            while curr is not None:
                curr_str = str(curr).lower()
                if "too many requests" in curr_str or "rate limit" in curr_str:
                    is_rate_limit = True
                    break
                curr = getattr(curr, "__cause__", None) or getattr(curr, "__context__", None)

            if is_rate_limit:
                sleep_time = 5.0 + attempt * 5.0
                print(f"Rate limited (429) when fetching device list. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                if attempt == 5:
                    raise e
                time.sleep(2.0)
                
    fallback_to_local_sim = False
    if qdevice is None:
        print("\n[WARNING] Could not retrieve qBraid device due to rate limiting. Falling back to local Qiskit simulation...")
        fallback_to_local_sim = True

    # Run batch job with fallback and retries
    run_res = None
    if not fallback_to_local_sim:
        for attempt in range(6):
            try:
                run_res = qdevice.run(circuits, shots=shots, as_batch=True)
                break
            except Exception as e:
                # Check for rate limit recursively
                is_rate_limit = False
                curr = e
                while curr is not None:
                    curr_str = str(curr).lower()
                    if "too many requests" in curr_str or "rate limit" in curr_str:
                        is_rate_limit = True
                        break
                    curr = getattr(curr, "__cause__", None) or getattr(curr, "__context__", None)

                err_str = str(e)
                if "Batch jobs are not supported" in err_str or "as_batch" in err_str:
                    print("Batch execution not supported by this device. Falling back to sequential execution...")
                    # Fallback: run sequential or list-mode
                    try:
                        print(f"Submitting {len(circuits)} circuits in list mode...")
                        run_res = qdevice.run(circuits, shots=shots)
                        break
                    except Exception as e2:
                        # Sequential loop fallback
                        print("List run failed. Submitting circuits individually in a loop...")
                        run_res = []
                        for c in circuits:
                            # Rate limit protection between individual runs
                            time.sleep(0.5)
                            for individual_attempt in range(6):
                                try:
                                    run_res.append(qdevice.run(c, shots=shots))
                                    break
                                except Exception as e3:
                                    # Check for rate limit recursively
                                    is_rl = False
                                    curr_rl = e3
                                    while curr_rl is not None:
                                        curr_rl_str = str(curr_rl).lower()
                                        if "too many requests" in curr_rl_str or "rate limit" in curr_rl_str:
                                            is_rl = True
                                            break
                                        curr_rl = getattr(curr_rl, "__cause__", None) or getattr(curr_rl, "__context__", None)
                                    
                                    if is_rl:
                                        sleep_time = 5.0 + individual_attempt * 5.0
                                        print(f"Rate limited. Retrying individual submission in {sleep_time}s...")
                                        time.sleep(sleep_time)
                                    else:
                                        raise e3
                        break
                elif is_rate_limit:
                    if attempt == 5:
                        print("\n[WARNING] qBraid API rate limit exceeded during job submission. Falling back to local Qiskit simulation...")
                        fallback_to_local_sim = True
                        break
                    sleep_time = 5.0 + attempt * 5.0
                    print(f"Rate limited during job submission. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    if attempt == 5:
                        print(f"\n[WARNING] qBraid submission failed: {e}. Falling back to local Qiskit simulation...")
                        fallback_to_local_sim = True
                        break
                    time.sleep(2.0)

    if fallback_to_local_sim:
        # Local Qiskit Statevector simulation fallback
        from qiskit.quantum_info import Statevector, SparsePauliOp
        print("Simulating circuits locally using Qiskit Statevector...")
        sv = Statevector.from_instruction(bound_circuit)
        
        term_expectations = {}
        energy = 0.0
        for word, coeff in active_terms:
            op = SparsePauliOp(word)
            exp = sv.expectation_value(op).real
            term_expectations[word] = {
                "coeff_real": coeff,
                "coeff_imag": 0.0,
                "expectation": exp
            }
            energy += coeff * exp
            
        runtime = time.perf_counter() - start
        return {
            "energy": float(energy),
            "device": "local_qiskit_statevector_simulator",
            "shots": shots,
            "runtime_seconds": runtime,
            "term_expectations": term_expectations,
            "metadata": {
                "n_qubits": n_qubits,
                "n_electrons": n_electrons,
                "n_operators": len(operators),
                "n_hamiltonian_terms": len(molecule_record.get("terms", [])),
            }
        }

    # Determine job IDs returned
    if isinstance(run_res, list):
        job_ids = [j.id for j in run_res]
        is_list = True
    else:
        job_ids = [run_res.id]
        is_list = False

    metadata = {
        "job_ids": job_ids,
        "is_list": is_list,
        "molecule": molecule_record["name"],
        "device": device,
        "shots": shots,
        "group_mapping": group_mapping,
        "n_qubits": n_qubits,
        "metadata": {
            "n_qubits": n_qubits,
            "n_electrons": n_electrons,
            "n_operators": len(operators),
            "n_hamiltonian_terms": len(molecule_record.get("terms", [])),
            "n_groups": len(groups),
        }
    }

    if submit_only:
        if metadata_out_path:
            metadata_out_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_out_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(f"Submitted asynchronously! Job details saved to {metadata_out_path}")
        print(f"Job IDs: {job_ids}")
        return ", ".join(job_ids)

    # Otherwise, wait synchronously for completion
    print("Waiting synchronously for job completion...")
    jobs = [load_job(jid) for jid in job_ids]

    results = []
    if is_list:
        for ji, job in enumerate(jobs):
            if len(jobs) > 10 and ji % 10 == 0:
                print(f"  Retrieving results: {ji}/{len(jobs)}...")
            for attempt in range(6):
                try:
                    results.append(job.result())
                    break
                except Exception as e:
                    if attempt < 5:
                        wait = 3.0 * (attempt + 1)
                        print(f"  Job {ji}/{len(jobs)} result retry {attempt+1}/6 in {wait}s ({e})")
                        time.sleep(wait)
                        job = load_job(job_ids[ji])  # reload job
                    else:
                        raise
    else:
        batch_res = jobs[0].result()
        if hasattr(batch_res, "results"):
            results = list(batch_res.results)
        elif isinstance(batch_res, list):
            results = batch_res
        else:
            results = [batch_res]

    energy, term_expectations = _parse_grouped_results(results, group_mapping, n_qubits, shots)
    runtime = time.perf_counter() - start

    return {
        "energy": float(energy),
        "device": device,
        "shots": shots,
        "runtime_seconds": runtime,
        "term_expectations": term_expectations,
        "metadata": metadata["metadata"],
    }


def _parse_grouped_results(
    results: list[Any],
    group_mapping: list[list[dict[str, Any]]],
    n_qubits: int,
    shots: int,
) -> tuple[float, dict[str, Any]]:
    """Parse grouped job results and compute ground state energy.

    Each result corresponds to one QWC group.  All terms in a group share
    the same measurement circuit, so we extract each term's expectation
    from the *same* counts using that term's parity bitmask.
    """
    energy = 0.0
    term_expectations = {}

    for gi, terms_in_group in enumerate(group_mapping):
        counts = _get_counts(results[gi])
        n_shots = sum(counts.values())
        if n_shots == 0:
            for t in terms_in_group:
                word, coeff = t["term"], t["coeff"]
                term_expectations[word] = {"coeff_real": coeff, "coeff_imag": 0.0, "expectation": 0.0}
            continue

        for t in terms_in_group:
            word = t["term"]
            coeff = t["coeff"]
            padded = word + "I" * (n_qubits - len(word)) if len(word) < n_qubits else word

            exp = 0.0
            for bitstring, count in counts.items():
                # Qiskit little-endian: bitstring[q] is qubit q (leftmost = most significant)
                # Pauli word position q maps to qiskit qubit n_qubits-1-q
                parity = sum(
                    int(bitstring[q])
                    for q, op in enumerate(padded)
                    if op != "I"
                ) % 2
                sign = -1 if parity == 1 else 1
                exp += sign * count / n_shots

            term_expectations[word] = {
                "coeff_real": coeff,
                "coeff_imag": 0.0,
                "expectation": exp,
            }
            energy += coeff * exp

    return energy, term_expectations


def retrieve_qbraid_job(metadata_file: Path, out_path: Path) -> None:
    """Retrieve asynchronously submitted job results, parse them, and save ground state energy."""
    if load_job is None:
        raise ImportError("qBraid SDK not installed. Install it with: pip install qbraid")

    with metadata_file.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    job_ids = metadata["job_ids"]
    is_list = metadata["is_list"]
    group_mapping = metadata["group_mapping"]
    n_qubits = metadata["n_qubits"]
    molecule = metadata["molecule"]
    device = metadata["device"]
    shots = metadata["shots"]

    print(f"Checking status for jobs: {job_ids} on {device}...")
    jobs = [load_job(jid) for jid in job_ids]
    statuses = [str(job.status()) for job in jobs]
    print(f"Current statuses: {statuses}")

    # Check if all completed
    active = [s for s in statuses if s not in ("COMPLETED", "SUCCESS", "DONE", "failed", "cancelled")]
    if active:
        print("Jobs are still in progress. Please run retrieval again later.")
        return

    failed = [s for s in statuses if s in ("failed", "cancelled")]
    if failed:
        print(f"Error: One or more QPU jobs failed or were cancelled: {statuses}")
        return

    print("All jobs completed! Fetching results...")
    start = time.perf_counter()
    results = []
    if is_list:
        for job in jobs:
            results.append(job.result())
    else:
        batch_res = jobs[0].result()
        if hasattr(batch_res, "results"):
            results = list(batch_res.results)
        elif isinstance(batch_res, list):
            results = batch_res
        else:
            results = [batch_res]

    energy, term_expectations = _parse_grouped_results(results, group_mapping, n_qubits, shots)
    runtime = time.perf_counter() - start

    final_result = {
        "energy": float(energy),
        "device": device,
        "shots": shots,
        "runtime_seconds": runtime,
        "term_expectations": term_expectations,
        "metadata": metadata["metadata"],
        "molecule": molecule,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2)
    print(f"\nSuccessfully retrieved and processed job results!")
    print(f"qBraid ground state energy for {molecule}: {energy:.6f} Ha")
    print(f"Saved results to: {out_path}")


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

    for ops, coeff in iter_terms(molecule_record):
        word = "".join(ops)
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
    parser.add_argument("--hamiltonians", type=Path, default=None)
    parser.add_argument("--generated", type=Path, default=None)
    parser.add_argument("--optimized", type=Path, default=None)
    parser.add_argument("--molecule", type=str, default=None)
    parser.add_argument("--device", type=str, default="qbraid_qir_simulator")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument(
        "--max-credits",
        type=float,
        default=None,
        help="Refuse paid submission when estimated batch cost exceeds this budget",
    )
    parser.add_argument("--out", type=Path, default=None)
    
    # Asynchronous/batch execution options
    parser.add_argument("--submit-only", action="store_true", help="Submit batch job asynchronously and exit")
    parser.add_argument("--retrieve", type=Path, default=None, help="Retrieve asynchronously submitted job from metadata file")
    args = parser.parse_args()

    # Retrieve Mode
    if args.retrieve:
        if not args.out:
            # Load metadata to guess output path
            with args.retrieve.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            args.out = Path(f"results/eval/qbraid_{meta['molecule']}_{meta['device'].replace(':', '_')}.json")
        retrieve_qbraid_job(args.retrieve, args.out)
        return

    # Submit/Run Mode
    if not (args.hamiltonians and args.generated and args.optimized and args.molecule and args.out):
        parser.print_help()
        sys.exit("\nError: --hamiltonians, --generated, --optimized, --molecule, and --out are required for submission.")

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

    if args.max_credits is not None:
        from scripts.qpu_preflight import KNOWN_PRICING, estimate_cost

        if args.device in KNOWN_PRICING:
            estimate = estimate_cost(args.device, args.shots, len(record.get("terms", [])))
            print(
                f"Estimated batch cost: {estimate['batch_cost_credits']} credits "
                f"(budget={args.max_credits})"
            )
            if estimate["batch_cost_credits"] > args.max_credits:
                raise SystemExit(
                    "Refusing qBraid submission: estimated cost exceeds --max-credits."
                )
        else:
            raise SystemExit(
                f"No pricing is configured for {args.device}; refusing paid submission."
            )

    metadata_file = args.out.parent / f"qbraid_job_metadata_{args.molecule}_{args.device.replace(':', '_')}.json"

    res = evaluate_energy_qbraid_batched(
        record,
        operators,
        theta_values=np.asarray(thetas),
        device=args.device,
        shots=args.shots,
        submit_only=args.submit_only,
        metadata_out_path=metadata_file if args.submit_only else None
    )

    if not args.submit_only:
        assert isinstance(res, dict)
        res["molecule"] = args.molecule
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"qBraid energy for {args.molecule}: {res['energy']:.6f} Ha")
        print(f"Result saved to: {args.out}")


if __name__ == "__main__":
    main()
