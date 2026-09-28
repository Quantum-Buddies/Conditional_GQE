"""CUDA-Q hardware-efficient VQE baseline over generated Hamiltonians."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm

try:  # pragma: no cover - runtime dependency
    import cudaq  # type: ignore[import-untyped]
    import cudaq_solvers as solvers  # type: ignore[import-untyped]
except Exception as exc:  # pragma: no cover - handled via helper
    cudaq = None  # type: ignore[assignment]
    solvers = None  # type: ignore[assignment]
    _CUDAQ_IMPORT_ERROR: Optional[Exception] = exc
else:  # pragma: no cover - executed at runtime
    _CUDAQ_IMPORT_ERROR = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.hamiltonian_utils import (  # noqa: E402
    exact_diagonalize_hamiltonian,
    hamiltonian_to_spin_operator,
    load_hamiltonian_records,
)


def _ensure_cudaq_available() -> None:
    if cudaq is None or solvers is None:
        raise RuntimeError(
            "cudaq and cudaq_solvers are required for this baseline."
        ) from _CUDAQ_IMPORT_ERROR


def _configure_target(preferred: str) -> str:
    assert cudaq is not None
    try:
        cudaq.set_target(preferred)
        return preferred
    except RuntimeError:
        if preferred != "qpp-cpu":
            cudaq.set_target("qpp-cpu")
            return "qpp-cpu"
        raise


def _build_hwe_ansatz(n_qubits: int, reps: int):
    kernel, thetas = cudaq.make_kernel(list)  # type: ignore[union-attr]
    qubits = kernel.qalloc(n_qubits)
    idx = 0
    for _ in range(reps):
        for q in range(n_qubits):
            kernel.ry(thetas[idx], qubits[q])
            idx += 1
        for q in range(n_qubits - 1):
            kernel.cx(qubits[q], qubits[q + 1])
    return kernel, idx


def _run_record(
    record: Dict[str, Any],
    *,
    maxiter: int,
    method: str,
    reps: int,
    target: str,
) -> Dict[str, Any]:
    _ensure_cudaq_available()
    configured_target = _configure_target(target)
    n_qubits = int(record["n_qubits"])
    spin_op = hamiltonian_to_spin_operator(record)
    ansatz, n_params = _build_hwe_ansatz(n_qubits, reps)
    init_params = [0.0] * n_params

    vqe_kwargs: Dict[str, Any] = {
        "optimizer": minimize,
        "method": method,
        "options": {"maxiter": maxiter},
    }
    if method.upper() == "L-BFGS-B":
        vqe_kwargs["jac"] = "3-point"
        vqe_kwargs["tol"] = 1e-4

    start = time.perf_counter()
    energy, _, _ = solvers.vqe(ansatz, spin_op, init_params, **vqe_kwargs)
    runtime = time.perf_counter() - start
    baseline_energy = float(np.real(energy))

    reference_energy = None
    try:
        reference_energy, _ = exact_diagonalize_hamiltonian(record)
    except Exception:
        reference_energy = None

    delta = None
    if reference_energy is not None:
        delta = abs(baseline_energy - reference_energy)

    return {
        "system": record.get("name", "unknown"),
        "baseline": "cudaq_vqe",
        "reference_energy": reference_energy,
        "baseline_energy": baseline_energy,
        "delta_energy": delta,
        "n_spin_orbitals": n_qubits,
        "n_pauli_terms": len(record.get("terms", [])),
        "mode": "cudaq_hwe_vqe",
        "ansatz_reps": reps,
        "optimizer_method": method,
        "maxiter": maxiter,
        "runtime_sec": runtime,
        "target": configured_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CUDA-Q VQE on generated Hamiltonians.")
    parser.add_argument("--ham", type=Path, required=True, help="Path to hamiltonians.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--molecule", type=str, default=None, help="Optional molecule filter")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--method", type=str, default="COBYLA", help="scipy optimizer method")
    parser.add_argument("--ansatz-reps", type=int, default=2)
    parser.add_argument("--target", type=str, default="nvidia", help="Preferred cudaq target (e.g., nvidia, nvidia-mqpu, qpp-cpu)")
    args = parser.parse_args()

    records = load_hamiltonian_records(args.ham)
    if args.molecule:
        records = [r for r in records if r.get("name") == args.molecule]

    results: List[Dict[str, Any]] = []
    for rec in tqdm(records, desc="CUDA-Q VQE", unit="system", dynamic_ncols=True):
        try:
            results.append(
                _run_record(
                    rec,
                    maxiter=args.maxiter,
                    method=args.method,
                    reps=args.ansatz_reps,
                    target=args.target,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "system": rec.get("name", "unknown"),
                    "baseline": "cudaq_vqe",
                    "status": f"error: {exc}",
                    "n_spin_orbitals": rec.get("n_qubits"),
                    "n_pauli_terms": len(rec.get("terms", [])),
                    "mode": "cudaq_hwe_vqe",
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Wrote CUDA-Q baseline results to: {args.out}")


if __name__ == "__main__":
    main()

