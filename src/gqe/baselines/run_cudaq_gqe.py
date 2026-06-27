"""CUDA-Q Generative Quantum Eigensolver (GQE) over dataset Hamiltonians."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

# cudaq-solvers[gqe] imports PyTorch; on HPC nodes the bundled PyTorch CUDA
# build may not match the host driver, which emits a noisy UserWarning even
# when CUDA-Q itself runs fine. Silence only these known messages.
warnings.filterwarnings(
    "ignore",
    message=r".*NVIDIA driver on your system is too old.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*CUDA initialization:.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*PyTorch wheel with CUDA.*",
    category=UserWarning,
)

for _log in ("torch", "torch.cuda"):
    logging.getLogger(_log).setLevel(logging.ERROR)

# Workaround for PyTorch issue #95648: sm_89 (Ada Lovelace / L40S) is
# functionally identical to sm_86 but not listed in get_arch_list(), which
# causes cudaq-solvers GQE to refuse GPU execution.
import torch  # noqa: E402
_orig_get_arch_list = torch.cuda.get_arch_list
def _patched_get_arch_list() -> list[str]:
    archs = list(_orig_get_arch_list())
    if "sm_89" not in archs:
        archs.append("sm_89")
    return archs
torch.cuda.get_arch_list = _patched_get_arch_list  # type: ignore[assignment]

try:  # pragma: no cover - resolved at runtime
    import cudaq  # type: ignore[import-untyped]
    import cudaq_solvers as solvers  # type: ignore[import-untyped]
    from cudaq_solvers.gqe_algorithm.gqe import get_default_config  # type: ignore[import-untyped]
except Exception as exc:  # pragma: no cover
    cudaq = None  # type: ignore[assignment]
    solvers = None  # type: ignore[assignment]
    get_default_config = None  # type: ignore[assignment]
    _CUDAQ_IMPORT_ERROR: Optional[Exception] = exc
else:  # pragma: no cover
    _CUDAQ_IMPORT_ERROR = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.hamiltonian_utils import (  # noqa: E402
    exact_diagonalize_hamiltonian,
    hamiltonian_to_spin_operator,
    iter_terms,
    load_hamiltonian_records,
    pauli_ops_to_spin_term,
)
from common.operator_pool import build_uccsd_operator_pool  # noqa: E402


PERIODIC_TABLE = {
    "H": 1,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "I": 53,
}


def _ensure_cudaq_available() -> None:
    if cudaq is None or solvers is None or get_default_config is None:
        msg = (
            "CUDA-Q GQE baseline requires `cudaq` and `cudaq-solvers[gqe]`.\n"
            "Install into the experiment environment, for example:\n"
            "  pip install cudaq-solvers[gqe]"
        )
        raise RuntimeError(msg) from _CUDAQ_IMPORT_ERROR


def _ensure_cuda_context() -> None:
    """Create a CUDA context on the GPU assigned to this MPI rank.

    Open MPI's smcuda BTL needs each rank to have a CUDA context before
    MPI_Init() so it can set up GPU-buffer communication. Without this,
    it falls back to TCP which cannot handle device pointers.
    """
    import ctypes
    import os

    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    libcudart = ctypes.CDLL("/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib/libcudart.so")
    libcudart.cudaSetDevice(local_rank)
    d = ctypes.c_void_p()
    libcudart.cudaMalloc(ctypes.byref(d), 4)
    libcudart.cudaFree(d)


def _configure_target(preferred: str, option: str | None = None) -> str:
    assert cudaq is not None
    try:
        needs_mpi = (option and "mgpu" in option) or preferred == "tensornet"
        if needs_mpi:
            if not cudaq.mpi.is_initialized():
                _ensure_cuda_context()
                cudaq.mpi.initialize()
                print(f"MPI initialized: rank={cudaq.mpi.rank()}, num_ranks={cudaq.mpi.num_ranks()}")
        if option:
            cudaq.set_target(preferred, option=option)
        else:
            cudaq.set_target(preferred)
        return preferred
    except RuntimeError:
        if preferred != "qpp-cpu":
            cudaq.set_target("qpp-cpu")
            return "qpp-cpu"
        raise


def _estimate_active_electrons(record: Dict[str, Any]) -> int:
    active = record.get("active_space") or {}
    for key in ("n_active_electrons", "n_electrons"):
        value = active.get(key) if isinstance(active, dict) else None
        if value is None:
            value = record.get(key)
        if isinstance(value, int) and value > 0:
            return min(value, int(record["n_qubits"]))

    geometry = record.get("geometry", [])
    total = 0
    for entry in geometry:
        if not isinstance(entry, list) or not entry:
            continue
        symbol = str(entry[0]).capitalize()
        total += PERIODIC_TABLE.get(symbol, 0)
    electrons = total - int(record.get("charge", 0))
    electrons = max(1, min(electrons, int(record["n_qubits"])))
    return electrons


def _serialize_selected_operators(
    op_pool: Sequence[tuple[Any, complex, str]], indices: Iterable[int], n_qubits: int
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx in indices:
        op, coeff, pw_str = op_pool[int(idx)]
        records.append(
            {
                "index": int(idx),
                "coefficient_real": float(coeff.real),
                "coefficient_imag": float(coeff.imag),
                "pauli_word": pw_str,
            }
        )
    return records

def _build_operator_pool(
    record: Dict[str, Any],
    *,
    max_terms: int,
    scale_factors: Sequence[float],
) -> List[tuple[Any, complex, str]]:
    return build_uccsd_operator_pool(record, scale_factors=scale_factors)


def _energy_shift_and_scale(record: Dict[str, Any]) -> tuple[float, float]:
    shift = 0.0
    l1_norm = 0.0
    for ops, coeff in iter_terms(record):
        coeff_real = float(np.real(coeff))
        l1_norm += abs(coeff_real)
        if all(op == "I" for op in ops):
            shift += coeff_real
    return shift, max(1.0, l1_norm)


def _run_record(
    record: Dict[str, Any],
    *,
    max_iters: int,
    ngates: int,
    max_terms: int,
    scale_factors: Sequence[float],
    target: str,
    target_option: str | None = None,
) -> Dict[str, Any]:
    _ensure_cudaq_available()
    configured_target = _configure_target(target, target_option)
    n_qubits = int(record["n_qubits"])
    n_electrons = _estimate_active_electrons(record)
    spin_ham = hamiltonian_to_spin_operator(record)
    op_pool = _build_operator_pool(record, max_terms=max_terms, scale_factors=scale_factors)
    if not op_pool:
        raise RuntimeError("Operator pool is empty; increase --max-terms or adjust scales.")
    energy_shift, energy_scale = _energy_shift_and_scale(record)

    @cudaq.kernel  # type: ignore[union-attr]
    def kernel(
        n_qubits_kernel: int,
        n_electrons_kernel: int,
        coeffs: List[float],
        words: List[cudaq.pauli_word],
    ) -> None:
        q = cudaq.qvector(n_qubits_kernel)  # type: ignore[union-attr]
        for i in range(n_electrons_kernel):
            x(q[i])  # type: ignore[name-defined]
        for j in range(len(coeffs)):
            exp_pauli(coeffs[j], q, words[j])  # type: ignore[name-defined]

    # Build lookup: SpinOperator id -> (coefficient, pauli_string)
    _op_to_data: dict[int, tuple[complex, str]] = {}
    for _op, _c, _pstr in op_pool:
        _op_to_data[id(_op)] = (_c, _pstr)

    def _raw_cost(sampled_ops: List[Any]) -> float:
        coeffs: List[float] = []
        words: List[cudaq.pauli_word] = []  # type: ignore[name-defined]
        for op in sampled_ops:
            data = _op_to_data.get(id(op))
            if data is None:
                raise RuntimeError("Sampled operator was not found in the operator pool lookup.")
            _c, _pstr = data
            coeffs.append(float(np.real(_c)))
            words.append(_pstr)
        result = cudaq.observe(  # type: ignore[union-attr]
            kernel,
            spin_ham,
            n_qubits,
            n_electrons,
            coeffs,
            words,
        )
        return float(result.expectation())

    def cost(sampled_ops: List[Any], **_: Any) -> float:
        raw_energy = _raw_cost(sampled_ops)
        return (raw_energy - energy_shift) / energy_scale

    cfg = get_default_config()  # type: ignore[operator]
    cfg.use_fabric_logging = False
    cfg.save_trajectory = False
    cfg.verbose = False

    operators_only = [op for op, _coeff, _pstr in op_pool]
    min_energy, best_indices = solvers.gqe(  # type: ignore[union-attr]
        cost,
        operators_only,
        max_iters=max_iters,
        ngates=ngates,
        config=cfg,
    )

    reference_energy = None
    try:
        reference_energy, _ = exact_diagonalize_hamiltonian(record)
    except Exception:
        reference_energy = None

    best_ops = [operators_only[int(i)] for i in best_indices]
    baseline_energy = _raw_cost(best_ops)
    delta = None
    if reference_energy is not None:
        delta = abs(baseline_energy - reference_energy)

    return {
        "system": record.get("name", "unknown"),
        "baseline": "cudaq_gqe",
        "reference_energy": reference_energy,
        "baseline_energy": baseline_energy,
        "delta_energy": delta,
        "n_spin_orbitals": n_qubits,
        "n_pauli_terms": len(record.get("terms", [])),
        "mode": "cudaq_gqe",
        "gqe_config": {
            "max_iters": max_iters,
            "ngates": ngates,
            "num_samples": int(getattr(cfg, "num_samples", 5)),
            "objective_shift": energy_shift,
            "objective_scale": energy_scale,
            "min_scaled_energy": float(min_energy),
        },
        "n_electrons": n_electrons,
        "target": configured_target,
        "target_option": target_option,
        "gqe_selected_operators": _serialize_selected_operators(
            op_pool,
            (int(i) for i in best_indices),
            n_qubits,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CUDA-Q GQE baseline on dataset Hamiltonians.")
    parser.add_argument("--ham", type=Path, required=True, help="Path to hamiltonians.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--molecule", type=str, default=None, help="Optional molecule filter")
    parser.add_argument("--max-iters", type=int, default=25)
    parser.add_argument("--ngates", type=int, default=10)
    parser.add_argument("--max-terms", type=int, default=32, help="Number of Pauli terms to seed the operator pool")
    parser.add_argument(
        "--pool-scale",
        type=float,
        nargs="*",
        default=(0.0125, -0.0125, 0.025, -0.025, 0.05, -0.05),
        help="Scale factors applied to each selected Pauli term when building the operator pool",
    )
    parser.add_argument("--target", type=str, default="nvidia", help="Preferred cudaq target (e.g., nvidia, nvidia-mqpu, qpp-cpu)")
    parser.add_argument("--target-option", type=str, default=None, help="Target option string (e.g., mqpu,fp32)")
    parser.add_argument("--max-qubits", type=int, default=16, help="Skip systems with more than this many qubits")
    args = parser.parse_args()

    records = load_hamiltonian_records(args.ham)
    if args.molecule:
        records = [r for r in records if r.get("name") == args.molecule]

    results: List[Dict[str, Any]] = []
    for rec in tqdm(records, desc="CUDA-Q GQE", unit="system", dynamic_ncols=True):
        n_qubits = int(rec.get("n_qubits", 0))
        if n_qubits > args.max_qubits:
            results.append(
                {
                    "system": rec.get("name", "unknown"),
                    "baseline": "cudaq_gqe",
                    "status": f"skipped_qubits>{args.max_qubits}",
                    "n_spin_orbitals": n_qubits,
                    "n_pauli_terms": len(rec.get("terms", [])),
                }
            )
            continue
        try:
            results.append(
                _run_record(
                    rec,
                    max_iters=args.max_iters,
                    ngates=args.ngates,
                    max_terms=args.max_terms,
                    scale_factors=args.pool_scale,
                    target=args.target,
                    target_option=args.target_option,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "system": rec.get("name", "unknown"),
                    "baseline": "cudaq_gqe",
                    "status": f"error: {exc}",
                    "n_spin_orbitals": rec.get("n_qubits"),
                    "n_pauli_terms": len(rec.get("terms", [])),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Wrote CUDA-Q GQE baseline results to: {args.out}")


if __name__ == "__main__":
    main()

