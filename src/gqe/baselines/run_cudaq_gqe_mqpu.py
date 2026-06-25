"""CUDA-Q GQE with multi-QPU (mqpu) backend for Hamiltonian batching across GPUs.

Usage (single-node multi-GPU):
    CUDA-Q supports `mqpu` backend which distributes Hamiltonian terms across
    GPUs during expectation value evaluation.

    python -m src.gqe.baselines.run_cudaq_gqe_mqpu \
        --ham results/data/hamiltonians.json \
        --out results/mqpu_gqe.json \
        --target nvidia --target-option mqpu
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=r".*NVIDIA driver on your system is too old.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", message=r".*CUDA initialization:.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*PyTorch wheel with CUDA.*", category=UserWarning)

for _log in ("torch", "torch.cuda"):
    logging.getLogger(_log).setLevel(logging.ERROR)

import torch  # noqa: E402

_orig_get_arch_list = torch.cuda.get_arch_list

def _patched_get_arch_list() -> list[str]:
    archs = list(_orig_get_arch_list())
    if "sm_89" not in archs:
        archs.append("sm_89")
    return archs


torch.cuda.get_arch_list = _patched_get_arch_list  # type: ignore[assignment]

try:
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


PERIODIC_TABLE = {
    "H": 1,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
}


N_ELECTRONS_TABLE = PERIODIC_TABLE


def _ensure_cudaq_available() -> None:
    if cudaq is None:
        msg = f"cudaq / cudaq-solvers not available: {_CUDAQ_IMPORT_ERROR}"
        raise RuntimeError(msg)


def _configure_target(preferred: str, option: str | None = None) -> str:
    assert cudaq is not None
    try:
        if option:
            cudaq.set_target(preferred, option=option)
        else:
            cudaq.set_target(preferred)
        return preferred
    except Exception as exc:
        if preferred == "nvidia":
            try:
                cudaq.set_target("qpp-cpu")
                return "qpp-cpu"
            except Exception:
                pass
        raise RuntimeError(f"Failed to set target {preferred}: {exc}") from exc


def _estimate_active_electrons(record: Dict[str, Any]) -> int:
    """Estimate total active electrons from geometry and charge."""
    charge = float(record.get("charge", 0.0))
    geometry = record.get("geometry", {})
    if geometry and "coordinates" in geometry:
        atoms = geometry.get("atoms", [])
    else:
        atoms = record.get("atoms", [])
    if not atoms and "name" in record:
        return max(1, int(record.get("n_qubits", 1)) // 2)
    total_electrons = sum(N_ELECTRONS_TABLE.get(atom.get("element", "H"), 1) for atom in atoms)
    return max(1, int(total_electrons - charge))


def _build_operator_pool(
    record: Dict[str, Any],
    *,
    max_terms: int = 32,
    scale_factors: Sequence[float] = (0.0125, -0.0125, 0.025, -0.025, 0.05, -0.05),
) -> List[Any]:
    """Build a pool of operators for GQE from Hamiltonian terms."""
    _ensure_cudaq_available()
    entries = sorted(iter_terms(record), key=lambda item: abs(item[1]), reverse=True)
    pool: List[Any] = []
    used = 0
    for ops, coeff in entries:
        if used >= max_terms:
            break
        term_op = pauli_ops_to_spin_term(ops)
        if term_op is None:
            continue
        used += 1
        sign = 1.0 if coeff.real >= 0 else -1.0
        pauli_str = "".join(ops)
        for scale in scale_factors:
            pool.append((scale * sign * term_op, complex(scale * sign * abs(coeff)), pauli_str))
    return pool


def _serialize_selected_operators(
    op_pool: List[Any],
    indices: Iterable[int],
    n_qubits: int,
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


def _energy_shift_and_scale(record: Dict[str, Any]) -> tuple[float, float]:
    """Extract energy shift (identity term) and scaling factor."""
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
        "baseline": "cudaq_gqe_mqpu",
        "reference_energy": reference_energy,
        "baseline_energy": baseline_energy,
        "delta_energy": delta,
        "n_spin_orbitals": n_qubits,
        "n_pauli_terms": len(record.get("terms", [])),
        "mode": "cudaq_gqe_mqpu",
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
    parser = argparse.ArgumentParser(description="Run CUDA-Q GQE baseline with mqpu multi-GPU support.")
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
    parser.add_argument("--target", type=str, default="nvidia", help="Preferred cudaq target")
    parser.add_argument("--target-option", type=str, default=None, help="Target option (e.g., mqpu, mgpu)")
    parser.add_argument("--max-qubits", type=int, default=16, help="Skip systems with more than this many qubits")
    args = parser.parse_args()

    records = load_hamiltonian_records(args.ham)
    if args.molecule:
        records = [r for r in records if r.get("name") == args.molecule]

    results: List[Dict[str, Any]] = []
    for rec in tqdm(records, desc="CUDA-Q GQE (mqpu)", unit="system", dynamic_ncols=True):
        n_qubits = int(rec.get("n_qubits", 0))
        if n_qubits > args.max_qubits:
            results.append(
                {
                    "system": rec.get("name", "unknown"),
                    "baseline": "cudaq_gqe_mqpu",
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
                    "baseline": "cudaq_gqe_mqpu",
                    "status": f"error: {exc}",
                    "n_spin_orbitals": rec.get("n_qubits"),
                    "n_pauli_terms": len(rec.get("terms", [])),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Wrote CUDA-Q GQE mqpu results to: {args.out}")


if __name__ == "__main__":
    main()
