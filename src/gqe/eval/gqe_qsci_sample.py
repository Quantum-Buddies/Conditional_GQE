"""Sample a CUDA-Q Solvers GQE circuit and classically QSCI-diagonalize.

Train GQE on GPU with ``run_cudaq_gqe.py``. This script **samples** the
emitted HF + ``exp_pauli`` kernel (``cudaq.sample`` / ``sample_async``) and
does **not** call ``solvers.gqe`` on a QPU.

Backends:
  nvidia / nvidia-fp64  — AIRE L40S
  qbraid-qir            — ``qbraid:qbraid:sim:qir-sv`` (free, ≤2000 shots)
  qbraid-cepheus        — ``aws:rigetti:qpu:cepheus-1-108q`` (needs CUDAQ_ALLOW_QPU=1)

Never pass API keys. Never re-rank DFT ligands.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import (
    find_record_by_name,
    get_active_electron_count,
    load_hamiltonian_records,
)
from src.gqe.eval.qsci_map import cudaq_counts_to_lsb, unique_bitstrings_from_counts
from src.gqe.eval.qsci_postprocess import qsci_energy_from_bitstrings
from src.gqe.eval.sqd import (
    run_sqd,
    self_consistent_config_recovery,
)

GATE1_CASCI_HA = -288.10912104
CHEMICAL_ACCURACY_MHA = 1.6
N_FCI_SNO14 = 49
DEFAULT_MAX_TWO_QUBIT = 400


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_gqe_operators(
    gqe_json: Path,
    molecule: str,
) -> tuple[list[str], list[float], dict[str, Any]]:
    """Read Pauli words and coefficients from ``run_cudaq_gqe.py`` output."""
    data = json.loads(gqe_json.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]]
    if isinstance(data, list):
        rows = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict):
        rows = [r for r in data.get("results", []) if isinstance(r, dict)]
        if not rows and data.get("gqe_selected_operators"):
            rows = [data]
    else:
        rows = []
    chosen = None
    for row in rows:
        name = str(row.get("system") or row.get("molecule") or "")
        if name == molecule:
            chosen = row
            break
    if chosen is None and len(rows) == 1:
        chosen = rows[0]
    if chosen is None:
        raise FileNotFoundError(
            f"No GQE result for molecule {molecule!r} in {gqe_json}"
        )
    ops = chosen.get("gqe_selected_operators") or []
    words = [str(item["pauli_word"]) for item in ops]
    thetas = [float(item["coefficient_real"]) for item in ops]
    if not words:
        raise ValueError(f"GQE JSON {gqe_json} has empty operator list for {molecule}")
    return words, thetas, chosen


def estimate_two_qubit_from_pauli_words(words: list[str]) -> int:
    """Lower-bound CNOT count: each weight-w ``exp_pauli`` needs 2*(w-1) CNOTs."""
    n = 0
    for word in words:
        support = sum(ch not in {"I", "i"} for ch in str(word))
        if support >= 2:
            n += 2 * (support - 1)
    return n


def count_two_qubit_gates_qasm(qasm: str) -> int:
    """Count CX/CZ/CNOT/CP/ISWAP-like ops in OpenQASM text."""
    n = 0
    for line in qasm.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("//") or stripped.startswith("include") or stripped.startswith("qreg"):
            continue
        if re.search(r"\b(cx|cz|cy|cnot|cp|crx|cry|crz|iswap|swap|ccx)\b", stripped):
            n += 1
    return n


def translate_two_qubit_count(kernel: Any, kernel_args: tuple) -> int | None:
    try:
        import cudaq
    except ImportError:
        return None
    translate = getattr(cudaq, "translate", None)
    if translate is None:
        return None
    try:
        text = translate(kernel, *kernel_args)
    except TypeError:
        try:
            text = translate(kernel)
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(text, str):
        text = str(text)
    return count_two_qubit_gates_qasm(text)


def qsci_from_cudaq_counts(
    record: dict[str, Any],
    cudaq_counts: dict[str, int],
    *,
    n_samples: int,
    n_electrons: int,
    recover: bool,
) -> dict[str, Any]:
    lsb_counts = cudaq_counts_to_lsb(cudaq_counts)
    if recover:
        lsb_counts = self_consistent_config_recovery(
            lsb_counts,
            int(record["n_qubits"]),
            n_electrons,
        )
    unique = unique_bitstrings_from_counts(lsb_counts)
    chosen = unique[: max(1, int(n_samples))]
    energy = qsci_energy_from_bitstrings(record, chosen)
    sqd = run_sqd(
        record,
        lsb_counts,
        n_electrons=n_electrons,
        subspace_size=int(n_samples),
        particle_number_tol=0,
        spin_parity=0,
        n_recovered=0,
        return_details=True,
        reverse_bit_order=False,
    )
    return {
        "qsci_energy_ha": float(energy),
        "n_unique_lsb": len(unique),
        "n_dets_used": len(chosen),
        "bitstrings": chosen,
        "sqd": {
            "energy": sqd.get("energy"),
            "n_bitstrings": sqd.get("n_bitstrings"),
            "n_unique_raw": sqd.get("n_unique_raw"),
            "rejection": sqd.get("rejection"),
        },
        "recovered": bool(recover),
    }


def evaluate_delta_mha(energy_ha: float, casci_ha: float) -> float:
    return (float(energy_ha) - float(casci_ha)) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="QSCI-sample a CUDA-Q Solvers GQE circuit")
    parser.add_argument("--gqe-json", type=Path, required=True)
    parser.add_argument("--hamiltonians", type=Path, required=True)
    parser.add_argument("--molecule", type=str, default="sno_14q")
    parser.add_argument(
        "--backend",
        type=str,
        default="nvidia",
        help="nvidia | nvidia-fp64 | qbraid | qbraid-qir | qbraid-cepheus | qpp-cpu",
    )
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--casci", type=float, default=GATE1_CASCI_HA)
    parser.add_argument("--n-fci", type=int, default=N_FCI_SNO14)
    parser.add_argument("--pass-mha", type=float, default=CHEMICAL_ACCURACY_MHA)
    parser.add_argument("--recover", action="store_true", help="S-CORE particle-number repair")
    parser.add_argument("--async-submit", action="store_true")
    parser.add_argument("--future-path", type=Path, default=None)
    parser.add_argument("--max-two-qubit-gates", type=int, default=DEFAULT_MAX_TWO_QUBIT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.backend in {"qbraid-qir", "qbraid"} and args.shots > 2000:
        print("Clamping shots to 2000 for qBraid QIR-SV")
        args.shots = 2000

    records = load_hamiltonian_records(args.hamiltonians)
    record = find_record_by_name(records, args.molecule)
    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)
    words, thetas, gqe_row = load_gqe_operators(args.gqe_json, args.molecule)

    from src.gqe.eval.cudaq_qbraid_target import persist_async_result
    from src.gqe.eval.qsci import _make_hcgqe_kernel, _set_cudaq_target, sample_counts_cudaq

    os.environ.setdefault("CUDAQ_NVIDIA_OPTION", "fp64")
    _set_cudaq_target(args.backend)

    kernel, pauli_words, thetas_used = _make_hcgqe_kernel(
        words, thetas, n_qubits, n_electrons
    )
    kernel_args = (n_qubits, n_electrons, pauli_words, thetas_used)

    n2q_qasm = translate_two_qubit_count(kernel, kernel_args)
    n2q_pauli = estimate_two_qubit_from_pauli_words(words)
    n2q = n2q_qasm if n2q_qasm not in (None, 0) else n2q_pauli
    hardware_sample = args.backend == "qbraid-cepheus" or (
        args.backend == "qbraid"
        and "cepheus" in os.environ.get("CUDAQ_QBRAID_MACHINE", "")
    )
    if n2q is not None and hardware_sample and n2q > int(args.max_two_qubit_gates):
        raise SystemExit(
            f"Abort: translated two-qubit count {n2q} > {args.max_two_qubit_gates}"
        )

    future_path = args.future_path
    cudaq_counts: dict[str, int] | None = None
    use_async = bool(args.async_submit) or args.backend.startswith("qbraid")
    try:
        if use_async:
            import cudaq

            future = cudaq.sample_async(
                kernel, *kernel_args, shots_count=int(args.shots)
            )
            if future_path is None:
                future_path = args.out.with_suffix(".future.json")
            persist_async_result(future, future_path)
            tqdm.write(f"Submitted sample_async → {future_path}")
            result = future.get()
            cudaq_counts = {str(bs): int(c) for bs, c in result.items()}
        else:
            cudaq_counts = sample_counts_cudaq(
                kernel,
                kernel_args,
                n_shots=int(args.shots),
                seed=int(args.seed),
            )
    except Exception as exc:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        err_payload = {
            "experiment": "cudaq-gqe-cepheus-sno14",
            "updated_at": _now_iso(),
            "molecule": args.molecule,
            "backend": args.backend,
            "qbraid_machine": os.environ.get("CUDAQ_QBRAID_MACHINE"),
            "n_ops": len(words),
            "two_qubit_gates": n2q,
            "two_qubit_gates_qasm": n2q_qasm,
            "two_qubit_gates_pauli_estimate": n2q_pauli,
            "pass_chemical_accuracy": False,
            "sample_error": f"{type(exc).__name__}: {exc}",
            "future_path": str(future_path) if future_path else None,
            "note": (
                "qBraid sample failed before QSCI. Cepheus stays gated. "
                "Do not set CUDAQ_ALLOW_QPU."
            ),
        }
        args.out.write_text(json.dumps(err_payload, indent=2), encoding="utf-8")
        print(f"SAMPLE_FAIL {type(exc).__name__}: {exc} → {args.out}")
        sys.exit(3)

    scored = qsci_from_cudaq_counts(
        record,
        cudaq_counts,
        n_samples=int(args.n_samples),
        n_electrons=n_electrons,
        recover=bool(args.recover),
    )
    delta = evaluate_delta_mha(scored["qsci_energy_ha"], float(args.casci))
    n_dets = int(scored["n_dets_used"])
    passed = bool(delta <= float(args.pass_mha) and n_dets < int(args.n_fci))

    payload = {
        "experiment": "cudaq-gqe-cepheus-sno14",
        "updated_at": _now_iso(),
        "molecule": args.molecule,
        "backend": args.backend,
        "qbraid_machine": os.environ.get("CUDAQ_QBRAID_MACHINE"),
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "n_ops": len(words),
        "operators": words,
        "thetas": thetas,
        "shots": int(args.shots),
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "casci_ha": float(args.casci),
        "qsci_energy_ha": scored["qsci_energy_ha"],
        "delta_e_qsci_mha": delta,
        "n_dets": n_dets,
        "n_fci": int(args.n_fci),
        "n_unique_raw_cudaq": len(cudaq_counts),
        "two_qubit_gates_qasm": n2q_qasm,
        "two_qubit_gates": n2q,
        "two_qubit_gates_pauli_estimate": n2q_pauli,
        "pass_chemical_accuracy": passed,
        "filled_full_ci": n_dets >= int(args.n_fci),
        "recovered": bool(args.recover),
        "gqe_baseline_energy": gqe_row.get("baseline_energy"),
        "future_path": str(future_path) if future_path else None,
        "cudaq_allow_qpu": os.environ.get("CUDAQ_ALLOW_QPU", ""),
        "note": (
            "Library GQE proposed the circuit; energy is classical QSCI vs CASCI. "
            "Not a DFT ΔBDE. Not trained 14-char H-cGQE unless words came from that checkpoint."
        ),
        "sqd": scored["sqd"],
        "bitstrings": scored["bitstrings"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"QSCI E={payload['qsci_energy_ha']:.8f} Ha  "
        f"ΔE={delta:.3f} mHa  dets={n_dets}/{args.n_fci}  "
        f"pass={passed}  → {args.out}"
    )
    if not passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
