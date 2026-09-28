"""Export a GQE-optimized circuit as an OpenQASM demo pack for Quaggle.

Takes a precomputed (operators + thetas) pair from the H-cGQE pipeline and
produces a self-contained OpenQASM 2.0 file + metadata JSON that can be
imported into Quaggle's Circuit Builder or used as a standalone repro artifact.

Usage:
    python scripts/export_quaggle_demo.py \
        --optimized results/eval/h_cgqe_optimized.json \
        --molecule h2 \
        --out results/quaggle/

    # Or specify operators/thetas directly
    python scripts/export_quaggle_demo.py \
        --operators YZYI XXYY XZXI \
        --thetas 0.01 0.02 0.03 \
        --n-qubits 4 --n-electrons 2 \
        --molecule h2 \
        --out results/quaggle/h2_demo.qasm
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps


def _pad_pauli_word(word: str, n_qubits: int) -> str:
    if len(word) == n_qubits:
        return word
    if len(word) < n_qubits:
        return word + "I" * (n_qubits - len(word))
    return word[:n_qubits]


def build_ansatz_circuit(
    operators: list[str],
    thetas: list[float],
    n_qubits: int,
    n_electrons: int = 0,
) -> QuantumCircuit:
    """Build a Qiskit QuantumCircuit from GQE operator sequence + angles.

    Each Pauli word is decomposed into basis-change gates + entangling CNOT
    ladder + RZ rotation, following the standard Pauli-evolution gate
    decomposition used in VQE ansätze.
    """
    qc = QuantumCircuit(n_qubits, name="h_cgqe_ansatz")

    # Hartree-Fock initial state: X on first n_electrons qubits
    for i in range(min(n_electrons, n_qubits)):
        qc.x(i)

    for op_word, theta in zip(operators, thetas):
        word = _pad_pauli_word(op_word, n_qubits)

        # Find non-identity qubits
        active = [(i, ch) for i, ch in enumerate(word) if ch != "I"]
        if not active:
            continue

        # Basis change: H for X, Sdg-H for Y, nothing for Z
        for i, ch in active:
            if ch == "X":
                qc.h(i)
            elif ch == "Y":
                qc.sdg(i)
                qc.h(i)

        # CNOT ladder: chain from first to last active qubit
        active_qubits = [i for i, _ in active]
        for i in range(len(active_qubits) - 1):
            qc.cx(active_qubits[i], active_qubits[i + 1])

        # RZ rotation on last active qubit
        last_q = active_qubits[-1]
        qc.rz(2.0 * theta, last_q)

        # Undo CNOT ladder
        for i in range(len(active_qubits) - 2, -1, -1):
            qc.cx(active_qubits[i], active_qubits[i + 1])

        # Undo basis change
        for i, ch in active:
            if ch == "X":
                qc.h(i)
            elif ch == "Y":
                qc.h(i)
                qc.s(i)

    return qc


def export_demo_pack(
    operators: list[str],
    thetas: list[float],
    n_qubits: int,
    n_electrons: int,
    molecule: str,
    energy: float | None = None,
    hf_energy: float | None = None,
    fci_energy: float | None = None,
    source_checkpoint: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Export QASM + metadata for Quaggle import.

    Returns dict of {artifact_name: path}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    qc = build_ansatz_circuit(operators, thetas, n_qubits, n_electrons)

    # Add measurements for Quaggle compatibility
    qc_measured = qc.copy()
    qc_measured.measure_all()

    # Export QASM 2.0
    qasm_str = qasm2_dumps(qc_measured)
    qasm_path = out_dir / f"{molecule}_gqe_demo.qasm"
    qasm_path.write_text(qasm_str, encoding="utf-8")

    # Also export without measurements (for ansatz-only display)
    qasm_clean = qasm2_dumps(qc)
    qasm_clean_path = out_dir / f"{molecule}_gqe_ansatz.qasm"
    qasm_clean_path.write_text(qasm_clean, encoding="utf-8")

    # Metadata JSON
    metadata = {
        "molecule": molecule,
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "circuit_depth": qc.depth(),
        "gate_count": qc.size(),
        "n_gates_per_type": qc.count_ops(),
        "operators": operators,
        "thetas": thetas,
        "energy_ha": energy,
        "hf_energy_ha": hf_energy,
        "fci_energy_ha": fci_energy,
        "source": "Conditional-GQE (H-cGQE Transformer + L-BFGS-B)",
        "source_checkpoint": source_checkpoint,
        "qasm_file": qasm_path.name,
        "qasm_ansatz_file": qasm_clean_path.name,
        "quaggle_import_instructions": (
            "1. Open Quaggle Circuit Builder\n"
            "2. Import the QASM file (OpenQASM 2.0 format)\n"
            "3. The circuit includes Hartree-Fock initial state + GQE ansatz\n"
            "4. Run on simulator or export to QPU\n"
            "5. Reference energy values in metadata JSON for comparison"
        ),
    }

    meta_path = out_dir / f"{molecule}_gqe_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "qasm": qasm_path,
        "qasm_ansatz": qasm_clean_path,
        "metadata": meta_path,
    }


def _load_optimized_json(path: Path, molecule: str) -> dict[str, Any]:
    """Load a specific molecule from an optimized results JSON."""
    with path.open() as f:
        data = json.load(f)
    for r in data:
        if r["molecule"] == molecule:
            return r
    available = [r["molecule"] for r in data]
    raise ValueError(f"Molecule '{molecule}' not found in {path}. Available: {available}")


def _load_hamiltonian_meta(ham_path: Path, molecule: str) -> dict[str, Any]:
    """Load HF/FCI energies and electron count from Hamiltonians JSON."""
    with ham_path.open() as f:
        data = json.load(f)
    for r in data.get("records", []):
        if r["name"] == molecule:
            return {
                "n_qubits": int(r["n_qubits"]),
                "n_electrons": r.get("active_space", {}).get("n_active_electrons", 2),
                "hf_energy": r.get("hf_energy"),
                "fci_energy": r.get("fci_energy"),
            }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GQE circuit as OpenQASM demo pack for Quaggle"
    )
    parser.add_argument(
        "--optimized",
        type=Path,
        default=None,
        help="Path to optimized results JSON (from optimize_h_cgqe_coefficients.py)",
    )
    parser.add_argument("--molecule", type=str, required=True, help="Molecule name")
    parser.add_argument(
        "--hamiltonians",
        type=Path,
        default=Path("results/data/hamiltonians.json"),
        help="Hamiltonians JSON for metadata (HF/FCI energies, electron count)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory or file")

    # Direct specification (alternative to --optimized)
    parser.add_argument("--operators", nargs="+", type=str, default=None)
    parser.add_argument("--thetas", nargs="+", type=float, default=None)
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--n-electrons", type=int, default=0)
    parser.add_argument("--energy", type=float, default=None)

    # Optional: source checkpoint for provenance
    parser.add_argument("--source-checkpoint", type=str, default=None)

    args = parser.parse_args()

    # Load from optimized JSON or direct args
    if args.optimized:
        result = _load_optimized_json(args.optimized, args.molecule)
        operators = result["best_operators"]
        thetas = result["best_thetas"]
        energy = result.get("best_energy")
        n_qubits = result.get("n_qubits", 4)
    else:
        if not args.operators or not args.thetas:
            parser.error("Must provide --optimized or both --operators and --thetas")
        operators = args.operators
        thetas = args.thetas
        energy = args.energy
        n_qubits = args.n_qubits or len(operators[0]) if operators else 4

    # Load Hamiltonian metadata
    ham_meta = _load_hamiltonian_meta(args.hamiltonians, args.molecule)
    n_electrons = args.n_electrons or ham_meta.get("n_electrons") or 2
    hf_energy = ham_meta.get("hf_energy")
    fci_energy = ham_meta.get("fci_energy")
    if n_qubits is None:
        n_qubits = ham_meta.get("n_qubits", len(operators[0]))

    # Determine output path
    if args.out.suffix == ".qasm":
        out_dir = args.out.parent
    else:
        out_dir = args.out

    print(f"Exporting {args.molecule} ({n_qubits}q, {n_electrons}e) to {out_dir}/")
    print(f"  Operators: {len(operators)}")
    print(f"  Thetas: {len(thetas)}")
    print(f"  Energy: {energy}")

    paths = export_demo_pack(
        operators=operators,
        thetas=thetas,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        molecule=args.molecule,
        energy=energy,
        hf_energy=hf_energy,
        fci_energy=fci_energy,
        source_checkpoint=args.source_checkpoint,
        out_dir=out_dir,
    )

    print(f"\nExported:")
    for name, p in paths.items():
        print(f"  {name}: {p}")

    # Print circuit stats
    qc = build_ansatz_circuit(operators, thetas, n_qubits, n_electrons)
    print(f"\nCircuit stats:")
    print(f"  Depth: {qc.depth()}")
    print(f"  Gates: {qc.size()}")
    print(f"  Gate types: {qc.count_ops()}")

    print(f"\nQASM preview (first 20 lines):")
    qasm_lines = paths["qasm"].read_text().splitlines()
    for line in qasm_lines[:20]:
        print(f"  {line}")
    if len(qasm_lines) > 20:
        print(f"  ... ({len(qasm_lines)} lines total)")


if __name__ == "__main__":
    main()
