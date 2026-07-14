"""MPS scaling curve: compare statevector vs MPS simulation across qubit counts.

Key points:
- Statevector (nvidia backend): exact but limited to <=24 qubits on L40S (48GB)
- MPS (tensornet-mps backend): approximate, single-GPU only, handles 60+ qubits
- Bond dimension controlled by CUDAQ_MPS_MAX_BOND env var (default 64)
- For HF states (no entanglement), MPS is exact at any D
- We add a simple entangling layer to demonstrate bond dimension effects
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cudaq
except ImportError:
    cudaq = None

try:
    from src.gqe.common.hamiltonian_utils import (
        load_hamiltonian_records,
        hamiltonian_to_spin_operator,
        get_active_electron_count,
    )
except ImportError:
    from gqe.common.hamiltonian_utils import (
        load_hamiltonian_records,
        hamiltonian_to_spin_operator,
        get_active_electron_count,
    )


def _make_entangled_kernel():
    """Create a CUDA-Q kernel with HF state + entangling CNOT layer."""
    @cudaq.kernel
    def entangled_kernel(n_qubits: int, n_electrons: int):
        q = cudaq.qvector(n_qubits)
        for i in range(n_electrons):
            x(q[i])
        # Full CNOT chain: creates GHZ-like state with maximal entanglement
        for i in range(n_qubits - 1):
            x.ctrl(q[i], q[i + 1])
    return entangled_kernel


def _run_statevector(record: dict) -> tuple[float | None, float]:
    """Run statevector simulation using nvidia backend."""
    n_qubits = int(record["n_qubits"])
    if n_qubits > 24:
        return None, 0.0
    try:
        cudaq.set_target("nvidia")
        spin_ham = hamiltonian_to_spin_operator(record)
        kernel = _make_entangled_kernel()
        n_electrons = get_active_electron_count(record)
        t0 = time.time()
        result = cudaq.observe(kernel, spin_ham, n_qubits, n_electrons)
        runtime = time.time() - t0
        return float(result.expectation()), runtime
    except Exception as e:
        print(f"    Statevector failed: {e}")
        return None, 0.0


def _run_mps(record: dict, bond_dim: int) -> tuple[float | None, float]:
    """Run MPS simulation with given bond dimension."""
    n_qubits = int(record["n_qubits"])
    os.environ["CUDAQ_MPS_MAX_BOND"] = str(bond_dim)
    try:
        cudaq.set_target("tensornet-mps")
        spin_ham = hamiltonian_to_spin_operator(record)
        kernel = _make_entangled_kernel()
        n_electrons = get_active_electron_count(record)
        t0 = time.time()
        result = cudaq.observe(kernel, spin_ham, n_qubits, n_electrons)
        runtime = time.time() - t0
        return float(result.expectation()), runtime
    except Exception as e:
        print(f"    MPS D={bond_dim} failed: {e}")
        return None, 0.0


def run_mps_scaling(
    config_path: str,
    out_path: str,
) -> dict[str, Any]:
    """Run MPS scaling curve experiment."""

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    molecules = config.get("molecules", [])
    bond_dims = config.get("bond_dimensions", [32, 64, 128, 256])

    ham_files = [
        "results/data/hamiltonians_phase3.json/hamiltonians.json",
        "results/data/hamiltonians_scaling.json/hamiltonians.json",
        "results/data/hamiltonians_40plus.json/hamiltonians.json",
    ]

    all_records = {}
    for hf in ham_files:
        p = Path(hf)
        if p.exists():
            records = load_hamiltonian_records(p)
            for r in records:
                all_records[r["name"]] = r

    results = []
    header = f"{'Molecule':20s} {'Qubits':>6s} {'SV (Ha)':>14s} {'SV time':>8s}"
    for D in bond_dims:
        header += f" {'MPS D='+str(D):>12s}"
    print(header)
    print("-" * len(header))

    for mol_cfg in molecules:
        name = mol_cfg["name"]
        record = all_records.get(name)
        if record is None:
            print(f"  {name}: NOT FOUND in Hamiltonian data")
            continue

        n_qubits = int(record["n_qubits"])
        print(f"{name:20s} {n_qubits:>6d}", end="", flush=True)

        sv_energy, sv_runtime = _run_statevector(record)
        if sv_energy is not None:
            print(f" {sv_energy:>14.6f} {sv_runtime:>7.2f}s", end="", flush=True)
        else:
            print(f" {'N/A':>14s} {'N/A':>8s}", end="", flush=True)

        mps_energies = {}
        mps_runtimes = {}
        for D in bond_dims:
            E, rt = _run_mps(record, D)
            mps_energies[D] = E
            mps_runtimes[D] = rt
            if E is not None:
                print(f" {E:>12.6f}", end="", flush=True)
            else:
                print(f" {'N/A':>12s}", end="", flush=True)

        print()

        # Save incrementally after each molecule
        results.append({
            "molecule": name,
            "n_qubits": n_qubits,
            "statevector_energy": sv_energy,
            "statevector_runtime": sv_runtime,
            "mps_energies": mps_energies,
            "mps_runtimes": mps_runtimes,
            "errors_vs_sv": {
                f"D={D}": abs(mps_energies[D] - sv_energy) * 1000
                          if mps_energies.get(D) is not None and sv_energy is not None
                          else None
                for D in bond_dims
            },
        })

        # Incremental save
        partial = {
            "experiment": "mps_scaling",
            "description": "Statevector vs MPS comparison across qubit counts with entangling layer",
            "bond_dimensions": bond_dims,
            "backend_info": {
                "statevector": "nvidia (single L40S, <=24 qubits)",
                "mps": "tensornet-mps (single GPU, CUDAQ_MPS_MAX_BOND env var)",
                "note": "tensornet-mps is single-GPU only per CUDA-Q docs",
            },
            "results": results,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(partial, f, indent=2)

    # Save final results
    return {
        "experiment": "mps_scaling",
        "description": "Statevector vs MPS comparison across qubit counts with entangling layer",
        "bond_dimensions": bond_dims,
        "backend_info": {
            "statevector": "nvidia (single L40S, <=24 qubits)",
            "mps": "tensornet-mps (single GPU, CUDAQ_MPS_MAX_BOND env var)",
            "note": "tensornet-mps is single-GPU only per CUDA-Q docs",
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MPS scaling curve experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = run_mps_scaling(str(args.config), str(args.out))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
