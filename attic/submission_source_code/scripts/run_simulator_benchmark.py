#!/usr/bin/env python3
"""Benchmark H-cGQE circuits on free qBraid quantum simulators.

Measures what matters on simulators (not QPUs):
  - Energy accuracy vs GPU reference and FCI (validates QWC grouping + bit order)
  - Shot noise convergence at 1024 / 4096 / 8192 shots
  - SQD recovery from simulator measurement counts
  - QWC grouping effectiveness (term reduction ratio)
  - Cross-device consistency (IonQ sim vs qBraid QIR sim vs AWS SV1)
  - Circuit metrics (depth, gate count, execution time)

Does NOT measure (irrelevant on simulators):
  - Gate fidelity / error rates (simulators are ideal)
  - ZNE / REM (no noise to mitigate)
  - Queue depth (simulators have no queue)

Available free simulators (as of Jul 2026):
  - ionq:ionq:sim:simulator  — FREE, 29q, supports noise models
  - qbraid:qbraid:sim:qir-sv  — FREE, 30q (sparse state vector)
  - aws:aws:sim:sv1           — 7.5 credits/min, 34q (free first 12 months)

Usage:
    python scripts/run_simulator_benchmark.py \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --optimized results/eval/h_cgqe_optimized.json \
        --molecules h2 lih \
        --devices ionq:ionq:sim:simulator qbraid:qbraid:sim:qir-sv \
        --shots 1024 4096 8192 \
        --out results/eval/simulator_benchmark.json

    # Quick H2-only test on IonQ free sim:
    python scripts/run_simulator_benchmark.py \
        --molecules h2 \
        --devices ionq:ionq:sim:simulator \
        --shots 4096 \
        --out results/eval/sim_bench_h2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gqe.common.hamiltonian_utils import (
    load_hamiltonian_records,
    find_record_by_name,
    get_active_electron_count,
)
from src.gqe.eval.qbraid_backend import (
    evaluate_energy_qbraid_batched,
    _build_ansatz_circuit,
    _group_qwc_terms,
    _circuit_to_qasm,
    export_sqd_sampling_circuit,
)
from src.gqe.common.hamiltonian_utils import iter_terms

try:
    from src.gqe.eval.sqd import sqd_energy_from_counts, apply_symmetry_filters
    _SQD_AVAILABLE = True
except ImportError:
    _SQD_AVAILABLE = False


# Simulator device specs
SIMULATOR_SPECS = {
    "ionq:ionq:sim:simulator": {"max_qubits": 29, "max_shots": 10000, "cost": "free", "batch_support": False},
    "qbraid:qbraid:sim:qir-sv": {"max_qubits": 30, "max_shots": 2000, "cost": "free", "batch_support": False},
    "aws:aws:sim:sv1":          {"max_qubits": 34, "max_shots": 100000, "cost": "7.5 credits/min", "batch_support": False},
    "aws:aws:sim:dm1":          {"max_qubits": 17, "max_shots": 100000, "cost": "7.5 credits/min", "batch_support": False},
}


def _load_optimized(path: Path) -> dict[str, dict[str, Any]]:
    """Load optimized results and return {molecule: entry} dict."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    results = data if isinstance(data, list) else data.get("results", [])
    out = {}
    for entry in results:
        name = entry.get("molecule", "")
        if name:
            out[name] = entry
    return out


def _get_circuit_metrics(circuit) -> dict[str, int]:
    """Compute circuit complexity metrics."""
    try:
        decomposed = circuit.decompose(reps=3)
        two_qubit = sum(1 for inst, _, _ in decomposed.data if inst.num_qubits == 2)
        return {
            "depth": int(decomposed.depth()),
            "two_qubit_gates": two_qubit,
            "total_gates": len(decomposed.data),
            "n_qubits": circuit.num_qubits,
        }
    except Exception:
        return {"depth": int(circuit.depth()), "two_qubit_gates": 0, "total_gates": len(circuit.data), "n_qubits": circuit.num_qubits}


def _compute_qwc_stats(record: dict[str, Any]) -> dict[str, Any]:
    """Compute QWC grouping statistics for a Hamiltonian."""
    active_terms = []
    for ops, coeff in iter_terms(record):
        word = "".join(ops)
        active_terms.append((word, coeff.real))
    groups = _group_qwc_terms(active_terms)
    return {
        "n_hamiltonian_terms": len(active_terms),
        "n_qwc_groups": len(groups),
        "reduction_ratio": len(active_terms) / len(groups) if groups else 0,
        "group_sizes": [len(g) for g in groups],
    }


def _run_sqd_on_counts(
    record: dict[str, Any],
    counts: dict[str, int],
    n_electrons: int,
    subspace_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Run SQD post-processing on measurement counts from simulator."""
    if not _SQD_AVAILABLE:
        return {"error": "SQD module not available"}

    if subspace_sizes is None:
        subspace_sizes = [16, 32, 64, 128, 256]

    n_qubits = int(record["n_qubits"])
    results = {}

    # Unfiltered SQD
    energies = []
    for k in subspace_sizes:
        try:
            e = sqd_energy_from_counts(record, counts, subspace_size=k)
            energies.append({"subspace_size": k, "energy": e})
        except Exception as ex:
            energies.append({"subspace_size": k, "error": str(ex)})
    results["unfiltered"] = energies

    # Symmetry-filtered SQD
    filtered_energies = []
    for k in subspace_sizes:
        try:
            e = sqd_energy_from_counts(
                record, counts,
                subspace_size=k,
                n_electrons=n_electrons,
                particle_number_tol=0,
            )
            filtered_energies.append({"subspace_size": k, "energy": e})
        except Exception as ex:
            filtered_energies.append({"subspace_size": k, "error": str(ex)})
    results["symmetry_filtered"] = filtered_energies

    # Best energies
    all_e = [e["energy"] for e in energies if "energy" in e]
    filt_e = [e["energy"] for e in filtered_energies if "energy" in e]
    results["best_unfiltered"] = min(all_e) if all_e else None
    results["best_filtered"] = min(filt_e) if filt_e else None

    return results


def _extract_counts_from_result(result: Any) -> dict[str, int]:
    """Extract raw counts from a qBraid job result."""
    try:
        return result.data.get_counts()
    except Exception:
        pass
    try:
        return dict(result.measurement_counts())
    except Exception:
        pass
    try:
        return dict(result.get_counts())
    except Exception:
        pass
    return {}


def benchmark_molecule(
    mol_name: str,
    record: dict[str, Any],
    operators: list[str],
    thetas: list[float],
    gpu_energy: float | None,
    fci_energy: float | None,
    hf_energy: float | None,
    devices: list[str],
    shot_counts: list[int],
    n_electrons: int,
    sqd_only: bool = False,
    max_qwc_circuits: int = 20,
    skip_sqd: bool = False,
) -> dict[str, Any]:
    """Run comprehensive simulator benchmark for one molecule."""
    n_qubits = int(record["n_qubits"])
    print(f"\n{'='*60}")
    print(f"  {mol_name} ({n_qubits}q, {n_electrons}e)")
    print(f"{'='*60}")

    # Circuit metrics
    circuit, _, param_symbols = _build_ansatz_circuit(n_qubits, n_electrons, operators)
    theta_arr = np.asarray(thetas) if thetas else np.zeros(len(operators))
    if hasattr(circuit, "assign_parameters"):
        bound = circuit.assign_parameters({s: float(v) for s, v in zip(param_symbols, theta_arr)})
    else:
        bound = circuit.bind_parameters({s: float(v) for s, v in zip(param_symbols, theta_arr)})

    circ_metrics = _get_circuit_metrics(bound)
    qwc_stats = _compute_qwc_stats(record)

    print(f"  Circuit: depth={circ_metrics['depth']}, 2q gates={circ_metrics['two_qubit_gates']}")
    print(f"  QWC: {qwc_stats['n_hamiltonian_terms']} terms -> {qwc_stats['n_qwc_groups']} groups "
          f"({qwc_stats['reduction_ratio']:.1f}x reduction)")
    if gpu_energy is not None:
        print(f"  GPU energy:  {gpu_energy:.6f} Ha")
    if fci_energy is not None:
        print(f"  FCI energy:  {fci_energy:.6f} Ha")
    if hf_energy is not None:
        print(f"  HF energy:   {hf_energy:.6f} Ha")

    mol_result = {
        "molecule": mol_name,
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "operators": operators,
        "n_operators": len(operators),
        "circuit_metrics": circ_metrics,
        "qwc_stats": qwc_stats,
        "gpu_energy_ha": gpu_energy,
        "fci_energy_ha": fci_energy,
        "hf_energy_ha": hf_energy,
        "device_results": {},
    }

    # Decide whether to do full QWC energy evaluation or SQD-only
    n_qwc_groups = qwc_stats["n_qwc_groups"]
    do_full_qwc = not sqd_only and n_qwc_groups <= max_qwc_circuits
    if not do_full_qwc:
        print(f"  Mode: SQD-only (QWC groups={n_qwc_groups} > threshold={max_qwc_circuits})")
    else:
        print(f"  Mode: Full QWC + SQD ({n_qwc_groups} groups <= threshold={max_qwc_circuits})")

    # Run on each device at each shot count
    for device_id in devices:
        spec = SIMULATOR_SPECS.get(device_id, {})
        max_q = spec.get("max_qubits", 30)
        if n_qubits > max_q:
            print(f"\n  [{device_id}] SKIP: {n_qubits}q > {max_q}q limit")
            mol_result["device_results"][device_id] = {"skipped": f"{n_qubits}q exceeds {max_q}q limit"}
            continue

        print(f"\n  [{device_id}] ({spec.get('cost', '?')})")
        max_shots = spec.get("max_shots", 10000)
        device_data = {
            "device": device_id,
            "cost": spec.get("cost", "unknown"),
            "max_qubits": max_q,
            "max_shots": max_shots,
            "mode": "sqd_only" if not do_full_qwc else "full_qwc_plus_sqd",
            "shot_results": [],
        }

        for shots in shot_counts:
            if shots > max_shots:
                print(f"    Shots={shots} SKIP: exceeds {max_shots} shot limit")
                device_data["shot_results"].append({
                    "shots": shots,
                    "status": "skipped",
                    "error": f"shots={shots} exceeds device limit={max_shots}",
                })
                continue
            print(f"    Shots={shots} ...")
            t0 = time.perf_counter()

            shot_entry = {"shots": shots, "status": "ok"}

            # --- Full QWC energy evaluation (if enabled) ---
            if do_full_qwc:
                try:
                    result = evaluate_energy_qbraid_batched(
                        record,
                        operators,
                        theta_values=theta_arr,
                        device=device_id,
                        shots=shots,
                        submit_only=False,
                    )
                    runtime = time.perf_counter() - t0

                    sim_energy = result["energy"]
                    device_used = result.get("device", device_id)

                    sim_gpu_diff = abs(sim_energy - gpu_energy) * 1000 if gpu_energy is not None else None
                    sim_fci_err = abs(sim_energy - fci_energy) * 1000 if fci_energy is not None else None
                    gpu_fci_err = abs(gpu_energy - fci_energy) * 1000 if gpu_energy is not None and fci_energy is not None else None

                    shot_entry.update({
                        "sim_energy_ha": sim_energy,
                        "device_used": device_used,
                        "runtime_seconds": round(runtime, 2),
                        "sim_gpu_diff_mha": round(sim_gpu_diff, 4) if sim_gpu_diff is not None else None,
                        "sim_fci_err_mha": round(sim_fci_err, 4) if sim_fci_err is not None else None,
                        "gpu_fci_err_mha": round(gpu_fci_err, 4) if gpu_fci_err is not None else None,
                        "n_qwc_groups": result.get("metadata", {}).get("n_groups", n_qwc_groups),
                    })

                    print(f"      QWC Energy: {sim_energy:.6f} Ha | sim-GPU: {sim_gpu_diff:.3f} mHa | runtime: {runtime:.1f}s")

                except Exception as e:
                    runtime = time.perf_counter() - t0
                    shot_entry["qwc_error"] = str(e)
                    shot_entry["runtime_seconds"] = round(runtime, 2)
                    print(f"      QWC FAILED: {e}")
            else:
                shot_entry["sim_energy_ha"] = None
                shot_entry["mode"] = "sqd_only"

            # --- SQD Z-basis sampling + recovery (always, if enabled) ---
            if _SQD_AVAILABLE and not skip_sqd and shots >= 1024:
                try:
                    print(f"      SQD Z-basis sampling ...")
                    manifest = export_sqd_sampling_circuit(
                        record, operators,
                        theta_values=theta_arr,
                        device=device_id,
                        shots=shots,
                    )

                    from qbraid import QbraidProvider
                    provider = QbraidProvider()
                    devices_list = provider.get_devices()
                    qdevice = next((d for d in devices_list if d.id == device_id), None)
                    if qdevice is None:
                        qdevice = next((d for d in devices_list if device_id in d.id), None)

                    if qdevice is not None:
                        qasm_str = manifest.get("circuit_qasm", "")
                        try:
                            from qiskit.qasm2 import loads as qasm2_loads
                            sqd_circuit = qasm2_loads(qasm_str)
                        except Exception:
                            from qiskit import QuantumCircuit
                            sqd_circuit = QuantumCircuit.from_qasm_str(qasm_str)

                        sqd_job = qdevice.run(sqd_circuit, shots=shots)
                        sqd_result = sqd_job.result()
                        sqd_counts = _extract_counts_from_result(sqd_result)

                        if sqd_counts:
                            sqd_data = _run_sqd_on_counts(record, sqd_counts, n_electrons)
                            shot_entry["sqd_recovery"] = sqd_data
                            best_sqd = sqd_data.get("best_filtered") or sqd_data.get("best_unfiltered")
                            if best_sqd is not None:
                                sqd_fci_err = abs(best_sqd - fci_energy) * 1000 if fci_energy else None
                                sqd_gpu_diff = abs(best_sqd - gpu_energy) * 1000 if gpu_energy else None
                                shot_entry["sqd_best_energy_ha"] = best_sqd
                                shot_entry["sqd_fci_err_mha"] = round(sqd_fci_err, 4) if sqd_fci_err is not None else None
                                shot_entry["sqd_gpu_diff_mha"] = round(sqd_gpu_diff, 4) if sqd_gpu_diff is not None else None
                                print(f"      SQD best: {best_sqd:.6f} Ha | FCI err: {sqd_fci_err:.3f} mHa")
                            else:
                                print(f"      SQD: no valid energy recovered")
                        else:
                            shot_entry["sqd_recovery"] = {"error": "No counts extracted"}
                            print(f"      SQD: no counts extracted")
                    else:
                        shot_entry["sqd_recovery"] = {"error": "Device not found"}
                except Exception as e:
                    shot_entry["sqd_recovery"] = {"error": str(e)}
                    print(f"      SQD failed: {e}")

            if "runtime_seconds" not in shot_entry:
                shot_entry["runtime_seconds"] = round(time.perf_counter() - t0, 2)

            device_data["shot_results"].append(shot_entry)
            time.sleep(2.0)

        mol_result["device_results"][device_id] = device_data

    return mol_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark H-cGQE circuits on free qBraid simulators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hamiltonians", type=Path,
                        default=Path("results/data/hamiltonians_gic2026/hamiltonians.json"))
    parser.add_argument("--optimized", type=Path,
                        default=Path("results/eval/h_cgqe_optimized.json"),
                        help="Optimized coefficient results JSON")
    parser.add_argument("--molecules", nargs="+", default=["h2", "lih"],
                        help="Molecule names to benchmark")
    parser.add_argument("--devices", nargs="+",
                        default=["ionq:ionq:sim:simulator", "qbraid:qbraid:sim:qir-sv"],
                        help="qBraid simulator device IDs")
    parser.add_argument("--shots", nargs="+", type=int, default=[4096],
                        help="Shot counts to test (default: 4096)")
    parser.add_argument("--out", type=Path,
                        default=Path("results/eval/simulator_benchmark.json"))
    parser.add_argument("--skip-sqd", action="store_true",
                        help="Skip SQD recovery from simulator counts")
    parser.add_argument("--sqd-only", action="store_true",
                        help="Only do SQD Z-basis sampling (skip full QWC energy evaluation)")
    parser.add_argument("--max-qwc-circuits", type=int, default=20,
                        help="Auto-switch to sqd-only above this many QWC circuits (default: 20)")
    args = parser.parse_args()

    print("=" * 60)
    print("  qBraid Simulator Benchmark for H-cGQE")
    print("=" * 60)
    print(f"  Hamiltonians: {args.hamiltonians}")
    print(f"  Optimized:    {args.optimized}")
    print(f"  Molecules:    {args.molecules}")
    print(f"  Devices:      {args.devices}")
    print(f"  Shots:        {args.shots}")
    print(f"  SQD recovery: {'disabled' if args.skip_sqd else 'enabled'}")

    # Load data
    records = load_hamiltonian_records(args.hamiltonians)
    optimized = _load_optimized(args.optimized)

    all_results = []

    for mol_name in args.molecules:
        record = find_record_by_name(records, mol_name)
        if record is None:
            print(f"\n  SKIP: {mol_name} not found in Hamiltonians")
            continue

        opt = optimized.get(mol_name, {})
        operators = opt.get("best_operators", opt.get("operators", []))
        thetas = opt.get("best_thetas", opt.get("thetas", []))
        gpu_energy = opt.get("best_energy", opt.get("energy"))
        fci_energy = record.get("fci_energy")
        hf_energy = record.get("hf_energy")
        n_electrons = get_active_electron_count(record)

        if not operators:
            print(f"\n  SKIP: No operators for {mol_name}, using HF-only (empty sequence)")
            operators = []
            thetas = []

        result = benchmark_molecule(
            mol_name, record, operators, thetas,
            gpu_energy, fci_energy, hf_energy,
            args.devices, args.shots, n_electrons,
            sqd_only=args.sqd_only,
            max_qwc_circuits=args.max_qwc_circuits,
            skip_sqd=args.skip_sqd,
        )
        all_results.append(result)

    # Summary
    print("\n\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"{'Molecule':12s} {'Device':28s} {'Shots':>6s} {'Sim E':>12s} {'GPU E':>12s} {'diff mHa':>9s} {'FCI err':>9s}")
    print("-" * 88)

    for mol in all_results:
        mol_name = mol["molecule"]
        gpu_e = mol.get("gpu_energy_ha")
        fci_e = mol.get("fci_energy_ha")

        for dev_id, dev_data in mol.get("device_results", {}).items():
            if "skipped" in dev_data:
                print(f"{mol_name:12s} {dev_id:28s}  SKIP: {dev_data['skipped']}")
                continue

            for shot_entry in dev_data.get("shot_results", []):
                if shot_entry.get("status") not in ("ok", "skipped"):
                    print(f"{mol_name:12s} {dev_id:28s} {shot_entry.get('shots', 0):>6d}  FAILED: {shot_entry.get('error', shot_entry.get('qwc_error', ''))[:50]}")
                    continue
                if shot_entry.get("status") == "skipped":
                    print(f"{mol_name:12s} {dev_id:28s} {shot_entry.get('shots', 0):>6d}  SKIP: {shot_entry.get('error', '')[:40]}")
                    continue

                sim_e = shot_entry.get("sim_energy_ha")
                sqd_e = shot_entry.get("sqd_best_energy_ha")
                mode = shot_entry.get("mode", "")

                if sim_e is not None:
                    diff = shot_entry.get("sim_gpu_diff_mha", 0) or 0
                    fci_err = shot_entry.get("sim_fci_err_mha", 0) or 0
                    print(f"{mol_name:12s} {dev_id:28s} {shot_entry['shots']:>6d} "
                          f"{sim_e:>12.6f} {(gpu_e or 0):>12.6f} {diff:>9.3f} {fci_err:>9.3f}")
                elif sqd_e is not None:
                    print(f"{mol_name:12s} {dev_id:28s} {shot_entry['shots']:>6d} "
                          f"{'(sqd-only)':>12s} {(gpu_e or 0):>12.6f} {'':>9s} {'':>9s}")
                else:
                    print(f"{mol_name:12s} {dev_id:28s} {shot_entry['shots']:>6d}  No energy recovered")

                # SQD result if available
                if sqd_e is not None:
                    sqd_err = shot_entry.get("sqd_fci_err_mha", 0) or 0
                    sqd_diff = shot_entry.get("sqd_gpu_diff_mha", 0) or 0
                    print(f"  {'→ SQD':40s} {sqd_e:>12.6f} {'':>12s} {sqd_diff:>9.3f} {sqd_err:>9.3f}")

    # Save results
    payload = {
        "description": "qBraid simulator benchmark: energy accuracy, shot noise, SQD recovery",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "hamiltonians": str(args.hamiltonians),
            "optimized": str(args.optimized),
            "molecules": args.molecules,
            "devices": args.devices,
            "shots": args.shots,
            "sqd_enabled": not args.skip_sqd,
        },
        "simulator_specs": SIMULATOR_SPECS,
        "results": all_results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {args.out}")

    # Also save a simplified version for build_gic_benchmark.py consumption
    bench_entries = []
    for mol in all_results:
        mol_name = mol["molecule"]
        for dev_id, dev_data in mol.get("device_results", {}).items():
            if "skipped" in dev_data:
                continue
            for shot_entry in dev_data.get("shot_results", []):
                if shot_entry.get("status") != "ok":
                    continue
                entry = {
                    "molecule": mol_name,
                    "device": dev_id,
                    "sim_energy": shot_entry.get("sim_energy_ha"),
                    "shots": shot_entry["shots"],
                    "runtime_seconds": shot_entry["runtime_seconds"],
                    "mode": shot_entry.get("mode", "full_qwc"),
                    "sim_gpu_diff_mha": shot_entry.get("sim_gpu_diff_mha"),
                    "sim_fci_err_mha": shot_entry.get("sim_fci_err_mha"),
                }
                if "sqd_best_energy_ha" in shot_entry:
                    entry["sqd_energy"] = shot_entry["sqd_best_energy_ha"]
                    entry["sqd_fci_err_mha"] = shot_entry.get("sqd_fci_err_mha")
                    entry["sqd_gpu_diff_mha"] = shot_entry.get("sqd_gpu_diff_mha")
                bench_entries.append(entry)

    bench_path = args.out.parent / "simulator_validation.json"
    with open(bench_path, "w") as f:
        json.dump(bench_entries, f, indent=2)
    print(f"Benchmark entries (for pipeline) saved to: {bench_path}")


if __name__ == "__main__":
    main()
