#!/usr/bin/env python3
"""Build consolidated GIC benchmark table from all result sources.

Merges:
  - RL best circuits (extract_best_circuits.py output)
  - RL training metrics (checkpoint best_energies)
  - GQE baseline (cudaq_gqe_uccsd_3gpu.json or similar)
  - VQE baselines (UCCSD-VQE, ADAPT-VQE)
  - SQD pilot results (sqd_pilot output)
  - QPU / simulator validation results
  - Hamiltonian records (FCI / HF reference energies)

Produces:
  - results/eval/gic_benchmark_consolidated.json
  - results/eval/gic_benchmark_consolidated.csv

Usage:
    python scripts/build_gic_benchmark.py \
        --best-circuits results/train/h_cgqe_model_qbraid_rl_best_circuits.json \
        --rl-metrics results/train/h_cgqe_model_qbraid_rl_rl_metrics.json \
        --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
        --gqe-baseline results/baselines/cudaq_gqe_uccsd_3gpu.json \
        --vqe-baseline results/baselines/cudaq_vqe_results.json \
        --adapt-vqe-baseline results/baselines/adapt_vqe_results.json \
        --sqd-results results/eval/sqd_pilot_results.json \
        --qpu-results results/eval/simulator_validation.json \
        --out results/eval/gic_benchmark_consolidated.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CHEMICAL_ACCURACY_MHA = 1.6


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--best-circuits", type=Path, required=True,
                   help="extract_best_circuits.py output JSON")
    p.add_argument("--rl-metrics", type=Path, default=None,
                   help="RL checkpoint metrics JSON (best_energies)")
    p.add_argument("--hamiltonians", type=Path, required=True,
                   help="Hamiltonian records JSON")
    p.add_argument("--gqe-baseline", type=Path, default=None,
                   help="GQE baseline results JSON")
    p.add_argument("--vqe-baseline", type=Path, default=None,
                   help="UCCSD-VQE baseline results JSON")
    p.add_argument("--adapt-vqe-baseline", type=Path, default=None,
                   help="ADAPT-VQE baseline results JSON")
    p.add_argument("--sqd-results", type=Path, default=None,
                   help="SQD pilot results JSON (file or directory)")
    p.add_argument("--qpu-results", type=Path, default=None,
                   help="QPU/simulator validation results JSON (file or directory)")
    p.add_argument("--evaluation", type=Path, default=None,
                   help="Optional: evaluate_h_cgqe.py output for additional GPU energies")
    p.add_argument("--optimized-results", type=Path, default=None,
                   help="Optimized coefficient results JSON (from optimize_h_cgqe_coefficients.py)")
    p.add_argument("--out", type=Path, default=Path("results/eval/gic_benchmark_consolidated.json"))
    p.add_argument("--csv-out", type=Path, default=None,
                   help="CSV output path (default: <out>.csv)")
    return p.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_hamiltonian_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load hamiltonians JSON and return {name: record} dict."""
    data = _load_json(path)
    records = data.get("records", data if isinstance(data, list) else [])
    return {r["name"]: r for r in records}


def _load_baseline(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load a baseline results file and return {molecule_name: entry} dict.

    Works for GQE baseline, VQE baseline, ADAPT-VQE baseline, etc.
    """
    if path is None or not path.exists():
        return {}
    data = _load_json(path)
    results = data.get("results", data if isinstance(data, list) else [])
    out: dict[str, dict[str, Any]] = {}
    for r in results:
        name = r.get("system") or r.get("molecule") or r.get("name", "")
        name = _normalize_mol_name(name)
        out[name] = r
    return out


def _load_sqd_results(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load SQD pilot results and return {molecule_name: entry} dict.

    Accepts a single JSON file or a directory of JSON files.
    """
    if path is None or not path.exists():
        return {}

    files: list[Path] = []
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    else:
        files = [path]

    out: dict[str, dict[str, Any]] = {}
    for f in files:
        try:
            data = _load_json(f)
        except (json.JSONDecodeError, KeyError):
            continue
        if isinstance(data, dict) and "results" in data:
            for entry in data["results"]:
                name = entry.get("molecule") or entry.get("name", "")
                name = _normalize_mol_name(name)
                if name:
                    out[name] = entry
        elif isinstance(data, dict) and "molecule" in data:
            name = _normalize_mol_name(data["molecule"])
            out[name] = data
    return out


def _normalize_mol_name(name: str) -> str:
    """Strip common suffixes like _0.74, _eq etc. from molecule names."""
    for suffix in ("_0.74", "_eq", "_1.0A"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _load_qpu_results(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load QPU/simulator validation results.

    Accepts a single JSON file or a directory of JSON files.
    Returns {molecule_name: entry}.
    """
    if path is None or not path.exists():
        return {}

    files: list[Path] = []
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    else:
        files = [path]

    out: dict[str, dict[str, Any]] = {}
    for f in files:
        try:
            data = _load_json(f)
        except (json.JSONDecodeError, KeyError):
            continue
        if isinstance(data, list):
            for entry in data:
                name = entry.get("molecule") or entry.get("name", "")
                name = _normalize_mol_name(name)
                if name:
                    out[name] = entry
        elif isinstance(data, dict):
            if "molecule" in data:
                name = _normalize_mol_name(data["molecule"])
                out[name] = data
            elif "results" in data:
                for entry in data["results"]:
                    name = entry.get("molecule") or entry.get("name", "")
                    name = _normalize_mol_name(name)
                    if name:
                        out[name] = entry
    return out


def _load_evaluation(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load evaluate_h_cgqe.py output for additional GPU energies."""
    if path is None or not path.exists():
        return {}
    data = _load_json(path)
    results = data if isinstance(data, list) else data.get("results", [])
    out: dict[str, dict[str, Any]] = {}
    for r in results:
        name = r.get("molecule") or r.get("system") or r.get("name", "")
        name = _normalize_mol_name(name)
        out[name] = r
    return out


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    # Load all sources
    best_data = _load_json(args.best_circuits)
    if isinstance(best_data, list):
        best_circuits = {}
        for entry in best_data:
            name = _normalize_mol_name(entry.get("molecule", entry.get("name", "")))
            if name:
                best_circuits[name] = entry
    else:
        best_circuits: dict[str, Any] = best_data.get("best_circuits", best_data)

    rl_metrics: dict[str, Any] = {}
    if args.rl_metrics and args.rl_metrics.exists():
        rl_metrics = _load_json(args.rl_metrics).get("best_energies", {})

    optimized_results: dict[str, dict[str, Any]] = {}
    if args.optimized_results and args.optimized_results.exists():
        opt_data = _load_json(args.optimized_results)
        opt_list = opt_data if isinstance(opt_data, list) else opt_data.get("results", [])
        for entry in opt_list:
            name = _normalize_mol_name(entry.get("molecule", ""))
            if name:
                optimized_results[name] = entry

    ham_records = _load_hamiltonian_records(args.hamiltonians)
    gqe_baseline = _load_baseline(args.gqe_baseline)
    vqe_baseline = _load_baseline(args.vqe_baseline)
    adapt_vqe_baseline = _load_baseline(args.adapt_vqe_baseline)
    sqd_results = _load_sqd_results(args.sqd_results)
    qpu_results = _load_qpu_results(args.qpu_results)
    eval_results = _load_evaluation(args.evaluation)

    # Build per-molecule rows
    all_molecules = set(best_circuits.keys()) | set(ham_records.keys())
    rows: list[dict[str, Any]] = []

    for mol_name in sorted(all_molecules, key=lambda m: (ham_records.get(m, {}).get("n_qubits", 999), m)):
        ham = ham_records.get(mol_name, {})
        bc = best_circuits.get(mol_name, {})
        gqe = gqe_baseline.get(mol_name, {})
        vqe = vqe_baseline.get(mol_name, {})
        adapt_vqe = adapt_vqe_baseline.get(mol_name, {})
        sqd = sqd_results.get(mol_name, {})
        qpu = qpu_results.get(mol_name, {})
        ev = eval_results.get(mol_name, {})

        n_qubits = ham.get("n_qubits") or bc.get("n_qubits")
        n_electrons = ham.get("n_electrons") or bc.get("n_electrons")

        fci = _safe_float(ham.get("fci_energy")) or _safe_float(bc.get("fci_energy"))
        hf = _safe_float(ham.get("hf_energy")) or _safe_float(bc.get("hf_energy"))

        # GPU energy: prefer checkpoint best (optimized thetas during RL),
        # fallback to resampled energy (fixed θ=0.01), then eval, then rl_metrics
        ckpt_best = _safe_float(bc.get("checkpoint_best_energy")) or _safe_float(rl_metrics.get(mol_name))
        resampled_energy = _safe_float(bc.get("energy"))
        gpu_energy = ckpt_best
        energy_provenance = "checkpoint_best"
        if gpu_energy is None:
            gpu_energy = resampled_energy
            energy_provenance = "resampled_fixed_theta"
        if gpu_energy is None and ev:
            gpu_energy = _safe_float(ev.get("best_generated_energy"))
            energy_provenance = "evaluation_output"
        if gpu_energy is None and mol_name in rl_metrics:
            gpu_energy = _safe_float(rl_metrics[mol_name])
            energy_provenance = "rl_metrics"

        # Optimized energy (from L-BFGS-B coefficient optimization)
        optimized_energy = None
        if mol_name in optimized_results:
            optimized_energy = _safe_float(optimized_results[mol_name].get("best_energy"))
            if optimized_energy is not None and (gpu_energy is None or optimized_energy < gpu_energy):
                gpu_energy = optimized_energy
                energy_provenance = "l_bfgs_b_optimized"

        # QPU / simulator energy (extracted early for execution_type logic)
        qpu_energy = _safe_float(qpu.get("sim_energy")) or _safe_float(qpu.get("qpu_energy"))
        sim_sqd_energy = _safe_float(qpu.get("sqd_energy"))
        sim_mode = qpu.get("mode", "full_qwc")
        sim_device = qpu.get("device")
        sim_shots = qpu.get("shots")

        # SQD energy (from pilot or QPU counts post-processing)
        sqd_energy = _safe_float(sqd.get("sqd_energy")) or _safe_float(sqd.get("energy"))
        sqd_recovered_energy = _safe_float(sqd.get("sqd_recovered"))

        # Determine execution type and device
        execution_type = "gpu_statevector"
        device = "L40S_nvidia"
        if qpu_energy is not None:
            execution_type = "qpu_simulator"
            device = sim_device or qpu.get("device_id", qpu.get("device", "unknown"))
        elif sim_sqd_energy is not None:
            execution_type = "sim_sqd_only"
            device = sim_device or "unknown"
        elif sqd_energy is not None:
            execution_type = "sqd_classical_postprocess"
            device = "classical"

        # Training exposure: was this molecule seen during RL training?
        if mol_name in rl_metrics:
            training_exposure = "seen_in_rl"
        elif mol_name in optimized_results:
            training_exposure = "optimized_only"
        elif mol_name in best_circuits:
            training_exposure = "extracted"
        else:
            training_exposure = "unseen"

        # GQE baseline energy
        gqe_energy = _safe_float(gqe.get("baseline_energy"))

        # VQE baseline energies
        vqe_energy = _safe_float(vqe.get("baseline_energy")) or _safe_float(vqe.get("vqe_energy"))
        adapt_vqe_energy = _safe_float(adapt_vqe.get("baseline_energy")) or _safe_float(adapt_vqe.get("vqe_energy"))

        # Compute errors
        err_vs_fci_mha = None
        if fci is not None and gpu_energy is not None:
            err_vs_fci_mha = abs(gpu_energy - fci) * 1000.0

        gqe_err_mha = None
        if fci is not None and gqe_energy is not None:
            gqe_err_mha = abs(gqe_energy - fci) * 1000.0

        vqe_err_mha = None
        if fci is not None and vqe_energy is not None:
            vqe_err_mha = abs(vqe_energy - fci) * 1000.0

        adapt_vqe_err_mha = None
        if fci is not None and adapt_vqe_energy is not None:
            adapt_vqe_err_mha = abs(adapt_vqe_energy - fci) * 1000.0

        sqd_err_mha = None
        if fci is not None and sqd_energy is not None:
            sqd_err_mha = abs(sqd_energy - fci) * 1000.0

        qpu_gpu_delta_mha = None
        if qpu_energy is not None and gpu_energy is not None:
            qpu_gpu_delta_mha = abs(qpu_energy - gpu_energy) * 1000.0

        qpu_err_mha = None
        if fci is not None and qpu_energy is not None:
            qpu_err_mha = abs(qpu_energy - fci) * 1000.0

        sim_sqd_err_mha = None
        if fci is not None and sim_sqd_energy is not None:
            sim_sqd_err_mha = abs(sim_sqd_energy - fci) * 1000.0

        chem_acc = err_vs_fci_mha is not None and err_vs_fci_mha <= CHEMICAL_ACCURACY_MHA

        improvement_mha = None
        if gqe_err_mha is not None and err_vs_fci_mha is not None:
            improvement_mha = gqe_err_mha - err_vs_fci_mha

        row = {
            "molecule": mol_name,
            "n_qubits": n_qubits,
            "n_electrons": n_electrons,
            "fci_energy_ha": fci,
            "hf_energy_ha": hf,
            "gqe_baseline_ha": gqe_energy,
            "vqe_baseline_ha": vqe_energy,
            "adapt_vqe_baseline_ha": adapt_vqe_energy,
            "h_cgqe_gpu_ha": gpu_energy,
            "h_cgqe_resampled_ha": resampled_energy,
            "h_cgqe_optimized_ha": optimized_energy,
            "h_cgqe_qpu_ha": qpu_energy,
            "sim_sqd_energy_ha": sim_sqd_energy,
            "sim_mode": sim_mode,
            "sim_device": sim_device,
            "sim_shots": sim_shots,
            "sqd_energy_ha": sqd_energy,
            "sqd_recovered_energy_ha": sqd_recovered_energy,
            "energy_provenance": energy_provenance,
            "execution_type": execution_type,
            "device": device,
            "training_exposure": training_exposure,
            "err_vs_fci_mha": round(err_vs_fci_mha, 4) if err_vs_fci_mha is not None else None,
            "gqe_err_vs_fci_mha": round(gqe_err_mha, 4) if gqe_err_mha is not None else None,
            "vqe_err_vs_fci_mha": round(vqe_err_mha, 4) if vqe_err_mha is not None else None,
            "adapt_vqe_err_vs_fci_mha": round(adapt_vqe_err_mha, 4) if adapt_vqe_err_mha is not None else None,
            "sqd_err_vs_fci_mha": round(sqd_err_mha, 4) if sqd_err_mha is not None else None,
            "qpu_err_vs_fci_mha": round(qpu_err_mha, 4) if qpu_err_mha is not None else None,
            "sim_sqd_err_vs_fci_mha": round(sim_sqd_err_mha, 4) if sim_sqd_err_mha is not None else None,
            "qpu_gpu_delta_mha": round(qpu_gpu_delta_mha, 4) if qpu_gpu_delta_mha is not None else None,
            "improvement_over_gqe_mha": round(improvement_mha, 4) if improvement_mha is not None else None,
            "chemical_accuracy": chem_acc,
            "n_ops": bc.get("n_ops"),
            "operators": bc.get("operators", []),
            "checkpoint_best_energy": _safe_float(bc.get("checkpoint_best_energy")) or _safe_float(rl_metrics.get(mol_name)),
            "source": {
                "best_circuits": mol_name in best_circuits,
                "gqe_baseline": mol_name in gqe_baseline,
                "vqe_baseline": mol_name in vqe_baseline,
                "adapt_vqe_baseline": mol_name in adapt_vqe_baseline,
                "sqd_results": mol_name in sqd_results,
                "qpu_validation": mol_name in qpu_results,
                "evaluation": mol_name in eval_results,
                "optimized_results": mol_name in optimized_results,
            },
        }
        rows.append(row)

    # Summary statistics
    gpu_errs = [r["err_vs_fci_mha"] for r in rows if r["err_vs_fci_mha"] is not None]
    gqe_errs = [r["gqe_err_vs_fci_mha"] for r in rows if r["gqe_err_vs_fci_mha"] is not None]
    vqe_errs = [r["vqe_err_vs_fci_mha"] for r in rows if r["vqe_err_vs_fci_mha"] is not None]
    adapt_vqe_errs = [r["adapt_vqe_err_vs_fci_mha"] for r in rows if r["adapt_vqe_err_vs_fci_mha"] is not None]
    sqd_errs = [r["sqd_err_vs_fci_mha"] for r in rows if r["sqd_err_vs_fci_mha"] is not None]
    qpu_errs = [r["qpu_err_vs_fci_mha"] for r in rows if r["qpu_err_vs_fci_mha"] is not None]
    chem_acc_count = sum(1 for r in rows if r["chemical_accuracy"])
    improvements = [r["improvement_over_gqe_mha"] for r in rows if r["improvement_over_gqe_mha"] is not None]

    summary = {
        "total_molecules": len(rows),
        "molecules_with_gpu_energy": sum(1 for r in rows if r["h_cgqe_gpu_ha"] is not None),
        "molecules_with_gqe_baseline": sum(1 for r in rows if r["gqe_baseline_ha"] is not None),
        "molecules_with_vqe_baseline": sum(1 for r in rows if r["vqe_baseline_ha"] is not None),
        "molecules_with_adapt_vqe_baseline": sum(1 for r in rows if r["adapt_vqe_baseline_ha"] is not None),
        "molecules_with_sqd_results": sum(1 for r in rows if r["sqd_energy_ha"] is not None),
        "molecules_with_qpu_validation": sum(1 for r in rows if r["h_cgqe_qpu_ha"] is not None),
        "molecules_with_sim_sqd": sum(1 for r in rows if r["sim_sqd_energy_ha"] is not None),
        "chemical_accuracy_count": chem_acc_count,
        "chemical_accuracy_pct": round(100.0 * chem_acc_count / len(rows), 1) if rows else 0,
        "mean_gpu_error_mha": round(sum(gpu_errs) / len(gpu_errs), 4) if gpu_errs else None,
        "median_gpu_error_mha": round(sorted(gpu_errs)[len(gpu_errs) // 2], 4) if gpu_errs else None,
        "max_gpu_error_mha": round(max(gpu_errs), 4) if gpu_errs else None,
        "min_gpu_error_mha": round(min(gpu_errs), 4) if gpu_errs else None,
        "mean_gqe_error_mha": round(sum(gqe_errs) / len(gqe_errs), 4) if gqe_errs else None,
        "mean_vqe_error_mha": round(sum(vqe_errs) / len(vqe_errs), 4) if vqe_errs else None,
        "mean_adapt_vqe_error_mha": round(sum(adapt_vqe_errs) / len(adapt_vqe_errs), 4) if adapt_vqe_errs else None,
        "mean_sqd_error_mha": round(sum(sqd_errs) / len(sqd_errs), 4) if sqd_errs else None,
        "mean_qpu_error_mha": round(sum(qpu_errs) / len(qpu_errs), 4) if qpu_errs else None,
        "mean_improvement_over_gqe_mha": round(sum(improvements) / len(improvements), 4) if improvements else None,
        "qubit_range": {
            "min": min((r["n_qubits"] for r in rows if r["n_qubits"]), default=None),
            "max": max((r["n_qubits"] for r in rows if r["n_qubits"]), default=None),
        },
    }

    # Generalization analysis: seen vs unseen
    # Molecules seen during RL training (from best_energies in rl_metrics)
    seen_mols = set(rl_metrics.keys()) if rl_metrics else set(best_circuits.keys())
    unseen_errs = [r["err_vs_fci_mha"] for r in rows
                   if r["err_vs_fci_mha"] is not None and r["molecule"] not in seen_mols]
    seen_errs = [r["err_vs_fci_mha"] for r in rows
                 if r["err_vs_fci_mha"] is not None and r["molecule"] in seen_mols]

    summary["generalization"] = {
        "seen_molecules": len(seen_mols),
        "unseen_molecules": len(rows) - len(seen_mols),
        "seen_mean_error_mha": round(sum(seen_errs) / len(seen_errs), 4) if seen_errs else None,
        "unseen_mean_error_mha": round(sum(unseen_errs) / len(unseen_errs), 4) if unseen_errs else None,
    }

    payload = {
        "description": "Consolidated GIC 2026 benchmark: H-cGQE (RL-tuned) vs GQE/VQE/ADAPT-VQE baselines vs SQD vs FCI",
        "chemical_accuracy_threshold_mha": CHEMICAL_ACCURACY_MHA,
        "sources": {
            "best_circuits": str(args.best_circuits),
            "rl_metrics": str(args.rl_metrics) if args.rl_metrics else None,
            "hamiltonians": str(args.hamiltonians),
            "gqe_baseline": str(args.gqe_baseline) if args.gqe_baseline else None,
            "vqe_baseline": str(args.vqe_baseline) if args.vqe_baseline else None,
            "adapt_vqe_baseline": str(args.adapt_vqe_baseline) if args.adapt_vqe_baseline else None,
            "sqd_results": str(args.sqd_results) if args.sqd_results else None,
            "qpu_results": str(args.qpu_results) if args.qpu_results else None,
            "evaluation": str(args.evaluation) if args.evaluation else None,
            "optimized_results": str(args.optimized_results) if args.optimized_results else None,
        },
        "summary": summary,
        "rows": rows,
    }
    return payload


def write_csv(payload: dict[str, Any], csv_path: Path) -> None:
    """Write benchmark rows to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "molecule", "n_qubits", "n_electrons",
        "fci_energy_ha", "hf_energy_ha",
        "gqe_baseline_ha", "vqe_baseline_ha", "adapt_vqe_baseline_ha",
        "h_cgqe_gpu_ha", "h_cgqe_resampled_ha", "h_cgqe_optimized_ha", "h_cgqe_qpu_ha",
        "sqd_energy_ha", "sqd_recovered_energy_ha",
        "energy_provenance", "execution_type", "device", "training_exposure",
        "err_vs_fci_mha", "gqe_err_vs_fci_mha", "vqe_err_vs_fci_mha",
        "adapt_vqe_err_vs_fci_mha", "sqd_err_vs_fci_mha", "qpu_err_vs_fci_mha",
        "qpu_gpu_delta_mha", "improvement_over_gqe_mha",
        "chemical_accuracy", "n_ops",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow(row)
    print(f"CSV → {csv_path}")


def main() -> None:
    args = _parse_args()
    payload = build_benchmark(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"JSON → {args.out}")

    csv_out = args.csv_out or args.out.with_suffix(".csv")
    write_csv(payload, csv_out)

    s = payload["summary"]
    print(f"\n=== Benchmark Summary ===")
    print(f"  Molecules: {s['total_molecules']} ({s['molecules_with_gpu_energy']} with GPU energy)")
    print(f"  Qubit range: {s['qubit_range']['min']}–{s['qubit_range']['max']}")
    print(f"  Chemical accuracy: {s['chemical_accuracy_count']}/{s['total_molecules']} ({s['chemical_accuracy_pct']}%)")
    print(f"  Mean GPU error: {s['mean_gpu_error_mha']} mHa")
    if s["mean_gqe_error_mha"] is not None:
        print(f"  Mean GQE error: {s['mean_gqe_error_mha']} mHa")
        print(f"  Mean improvement: {s['mean_improvement_over_gqe_mha']} mHa")
    if s.get("mean_vqe_error_mha") is not None:
        print(f"  Mean VQE error: {s['mean_vqe_error_mha']} mHa")
    if s.get("mean_adapt_vqe_error_mha") is not None:
        print(f"  Mean ADAPT-VQE error: {s['mean_adapt_vqe_error_mha']} mHa")
    if s.get("mean_sqd_error_mha") is not None:
        print(f"  Mean SQD error: {s['mean_sqd_error_mha']} mHa")
    if s["molecules_with_qpu_validation"] > 0:
        print(f"  QPU validated: {s['molecules_with_qpu_validation']} molecules, mean error: {s['mean_qpu_error_mha']} mHa")
    g = s["generalization"]
    if g["unseen_mean_error_mha"] is not None:
        print(f"  Generalization: seen={g['seen_mean_error_mha']} mHa, unseen={g['unseen_mean_error_mha']} mHa")


if __name__ == "__main__":
    main()
