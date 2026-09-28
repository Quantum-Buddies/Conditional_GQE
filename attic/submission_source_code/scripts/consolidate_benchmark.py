#!/usr/bin/env python3
"""Consolidate all pipeline results into a single benchmark JSON for the GIC 2026 submission.

Collects:
- H-cGQE optimized circuit data (local GPU L-BFGS-B)
- Local SQD pilot results (L40S)
- AWS SV1 simulator SQD results
- Rigetti Cepheus-1-108Q QPU SQD + QWC results
- Ledger cost accounting
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    out_dir = ROOT / "results/eval/benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. H-cGQE optimized circuits (supervised checkpoint)
    optimized = load_json(ROOT / "results/eval/h_cgqe_optimized.json") or []

    # 1b. H-cGQE RL-optimized circuits
    rl_optimized = load_json(ROOT / "results/eval/h_cgqe_rl_optimized.json") or []

    # 2. Local SQD pilot
    local_sqd = load_json(ROOT / "results/eval/sqd_pilot/sqd_pilot_consolidated.json")

    # 3. AWS SV1 simulator results
    sv1_results = load_json(ROOT / "results/qpu/h2_sv1_sqd_results.json")

    # 4. Cepheus QPU results (supervised checkpoint)
    cepheus_results = load_json(ROOT / "results/qpu/cepheus_sqd_results.json")

    # 4b. Cepheus QPU results (RL checkpoint)
    cepheus_rl_results = load_json(ROOT / "results/qpu/cepheus_rl_sqd_results.json")

    # 5. Cepheus submission metadata
    cepheus_meta = load_json(ROOT / "results/qpu/cepheus_submission_meta.json")
    cepheus_rl_meta = load_json(ROOT / "results/qpu/cepheus_rl_submission_meta.json")

    # 6. Ledger summary
    from src.gqe.eval.qpu_ledger import QpuLedger
    ledger = QpuLedger(ROOT / "results/eval/qpu_jobs.sqlite")
    ledger_entries = []
    total_cost = 0.0
    for entry in ledger.list_jobs():
        ledger_entries.append({
            "job_id": entry.job_id,
            "molecule": entry.molecule,
            "device": entry.device_id,
            "status": entry.status.value,
            "shots": entry.shots,
            "n_qubits": entry.n_qubits,
            "measurement_basis": entry.measurement_basis,
            "pipeline_stage": entry.pipeline_stage,
            "estimated_cost_credits": entry.estimated_cost_credits,
        })
        total_cost += entry.estimated_cost_credits or 0
    ledger.close()

    # Build consolidated benchmark
    benchmark = {
        "experiment": "gic2026_nisq_pipeline",
        "description": "H-cGQE + L-BFGS-B + SQD pipeline: AI inference (local GPU) → QPU execution (Rigetti Cepheus-1-108Q) → classical post-processing (SQD)",
        "timestamp": cepheus_meta.get("timestamp") if cepheus_meta else None,

        "pipeline_stages": {
            "stage_1_ai_inference": {
                "description": "H-cGQE transformer generates operator sequences",
                "model": "h_cgqe_rl",
                "n_molecules_optimized": len(optimized),
                "molecules": [
                    {
                        "name": r["molecule"],
                        "n_qubits": r["n_qubits"],
                        "n_operators": len(r.get("best_operators", [])),
                        "optimized_energy": r["best_energy"],
                        "n_starts": r.get("optimization_metadata", {}).get("n_starts", 0),
                    }
                    for r in optimized
                ],
            },
            "stage_2_classical_optimization": {
                "description": "L-BFGS-B coefficient optimization on L40S GPU via CUDA-Q",
                "optimizer": "L-BFGS-B",
                "backend": "CUDA-Q nvidia target (L40S)",
                "n_molecules": len(optimized),
            },
            "stage_3_qpu_execution": {
                "description": "SQD sampling circuits submitted to Rigetti Cepheus-1-108Q via qBraid",
                "device": "aws:rigetti:qpu:cepheus-1-108q",
                "n_sqd_jobs": len(cepheus_meta.get("sqd_jobs", {})) if cepheus_meta else 0,
                "n_qwc_jobs": sum(len(v) for v in cepheus_meta.get("qwc_jobs", {}).values()) if cepheus_meta else 0,
                "shots_per_circuit": 4096,
            },
            "stage_4_post_processing": {
                "description": "SQD classical diagonalization from QPU measurement counts",
                "method": "sample-based quantum diagonalization",
                "symmetry_filters": ["particle_number", "spin_parity"],
            },
        },

        "results": {
            "local_sqd_pilot": {},
            "sv1_simulator": {},
            "cepheus_qpu_sqd": {},
            "cepheus_qpu_qwc": {},
            "cepheus_qpu_rl_sqd": {},
        },
        "rl_checkpoint": {
            "optimized_circuits": [
                {
                    "name": r["molecule"],
                    "n_qubits": r["n_qubits"],
                    "n_operators": r["n_operators"],
                    "rl_unoptimized_energy": r.get("rl_unoptimized_energy"),
                    "optimized_energy": r["optimized_energy"],
                    "checkpoint": r.get("checkpoint", "rl"),
                }
                for r in rl_optimized
            ],
        },

        "cost_accounting": {
            "budget_credits": 13400.0,
            "total_estimated_credits": total_cost,
            "remaining_credits": 13400.0 - total_cost,
            "ledger_entries": ledger_entries,
        },
    }

    # Parse local SQD pilot
    if local_sqd:
        for r in local_sqd.get("results", []):
            mol = r["molecule"]
            controls = r.get("controls", {})
            for ctrl_name, ctrl_data in controls.items():
                analysis = ctrl_data.get("analysis", {})
                benchmark["results"]["local_sqd_pilot"][f"{mol}_{ctrl_name}"] = {
                    "molecule": mol,
                    "control": ctrl_name,
                    "n_qubits": r["n_qubits"],
                    "fci_energy": r.get("fci_energy"),
                    "sqd_energy": analysis.get("best_energy"),
                    "error_mha": analysis.get("error_vs_fci_mha"),
                    "variational_bound": analysis.get("variational_bound_satisfied"),
                }

    # Parse SV1 results
    if sv1_results:
        for mol, data in sv1_results.items():
            if data.get("status") == "completed":
                sqd = data.get("sqd_analysis", {})
                benchmark["results"]["sv1_simulator"][mol] = {
                    "molecule": mol,
                    "device": "aws:aws:sim:sv1",
                    "fci_energy": sqd.get("fci_energy"),
                    "sqd_energy": sqd.get("sqd_energy"),
                    "error_mha": sqd.get("error_vs_fci_mha"),
                    "variational_bound": sqd.get("variational_bound_satisfied"),
                    "n_bitstrings": sqd.get("n_symmetry_filtered"),
                }

    # Parse Cepheus SQD results (supervised checkpoint)
    if cepheus_results:
        for mol, data in cepheus_results.items():
            if mol.endswith("_qwc"):
                continue
            if data.get("status") == "completed":
                sqd = data.get("sqd_analysis", {})
                benchmark["results"]["cepheus_qpu_sqd"][mol] = {
                    "molecule": mol,
                    "device": "aws:rigetti:qpu:cepheus-1-108q",
                    "checkpoint": "supervised",
                    "fci_energy": sqd.get("fci_energy"),
                    "sqd_energy": sqd.get("sqd_energy"),
                    "error_mha": sqd.get("error_vs_fci_mha"),
                    "variational_bound": sqd.get("variational_bound_satisfied"),
                    "n_bitstrings": sqd.get("n_symmetry_filtered"),
                    "n_unique_raw": sqd.get("n_unique_raw"),
                }

        # Parse QWC results
        for mol, data in cepheus_results.items():
            if not mol.endswith("_qwc"):
                continue
            if data.get("status") == "completed":
                base_mol = mol.replace("_qwc", "")
                benchmark["results"]["cepheus_qpu_qwc"][base_mol] = {
                    "molecule": base_mol,
                    "device": "aws:rigetti:qpu:cepheus-1-108q",
                    "qwc_energy": data.get("qwc_energy"),
                    "fci_energy": data.get("fci_energy"),
                    "error_mha": data.get("error_vs_fci_mha"),
                }

    # Parse Cepheus RL SQD results
    if cepheus_rl_results:
        for mol, data in cepheus_rl_results.items():
            if data.get("status") == "completed":
                sqd = data.get("sqd_analysis", {})
                benchmark["results"]["cepheus_qpu_rl_sqd"][mol] = {
                    "molecule": mol,
                    "device": "aws:rigetti:qpu:cepheus-1-108q",
                    "checkpoint": "rl",
                    "fci_energy": sqd.get("fci_energy"),
                    "sqd_energy": sqd.get("sqd_energy"),
                    "error_mha": sqd.get("error_vs_fci_mha"),
                    "variational_bound": sqd.get("variational_bound_satisfied"),
                    "n_bitstrings": sqd.get("n_symmetry_filtered"),
                    "n_unique_raw": sqd.get("n_unique_raw"),
                }

    # Summary table
    print("\n" + "=" * 80)
    print("GIC 2026 NISQ Pipeline — Consolidated Benchmark")
    print("=" * 80)

    print("\n--- H-cGQE Optimized Circuits (Supervised checkpoint) ---")
    for m in benchmark["pipeline_stages"]["stage_1_ai_inference"]["molecules"]:
        print(f"  {m['name']:15s}: {m['n_qubits']:2d}q, {m['n_operators']:2d} ops, E={m['optimized_energy']:.6f}")

    print("\n--- H-cGQE RL-Optimized Circuits (RL checkpoint) ---")
    for m in benchmark.get("rl_checkpoint", {}).get("optimized_circuits", []):
        print(f"  {m['name']:15s}: {m['n_qubits']:2d}q, {m['n_operators']:2d} ops, "
              f"RL_E={m['rl_unoptimized_energy']:.6f}, Opt_E={m['optimized_energy']:.6f}")

    print("\n--- SQD Results ---")
    print(f"  {'Molecule':15s} {'Source':25s} {'SQD Energy':>12s} {'FCI Energy':>12s} {'Error (mHa)':>12s} {'Var Bound':>10s}")
    print(f"  {'-'*15} {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for source, results_dict in [
        ("Local L40S", benchmark["results"]["local_sqd_pilot"]),
        ("AWS SV1 sim", benchmark["results"]["sv1_simulator"]),
        ("Cepheus QPU (supervised)", benchmark["results"]["cepheus_qpu_sqd"]),
        ("Cepheus QPU (RL)", benchmark["results"]["cepheus_qpu_rl_sqd"]),
    ]:
        for key, r in sorted(results_dict.items()):
            print(f"  {r.get('molecule', key):15s} {source:25s} {r.get('sqd_energy', 0):>12.6f} "
                  f"{r.get('fci_energy', 0):>12.6f} {r.get('error_mha', 0):>12.3f} "
                  f"{str(r.get('variational_bound', '')):>10s}")

    print(f"\n--- QWC Results (Cepheus QPU) ---")
    for mol, r in benchmark["results"]["cepheus_qpu_qwc"].items():
        print(f"  {mol:15s} QWC energy={r.get('qwc_energy', 0):.6f} FCI={r.get('fci_energy', 0):.6f} "
              f"error={r.get('error_mha', 0):.3f} mHa")

    print(f"\n--- Cost Accounting ---")
    print(f"  Budget: {benchmark['cost_accounting']['budget_credits']:.0f} credits")
    print(f"  Spent:  {benchmark['cost_accounting']['total_estimated_credits']:.2f} credits")
    print(f"  Remaining: {benchmark['cost_accounting']['remaining_credits']:.2f} credits")

    # Save
    out_path = out_dir / "gic2026_consolidated_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\nBenchmark saved: {out_path}")


if __name__ == "__main__":
    main()
