#!/usr/bin/env python3
"""Consolidate all GIC 2026 benchmark results into a single submission-ready JSON."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

def load_json(p):
    with open(ROOT / p) as f:
        return json.load(f)

def main():
    eval_data = load_json("results/eval/h_cgqe_evaluation_gic2026.json")
    opt_data = load_json("results/eval/h_cgqe_optimized_gic2026.json")
    qpu_data = load_json("results/qpu/cepheus_rl_sqd_results.json")
    rl_opt = load_json("results/eval/h_cgqe_rl_optimized.json")
    qpu_counts = load_json("results/qpu/cepheus_rl_counts.json")

    # Build GPU benchmark table
    opt_map = {r["molecule"]: r for r in opt_data}
    eval_map = {e["molecule"]: e for e in eval_data}

    gpu_benchmark = []
    for mol, o in opt_map.items():
        e = eval_map.get(mol, {})
        ref = e.get("reference_energy")
        gqe = e.get("baseline_energy")
        gpu_benchmark.append({
            "molecule": mol,
            "n_qubits": o["n_qubits"],
            "n_operators": len(o["best_operators"]),
            "best_operators": o["best_operators"],
            "h_cgqe_unoptimized_energy": e.get("best_generated_energy"),
            "h_cgqe_optimized_energy": o["best_energy"],
            "cudaq_gqe_energy": gqe,
            "reference_energy": ref,
            "error_vs_reference_mha": round(abs(o["best_energy"] - ref) * 1000, 2) if ref else None,
            "improvement_over_gqe_mha": round((gqe - o["best_energy"]) * 1000, 2) if gqe else None,
        })

    # Build QPU results table
    qpu_results = []
    for mol, r in qpu_data.items():
        qc = qpu_counts.get(mol, {})
        qpu_results.append({
            "molecule": mol,
            "n_qubits": r["n_qubits"],
            "n_electrons": r["n_electrons"],
            "device": "aws:rigetti:qpu:cepheus-1-108q",
            "shots": r["n_shots"],
            "n_unique_bitstrings": r["n_unique_bitstrings"],
            "hf_energy": r["hf_energy"],
            "sqd_energy": r["sqd_energy"],
            "fci_energy": r["fci_energy"],
            "error_vs_fci_mha": r.get("error_vs_fci_mha"),
            "improvement_over_hf_mha": round(r["improvement_over_hf_mha"], 2),
            "job_id": qc.get("job_id"),
        })

    # Build RL-optimized circuit table
    rl_circuits = []
    for r in rl_opt:
        rl_circuits.append({
            "molecule": r["molecule"],
            "n_qubits": r["n_qubits"],
            "n_operators": len(r["best_operators"]),
            "rl_unoptimized_energy": r["rl_unoptimized_energy"],
            "optimized_energy": r["optimized_energy"],
            "best_operators": r["best_operators"],
            "best_thetas": r["best_thetas"],
            "optimization_time_seconds": r["optimization_time_seconds"],
        })

    # Summary stats
    chemically_accurate = sum(
        1 for b in gpu_benchmark
        if b["error_vs_reference_mha"] is not None and b["error_vs_reference_mha"] <= 1.6
    )
    total_molecules = len(gpu_benchmark)

    consolidated = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "challenge": "GIC 2026",
        "team": "Quantum-Buddies",
        "model": "H-cGQE (Hierarchical conditional Generative Quantum Eigensolver)",
        "hardware": {
            "gpu": "NVIDIA L40S (48GB, PCIe)",
            "qpu": "Rigetti Cepheus-1-108Q (108q superconducting)",
            "qpu_access": "qBraid SDK",
            "classical_optimizer": "L-BFGS-B (scipy.optimize)",
        },
        "summary": {
            "total_molecules_benchmarked": total_molecules,
            "chemically_accurate_count": chemically_accurate,
            "chemical_accuracy_threshold_mha": 1.6,
            "qpu_molecules_run": len(qpu_results),
            "qpu_chemically_accurate": sum(1 for q in qpu_results if q.get("error_vs_fci_mha") is not None and q["error_vs_fci_mha"] <= 1.6),
            "best_gpu_error_mha": min(b["error_vs_reference_mha"] for b in gpu_benchmark if b["error_vs_reference_mha"] is not None),
            "best_qpu_error_mha": min(q["error_vs_fci_mha"] for q in qpu_results if q.get("error_vs_fci_mha") is not None),
        },
        "gpu_benchmark": gpu_benchmark,
        "qpu_results": qpu_results,
        "rl_optimized_circuits": rl_circuits,
        "pipeline_description": {
            "stage_1": "H-cGQE Transformer autoregressively generates Pauli operator sequences (circuit architecture)",
            "stage_2": "L-BFGS-B classical optimization of rotation coefficients (thetas) on GPU via CUDA-Q statevector",
            "stage_3": "RL fine-tuning (DAPO) with energy-based rewards for improved operator selection",
            "stage_4": "QPU sampling on Rigetti Cepheus-1-108Q + SQD post-processing for ground state energy",
        },
    }

    out_path = ROOT / "results/phase3_final/consolidated_results_gic2026.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"Consolidated results saved: {out_path}")
    print(f"  GPU benchmark: {total_molecules} molecules, {chemically_accurate} chemically accurate")
    print(f"  QPU results: {len(qpu_results)} molecules on Cepheus-1-108Q")
    print(f"  Best GPU error: {consolidated['summary']['best_gpu_error_mha']:.2f} mHa")
    print(f"  Best QPU error: {consolidated['summary']['best_qpu_error_mha']:.2f} mHa")

if __name__ == "__main__":
    main()
