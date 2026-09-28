#!/usr/bin/env python3
"""Submit remaining molecules to Rigetti Cepheus QPU via qBraid.

Runs 9 molecules at 4096 shots each = ~1,836 credits total.
All molecules have direct GPU counterparts for comparison.

Usage (from qBraid Lab):
  python scripts/submit_remaining_qpu.py --shots 4096
  python scripts/submit_remaining_qpu.py --shots 4096 --dry-run  # export only
  python scripts/submit_remaining_qpu.py --shots 4096 --molecules h2_0.74 h2_1.0  # subset
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Molecules to submit (all have GPU counterparts)
TARGET_MOLECULES = [
    # H2 dissociation curve (4q) — our best GPU molecule
    {"name": "h2_0.5",   "n_qubits": 4,  "gpu_error_mha": 0.77,  "geometry": [["H", [0,0,0]], ["H", [0,0,0.5]]]},
    {"name": "h2_0.74",  "n_qubits": 4,  "gpu_error_mha": 0.15,  "geometry": [["H", [0,0,0]], ["H", [0,0,0.74]]]},
    {"name": "h2_1.0",   "n_qubits": 4,  "gpu_error_mha": 0.00,  "geometry": [["H", [0,0,0]], ["H", [0,0,1.0]]]},
    {"name": "h2_1.5",   "n_qubits": 4,  "gpu_error_mha": 19.93, "geometry": [["H", [0,0,0]], ["H", [0,0,1.5]]]},
    {"name": "h2_2.0",   "n_qubits": 4,  "gpu_error_mha": 66.54, "geometry": [["H", [0,0,0]], ["H", [0,0,2.0]]]},
    # LiH at 2 geometries (8q) — 2nd best molecule
    {"name": "lih_1.2",  "n_qubits": 8,  "gpu_error_mha": 2.09,  "geometry": [["Li", [0,0,0]], ["H", [0,0,1.2]]]},
    {"name": "lih_2.0",  "n_qubits": 8,  "gpu_error_mha": 2.01,  "geometry": [["Li", [0,0,0]], ["H", [0,0,2.0]]]},
    # EUV photoresist molecules (8q) — direct GPU comparison
    {"name": "methyl_iodide", "n_qubits": 8, "gpu_error_mha": 1.59,
     "geometry": [["C", [0,0,0]], ["H", [0,0.63,0.63]], ["H", [0,-0.63,0.63]], ["H", [0.63,0,-0.63]], ["I", [0,0,2.14]]]},
    {"name": "imeph",         "n_qubits": 8, "gpu_error_mha": 24.78,
     "geometry": [["I", [0,0,0]], ["C", [0,0,2.14]], ["C", [1.40,0,2.74]], ["C", [1.40,0,4.14]], ["C", [0,0,4.74]], ["C", [-1.40,0,4.14]], ["C", [-1.40,0,2.74]], ["H", [0,0,5.83]], ["H", [2.49,0,2.24]], ["H", [2.49,0,4.64]], ["H", [-2.49,0,4.64]], ["H", [-2.49,0,2.24]]]},
]


def main():
    parser = argparse.ArgumentParser(description="Submit remaining molecules to Cepheus QPU")
    parser.add_argument("--shots", type=int, default=4096, help="Shots per molecule (default 4096)")
    parser.add_argument("--device", type=str, default="aws:rigetti:qpu:cepheus-1-108q")
    parser.add_argument("--dry-run", action="store_true", help="Export manifests only, don't submit")
    parser.add_argument("--molecules", nargs="*", help="Subset of molecule names to submit")
    args = parser.parse_args()

    targets = TARGET_MOLECULES
    if args.molecules:
        targets = [t for t in TARGET_MOLECULES if t["name"] in args.molecules]

    # Cost calculation
    task_credits = 30
    shot_credits = args.shots * 0.0425
    per_mol = task_credits + shot_credits
    total = per_mol * len(targets)

    print(f"\n{'='*70}")
    print(f"QPU Submission Plan: {len(targets)} molecules × {args.shots} shots")
    print(f"Device: {args.device}")
    print(f"{'='*70}")
    print(f"\n{'Molecule':25s} {'Q':>3s} {'GPU err':>10s} {'Credits':>10s}")
    print("-" * 55)
    for t in targets:
        print(f"{t['name']:25s} {t['n_qubits']:>3d}q {t['gpu_error_mha']:>8.2f} mHa {per_mol:>8.1f}")
    print(f"\n{'Total:':>35s} {total:>8.1f} credits")
    print(f"Remaining after: {1925 - total:.1f} credits")
    print(f"Cost in USD: ${total * 0.01:.2f}")
    print()

    if args.dry_run:
        print("DRY RUN — no QPU submission. Manifests will be exported.")
    else:
        print("⚠  This will use REAL qBraid credits!")
        confirm = input(f"  Submit {len(targets)} jobs for {total:.0f} credits? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return

    # Try to import qBraid
    try:
        from qbraid import QbraidProvider
        from qbraid.runtime import QuantumDevice
        HAS_QBRAID = True
    except ImportError:
        print("\n⚠ qBraid SDK not installed. Exporting manifests only.")
        print("  Run this from qBraid Lab for actual QPU submission.")
        HAS_QBRAID = False

    # Load Hamiltonians
    ham_path = ROOT / "results/data/hamiltonians_gic2026/hamiltonians.json"
    with ham_path.open() as f:
        hamiltonians = json.load(f)
    ham_records = {h["name"]: h for h in hamiltonians}

    # Load GPU evaluation results for circuit data
    eval_path = ROOT / "results/phase3_final/consolidated_results_gic2026.json"
    with eval_path.open() as f:
        consolidated = json.load(f)
    gpu_results = {r["molecule"]: r for r in consolidated.get("gpu_benchmark", [])}

    results = []
    manifest_dir = ROOT / "results/qpu/remaining_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for t in targets:
        name = t["name"]
        print(f"\n--- {name} ({t['n_qubits']}q) ---")

        # Get Hamiltonian
        ham = ham_records.get(name)
        if ham is None:
            # Try without bond distance suffix
            base_name = name.split("_")[0]
            ham = ham_records.get(base_name)
        if ham is None:
            print(f"  ⚠ No Hamiltonian found for {name}, skipping")
            continue

        # Get GPU circuit
        gpu = gpu_results.get(name, {})
        operators = gpu.get("best_operators", [])
        print(f"  Hamiltonian: {ham['n_qubits']}q, {ham.get('n_pauli_terms', '?')} terms")
        print(f"  GPU operators: {operators}")
        print(f"  GPU energy: {gpu.get('h_cgqe_optimized_energy', 'N/A')}")
        print(f"  GPU error: {t['gpu_error_mha']:.2f} mHa")

        # Export manifest
        manifest = {
            "molecule": name,
            "n_qubits": ham["n_qubits"],
            "operators": operators,
            "thetas": gpu.get("best_thetas", []),
            "hamiltonian_terms": ham.get("terms", [])[:20],  # First 20 for preview
            "gpu_reference": {
                "energy": gpu.get("h_cgqe_optimized_energy"),
                "error_mha": t["gpu_error_mha"],
                "fci_energy": gpu.get("reference_energy"),
            },
            "shots": args.shots,
            "device": args.device,
        }
        manifest_path = manifest_dir / f"{name}_manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Manifest: {manifest_path}")

        if args.dry_run or not HAS_QBRAID:
            results.append({"molecule": name, "status": "manifest_exported", "manifest": str(manifest_path)})
            continue

        # Submit to QPU
        try:
            provider = QbraidProvider()
            device = provider.get_device(args.device)

            # Build circuit from operators + thetas
            # This uses the QWC grouping from qbraid_backend
            sys.path.insert(0, str(ROOT / "src"))
            from gqe.eval.qbraid_backend import build_qwc_manifest, submit_qwc_grouped

            qwc_manifest = build_qwc_manifest(ham, gpu)
            print(f"  QWC groups: {len(qwc_manifest.get('groups', []))}")

            job_result = submit_qwc_grouped(
                device=device,
                manifest=qwc_manifest,
                shots=args.shots,
            )

            print(f"  ✓ Job submitted: {job_result.get('job_id', 'N/A')}")
            results.append({
                "molecule": name,
                "status": "submitted",
                "job_id": job_result.get("job_id"),
                "gpu_error_mha": t["gpu_error_mha"],
            })

        except Exception as e:
            print(f"  ✗ Submission failed: {e}")
            results.append({"molecule": name, "status": "failed", "error": str(e)})

    # Save submission ledger
    ledger_path = ROOT / "results/qpu/remaining_submissions.json"
    with ledger_path.open("w") as f:
        json.dump({
            "timestamp": "2026-07-26T22:00:00Z",
            "device": args.device,
            "shots": args.shots,
            "total_credits_estimated": total,
            "n_molecules": len(targets),
            "submissions": results,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Submission ledger: {ledger_path}")
    n_submitted = sum(1 for r in results if r["status"] == "submitted")
    n_exported = sum(1 for r in results if r["status"] == "manifest_exported")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    print(f"  Submitted: {n_submitted}, Exported: {n_exported}, Failed: {n_failed}")
    if n_submitted > 0:
        print(f"\n  Retrieve results later with:")
        print(f"    python scripts/retrieve_and_sqd.py --ledger {ledger_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
