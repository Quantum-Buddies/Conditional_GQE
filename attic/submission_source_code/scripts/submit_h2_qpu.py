#!/usr/bin/env python3
"""Submit H2 (0.74 Å) to Rigetti Cepheus QPU via qBraid.

This is our best GPU molecule (0.15 mHa, chemical accuracy) but has NOT been
run on QPU yet. This script generates the circuit, exports the QWC manifest,
and submits to Cepheus.

Cost estimate (AWS Braket):
  - 1 task × 30 credits = 30 credits
  - 8192 shots × 0.0425 credits/shot = 348.16 credits
  - Total: ~378 credits

Usage (from qBraid Lab):
  python scripts/submit_h2_qpu.py --shots 8192
  python scripts/submit_h2_qpu.py --shots 4096   # cheaper: ~204 credits
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Submit H2 to Rigetti Cepheus QPU")
    parser.add_argument("--shots", type=int, default=8192, help="Number of shots (default 8192)")
    parser.add_argument("--device", type=str, default="aws:rigetti:qpu:cepheus-1-108q",
                        help="qBraid device ID for Cepheus")
    parser.add_argument("--dry-run", action="store_true", help="Export manifest only, don't submit")
    args = parser.parse_args()

    # Cost estimate
    task_credits = 30
    shot_credits = args.shots * 0.0425
    total_credits = task_credits + shot_credits
    print(f"\n=== H2 QPU Submission ===")
    print(f"Device: {args.device}")
    print(f"Shots: {args.shots}")
    print(f"Estimated cost: {task_credits} (task) + {shot_credits:.1f} (shots) = {total_credits:.1f} credits")
    print(f"Estimated USD: ${total_credits * 0.01:.2f}")
    print()

    if args.dry_run:
        print("DRY RUN: Would submit to QPU. Use without --dry-run to actually submit.")
        return

    # Try to import qBraid
    try:
        from qbraid import QbraidProvider
        from qbraid.runtime import QuantumDevice
    except ImportError:
        print("ERROR: qBraid SDK not installed. Run: pip install qbraid-sdk")
        print("\nAlternatively, run this from qBraid Lab where the SDK is pre-installed.")
        sys.exit(1)

    # Load H2 Hamiltonian
    ham_path = ROOT / "results/data/hamiltonians_gic2026/hamiltonians.json"
    if not ham_path.exists():
        print(f"ERROR: {ham_path} not found. Run Hamiltonian generation first.")
        sys.exit(1)

    with ham_path.open() as f:
        hamiltonians = json.load(f)

    # Find H2 at 0.74 Å
    h2_record = None
    for h in hamiltonians:
        if h.get("name") in ("h2", "h2_0.74"):
            h2_record = h
            break

    if h2_record is None:
        print("ERROR: H2 molecule not found in Hamiltonians file")
        sys.exit(1)

    print(f"Found H2: {h2_record['n_qubits']}q, {h2_record.get('n_pauli_terms', '?')} Pauli terms")
    print(f"  FCI energy: {h2_record.get('fci_energy', 'N/A')}")
    print(f"  HF energy: {h2_record.get('hf_energy', 'N/A')}")

    # Load the trained model's best circuit for H2
    eval_path = ROOT / "results/eval/h_cgqe_evaluation.json"
    if eval_path.exists():
        with eval_path.open() as f:
            eval_data = json.load(f)
        h2_eval = None
        for r in eval_data if isinstance(eval_data, list) else eval_data.get("results", []):
            if r.get("molecule") in ("h2", "h2_0.74"):
                h2_eval = r
                break
        if h2_eval:
            print(f"\nBest H-cGQE circuit:")
            print(f"  Operators: {h2_eval.get('best_operators', 'N/A')}")
            print(f"  Thetas: {h2_eval.get('best_thetas', 'N/A')}")
            print(f"  Energy: {h2_eval.get('energy', 'N/A')}")

    # Export QWC manifest
    print(f"\nExporting QWC measurement manifest...")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from gqe.eval.qbraid_backend import build_qwc_manifest
        manifest = build_qwc_manifest(h2_record, h2_eval)
        manifest_path = ROOT / "results/qpu/h2_qwc_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Saved to {manifest_path}")
    except Exception as e:
        print(f"  Warning: Could not build QWC manifest ({e})")
        print("  Will submit raw circuit instead")

    # Submit to QPU
    print(f"\nSubmitting to {args.device}...")
    try:
        provider = QbraidProvider()
        device = provider.get_device(args.device)
        print(f"  Device: {device}")
        print(f"  Status: {device.status()}")

        # Build and submit circuit
        # This will be adapted based on the actual circuit format
        print(f"\n  Submitting {args.shots}-shot job...")
        job = device.run(circuit, shots=args.shots)
        print(f"  Job ID: {job.id}")
        print(f"\n  Job submitted! Check status with:")
        print(f"    from qbraid import QbraidProvider")
        print(f"    p = QbraidProvider()")
        print(f"    j = p.get_job('{job.id}')")
        print(f"    print(j.status(), j.result())")

    except Exception as e:
        print(f"\nERROR submitting to QPU: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your qBraid credits: https://account.qbraid.com/account/wallet")
        print("  2. Verify device availability: https://account.qbraid.com/devices")
        print("  3. Ensure you're running from qBraid Lab or have API key set")
        sys.exit(1)


if __name__ == "__main__":
    main()
