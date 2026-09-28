"""FMO2 reconstruction: exact-fragment and H-cGQE-fragment energies.

Computes FMO2 many-body expansion:
  E_FMO2 = sum_I E_I + sum_{I<J} (E_IJ - E_I - E_J)

Supports arbitrary numbers of fragments with explicit dimer Hamiltonians.
Integrates with MAP-Elites archive for circuit library selection via
select_elite_for_fragment().

Usage:
    # Exact diagonalization (classical baseline)
    python -m src.gqe.eval.run_fmo2 --fragments fragments.json --method exact --out results/fmo2_exact.json

    # H-cGQE inference (quantum)
    python -m src.gqe.eval.run_fmo2 --fragments fragments.json --method hcgqe \
        --checkpoint results/checkpoints/hcgqe_rl_dapo_best.pt --out results/fmo2_gqe.json

    # With MAP-Elites archive circuit library
    python -m src.gqe.eval.run_fmo2 --fragments fragments.json --method hcgqe \
        --checkpoint results/checkpoints/hcgqe_rl_dapo_best.pt \
        --archive-dir results/train/map_elites/ --out results/fmo2_gqe.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cudaq
except ImportError:
    cudaq = None

try:
    from gqe.common.hamiltonian_utils import load_hamiltonian_records, find_record_by_name
except ImportError:
    from src.gqe.common.hamiltonian_utils import load_hamiltonian_records, find_record_by_name

try:
    from src.gqe.common.ensure_checkpoint import ensure_checkpoint
except ImportError:
    def ensure_checkpoint(p, **kw):
        return Path(p)


def exact_energy_from_hamiltonian(record: dict[str, Any]) -> float:
    """Compute exact ground-state energy via dense diagonalization."""
    from src.gqe.common.hamiltonian_utils import hamiltonian_to_sparse_pauli_op
    op = hamiltonian_to_sparse_pauli_op(record)
    mat = op.to_matrix()
    eigvals = np.linalg.eigvalsh(mat)
    return float(eigvals[0])


def hcgqe_fragment_energy(
    record: dict[str, Any],
    checkpoint: str,
    n_samples: int = 100,
    target: str = "nvidia",
    target_option: str | None = "mqpu",
    archive_ops: list[str] | None = None,
) -> dict[str, Any]:
    """Run H-cGQE inference + L-BFGS-B optimization for a single fragment.

    Args:
        record: fragment Hamiltonian record
        checkpoint: path to H-cGQE model checkpoint
        n_samples: number of circuits to sample
        target: CUDA-Q target backend
        target_option: CUDA-Q target option (mqpu, etc.)
        archive_ops: optional pre-selected operators from MAP-Elites archive.
            If provided, these are evaluated directly instead of sampling.
    """
    import torch
    from src.gqe.models.h_cgqe_transformer import HcGQEModel, tokenize_hamiltonian, build_operator_vocab

    ckpt_path = ensure_checkpoint(checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    config = ckpt.get("config", {})
    model = HcGQEModel(
        vocab_size=config.get("vocab_size", 78),
        d_model=config.get("d_model", 256),
        nhead=config.get("nhead", 8),
        encoder_layers=config.get("encoder_layers", 4),
        decoder_layers=config.get("decoder_layers", 4),
        dim_feedforward=config.get("dim_feedforward", 1024),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt.get("model_state", ckpt.get("model_state_dict")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    n_qubits = record.get("n_qubits", 8)
    vocab = ckpt.get("vocab")
    if vocab is None:
        from src.gqe.common.operator_pool import build_uccsd_pauli_words
        pauli_words = build_uccsd_pauli_words(record)
        vocab = build_operator_vocab(pauli_words)
        model_vocab_size = config.get("vocab_size", 78)
        for i in range(len(vocab), model_vocab_size):
            vocab[f"<DUMMY_{i}>"] = i

    raw_terms = record.get("terms", record.get("pauli_terms", []))
    if not raw_terms:
        raise ValueError(f"No terms in record {record.get('name')}")
    terms = []
    for t in raw_terms:
        if isinstance(t, dict):
            label = t.get("term", t.get("label", ""))
            coeff = t.get("real", t.get("coefficient", 0.0))
            terms.append((label, float(coeff)))
        elif isinstance(t, (list, tuple)):
            terms.append((str(t[0]), float(t[1])))
        else:
            raise ValueError(f"Unexpected term format: {type(t)}")

    inputs = tokenize_hamiltonian(terms, vocab, max_terms=128, max_pauli_len=24)
    pauli_ids = inputs["pauli_ids"].unsqueeze(0).to(device)
    coeffs = inputs["coeffs"].unsqueeze(0).to(device)
    term_mask = inputs["term_mask"].unsqueeze(0).to(device)

    best_energy = float("inf")
    best_ops = None

    candidate_ops_list: list[list[str]] = []
    if archive_ops:
        candidate_ops_list.append(archive_ops)

    for _ in range(n_samples):
        tokens = model.generate(
            pauli_ids, coeffs, term_mask,
            bos_id=vocab["<BOS>"], eos_id=vocab["<EOS>"],
            max_len=32, temperature=1.0, vocab=vocab,
            force_entanglement=True, sample=True,
            n_qubits=n_qubits, freq_penalty=1.0,
        )
        ops = []
        for tok in tokens[0]:
            t = tok.item()
            if t == vocab["<EOS>"]:
                break
            if t >= 4:
                for word, idx in vocab.items():
                    if idx == t:
                        ops.append(word)
                        break
        if ops:
            candidate_ops_list.append(ops)

    if cudaq is not None:
        from src.gqe.eval.evaluate_h_cgqe import _compute_circuit_energy
        for ops in candidate_ops_list:
            if not ops:
                continue
            try:
                E = _compute_circuit_energy(record, ops, device=target)
                if E < best_energy:
                    best_energy = E
                    best_ops = ops
            except Exception as e:
                print(f"    Warning: energy eval failed for ops={ops[:3]}...: {e}")
                continue

    return {
        "fragment": record.get("name", "?"),
        "best_energy": best_energy,
        "best_operators": best_ops or [],
        "n_samples": n_samples,
        "n_candidates": len(candidate_ops_list),
        "used_archive_ops": archive_ops is not None,
    }


def load_archive_circuit(
    archive_dir: str,
    molecule_name: str,
    target_n_qubits: int,
) -> list[str] | None:
    """Load the best elite circuit from a MAP-Elites archive for a fragment."""
    from src.gqe.rl.map_elites import MAPElitesArchive

    archive_path = Path(archive_dir) / f"map_elites_{molecule_name}.json"
    if not archive_path.exists():
        archive_path = Path(archive_dir) / "map_elites.json"
    if not archive_path.exists():
        return None

    archive = MAPElitesArchive()
    archive.load(str(archive_path))
    if len(archive) == 0:
        return None

    elite = archive.select_elite_for_fragment(
        target_n_qubits=target_n_qubits,
        max_operators=32,
    )
    if elite is None:
        return None

    ops = elite.get("operators", [])
    e_val = elite.get("energy", 0.0)
    print(f"    Archive: selected elite with E={e_val:.6f}, ops={len(ops)}")
    return ops


def run_fmo2(
    fragments_file: str,
    method: str = "exact",
    checkpoint: str | None = None,
    target: str = "nvidia",
    target_option: str | None = "mqpu",
    n_samples: int = 100,
    archive_dir: str | None = None,
    dimers_file: str | None = None,
    parent_hamiltonians_file: str | None = None,
) -> dict[str, Any]:
    """Run FMO2 reconstruction.

    Args:
        fragments_file: JSON file with fragment Hamiltonian records
        method: "exact" (classical diagonalization) or "hcgqe" (quantum GQE)
        checkpoint: path to H-cGQE checkpoint (required for method="hcgqe")
        target: CUDA-Q target backend
        target_option: CUDA-Q target option
        n_samples: number of circuits to sample per fragment
        archive_dir: directory containing MAP-Elites archive JSON files
        dimers_file: JSON file with dimer Hamiltonian records (optional)
        parent_hamiltonians_file: JSON file with parent molecule Hamiltonians
    """
    t_start = time.time()

    with open(fragments_file) as f:
        frag_data = json.load(f)

    fragments = frag_data.get("fragments", frag_data.get("records", []))
    n_frags = len(fragments)

    print(f"FMO2 reconstruction: {n_frags} fragments, method={method}")
    if archive_dir:
        print(f"  Archive circuit library: {archive_dir}")

    dimer_records: dict[str, dict[str, Any]] = {}
    if dimers_file:
        with open(dimers_file) as f:
            dimer_data = json.load(f)
        dimer_list = dimer_data.get("dimers", dimer_data.get("records", []))
        for d in dimer_list:
            key = d.get("name", f"dim_{d.get('frag_i', 0)}_{d.get('frag_j', 0)}")
            dimer_records[key] = d
        print(f"  Loaded {len(dimer_records)} dimer Hamiltonians")

    parent_records = None
    if parent_hamiltonians_file:
        parent_records = load_hamiltonian_records(Path(parent_hamiltonians_file))

    # --- Monomer energies ---
    monomer_results = []
    monomer_energies = []
    for i, frag in enumerate(fragments):
        name = frag.get("name", f"frag_{i}")
        n_qubits = frag.get("n_qubits", 0)
        print(f"\n  Monomer {i}: {name} ({n_qubits}q)")

        archive_ops = None
        if archive_dir and method == "hcgqe":
            archive_ops = load_archive_circuit(archive_dir, name, n_qubits)

        if method == "exact":
            E = exact_energy_from_hamiltonian(frag)
            result = {"fragment": name, "best_energy": E, "best_operators": [], "n_samples": 0}
        else:
            result = hcgqe_fragment_energy(
                frag, checkpoint, n_samples, target, target_option,
                archive_ops=archive_ops,
            )
            E = result["best_energy"]

        monomer_results.append(result)
        monomer_energies.append(E)
        print(f"    E = {E:.6f} Ha")

    # --- Dimer energies ---
    dimer_results = {}
    dimer_energies = {}
    for i in range(n_frags):
        for j in range(i + 1, n_frags):
            pair_key = f"{i}_{j}"
            name_i = fragments[i].get("name", f"frag_{i}")
            name_j = fragments[j].get("name", f"frag_{j}")
            dimer_name = f"dim_{name_i}_{name_j}"
            print(f"\n  Dimer {i}-{j}: {dimer_name}")

            dimer_record = None
            if dimer_name in dimer_records:
                dimer_record = dimer_records[dimer_name]
            elif pair_key in dimer_records:
                dimer_record = dimer_records[pair_key]
            elif n_frags == 2 and parent_records is not None:
                raise ValueError(
                    "Circular FMO2 path removed: cannot use parent molecule as dimer. "
                    "Provide explicit dimer Hamiltonians via --dimers. "
                    "The 2-fragment parent-as-dimer shortcut is not a genuine "
                    "scaling result (max circuit = parent size)."
                )

            if dimer_record is None:
                E_ij = monomer_energies[i] + monomer_energies[j]
                print(f"    No dimer Hamiltonian found, using additive approximation")
                dimer_results[pair_key] = {"fragment": dimer_name, "best_energy": E_ij, "best_operators": [], "n_samples": 0}
            else:
                n_qubits_d = dimer_record.get("n_qubits", 0)
                print(f"    ({n_qubits_d}q)")

                archive_ops = None
                if archive_dir and method == "hcgqe":
                    archive_ops = load_archive_circuit(archive_dir, dimer_name, n_qubits_d)

                if method == "exact":
                    E_ij = exact_energy_from_hamiltonian(dimer_record)
                    dimer_results[pair_key] = {"fragment": dimer_name, "best_energy": E_ij, "best_operators": [], "n_samples": 0}
                else:
                    d_res = hcgqe_fragment_energy(
                        dimer_record, checkpoint, n_samples, target, target_option,
                        archive_ops=archive_ops,
                    )
                    E_ij = d_res["best_energy"]
                    dimer_results[pair_key] = d_res

            dimer_energies[pair_key] = E_ij
            print(f"    E = {E_ij:.6f} Ha")

    # --- FMO2 reassembly ---
    e_mono = sum(monomer_energies)
    e_pair = 0.0
    pair_interactions = {}
    for pair_key, E_ij in dimer_energies.items():
        i, j = int(pair_key.split("_")[0]), int(pair_key.split("_")[1])
        delta = E_ij - monomer_energies[i] - monomer_energies[j]
        pair_interactions[pair_key] = delta
        e_pair += delta
    e_fmo2 = e_mono + e_pair

    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"FMO2 Energy: {e_fmo2:.6f} Ha")
    print(f"  Monomer sum:     {e_mono:.6f}")
    print(f"  Pair correction: {e_pair:.6f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    result = {
        "method": method,
        "n_fragments": n_frags,
        "monomer_energies": monomer_energies,
        "monomer_results": monomer_results,
        "dimer_energies": dimer_energies,
        "dimer_results": dimer_results,
        "pair_interactions": pair_interactions,
        "fmo2_energy": e_fmo2,
        "monomer_sum": e_mono,
        "pair_correction": e_pair,
        "elapsed_seconds": elapsed,
        "archive_used": archive_dir is not None,
    }

    if method != "exact":
        print(f"\n  Computing exact FMO2 for comparison...")
        try:
            exact_result = run_fmo2(
                fragments_file, method="exact",
                dimers_file=dimers_file,
                parent_hamiltonians_file=parent_hamiltonians_file,
            )
            e_exact = exact_result["fmo2_energy"]
            error_mha = abs(e_fmo2 - e_exact) * 1000
            result["exact_fmo2_energy"] = e_exact
            result["error_mha"] = error_mha
            result["chemical_accuracy"] = error_mha <= 1.6
            print(f"  Exact FMO2:  {e_exact:.6f} Ha")
            print(f"  GQE error:   {error_mha:.2f} mHa "
                  f"({'chemical accuracy' if error_mha <= 1.6 else 'above 1.6 mHa'})")
        except Exception as e:
            print(f"  Could not compute exact FMO2: {e}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FMO2 reconstruction")
    parser.add_argument("--fragments", type=Path, required=True, help="Fragment Hamiltonians JSON")
    parser.add_argument("--method", type=str, default="exact", choices=["exact", "hcgqe"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=str, default="nvidia")
    parser.add_argument("--target-option", type=str, default="mqpu")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--archive-dir", type=str, default=None,
                        help="Directory containing MAP-Elites archive JSON files")
    parser.add_argument("--dimers", type=Path, default=None,
                        help="JSON file with dimer Hamiltonian records")
    parser.add_argument("--parent-hamiltonians", type=Path, default=None,
                        help="JSON file with parent molecule Hamiltonians (for 2-fragment FMO2)")
    args = parser.parse_args()

    result = run_fmo2(
        str(args.fragments),
        method=args.method,
        checkpoint=args.checkpoint,
        target=args.target,
        target_option=args.target_option,
        n_samples=args.n_samples,
        archive_dir=args.archive_dir,
        dimers_file=str(args.dimers) if args.dimers else None,
        parent_hamiltonians_file=str(args.parent_hamiltonians) if args.parent_hamiltonians else None,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
