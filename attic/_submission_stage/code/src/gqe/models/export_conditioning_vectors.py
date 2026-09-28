"""Export conditioning vectors from a trained chemistry encoder.

Loads a checkpoint produced by train_chemistry_encoder_ddp.py (or the single-GPU
version) and writes a JSON file mapping each Hamiltonian record name to its
latent conditioning vector.

Usage:
    python src/gqe/models/export_conditioning_vectors.py \
        --json results/data/hamiltonians.json \
        --checkpoint results/train/ddp_graph_conditioning.pt \
        --out results/train/ddp_conditioning_vectors.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from ..data.graph_dataset import HamiltonianGraphDataset, collate_graph_samples
    from .chemistry_encoder import ChemistryEncoder
    from .train_chemistry_encoder import FlatChemistryRegressor
except ImportError:
    from data.graph_dataset import HamiltonianGraphDataset, collate_graph_samples
    from models.chemistry_encoder import ChemistryEncoder
    from models.train_chemistry_encoder import FlatChemistryRegressor


def main() -> None:
    parser = argparse.ArgumentParser(description="Export conditioning vectors from trained encoder.")
    parser.add_argument("--json", type=Path, required=True, help="Hamiltonian dataset JSON")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON for conditioning vectors")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-fragments", action="store_true")
    parser.add_argument("--fragment-only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    mode = config.get("mode", "graph")

    dataset = HamiltonianGraphDataset(
        args.json,
        include_fragments=args.include_fragments,
        fragment_only=args.fragment_only,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples in {args.json}")

    sample = collate_graph_samples([dataset[0]])
    x_dim = int(sample["x"].shape[-1])
    e_dim = int(sample["edge_attr"].shape[-1])
    g_dim = int(sample["graph_attr"].shape[-1])

    if mode == "graph":
        encoder = ChemistryEncoder(
            node_feature_dim=x_dim,
            edge_feature_dim=e_dim,
            graph_feature_dim=g_dim,
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            conditioning_dim=config["conditioning_dim"],
            num_layers=config["num_layers"],
            dropout=config.get("dropout", 0.1),
        )
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        encoder.to(device)
        encoder.eval()
    else:
        flat_model = FlatChemistryRegressor(
            input_dim=g_dim,
            target_dim=len(ckpt["target_names"]),
            latent_dim=config["latent_dim"],
            head_hidden=config.get("head_hidden", 128),
            dropout=config.get("dropout", 0.1),
        )
        flat_model.load_state_dict(ckpt["encoder_state_dict"])
        flat_model.to(device)
        flat_model.eval()
        encoder = flat_model

    results: dict[str, Any] = {}
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample_raw = dataset[idx]
            batch = collate_graph_samples([sample_raw])
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            meta = batch["metadata"][0]
            name = meta.get("name", meta.get("parent_name", f"sample_{idx}"))
            if mode == "graph":
                z = encoder(batch)[0]
            else:
                z = encoder(batch)
            latent = z.cpu().numpy().tolist()
            results[name] = {
                "latent": latent,
                "conditioning_dim": len(latent),
                "metadata": {
                    "n_qubits": meta.get("n_qubits", meta.get("node_count", 0)),
                    "n_pauli_terms": meta.get("n_pauli_terms", 0),
                    "is_fragment": bool(meta.get("parent_name")),
                },
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Exported {len(results)} conditioning vectors to {args.out}")


if __name__ == "__main__":
    main()
