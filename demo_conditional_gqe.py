from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gqe.data.graph_dataset import HamiltonianGraphDataset, collate_graph_samples
from src.gqe.models.chemistry_encoder import ChemistryEncoder

TOKEN_VOCAB = ["I", "X", "Y", "Z", "CNOT", "RY", "RZ", "SWAP", "CZ", "H", "T", "S"]


class TinyConditionedCircuitGenerator(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, prefix: torch.Tensor, steps: int = 8) -> torch.Tensor:
        hidden = prefix.unsqueeze(0)
        token = torch.zeros((prefix.size(0), 1), dtype=torch.long, device=prefix.device)
        outputs: list[torch.Tensor] = []
        for _ in range(steps):
            embedded = self.embed(token)
            sequence, hidden = self.gru(embedded, hidden)
            logits = self.head(sequence[:, -1])
            token = torch.argmax(logits, dim=-1, keepdim=True)
            outputs.append(token.squeeze(-1))
        return torch.stack(outputs, dim=1)


def _resolve_dataset_path() -> tuple[Path, bool]:
    fmo_path = ROOT / "results" / "data" / "fragments" / "fmo_hamiltonians.json"
    parent_path = ROOT / "results" / "data" / "fragments" / "hamiltonians.json"
    if fmo_path.exists():
        return fmo_path, False
    return parent_path, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the chemistry-aware conditional GQE demo.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "train" / "conditional_gqe_demo.json",
        help="Output JSON artifact path for the demo summary.",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    dataset_path, include_fragments = _resolve_dataset_path()
    dataset = HamiltonianGraphDataset(dataset_path, include_fragments=include_fragments, fragment_only=False)
    if len(dataset) == 0:
        raise RuntimeError(f"No graph samples found in {dataset_path}")

    sample = dataset[0]
    batch = collate_graph_samples([sample])

    encoder = ChemistryEncoder(
        node_feature_dim=int(batch["x"].shape[-1]),
        edge_feature_dim=int(batch["edge_attr"].shape[-1]),
        graph_feature_dim=int(batch["graph_attr"].shape[-1]),
        hidden_dim=64,
        latent_dim=64,
        conditioning_dim=64,
        num_layers=3,
        dropout=0.0,
    )
    encoder.eval()

    with torch.no_grad():
        latent = encoder(batch)
        prefix = encoder.to_prefix_token(batch)
        gru_state = encoder.to_gru_state(batch, num_layers=2)

        generator = TinyConditionedCircuitGenerator(vocab_size=len(TOKEN_VOCAB), hidden_dim=prefix.size(-1))
        generator.eval()
        token_ids = generator(prefix, steps=8)

    token_names = [[TOKEN_VOCAB[int(token)] for token in row.tolist()] for row in token_ids]

    artifact = {
        "dataset_path": str(dataset_path),
        "sample_name": sample.name,
        "latent": latent.squeeze(0).detach().cpu().tolist(),
        "prefix": prefix.squeeze(0).detach().cpu().tolist(),
        "gru_state_shape": list(gru_state.shape),
        "token_trace": token_names,
        "metadata": sample.metadata,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print("=== Chemistry-aware conditional GQE demo ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Sample name: {sample.name}")
    print(f"Nodes: {sample.x.shape[0]}, Edges: {sample.edge_index.shape[1]}")
    print(f"Node feature dim: {sample.x.shape[-1]}, Edge feature dim: {sample.edge_attr.shape[-1]}")
    print(f"Latent shape: {tuple(latent.shape)}")
    print(f"Prefix shape: {tuple(prefix.shape)}")
    print(f"GRU state shape: {tuple(gru_state.shape)}")
    print("Generated conditioned token trace:")
    print(json.dumps(token_names, indent=2))
    print("\nMetadata summary:")
    print(json.dumps(sample.metadata, indent=2, default=str))
    print(f"\nWrote demo artifact to: {args.out}")


if __name__ == "__main__":
    main()
