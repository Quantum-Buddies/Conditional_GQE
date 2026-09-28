from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=r"CUDA initialization: The NVIDIA driver on your system is too old.*",
    category=UserWarning,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:  # pragma: no cover - package import path
    from ..data.graph_dataset import HamiltonianGraphDataset, GraphSample, collate_graph_samples
    from .chemistry_encoder import ChemistryEncoder
except ImportError:  # pragma: no cover - direct script execution fallback
    try:
        from data.graph_dataset import HamiltonianGraphDataset, GraphSample, collate_graph_samples
        from models.chemistry_encoder import ChemistryEncoder
    except ImportError:  # pragma: no cover - direct module execution fallback
        from data.graph_dataset import HamiltonianGraphDataset, GraphSample, collate_graph_samples  # type: ignore[no-redef]
        from chemistry_encoder import ChemistryEncoder  # type: ignore[no-redef]


TARGET_NAMES = [
    "n_qubits",
    "n_pauli_terms",
    "l1_norm",
    "mean_abs_coeff",
    "max_abs_coeff",
    "n_one_body_terms",
    "n_two_body_terms",
    "n_three_body_terms",
    "n_four_body_terms",
    "charge",
    "multiplicity",
    "n_active_electrons",
    "n_active_orbitals",
    "fragment_count",
    "is_fragment",
]


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _coerce_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return 0.0


def _target_vector(sample_or_meta: GraphSample | dict[str, Any]) -> np.ndarray:
    meta = sample_or_meta.metadata if isinstance(sample_or_meta, GraphSample) else sample_or_meta
    active = meta.get("active_space") or {}
    values = {
        "n_qubits": meta.get("n_qubits", meta.get("node_count", 0)),
        "n_pauli_terms": meta.get("n_pauli_terms", 0),
        "l1_norm": meta.get("l1_norm", 0.0),
        "mean_abs_coeff": meta.get("mean_abs_coeff", 0.0),
        "max_abs_coeff": meta.get("max_abs_coeff", 0.0),
        "n_one_body_terms": meta.get("n_one_body_terms", 0.0),
        "n_two_body_terms": meta.get("n_two_body_terms", 0.0),
        "n_three_body_terms": meta.get("n_three_body_terms", 0.0),
        "n_four_body_terms": meta.get("n_four_body_terms", 0.0),
        "charge": meta.get("charge", 0.0),
        "multiplicity": meta.get("multiplicity", 1.0),
        "n_active_electrons": active.get("n_active_electrons", 0.0),
        "n_active_orbitals": active.get("n_active_orbitals", 0.0),
        "fragment_count": meta.get("fragment_count", 0.0),
        "is_fragment": 1.0 if meta.get("parent_name") else 0.0,
    }
    return np.asarray([_coerce_value(values[name]) for name in TARGET_NAMES], dtype=np.float32)


class ChemistryRegressor(nn.Module):
    def __init__(self, encoder: ChemistryEncoder, target_dim: int, head_hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.latent_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, target_dim),
        )

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(batch)
        pred = self.head(latent)
        return latent, pred

    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.encoder(batch)

    def to_prefix_token(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.encoder.to_prefix_token(batch)


class FlatChemistryRegressor(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, latent_dim: int = 128, head_hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, target_dim),
        )

    @property
    def latent_dim(self) -> int:
        return int(self.head[0].in_features)

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(batch["graph_attr"])
        pred = self.head(latent)
        return latent, pred

    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.encoder(batch["graph_attr"])

    def to_prefix_token(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.encode(batch)


def _make_loader(dataset: HamiltonianGraphDataset, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graph_samples)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    torch,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            _, pred_std = model(batch)
            target = torch.stack([torch.as_tensor(_target_vector(sample), device=device) for sample in batch["metadata"]], dim=0)
            target_stdized = (target - target_mean) / target_std
            loss = nn.functional.mse_loss(pred_std, target_stdized)
            pred = pred_std * target_std + target_mean
            mae = nn.functional.l1_loss(pred, target)
            batch_size = int(target.size(0))
            total_loss += float(loss.item()) * batch_size
            total_mae += float(mae.item()) * batch_size
            total_count += batch_size
    return {
        "mse": total_loss / max(total_count, 1),
        "mae": total_mae / max(total_count, 1),
    }


def _encode_all_samples(
    model: nn.Module,
    dataset: HamiltonianGraphDataset,
    device: torch.device,
    torch,
) -> list[dict[str, Any]]:
    model.eval()
    payload = []
    with torch.no_grad():
        for sample in dataset:
            batch = collate_graph_samples([sample])
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            latent = model.encode(batch)
            prefix = model.to_prefix_token(batch)
            payload.append(
                {
                    "name": sample.name,
                    "latent": latent.squeeze(0).detach().cpu().tolist(),
                    "prefix": prefix.squeeze(0).detach().cpu().tolist(),
                    "metadata": sample.metadata,
                }
            )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the chemistry-conditioned GQE encoder.")
    parser.add_argument("--json", type=Path, required=True, help="Hamiltonian JSON dataset path.")
    parser.add_argument("--out", type=Path, required=True, help="Output marker file; metrics/model go next to it.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("graph", "flat"),
        default="graph",
        help="Encoder ablation mode: graph uses message passing, flat uses graph-level summary features only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--conditioning-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-hidden", type=int, default=128)
    parser.add_argument("--include-fragments", action="store_true", help="Also include nested fragment records.")
    parser.add_argument("--fragment-only", action="store_true", help="Train only on nested fragment records.")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available.")
    args = parser.parse_args()

    if not args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch

    _seed_everything(args.seed, torch)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    if args.use_cuda and device.type != "cuda":
        print("CUDA requested but unavailable. Falling back to CPU.")

    dataset = HamiltonianGraphDataset(
        args.json,
        include_fragments=args.include_fragments,
        fragment_only=args.fragment_only,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No graph samples found in {args.json}")

    samples = [dataset[i] for i in range(len(dataset))]
    target_matrix = torch.tensor(np.stack([_target_vector(sample) for sample in samples], axis=0), dtype=torch.float32)

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if len(indices) == 1:
        train_indices = indices
        val_indices: list[int] = []
    else:
        train_count = max(1, min(len(indices) - 1, int(round(len(indices) * args.train_fraction))))
        train_indices = indices[:train_count]
        val_indices = indices[train_count:]
        if not val_indices:
            val_indices = train_indices[-1:]

    train_targets = target_matrix[train_indices]
    target_mean = train_targets.mean(dim=0)
    target_std = train_targets.std(dim=0, unbiased=False).clamp_min(1e-6)

    train_loader = _make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False) if val_indices else None

    sample_batch = collate_graph_samples([samples[0]])
    if args.mode == "graph":
        encoder = ChemistryEncoder(
            node_feature_dim=int(sample_batch["x"].shape[-1]),
            edge_feature_dim=int(sample_batch["edge_attr"].shape[-1]),
            graph_feature_dim=int(sample_batch["graph_attr"].shape[-1]),
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            conditioning_dim=args.conditioning_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        model = ChemistryRegressor(encoder, target_dim=len(TARGET_NAMES), head_hidden=args.head_hidden, dropout=args.dropout)
    else:
        model = FlatChemistryRegressor(
            input_dim=int(sample_batch["graph_attr"].shape[-1]),
            target_dim=len(TARGET_NAMES),
            latent_dim=args.latent_dim,
            head_hidden=args.head_hidden,
            dropout=args.dropout,
        )
    model.to(device)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loss_history: list[float] = []
    train_mae_history: list[float] = []
    val_loss_history: list[float] = []
    val_mae_history: list[float] = []

    epoch_iter = tqdm(range(args.epochs), desc="Chemistry encoder", unit="epoch", dynamic_ncols=True, disable=None)
    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        count = 0
        for batch in train_loader:
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            _, pred_std = model(batch)
            target = torch.stack([torch.as_tensor(_target_vector(sample), device=device) for sample in batch["metadata"]], dim=0)
            target_stdized = (target - target_mean) / target_std
            loss = nn.functional.mse_loss(pred_std, target_stdized)
            pred = pred_std * target_std + target_mean
            mae = nn.functional.l1_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = int(target.size(0))
            running_loss += float(loss.item()) * batch_size
            running_mae += float(mae.item()) * batch_size
            count += batch_size
        train_loss = running_loss / max(count, 1)
        train_mae = running_mae / max(count, 1)
        train_loss_history.append(train_loss)
        train_mae_history.append(train_mae)

        if val_loader is not None:
            metrics = _evaluate(model, val_loader, target_mean, target_std, device, torch)
            val_loss_history.append(metrics["mse"])
            val_mae_history.append(metrics["mae"])
            epoch_iter.set_postfix_str(f"train_mse={train_loss:.4f} val_mse={metrics['mse']:.4f}")
        else:
            epoch_iter.set_postfix_str(f"train_mse={train_loss:.4f}")

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.mode}_conditioning.pt"
    metrics_path = out_dir / f"{args.mode}_conditioning_metrics.json"
    embeddings_path = out_dir / f"{args.mode}_conditioning_embeddings.json"

    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "head_state_dict": model.head.state_dict(),
            "target_names": TARGET_NAMES,
            "config": {
                "mode": args.mode,
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "conditioning_dim": args.conditioning_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "head_hidden": args.head_hidden,
                "train_fraction": args.train_fraction,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
            "target_mean": target_mean.detach().cpu(),
            "target_std": target_std.detach().cpu(),
        },
        model_path,
    )

    embeddings = _encode_all_samples(model, dataset, device, torch)
    with embeddings_path.open("w", encoding="utf-8") as f:
        json.dump({"records": embeddings}, f, indent=2)

    metrics_payload = {
        "mode": f"chemistry_conditioned_{args.mode}_regression",
        "dataset": str(args.json),
        "num_samples": len(dataset),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "target_names": TARGET_NAMES,
        "train_loss_history": train_loss_history,
        "train_mae_history": train_mae_history,
        "val_loss_history": val_loss_history,
        "val_mae_history": val_mae_history,
        "final_train_mse": train_loss_history[-1] if train_loss_history else None,
        "final_train_mae": train_mae_history[-1] if train_mae_history else None,
        "final_val_mse": val_loss_history[-1] if val_loss_history else None,
        "final_val_mae": val_mae_history[-1] if val_mae_history else None,
        "model_path": str(model_path),
        "embeddings_path": str(embeddings_path),
        "target_mean": target_mean.detach().cpu().tolist(),
        "target_std": target_std.detach().cpu().tolist(),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"status": "done", "metrics": str(metrics_path), "model": str(model_path)}, f, indent=2)

    print(f"Wrote chemistry encoder model to: {model_path}")
    print(f"Wrote chemistry encoder metrics to: {metrics_path}")
    print(f"Wrote chemistry embeddings to: {embeddings_path}")


if __name__ == "__main__":
    main()
