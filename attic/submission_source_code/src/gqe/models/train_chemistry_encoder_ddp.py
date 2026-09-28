"""Multi-GPU DistributedDataParallel training for the chemistry encoder.

Usage (single-node multi-GPU):
    torchrun --nproc_per_node=2 src/gqe/models/train_chemistry_encoder_ddp.py \
        --json results/data/hamiltonians.json \
        --out results/train/ddp_chemistry_encoder.done \
        --epochs 100 --batch-size 4 --hidden-dim 128 --latent-dim 128
"""
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
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=r"CUDA initialization: The NVIDIA driver on your system is too old.*",
    category=UserWarning,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from ..data.graph_dataset import HamiltonianGraphDataset, collate_graph_samples
    from .chemistry_encoder import ChemistryEncoder
    from .train_chemistry_encoder import (
        TARGET_NAMES,
        _coerce_value,
        ChemistryRegressor,
        FlatChemistryRegressor,
        _make_loader,
    )
except ImportError:
    from data.graph_dataset import HamiltonianGraphDataset, collate_graph_samples
    from models.chemistry_encoder import ChemistryEncoder
    from models.train_chemistry_encoder import (
        TARGET_NAMES,
        _coerce_value,
        ChemistryRegressor,
        FlatChemistryRegressor,
        _make_loader,
    )


def _target_vector(sample_or_meta: Any) -> np.ndarray:
    meta = sample_or_meta.metadata if hasattr(sample_or_meta, "metadata") else sample_or_meta
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


def _setup_distributed() -> tuple[int, int, torch.device]:
    import torch.distributed as dist

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    return rank, world_size, device


def _cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description="Multi-GPU DDP training for chemistry encoder.")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", type=str, choices=("graph", "flat"), default="graph")
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
    parser.add_argument("--include-fragments", action="store_true")
    parser.add_argument("--fragment-only", action="store_true")
    args = parser.parse_args()

    rank, world_size, device = _setup_distributed()
    is_main = rank == 0

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    dataset = HamiltonianGraphDataset(
        args.json,
        include_fragments=args.include_fragments,
        fragment_only=args.fragment_only,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No graph samples found in {args.json}")

    samples = [dataset[i] for i in range(len(dataset))]
    target_matrix = torch.tensor(np.stack([_target_vector(s) for s in samples], axis=0), dtype=torch.float32)

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if len(indices) == 1:
        train_indices = indices
        val_indices: list[int] = []
    else:
        train_count = max(1, min(len(indices) - 1, int(round(len(indices) * args.train_fraction))))
        train_indices = indices[:train_count]
        val_indices = indices[train_count:] or train_indices[-1:]

    train_targets = target_matrix[train_indices]
    target_mean = train_targets.mean(dim=0).to(device)
    target_std = train_targets.std(dim=0, unbiased=False).clamp_min(1e-6).to(device)

    # DDP-aware loaders
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices) if val_indices else None

    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 and val_subset else None

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_graph_samples,
        shuffle=(train_sampler is None),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate_graph_samples,
        shuffle=False,
    ) if val_subset else None

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

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[int(device.index)],
            output_device=int(device.index),
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_loss_history: list[float] = []
    train_mae_history: list[float] = []
    val_loss_history: list[float] = []
    val_mae_history: list[float] = []

    epoch_iter = tqdm(range(args.epochs), desc="DDP Chemistry encoder", unit="epoch", dynamic_ncols=True, disable=not is_main)
    for epoch in epoch_iter:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        running_mae = 0.0
        count = 0
        for batch in train_loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            _, pred_std = model(batch)
            target = torch.stack([torch.as_tensor(_target_vector(s), device=device) for s in batch["metadata"]], dim=0)
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

        # Gather metrics from all ranks
        if world_size > 1:
            metrics_tensor = torch.tensor([running_loss, running_mae, float(count)], device=device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            running_loss, running_mae, count = float(metrics_tensor[0]), float(metrics_tensor[1]), int(metrics_tensor[2])

        train_loss = running_loss / max(count, 1)
        train_mae = running_mae / max(count, 1)
        train_loss_history.append(train_loss)
        train_mae_history.append(train_mae)

        # Validation on main rank
        if val_loader is not None and is_main:
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                    _, pred_std = model(batch)
                    target = torch.stack([torch.as_tensor(_target_vector(s), device=device) for s in batch["metadata"]], dim=0)
                    target_stdized = (target - target_mean) / target_std
                    loss = nn.functional.mse_loss(pred_std, target_stdized)
                    pred = pred_std * target_std + target_mean
                    mae = nn.functional.l1_loss(pred, target)
                    batch_size = int(target.size(0))
                    val_loss += float(loss.item()) * batch_size
                    val_mae += float(mae.item()) * batch_size
                    val_count += batch_size
            val_loss_history.append(val_loss / max(val_count, 1))
            val_mae_history.append(val_mae / max(val_count, 1))
            epoch_iter.set_postfix_str(f"train_mse={train_loss:.4f} val_mse={val_loss_history[-1]:.4f}")
        else:
            epoch_iter.set_postfix_str(f"train_mse={train_loss:.4f}")

    if is_main:
        out_dir = args.out.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / f"ddp_{args.mode}_conditioning.pt"
        metrics_path = out_dir / f"ddp_{args.mode}_conditioning_metrics.json"

        torch.save(
            {
                "encoder_state_dict": model.module.encoder.state_dict() if isinstance(model, DistributedDataParallel) else model.encoder.state_dict(),
                "head_state_dict": model.module.head.state_dict() if isinstance(model, DistributedDataParallel) else model.head.state_dict(),
                "target_names": TARGET_NAMES,
                "config": {"mode": args.mode, "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim, "conditioning_dim": args.conditioning_dim, "num_layers": args.num_layers, "dropout": args.dropout, "head_hidden": args.head_hidden, "train_fraction": args.train_fraction, "learning_rate": args.learning_rate, "seed": args.seed},
                "target_mean": target_mean.cpu(),
                "target_std": target_std.cpu(),
            },
            model_path,
        )

        metrics_payload = {
            "mode": f"ddp_chemistry_conditioned_{args.mode}_regression",
            "dataset": str(args.json),
            "num_samples": len(dataset),
            "world_size": world_size,
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
            "target_mean": target_mean.cpu().tolist(),
            "target_std": target_std.cpu().tolist(),
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        with args.out.open("w", encoding="utf-8") as f:
            json.dump({"status": "done", "metrics": str(metrics_path), "model": str(model_path)}, f, indent=2)

        print(f"Wrote DDP model to: {model_path}")
        print(f"Wrote DDP metrics to: {metrics_path}")

    _cleanup_distributed()


if __name__ == "__main__":
    main()
