from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

try:  # pragma: no cover - package import path
    from ..data.graph_dataset import GraphSample, collate_graph_samples
except ImportError:  # pragma: no cover - direct script execution fallback
    try:
        from gqe.data.graph_dataset import GraphSample, collate_graph_samples
    except ImportError:  # pragma: no cover - direct module execution fallback
        from data.graph_dataset import GraphSample, collate_graph_samples  # type: ignore[no-redef]


@dataclass(frozen=True)
class ChemistryEncoderConfig:
    node_feature_dim: int
    edge_feature_dim: int
    graph_feature_dim: int = 0
    hidden_dim: int = 128
    latent_dim: int = 128
    conditioning_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1


class EdgeAwareMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_state: torch.Tensor, edge_index: torch.Tensor, edge_state: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.norm(node_state)

        src, dst = edge_index
        messages = self.message(torch.cat([node_state[src], node_state[dst], edge_state], dim=-1))
        agg = torch.zeros_like(node_state)
        agg.index_add_(0, dst, messages)

        degree = torch.zeros((node_state.size(0), 1), dtype=node_state.dtype, device=node_state.device)
        degree.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=node_state.dtype, device=node_state.device))
        agg = agg / degree.clamp_min(1.0)

        delta = self.update(torch.cat([node_state, agg], dim=-1))
        return self.norm(node_state + self.dropout(delta))


class ChemistryEncoder(nn.Module):
    """Encode a chemistry or fragment graph into a conditioning prior for GQE.

    The encoder is intentionally self-contained and does not require
    torch_geometric. It can operate on a single ``GraphSample``, a list of
    ``GraphSample`` objects, or the batched dictionary returned by
    ``collate_graph_samples``.
    """

    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_feature_dim: int,
        graph_feature_dim: int = 0,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        conditioning_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = ChemistryEncoderConfig(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            graph_feature_dim=graph_feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            conditioning_dim=conditioning_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.node_in = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_in = nn.Linear(edge_feature_dim, hidden_dim)
        self.graph_in = nn.Linear(graph_feature_dim, hidden_dim) if graph_feature_dim > 0 else None
        self.blocks = nn.ModuleList(
            EdgeAwareMessageBlock(hidden_dim=hidden_dim, edge_dim=hidden_dim, dropout=dropout)
            for _ in range(max(1, num_layers))
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.prefix_proj = nn.Sequential(
            nn.Linear(latent_dim, conditioning_dim),
            nn.GELU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        self.prefix_norm = nn.LayerNorm(conditioning_dim)

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    @property
    def conditioning_dim(self) -> int:
        return self.config.conditioning_dim

    def _coerce_batch(self, graphs: GraphSample | Sequence[GraphSample] | Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if isinstance(graphs, GraphSample):
            return {
                "x": graphs.x,
                "edge_index": graphs.edge_index,
                "edge_attr": graphs.edge_attr,
                "graph_attr": graphs.graph_attr.unsqueeze(0),
                "batch": torch.zeros((graphs.x.size(0),), dtype=torch.long, device=graphs.x.device),
            }
        if isinstance(graphs, Sequence) and graphs and isinstance(graphs[0], GraphSample):
            return collate_graph_samples(list(graphs))
        if isinstance(graphs, Mapping):
            x = graphs["x"]
            edge_index = graphs["edge_index"]
            edge_attr = graphs["edge_attr"]
            graph_attr = graphs.get("graph_attr")
            batch = graphs.get("batch")
            if batch is None:
                batch = torch.zeros((x.size(0),), dtype=torch.long, device=x.device)
            if graph_attr is None:
                graph_attr = torch.zeros((int(batch.max().item()) + 1 if batch.numel() else 1, 0), dtype=x.dtype, device=x.device)
            return {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "graph_attr": graph_attr,
                "batch": batch,
            }
        raise TypeError(
            "ChemistryEncoder expects a GraphSample, a list of GraphSample objects, or a batch dictionary."
        )

    def _pool_nodes(self, node_state: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        sums = torch.zeros((n_graphs, node_state.size(-1)), dtype=node_state.dtype, device=node_state.device)
        sums.index_add_(0, batch, node_state)
        counts = torch.zeros((n_graphs, 1), dtype=node_state.dtype, device=node_state.device)
        counts.index_add_(0, batch, torch.ones((batch.numel(), 1), dtype=node_state.dtype, device=node_state.device))
        mean = sums / counts.clamp_min(1.0)

        sq_sums = torch.zeros_like(sums)
        sq_sums.index_add_(0, batch, node_state.square())
        rms = torch.sqrt(sq_sums / counts.clamp_min(1.0))
        return mean, rms

    def _pool_edges(self, edge_state: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if edge_state.numel() == 0 or edge_index.numel() == 0:
            n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
            return torch.zeros((n_graphs, edge_state.size(-1) if edge_state.ndim else self.config.hidden_dim), dtype=edge_state.dtype, device=edge_state.device)

        edge_batch = batch[edge_index[0]]
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        sums = torch.zeros((n_graphs, edge_state.size(-1)), dtype=edge_state.dtype, device=edge_state.device)
        sums.index_add_(0, edge_batch, edge_state)
        counts = torch.zeros((n_graphs, 1), dtype=edge_state.dtype, device=edge_state.device)
        counts.index_add_(0, edge_batch, torch.ones((edge_batch.numel(), 1), dtype=edge_state.dtype, device=edge_state.device))
        return sums / counts.clamp_min(1.0)

    def forward(self, graphs: GraphSample | Sequence[GraphSample] | Mapping[str, Any]) -> torch.Tensor:
        batch = self._coerce_batch(graphs)
        x = batch["x"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        graph_attr = batch["graph_attr"]
        graph_batch = batch["batch"]

        node_state = self.node_in(x)
        edge_state = self.edge_in(edge_attr) if edge_attr.numel() else torch.zeros((0, node_state.size(-1)), dtype=node_state.dtype, device=node_state.device)

        if self.graph_in is not None and graph_attr.numel() > 0:
            graph_context = self.graph_in(graph_attr)
            node_state = node_state + graph_context[graph_batch]
        elif graph_attr.numel() == 0:
            graph_context = torch.zeros((int(graph_batch.max().item()) + 1 if graph_batch.numel() else 1, node_state.size(-1)), dtype=node_state.dtype, device=node_state.device)
        else:
            graph_context = torch.zeros((graph_attr.size(0), node_state.size(-1)), dtype=node_state.dtype, device=node_state.device)

        for block in self.blocks:
            node_state = block(node_state, edge_index, edge_state)

        mean_pool, rms_pool = self._pool_nodes(node_state, graph_batch)
        edge_pool = self._pool_edges(edge_state, edge_index, graph_batch)

        if graph_attr.numel() == 0:
            graph_context = torch.zeros((mean_pool.size(0), node_state.size(-1)), dtype=node_state.dtype, device=node_state.device)
        elif self.graph_in is None:
            graph_context = torch.zeros((graph_attr.size(0), node_state.size(-1)), dtype=node_state.dtype, device=node_state.device)
        elif graph_context.shape[-1] != node_state.size(-1):
            graph_context = self.graph_in(graph_attr)

        latent_input = torch.cat([mean_pool, rms_pool, edge_pool, graph_context], dim=-1)
        latent = self.latent_norm(self.readout(latent_input))
        return latent

    def encode(self, graphs: GraphSample | Sequence[GraphSample] | Mapping[str, Any]) -> torch.Tensor:
        return self.forward(graphs)

    def to_prefix_token(
        self,
        graphs: GraphSample | Sequence[GraphSample] | Mapping[str, Any],
        *,
        add_sequence_dim: bool = False,
    ) -> torch.Tensor:
        latent = self.encode(graphs)
        prefix = self.prefix_norm(self.prefix_proj(latent))
        return prefix.unsqueeze(1) if add_sequence_dim else prefix

    def to_gru_state(
        self,
        graphs: GraphSample | Sequence[GraphSample] | Mapping[str, Any],
        *,
        num_layers: int = 1,
    ) -> torch.Tensor:
        prefix = self.to_prefix_token(graphs, add_sequence_dim=False)
        return prefix.unsqueeze(0).repeat(max(1, num_layers), 1, 1)

    def describe(self, graphs: GraphSample | Sequence[GraphSample] | Mapping[str, Any]) -> dict[str, Any]:
        batch = self._coerce_batch(graphs)
        x = batch["x"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        graph_attr = batch["graph_attr"]
        graph_batch = batch["batch"]
        latent = self.encode(graphs)
        prefix = self.to_prefix_token(graphs)
        return {
            "latent_shape": tuple(latent.shape),
            "prefix_shape": tuple(prefix.shape),
            "node_count": int(x.size(0)),
            "edge_count": int(edge_index.size(1)),
            "node_feature_dim": int(x.size(-1)),
            "edge_feature_dim": int(edge_attr.size(-1)) if edge_attr.ndim > 1 else 0,
            "graph_feature_dim": int(graph_attr.size(-1)) if graph_attr.ndim > 1 else 0,
            "batch_size": int(graph_attr.size(0)) if graph_attr.ndim > 1 else 1,
            "num_graphs": int(graph_batch.max().item()) + 1 if graph_batch.numel() else 1,
        }


__all__ = ["ChemistryEncoder", "ChemistryEncoderConfig", "EdgeAwareMessageBlock"]
