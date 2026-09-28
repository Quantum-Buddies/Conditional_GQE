from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # pragma: no cover - package import path
    from ..common.hamiltonian_utils import iter_terms, load_hamiltonian_records
except ImportError:  # pragma: no cover - direct script execution fallback
    try:
        from gqe.common.hamiltonian_utils import iter_terms, load_hamiltonian_records
    except ImportError:  # pragma: no cover - direct module execution fallback
        from common.hamiltonian_utils import iter_terms, load_hamiltonian_records  # type: ignore[no-redef]

PERIODIC_TABLE = {
    "H": 1,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}

HALOGENS = {"F", "Cl", "Br", "I"}


@dataclass(frozen=True)
class GraphSample:
    name: str
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    graph_attr: torch.Tensor
    metadata: dict[str, Any]

    def to(self, device: torch.device | str) -> "GraphSample":
        return GraphSample(
            name=self.name,
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            graph_attr=self.graph_attr.to(device),
            metadata=self.metadata,
        )


def _atom_symbol(atom: Sequence[Any]) -> str:
    return str(atom[0]).capitalize()


def _atom_coords(atom: Sequence[Any]) -> np.ndarray:
    coords = np.asarray(atom[1], dtype=np.float32)
    if coords.shape != (3,):
        raise ValueError(f"Atom coordinates must be 3D, got shape {coords.shape!r}.")
    return coords


def _atomic_number(symbol: str) -> int:
    return PERIODIC_TABLE.get(symbol.capitalize(), 0)


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _rbf(distance: float, centers: np.ndarray, width: float) -> np.ndarray:
    return np.exp(-((distance - centers) ** 2) / max(width, 1e-6) ** 2)


def _term_stats(record: dict[str, Any]) -> dict[str, float]:
    abs_coeffs = []
    body_order_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    identity = 0.0
    for ops, coeff in iter_terms(record):
        coeff_real = float(np.real(coeff))
        abs_coeff = abs(coeff_real)
        abs_coeffs.append(abs_coeff)
        body_order = sum(1 for op in ops if op != "I")
        if body_order == 0:
            identity += coeff_real
        if body_order in body_order_counts:
            body_order_counts[body_order] += 1
    coeff_arr = np.asarray(abs_coeffs, dtype=np.float32)
    if coeff_arr.size == 0:
        coeff_arr = np.zeros(1, dtype=np.float32)
    return {
        "l1_norm": float(coeff_arr.sum()),
        "mean_abs_coeff": float(coeff_arr.mean()),
        "std_abs_coeff": float(coeff_arr.std()),
        "max_abs_coeff": float(coeff_arr.max()),
        "identity_coeff": float(identity),
        "n_one_body_terms": float(body_order_counts[1]),
        "n_two_body_terms": float(body_order_counts[2]),
        "n_three_body_terms": float(body_order_counts[3]),
        "n_four_body_terms": float(body_order_counts[4]),
    }


def _atom_features(record: dict[str, Any], coords: np.ndarray) -> torch.Tensor:
    atoms = list(record.get("geometry", []))
    n_atoms = max(len(atoms), 1)
    center = coords.mean(axis=0, keepdims=True)
    shifted = coords - center
    scale = float(np.max(np.linalg.norm(shifted, axis=1))) if n_atoms > 1 else 1.0
    scale = max(scale, 1e-6)
    normalized = shifted / scale
    distances = _pairwise_distances(coords)
    degrees = (distances < 1.8).sum(axis=1) - 1
    total_z = sum(_atomic_number(_atom_symbol(atom)) for atom in atoms) or 1
    charge = float(record.get("charge", 0))
    multiplicity = float(record.get("multiplicity", 1))
    active = record.get("active_space") or {}
    n_qubits = float(record.get("n_qubits", max(1, len(atoms))))
    n_pauli_terms = float(record.get("n_pauli_terms", 0))
    active_electrons = float(active.get("n_active_electrons") or 0)
    active_orbitals = float(active.get("n_active_orbitals") or 0)
    fragment_flag = 1.0 if record.get("parent_name") else 0.0
    fragment_charge = float(record.get("charge", 0))
    rows: list[list[float]] = []
    for idx, atom in enumerate(atoms):
        symbol = _atom_symbol(atom)
        atomic_number = float(_atomic_number(symbol))
        xyz = normalized[idx].tolist()
        r = float(np.linalg.norm(normalized[idx]))
        rows.append(
            [
                atomic_number / 100.0,
                atomic_number / float(total_z),
                xyz[0],
                xyz[1],
                xyz[2],
                r,
                float(degrees[idx] / max(n_atoms - 1, 1)),
                1.0 if symbol == "H" else 0.0,
                1.0 if symbol in HALOGENS else 0.0,
                1.0 if atomic_number >= 10 else 0.0,
                fragment_flag,
                fragment_charge / 10.0,
                charge / 10.0,
                multiplicity / 10.0,
                active_electrons / max(n_qubits, 1.0),
                active_orbitals / max(n_qubits, 1.0),
                n_qubits / 32.0,
                n_pauli_terms / 256.0,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def _graph_edges(coords: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    n_atoms = int(coords.shape[0])
    distances = _pairwise_distances(coords)
    if n_atoms <= 1:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 8), dtype=torch.float32)

    centers = np.linspace(0.5, 6.0, 4, dtype=np.float32)
    edges: list[list[int]] = []
    attrs: list[list[float]] = []
    scale = max(float(distances[distances > 0].mean()) if np.any(distances > 0) else 1.0, 1e-6)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            delta = coords[j] - coords[i]
            dist = float(np.linalg.norm(delta))
            rbf = _rbf(dist, centers, width=scale)
            edges.append([i, j])
            attrs.append(
                [
                    dist,
                    1.0 / max(dist, 1e-6),
                    delta[0] / scale,
                    delta[1] / scale,
                    delta[2] / scale,
                    float(rbf[0]),
                    float(rbf[1]),
                    float(rbf[2]),
                ]
            )
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), torch.tensor(attrs, dtype=torch.float32)


def record_to_graph_sample(record: dict[str, Any]) -> GraphSample:
    geometry = record.get("geometry")
    if not geometry:
        raise ValueError(f"Record {record.get('name', '<unnamed>')!r} does not contain geometry.")

    coords = np.stack([_atom_coords(atom) for atom in geometry], axis=0).astype(np.float32)
    x = _atom_features(record, coords)
    edge_index, edge_attr = _graph_edges(coords)
    stats = _term_stats(record)
    graph_attr = torch.tensor(
        [
            float(len(geometry) / 32.0),
            float(record.get("n_qubits", len(geometry))) / 32.0,
            float(record.get("n_pauli_terms", 0)) / 256.0,
            stats["l1_norm"] / 100.0,
            stats["mean_abs_coeff"],
            stats["std_abs_coeff"],
            stats["max_abs_coeff"],
            stats["identity_coeff"] / 100.0,
            stats["n_one_body_terms"] / 256.0,
            stats["n_two_body_terms"] / 256.0,
            stats["n_three_body_terms"] / 256.0,
            stats["n_four_body_terms"] / 256.0,
        ],
        dtype=torch.float32,
    )
    metadata = {
        "name": record.get("name", "unknown"),
        "split": record.get("split", "unspecified"),
        "charge": record.get("charge", 0),
        "multiplicity": record.get("multiplicity", 1),
        "active_space": record.get("active_space", {}),
        "fragment_count": record.get("fragment_count"),
        "parent_name": record.get("parent_name"),
        "parent_active_space": record.get("parent_active_space"),
        "n_qubits": record.get("n_qubits"),
        "n_pauli_terms": record.get("n_pauli_terms"),
        **stats,
    }
    return GraphSample(
        name=str(record.get("name", "unknown")),
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        graph_attr=graph_attr,
        metadata=metadata,
    )


class HamiltonianGraphDataset(Dataset):
    def __init__(
        self,
        json_path: str | Path | None = None,
        *,
        records: Sequence[dict[str, Any]] | None = None,
        include_fragments: bool = True,
        fragment_only: bool = False,
    ) -> None:
        if records is None:
            if json_path is None:
                raise ValueError("Provide either `json_path` or `records`.")
            payload_records = load_hamiltonian_records(Path(json_path))
        else:
            payload_records = [dict(rec) for rec in records]
        items: list[dict[str, Any]] = []
        for record in payload_records:
            if not fragment_only:
                items.append(record)
            if include_fragments:
                items.extend(dict(frag) for frag in record.get("fragments", []))
        self._records = items
        self._samples = [record_to_graph_sample(rec) for rec in self._records]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> GraphSample:
        return self._samples[index]


def collate_graph_samples(samples: Sequence[GraphSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    x_list = []
    edge_index_list = []
    edge_attr_list = []
    graph_attr_list = []
    batch_list = []
    metadata = []
    node_offset = 0
    for batch_idx, sample in enumerate(samples):
        x_list.append(sample.x)
        edge_index_list.append(sample.edge_index + node_offset)
        edge_attr_list.append(sample.edge_attr)
        graph_attr_list.append(sample.graph_attr)
        batch_list.append(torch.full((sample.x.shape[0],), batch_idx, dtype=torch.long))
        metadata.append(sample.metadata)
        node_offset += int(sample.x.shape[0])
    return {
        "x": torch.cat(x_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "graph_attr": torch.stack(graph_attr_list, dim=0),
        "batch": torch.cat(batch_list, dim=0),
        "names": [sample.name for sample in samples],
        "metadata": metadata,
    }


def load_graph_samples(json_path: str | Path, *, include_fragments: bool = True, fragment_only: bool = False) -> list[GraphSample]:
    return list(
        HamiltonianGraphDataset(
            json_path,
            include_fragments=include_fragments,
            fragment_only=fragment_only,
        )
    )
