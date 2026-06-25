from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

try:  # pragma: no cover - optional dependency
    from pyscf import gto
except Exception as exc:  # pragma: no cover - handled at runtime
    gto = None  # type: ignore[assignment]
    _PYSCF_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - import path only
    _PYSCF_IMPORT_ERROR = None

Geometry = Sequence[Sequence[Any]]
FragmentPlan = Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class ActiveSpaceSpec:
    mode: str = "full"
    n_active_electrons: int | None = None
    n_active_orbitals: int | None = None
    n_core_orbitals: int | None = None
    occupied_indices: tuple[int, ...] = ()
    active_indices: tuple[int, ...] = ()

    @property
    def has_selection(self) -> bool:
        return any(
            (
                self.n_active_electrons is not None,
                self.n_active_orbitals is not None,
                self.n_core_orbitals is not None,
                bool(self.occupied_indices),
                bool(self.active_indices),
            )
        )

    @property
    def is_full(self) -> bool:
        return self.mode == "full" and not self.has_selection

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.n_active_electrons is not None:
            kwargs["n_active_electrons"] = int(self.n_active_electrons)
        if self.n_active_orbitals is not None:
            kwargs["n_active_orbitals"] = int(self.n_active_orbitals)
        if self.occupied_indices:
            kwargs["occupied_indices"] = list(self.occupied_indices)
        if self.active_indices:
            kwargs["active_indices"] = list(self.active_indices)
        return kwargs

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_active_electrons": self.n_active_electrons,
            "n_active_orbitals": self.n_active_orbitals,
            "n_core_orbitals": self.n_core_orbitals,
            "occupied_indices": list(self.occupied_indices),
            "active_indices": list(self.active_indices),
        }


def _coerce_index_tuple(value: Sequence[int] | str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        tokens = [token for token in re.split(r"[\s,]+", value.strip()) if token]
        return tuple(int(token) for token in tokens)
    return tuple(int(item) for item in value)


def infer_electron_and_orbital_counts(
    geometry: Geometry,
    basis: str,
    charge: int,
    multiplicity: int,
) -> tuple[int, int]:
    if gto is None:
        raise RuntimeError(
            "PySCF is required to infer electron/orbital counts for active-space selection."
        ) from _PYSCF_IMPORT_ERROR

    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = int(charge)
    mol.spin = int(multiplicity) - 1
    mol.build(parse_arg=False)
    return int(mol.nelectron), int(mol.nao_nr())


def build_active_space_spec(
    *,
    mode: str | None = None,
    n_active_electrons: int | None = None,
    n_active_orbitals: int | None = None,
    n_core_orbitals: int | None = None,
    occupied_indices: Sequence[int] | str | None = None,
    active_indices: Sequence[int] | str | None = None,
    geometry: Geometry | None = None,
    basis: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
) -> ActiveSpaceSpec:
    normalized_mode = (mode or "").strip().lower().replace("-", "_")
    if not normalized_mode:
        normalized_mode = "full" if all(
            item is None
            for item in (
                n_active_electrons,
                n_active_orbitals,
                n_core_orbitals,
                occupied_indices,
                active_indices,
            )
        ) else "explicit"
    if normalized_mode in {"none", "full"}:
        normalized_mode = "full"
    elif normalized_mode in {"freeze_core", "frozen_core", "core_valence", "core"}:
        normalized_mode = "freeze_core"
    elif normalized_mode not in {"explicit", "full", "freeze_core"}:
        raise ValueError(f"Unsupported active-space mode: {mode!r}")

    occupied = _coerce_index_tuple(occupied_indices)
    active = _coerce_index_tuple(active_indices)

    if normalized_mode == "freeze_core" and (
        (n_active_electrons is None or n_active_orbitals is None)
        and n_core_orbitals is not None
        and geometry is not None
        and basis is not None
        and charge is not None
        and multiplicity is not None
    ):
        total_electrons, total_orbitals = infer_electron_and_orbital_counts(
            geometry=geometry,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
        )
        if n_active_electrons is None:
            n_active_electrons = max(0, total_electrons - 2 * int(n_core_orbitals))
        if n_active_orbitals is None:
            n_active_orbitals = max(0, total_orbitals - int(n_core_orbitals))

    return ActiveSpaceSpec(
        mode=normalized_mode,
        n_active_electrons=None if n_active_electrons is None else int(n_active_electrons),
        n_active_orbitals=None if n_active_orbitals is None else int(n_active_orbitals),
        n_core_orbitals=None if n_core_orbitals is None else int(n_core_orbitals),
        occupied_indices=occupied,
        active_indices=active,
    )


def _normalize_atom(atom: Sequence[Any]) -> list[Any]:
    if len(atom) != 2:
        raise ValueError(f"Invalid atom entry: {atom!r}")
    symbol = str(atom[0])
    coords = atom[1]
    if not isinstance(coords, Sequence):
        raise ValueError(f"Invalid coordinates for atom {atom!r}")
    return [symbol, [float(coord) for coord in coords]]


def fragment_geometry(geometry: Geometry, atom_indices: Sequence[int]) -> list[list[Any]]:
    atoms = list(geometry)
    if not atoms:
        raise ValueError("Cannot fragment an empty geometry")
    normalized: list[list[Any]] = []
    for raw_idx in atom_indices:
        idx = int(raw_idx)
        if idx < 0 or idx >= len(atoms):
            raise IndexError(
                f"Atom index {idx} out of range for geometry of length {len(atoms)}"
            )
        normalized.append(_normalize_atom(atoms[idx]))
    return normalized


def build_fragment_records(
    *,
    parent_name: str,
    geometry: Geometry,
    fragments: FragmentPlan,
    charge: int,
    multiplicity: int,
    basis: str,
    active_space: ActiveSpaceSpec | None = None,
) -> list[dict[str, Any]]:
    fragment_records: list[dict[str, Any]] = []
    for index, fragment in enumerate(fragments, start=1):
        atom_indices = fragment.get("atom_indices")
        fragment_geometry_value = fragment.get("geometry")
        if fragment_geometry_value is None and atom_indices is None:
            raise ValueError(
                "Each fragment must define either 'atom_indices' or an explicit 'geometry'."
            )
        if fragment_geometry_value is None:
            fragment_geometry_value = fragment_geometry(geometry, atom_indices)

        fragment_active_space = build_active_space_spec(
            **(fragment.get("active_space") or {}),
            geometry=fragment_geometry_value,
            basis=str(fragment.get("basis", basis)),
            charge=int(fragment.get("charge", charge)),
            multiplicity=int(fragment.get("multiplicity", multiplicity)),
        )
        fragment_name = str(fragment.get("name") or f"{parent_name}_fragment_{index}")
        fragment_records.append(
            {
                "name": fragment_name,
                "parent_name": parent_name,
                "geometry": fragment_geometry_value,
                "charge": int(fragment.get("charge", charge)),
                "multiplicity": int(fragment.get("multiplicity", multiplicity)),
                "basis": str(fragment.get("basis", basis)),
                "atom_indices": [] if atom_indices is None else [int(v) for v in atom_indices],
                "active_space": fragment_active_space.as_dict(),
                "parent_active_space": None if active_space is None else active_space.as_dict(),
            }
        )
    return fragment_records


def load_fragment_plan(value: str | Path) -> list[dict[str, Any]]:
    path = Path(value)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = yaml.safe_load(text)
    else:
        try:
            payload = json.loads(str(value))
        except json.JSONDecodeError:
            payload = yaml.safe_load(str(value))

    if isinstance(payload, dict) and "fragments" in payload:
        payload = payload["fragments"]
    if not isinstance(payload, list):
        raise ValueError("Fragment plan must be a JSON list or an object with a 'fragments' key.")
    return [dict(item) for item in payload]
