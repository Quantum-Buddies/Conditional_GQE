import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import yaml
from openfermion import count_qubits, get_fermion_operator
from openfermion.ops import QubitOperator
from openfermion.transforms import jordan_wigner
from openfermionpyscf import generate_molecular_hamiltonian
from tqdm.auto import tqdm

try:  # pragma: no cover - import path depends on execution mode
    from gqe.data.fragmentation import (
        ActiveSpaceSpec,
        build_active_space_spec,
        build_fragment_records,
        load_fragment_plan,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from fragmentation import (  # type: ignore[no-redef]
        ActiveSpaceSpec,
        build_active_space_spec,
        build_fragment_records,
        load_fragment_plan,
    )


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file {config_path} did not parse to a mapping.")
    return cfg


def _select_molecules(cfg: dict[str, Any], selected_name: str | None) -> list[dict[str, Any]]:
    dataset = cfg.get("dataset", {})
    if isinstance(dataset, list):
        molecules = [dict(m) for m in dataset]
    else:
        molecules = [dict(m) for m in dataset.get("molecules", [])]
    if selected_name is None:
        return molecules

    for molecule in molecules:
        if molecule.get("name") == selected_name:
            return [molecule]

    available = ", ".join(str(m.get("name", "<unnamed>")) for m in molecules)
    raise ValueError(f"Could not find molecule {selected_name!r}. Available molecules: {available}")


def _normalize_fragment_plan(plan_value: Any) -> list[dict[str, Any]] | None:
    if plan_value is None:
        return None
    if isinstance(plan_value, list):
        return [dict(item) for item in plan_value]
    if isinstance(plan_value, dict) and "fragments" in plan_value:
        fragments = plan_value["fragments"]
        if not isinstance(fragments, list):
            raise ValueError("Fragment plan 'fragments' key must contain a list.")
        return [dict(item) for item in fragments]
    return load_fragment_plan(plan_value)


def _prepare_generator_kwargs(
    *,
    geometry: list[Any],
    basis: str,
    multiplicity: int,
    charge: int,
    active_space: ActiveSpaceSpec,
) -> tuple[dict[str, Any], bool]:
    raw_kwargs: dict[str, Any] = {
        "geometry": geometry,
        "basis": basis,
        "multiplicity": multiplicity,
        "charge": charge,
    }
    raw_kwargs.update(active_space.as_kwargs())
    if active_space.n_core_orbitals is not None:
        raw_kwargs["n_core_orbitals"] = int(active_space.n_core_orbitals)

    signature = inspect.signature(generate_molecular_hamiltonian)
    call_kwargs: dict[str, Any] = {}
    used_active_space = False

    alias_map = {
        "occupied_indices": ("occupied_indices", "docc_mo_indices"),
        "active_indices": ("active_indices", "active_mo_indices", "active_orbital_indices"),
        "n_active_electrons": ("n_active_electrons",),
        "n_active_orbitals": ("n_active_orbitals",),
        "n_core_orbitals": ("n_core_orbitals", "n_core"),
    }

    for key in ("geometry", "basis", "multiplicity", "charge"):
        if key in signature.parameters:
            call_kwargs[key] = raw_kwargs[key]

    for canonical_key, aliases in alias_map.items():
        value = raw_kwargs.get(canonical_key)
        if value is None:
            continue
        for alias in aliases:
            if alias in signature.parameters:
                call_kwargs[alias] = value
                used_active_space = True
                break

    if active_space.has_selection and not used_active_space:
        raise RuntimeError(
            "Requested an active-space reduction, but the installed "
            "openfermionpyscf.generate_molecular_hamiltonian signature does not "
            "expose compatible active-space arguments. Please upgrade the chemistry stack."
        )

    return call_kwargs, used_active_space


def _generate_record(
    *,
    molecule: dict[str, Any],
    dataset_defaults: dict[str, Any],
    fragment_plan: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    name = str(molecule.get("name", "molecule"))
    basis = str(molecule.get("basis", dataset_defaults.get("basis", "sto-3g")))
    charge = int(molecule.get("charge", 0))
    multiplicity = int(molecule.get("multiplicity", 1))
    geometry = molecule.get("geometry")
    if geometry is None:
        raise ValueError(f"Molecule {name!r} is missing geometry information.")

    active_space_cfg = dict(dataset_defaults.get("active_space", {}))
    active_space_cfg.update(molecule.get("active_space", {}))
    active_space = build_active_space_spec(
        **active_space_cfg,
        geometry=geometry,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
    )

    call_kwargs, _ = _prepare_generator_kwargs(
        geometry=geometry,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        active_space=active_space,
    )
    mol_ham = generate_molecular_hamiltonian(**call_kwargs)
    fermion_ham = get_fermion_operator(mol_ham)
    qubit_ham = jordan_wigner(fermion_ham)
    terms = _to_serializable_terms(qubit_ham)

    fragments: list[dict[str, Any]] = []
    fragment_entries = fragment_plan or molecule.get("fragments") or dataset_defaults.get("fragments") or []
    if fragment_entries:
        fragments = build_fragment_records(
            parent_name=name,
            geometry=geometry,
            fragments=fragment_entries,
            charge=charge,
            multiplicity=multiplicity,
            basis=basis,
            active_space=active_space,
        )

    record: dict[str, Any] = {
        "name": name,
        "split": molecule.get("split", "unspecified"),
        "geometry": geometry,
        "basis": basis,
        "charge": charge,
        "multiplicity": multiplicity,
        "active_space": active_space.as_dict(),
        "n_qubits": int(count_qubits(fermion_ham)),
        "n_pauli_terms": len(terms),
        "terms": terms,
    }
    if fragments:
        record["fragment_count"] = len(fragments)
        record["fragments"] = fragments
    return record


def _to_serializable_terms(qubit_ham: QubitOperator) -> list[dict]:
    terms = []
    for pauli_term, coeff in qubit_ham.terms.items():
        label = " ".join([f"{p}{i}" for i, p in pauli_term]) if pauli_term else "I"
        terms.append({"term": label, "real": float(coeff.real), "imag": float(coeff.imag)})
    return terms


def generate_from_config(
    config_path: Path,
    output_dir: Path,
    *,
    selected_name: str | None = None,
    fragment_plan_value: Any = None,
    active_space_overrides: dict[str, Any] | None = None,
) -> Path:
    cfg = _load_config(config_path)
    dataset = cfg.get("dataset", {})
    if isinstance(dataset, list):
        dataset_meta: dict[str, Any] = {}
    else:
        dataset_meta = dataset
    fragment_plan_source = fragment_plan_value if fragment_plan_value is not None else dataset_meta.get("fragment_plan")
    fragment_plan = _normalize_fragment_plan(fragment_plan_source)
    molecules = _select_molecules(cfg, selected_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for m in tqdm(
        molecules,
        desc="Generating Hamiltonians",
        unit="system",
        dynamic_ncols=True,
        disable=None,
    ):
        if active_space_overrides:
            m = dict(m)
            m["active_space"] = {**m.get("active_space", {}), **active_space_overrides}
        record = _generate_record(
            molecule=m,
            dataset_defaults=dataset_meta,
            fragment_plan=fragment_plan,
        )
        records.append(record)

    out_file = output_dir / "hamiltonians.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({"records": records}, f, indent=2)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate molecular Hamiltonian dataset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--molecule", type=str, default=None, help="Optional molecule name to generate from the config.")
    parser.add_argument(
        "--fragment-plan",
        type=Path,
        default=None,
        help="Optional JSON fragment plan to attach to each generated molecule.",
    )
    parser.add_argument("--active-space-mode", type=str, default=None)
    parser.add_argument("--n-active-electrons", type=int, default=None)
    parser.add_argument("--n-active-orbitals", type=int, default=None)
    parser.add_argument("--n-core-orbitals", type=int, default=None)
    parser.add_argument("--occupied-indices", type=str, default=None)
    parser.add_argument("--active-indices", type=str, default=None)
    args = parser.parse_args()
    active_space_overrides = {
        key: value
        for key, value in {
            "mode": args.active_space_mode,
            "n_active_electrons": args.n_active_electrons,
            "n_active_orbitals": args.n_active_orbitals,
            "n_core_orbitals": args.n_core_orbitals,
            "occupied_indices": args.occupied_indices,
            "active_indices": args.active_indices,
        }.items()
        if value is not None
    }
    out = generate_from_config(
        args.config,
        args.out,
        selected_name=args.molecule,
        fragment_plan_value=args.fragment_plan,
        active_space_overrides=active_space_overrides or None,
    )
    print(f"Wrote Hamiltonian dataset to: {out}")


if __name__ == "__main__":
    main()

