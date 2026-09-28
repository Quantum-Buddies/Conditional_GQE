#!/usr/bin/env python3
"""Optional labelled-ours E0 column: AVAS → CASCI → Jordan–Wigner export.

This is **not** the DFT rank. ΔBDE / IP stay the acquisition function.
GQE does **not** propose molecules. Do not copy XYZ into tin_resist.yaml.

Recipe (Sayfutyarova AVAS, arXiv:1701.07862; MatGen-Q arXiv:2607.23988
describes the same shells on **their** unpublished methyltin XYZ):
  RHF (or UHF for charge +1) at **our** B3LYP geometry
  AVAS on Sn 5s/5p + the alpha (first) carbon 2p
  raise the projector threshold until 2 * ncas ≤ 24 (L40S statevector cap)
  CASCI in that space, then openfermion JW with the MatGen-Q integral
  transpose (0, 2, 3, 1)

The 14-character H-cGQE policy is **not** applied unless n_qubits == 14.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.mcscf import avas
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gqe.eval.tin_inform_dft import (  # noqa: E402
    BASIS,
    ECP,
    _parse_mem_mb,
    make_mol,
)

DEFAULT_OUT = Path("results/tin_ab/methyltin_ours")
DEFAULT_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80)
MAX_QUBITS_L40S = 24
MAX_FCI_DETS = 2_000_000
PYSCF_VERBOSE_DEFAULT = 3
FAIR_PAIR_IDS = {"me_cf3": ("me", "cf3")}
MATGENQ_DEMO = Path("/scratch/kcwp264/baselines/gqe-qsci-euv-photoresists/code")
POOL_NATIVE_ANGLES = (0.1, -0.1)
E0_KEYS = (
    "e0_casci_ha",
    "e0_qsci_ha",
    "delta_e_qsci_mha",
    "e0_n_qubits",
    "e0_ncas",
    "e0_nelecas",
)
FORBIDDEN = (
    "Do not write 72.6→21.2 kcal/mol as a GQE number; that is classical "
    "UCCSD(T) in arXiv:2607.23988, not this labelled-ours E0."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_xyz(path: Path | str) -> list[list[Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Not an XYZ file: {path}")
    n_atom = int(lines[0].strip())
    atoms: list[list[Any]] = []
    for line in lines[2 : 2 + n_atom]:
        parts = line.split()
        if len(parts) < 4:
            continue
        atoms.append(
            [str(parts[0]), [float(parts[1]), float(parts[2]), float(parts[3])]]
        )
    if len(atoms) != n_atom:
        raise ValueError(f"{path}: expected {n_atom} atoms, parsed {len(atoms)}")
    return atoms


def first_indices(atoms: Sequence[Sequence[Any]]) -> tuple[int, int]:
    sn_idx = next(i for i, row in enumerate(atoms) if str(row[0]) == "Sn")
    c_idx = next(i for i, row in enumerate(atoms) if str(row[0]) == "C")
    return sn_idx, c_idx


def fci_dimension(ncas: int, nelecas: Any) -> int:
    if isinstance(nelecas, (tuple, list, np.ndarray)):
        n_alpha, n_beta = int(nelecas[0]), int(nelecas[1])
    else:
        n_el = int(nelecas)
        n_alpha = n_el // 2 + n_el % 2
        n_beta = n_el // 2
    if min(ncas, n_alpha, n_beta) < 0 or n_alpha > ncas or n_beta > ncas:
        return 10**18
    return int(math.comb(ncas, n_alpha) * math.comb(ncas, n_beta))


def nelecas_total(nelecas: Any) -> int:
    if isinstance(nelecas, (tuple, list, np.ndarray)):
        return int(nelecas[0]) + int(nelecas[1])
    return int(nelecas)


def nelecas_json(nelecas: Any) -> int | list[int]:
    if isinstance(nelecas, (tuple, list, np.ndarray)):
        return [int(nelecas[0]), int(nelecas[1])]
    return int(nelecas)


def xyz_path_for(lig_dir: Path, score: Mapping[str, Any], charge: int) -> Path:
    paths = score.get("paths") if isinstance(score.get("paths"), Mapping) else {}
    key = "neutral_xyz" if int(charge) == 0 else "cation_xyz"
    raw = paths.get(key)
    if raw:
        candidate = Path(str(raw))
        if candidate.is_file():
            return candidate
    hits = sorted(lig_dir.glob("*_neutral.xyz" if int(charge) == 0 else "*_cation.xyz"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"No labelled-ours XYZ for charge={charge} under {lig_dir}. "
        "Score the ligand with tin_inform_dft.py first. Do not use MatGen-Q Table 1."
    )


def run_hf(mol: Any) -> Any:
    mf = scf.UHF(mol) if mol.spin else scf.RHF(mol)
    mf.conv_tol = 1e-8
    mf.max_cycle = 160
    mf.kernel()
    if not mf.converged:
        mf.level_shift = 0.4
        mf.kernel()
    if not mf.converged:
        raise RuntimeError(
            f"HF failed for charge={mol.charge} spin={mol.spin} "
            f"({mol.natm} atoms, {mol.nao} AOs)"
        )
    return mf


def ao_label_sets(sn_idx: int, c_idx: int) -> list[list[str]]:
    return [
        [f"{sn_idx} Sn 5s", f"{sn_idx} Sn 5p", f"{c_idx} C 2p"],
        ["Sn 5s", "Sn 5p", f"{c_idx} C 2p"],
        ["Sn 5s", "Sn 5p", "C 2p"],
    ]


def indexed_ao_labels(sn_idx: int, c_idx: int) -> list[str]:
    """Sn 5s/5p + first carbon 2p, atom-indexed. Same labels for a fair pair."""
    return [f"{sn_idx} Sn 5s", f"{sn_idx} Sn 5p", f"{c_idx} C 2p"]


def parse_fair_pair(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or str(raw).strip() == "":
        return None
    key = str(raw).strip().lower().replace(",", "_").replace("-", "_")
    if key in FAIR_PAIR_IDS:
        return FAIR_PAIR_IDS[key]
    parts = tuple(p for p in key.split("_") if p)
    if len(parts) >= 2:
        return parts
    raise ValueError(f"Unrecognised --fair-pair {raw!r}")


def backup_first_wave(lig_dir: Path, charge: int) -> dict[str, str]:
    """Preserve the auto-threshold first wave; do not overwrite an existing backup."""
    copied: dict[str, str] = {}
    pairs = (
        (lig_dir / "e0.json", lig_dir / "e0_thresh_auto.json"),
        (
            lig_dir / f"hamiltonian_c{int(charge)}.json",
            lig_dir / f"hamiltonian_c{int(charge)}_thresh_auto.json",
        ),
    )
    for src, dest in pairs:
        if src.is_file() and not dest.is_file():
            shutil.copy2(src, dest)
            copied[src.name] = str(dest)
    return copied


def avas_once(
    mf: Any,
    labels: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    ncas, nelecas, mo = avas.avas(
        mf,
        list(labels),
        threshold=float(threshold),
        canonicalize=True,
        openshell_option=3 if mf.mol.spin else 2,
    )
    ncas_i = int(ncas)
    return {
        "ao_labels": list(labels),
        "threshold": float(threshold),
        "ncas": ncas_i,
        "nelecas": nelecas_json(nelecas),
        "nelecas_raw": nelecas,
        "n_qubits": 2 * ncas_i,
        "n_fci": fci_dimension(ncas_i, nelecas),
        "mo_coeff": mo,
    }


def min_fitting_avas(
    mf: Any,
    labels: Sequence[str],
    *,
    thresholds: Sequence[float],
    max_qubits: int,
    max_fci: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    last_err: str | None = None
    chosen: dict[str, Any] | None = None
    for threshold in tqdm(
        list(thresholds),
        desc="AVAS thresholds",
        unit="thr",
        dynamic_ncols=True,
        disable=None,
    ):
        try:
            row = avas_once(mf, labels, float(threshold))
        except Exception as exc:  # noqa: BLE001 — AVAS label mismatch is expected
            last_err = f"{list(labels)} @ {threshold}: {exc}"
            attempts.append(
                {
                    "ao_labels": list(labels),
                    "threshold": float(threshold),
                    "error": str(exc),
                }
            )
            continue
        slim = {k: v for k, v in row.items() if k not in {"mo_coeff", "nelecas_raw"}}
        attempts.append(slim)
        fits = (
            row["ncas"] >= 2
            and row["n_qubits"] <= max_qubits
            and row["n_fci"] <= max_fci
        )
        if fits and chosen is None:
            chosen = {**row, "attempts": attempts}
            break
    if chosen is None:
        raise RuntimeError(
            "AVAS could not fit under "
            f"max_qubits={max_qubits} max_fci={max_fci} labels={list(labels)}. "
            f"attempts={attempts[-8:]}"
            + (f" last_err={last_err}" if last_err else "")
        )
    chosen["attempts"] = attempts
    return chosen


def agree_fair_threshold(
    *,
    out_dir: Path,
    ligand: str,
    pair: Sequence[str],
    probe: Mapping[str, Any],
    wait_sec: int,
) -> dict[str, Any]:
    """Share one AVAS threshold across independent array tasks (no mgpu)."""
    pair_key = "_".join(pair)
    probe_dir = out_dir / f"_fair_{pair_key}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    own = {
        "ligand": ligand,
        "fair_pair": pair_key,
        "ao_labels": list(probe["ao_labels"]),
        "threshold_start": float(probe["threshold_start"]),
        "min_threshold": float(probe["threshold"]),
        "ncas_at_min": int(probe["ncas"]),
        "nelecas_at_min": probe["nelecas"],
        "n_qubits_at_min": int(probe["n_qubits"]),
        "n_fci_at_min": int(probe["n_fci"]),
        "n_qubits_at_start": probe.get("n_qubits_at_start"),
        "ncas_at_start": probe.get("ncas_at_start"),
        "fits_at_start": bool(probe.get("fits_at_start")),
        "attempts": [
            {k: v for k, v in row.items() if k != "mo_coeff"}
            for row in (probe.get("attempts") or [])
        ],
        "updated_at": _now_iso(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    dump_json(probe_dir / f"{ligand}.json", own)
    partners = [p for p in pair if p != ligand]
    deadline = time.time() + max(int(wait_sec), 1)
    missing = list(partners)
    with tqdm(total=len(partners), desc="Fair-pair AVAS barrier", unit="lig", disable=None) as bar:
        seen = 0
        while missing and time.time() < deadline:
            still = []
            for partner in missing:
                if (probe_dir / f"{partner}.json").is_file():
                    continue
                still.append(partner)
            progressed = len(missing) - len(still)
            if progressed:
                bar.update(progressed)
                seen += progressed
            missing = still
            if missing:
                time.sleep(5)
        if missing:
            raise TimeoutError(
                f"Fair-pair barrier timed out waiting for {missing} under {probe_dir}"
            )
        if seen < len(partners):
            bar.update(len(partners) - seen)
    reports = {ligand: own}
    for partner in partners:
        reports[partner] = json.loads((probe_dir / f"{partner}.json").read_text(encoding="utf-8"))
    shared = max(float(reports[name]["min_threshold"]) for name in pair)
    agreed = {
        "fair_pair": pair_key,
        "charge": 0,
        "ao_labels_rule": "Sn 5s/5p + first-C 2p (atom-indexed)",
        "threshold_start": float(probe["threshold_start"]),
        "threshold": shared,
        "same_threshold_for": list(pair),
        "ligands": reports,
        "updated_at": _now_iso(),
    }
    dump_json(probe_dir / "shared_threshold.json", agreed)
    print(
        f"  Fair pair {pair_key}: shared threshold={shared} "
        f"(start={probe['threshold_start']}; "
        + ", ".join(
            f"{name} min={reports[name]['min_threshold']} "
            f"n_q={reports[name]['n_qubits_at_min']}"
            for name in pair
        )
        + ")"
    )
    return agreed


def select_avas(
    mf: Any,
    sn_idx: int,
    c_idx: int,
    *,
    max_qubits: int,
    max_fci: int,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    last_err: str | None = None
    attempts: list[dict[str, Any]] = []
    for labels in ao_label_sets(sn_idx, c_idx):
        for threshold in thresholds:
            try:
                ncas, nelecas, mo = avas.avas(
                    mf,
                    labels,
                    threshold=float(threshold),
                    canonicalize=True,
                    openshell_option=3 if mf.mol.spin else 2,
                )
            except Exception as exc:  # noqa: BLE001 — AVAS label mismatch is expected
                last_err = f"{labels} @ {threshold}: {exc}"
                continue
            ncas_i = int(ncas)
            n_qubits = 2 * ncas_i
            n_fci = fci_dimension(ncas_i, nelecas)
            row = {
                "ao_labels": list(labels),
                "threshold": float(threshold),
                "ncas": ncas_i,
                "nelecas": nelecas_json(nelecas),
                "n_qubits": n_qubits,
                "n_fci": n_fci,
            }
            attempts.append(row)
            if ncas_i < 2:
                continue
            if n_qubits <= max_qubits and n_fci <= max_fci:
                return {
                    **row,
                    "mo_coeff": mo,
                    "nelecas_raw": nelecas,
                    "attempts": attempts,
                }
    raise RuntimeError(
        "AVAS could not fit under "
        f"max_qubits={max_qubits} max_fci={max_fci}. "
        f"attempts={attempts[-8:]}"
        + (f" last_err={last_err}" if last_err else "")
    )


def casci_and_integrals(
    mf: Any, ncas: int, nelecas: Any, mo: Any, *, verbose: int = PYSCF_VERBOSE_DEFAULT
) -> Any:
    mc = mcscf.CASCI(mf, int(ncas), nelecas)
    mc.verbose = int(verbose)
    mc.fcisolver.conv_tol = 1e-10
    mc.kernel(mo)
    return mc


def restore_h2(mc: Any, ncas: int) -> np.ndarray:
    eri = mc.get_h2eff()
    return np.asarray(ao2mo.restore(1, eri, int(ncas)), dtype=float)


def to_pauli_terms(qubit_ham: Any) -> list[dict[str, float | str]]:
    terms: list[dict[str, float | str]] = []
    items = list(qubit_ham.terms.items())
    for pauli_term, coeff in tqdm(
        items,
        desc="Serializing Pauli terms",
        unit="term",
        dynamic_ncols=True,
        disable=None,
    ):
        label = " ".join([f"{p}{i}" for i, p in pauli_term]) if pauli_term else "I"
        terms.append(
            {"term": label, "real": float(coeff.real), "imag": float(coeff.imag)}
        )
    return terms


def jordan_wigner_cas(h1: np.ndarray, h2: np.ndarray, e_core: float) -> Any:
    from openfermion import InteractionOperator, jordan_wigner
    from openfermion.chem.molecular_data import spinorb_from_spatial

    two_body_of = np.asarray(h2.transpose(0, 2, 3, 1), order="C")
    one_body_so, two_body_so = spinorb_from_spatial(h1, two_body_of)
    interaction_op = InteractionOperator(
        float(e_core), one_body_so, 0.5 * two_body_so
    )
    return jordan_wigner(interaction_op)


def jw_casci_deviation_mha(qubit_ham: Any, n_qubits: int, casci_ha: float) -> float:
    from openfermion import get_sparse_operator
    from scipy.sparse.linalg import eigsh

    h_sparse = get_sparse_operator(qubit_ham, n_qubits=int(n_qubits))
    e_jw = float(eigsh(h_sparse, k=1, which="SA", return_eigenvectors=False)[0])
    return abs(e_jw - float(casci_ha)) * 1e3


def empty_e0_doc(ligand: str) -> dict[str, Any]:
    return {
        "label": "ours",
        "not_matgenq_table1": True,
        "gqe_proposes_molecules": False,
        "forbidden_gqe_bde_claim": FORBIDDEN,
        "ligand": ligand,
        "avas": {
            "method": "pyscf.mcscf.avas",
            "citation": "Sayfutyarova, Sun, Chan, Knizia, JCTC 13, 4063 (2017); arXiv:1701.07862",
            "shells": "Sn 5s, Sn 5p, alpha-C 2p (first carbon in labelled-ours XYZ)",
            "not_matgenq_xyz": True,
            "l40s_cap_qubits": MAX_QUBITS_L40S,
            "fair_pair": None,
        },
        "basis": BASIS,
        "ecp": dict(ECP),
        "charges": {},
        "updated_at": _now_iso(),
    }


def sync_top_level(doc: dict[str, Any]) -> dict[str, Any]:
    charges = doc.get("charges") or {}
    preferred = charges.get("0") or charges.get(0)
    if preferred is None and charges:
        preferred = next(iter(charges.values()))
    if isinstance(preferred, Mapping):
        doc["e0_casci_ha"] = preferred.get("e0_casci_ha")
        doc["e0_qsci_ha"] = preferred.get("e0_qsci_ha")
        doc["delta_e_qsci_mha"] = preferred.get("delta_e_qsci_mha")
        doc["e0_n_qubits"] = preferred.get("n_qubits")
        doc["e0_ncas"] = preferred.get("ncas")
        doc["e0_nelecas"] = preferred.get("nelecas")
        doc["e0_from_charge"] = preferred.get("charge", 0)
    doc["updated_at"] = _now_iso()
    return doc


def hamiltonian_record(
    *,
    ligand: str,
    charge: int,
    spin: int,
    atoms: Sequence[Sequence[Any]],
    mc: Any,
    mf: Any,
    avas_row: Mapping[str, Any],
    terms: list[dict[str, float | str]],
    smiles: str | None,
) -> dict[str, Any]:
    ncas = int(avas_row["ncas"])
    n_qubits = int(avas_row["n_qubits"])
    ncore = int(getattr(mc, "ncore", 0))
    name = f"ours_{ligand}_c{int(charge)}_{n_qubits}q"
    return {
        "name": name,
        "split": "ours",
        "ligand": ligand,
        "smiles": smiles,
        "geometry": [[str(sym), [float(x) for x in xyz]] for sym, xyz in atoms],
        "basis": BASIS,
        "ecp": dict(ECP),
        "charge": int(charge),
        "multiplicity": int(spin) + 1,
        "label": "ours",
        "not_matgenq_table1": True,
        "active_space": {
            "mode": "avas",
            "ao_labels": list(avas_row["ao_labels"]),
            "threshold": float(avas_row["threshold"]),
            "n_active_electrons": nelecas_total(avas_row["nelecas_raw"]),
            "n_active_orbitals": ncas,
            "n_core_orbitals": ncore,
            "n_fci": int(avas_row["n_fci"]),
            "occupied_indices": [],
            "active_indices": list(range(ncore, ncore + ncas)),
        },
        "n_qubits": n_qubits,
        "n_pauli_terms": len(terms),
        "hf_energy": float(mf.e_tot),
        "casci_energy": float(mc.e_tot),
        "source": "labelled-ours AVAS+CASCI at B3LYP/def2-SVP geometry",
        "arxiv_avas": "1701.07862",
        "arxiv_matgenq_recipe": "2607.23988",
        "fair_pair": avas_row.get("fair_pair"),
        "terms": terms,
    }


def run_casci_export(
    *,
    ligand: str,
    charge: int,
    out_dir: Path,
    max_qubits: int,
    max_fci: int,
    skip_jw_check: bool,
    threshold: float | None = None,
    fair_pair: Sequence[str] | None = None,
    pyscf_verbose: int = PYSCF_VERBOSE_DEFAULT,
    fair_wait_sec: int = 7200,
) -> dict[str, Any]:
    lig_dir = out_dir / ligand
    backups = backup_first_wave(lig_dir, charge)
    if backups:
        print(f"  Backed up first-wave files: {backups}")
    score_path = lig_dir / "score.json"
    if not score_path.is_file():
        raise FileNotFoundError(f"Missing {score_path}")
    score = json.loads(score_path.read_text(encoding="utf-8"))
    xyz = xyz_path_for(lig_dir, score, charge)
    atoms = read_xyz(xyz)
    sn_idx, c_idx = first_indices(atoms)
    spin = 0 if int(charge) == 0 else 1
    if fair_pair is not None and int(charge) != 0:
        raise ValueError("Fair-pair AVAS is charge 0 only")
    mol = make_mol(
        atoms,
        charge=int(charge),
        spin=spin,
        verbose=int(pyscf_verbose),
        max_memory=_parse_mem_mb(),
    )
    print(
        f"E0 CASCI ligand={ligand} charge={charge} xyz={xyz} "
        f"Sn={sn_idx} alphaC={c_idx} nao={mol.nao}"
    )
    mf = run_hf(mol)
    start_thr = 0.20 if threshold is None else float(threshold)
    if fair_pair is not None:
        labels = indexed_ao_labels(sn_idx, c_idx)
        ladder = [t for t in DEFAULT_THRESHOLDS if t + 1e-12 >= start_thr]
        if start_thr not in ladder:
            ladder = [start_thr, *ladder]
        probe = min_fitting_avas(
            mf,
            labels,
            thresholds=ladder,
            max_qubits=max_qubits,
            max_fci=max_fci,
        )
        try:
            at_start = avas_once(mf, labels, start_thr)
            probe["n_qubits_at_start"] = int(at_start["n_qubits"])
            probe["ncas_at_start"] = int(at_start["ncas"])
            probe["fits_at_start"] = (
                at_start["ncas"] >= 2
                and at_start["n_qubits"] <= max_qubits
                and at_start["n_fci"] <= max_fci
            )
        except Exception as exc:  # noqa: BLE001
            probe["n_qubits_at_start"] = None
            probe["fits_at_start"] = False
            probe["start_error"] = str(exc)
        probe["threshold_start"] = start_thr
        agreed = agree_fair_threshold(
            out_dir=out_dir,
            ligand=ligand,
            pair=fair_pair,
            probe=probe,
            wait_sec=fair_wait_sec,
        )
        shared_thr = float(agreed["threshold"])
        avas_row = avas_once(mf, labels, shared_thr)
        if (
            avas_row["ncas"] < 2
            or avas_row["n_qubits"] > max_qubits
            or avas_row["n_fci"] > max_fci
        ):
            raise RuntimeError(
                f"Shared fair threshold {shared_thr} does not fit L40S cap "
                f"for {ligand}: n_qubits={avas_row['n_qubits']} n_fci={avas_row['n_fci']}"
            )
        avas_row["attempts"] = probe.get("attempts") or []
        avas_row["fair_pair"] = agreed["fair_pair"]
        avas_row["threshold_start"] = start_thr
        avas_row["fair_agreed"] = agreed
    elif threshold is not None:
        labels = indexed_ao_labels(sn_idx, c_idx)
        avas_row = avas_once(mf, labels, float(threshold))
        if (
            avas_row["ncas"] < 2
            or avas_row["n_qubits"] > max_qubits
            or avas_row["n_fci"] > max_fci
        ):
            raise RuntimeError(
                f"--threshold {threshold} does not fit max_qubits={max_qubits} "
                f"for {ligand}: n_qubits={avas_row['n_qubits']} n_fci={avas_row['n_fci']}"
            )
        avas_row["attempts"] = [
            {k: v for k, v in avas_row.items() if k not in {"mo_coeff", "nelecas_raw"}}
        ]
        avas_row["fair_pair"] = None
        avas_row["threshold_start"] = float(threshold)
    else:
        avas_row = select_avas(
            mf,
            sn_idx,
            c_idx,
            max_qubits=max_qubits,
            max_fci=max_fci,
            thresholds=DEFAULT_THRESHOLDS,
        )
        avas_row["fair_pair"] = None
        avas_row["threshold_start"] = None
    print(
        f"  AVAS ncas={avas_row['ncas']} nelecas={avas_row['nelecas']} "
        f"n_qubits={avas_row['n_qubits']} N_FCI={avas_row['n_fci']} "
        f"threshold={avas_row['threshold']} labels={avas_row['ao_labels']}"
        + (
            f" fair_pair={avas_row.get('fair_pair')}"
            if avas_row.get("fair_pair")
            else ""
        )
    )
    mc = casci_and_integrals(
        mf,
        avas_row["ncas"],
        avas_row["nelecas_raw"],
        avas_row["mo_coeff"],
        verbose=int(pyscf_verbose),
    )
    h1, e_core = mc.get_h1eff()
    h2 = restore_h2(mc, avas_row["ncas"])
    qubit_ham = jordan_wigner_cas(np.asarray(h1), h2, float(e_core))
    n_qubits = int(avas_row["n_qubits"])
    jw_dev = None
    do_jw = (not skip_jw_check) and n_qubits <= 16
    if do_jw:
        jw_dev = jw_casci_deviation_mha(qubit_ham, n_qubits, float(mc.e_tot))
        print(f"  JW vs CASCI |dev| = {jw_dev:.3e} mHa")
        if jw_dev >= 1e-3:
            raise RuntimeError(f"Jordan-Wigner convention error: {jw_dev} mHa")
    terms = to_pauli_terms(qubit_ham)
    record = hamiltonian_record(
        ligand=ligand,
        charge=charge,
        spin=spin,
        atoms=atoms,
        mc=mc,
        mf=mf,
        avas_row=avas_row,
        terms=terms,
        smiles=score.get("smiles"),
    )
    ham_path = lig_dir / f"hamiltonian_c{int(charge)}.json"
    dump_json(
        ham_path,
        {
            "description": (
                "Labelled-ours AVAS+CASCI JW Hamiltonian. Not MatGen-Q Table 1. "
                "Not tin_resist.yaml."
            ),
            "records": [record],
        },
    )
    charge_block = {
        "charge": int(charge),
        "spin": spin,
        "xyz": str(xyz),
        "ao_labels": list(avas_row["ao_labels"]),
        "threshold": float(avas_row["threshold"]),
        "threshold_start": avas_row.get("threshold_start"),
        "ncas": int(avas_row["ncas"]),
        "nelecas": avas_row["nelecas"],
        "n_qubits": n_qubits,
        "n_fci": int(avas_row["n_fci"]),
        "fair_pair": avas_row.get("fair_pair"),
        "n_core": int(getattr(mc, "ncore", 0)),
        "hf_energy_ha": float(mf.e_tot),
        "casci_energy_ha": float(mc.e_tot),
        "e0_casci_ha": float(mc.e_tot),
        "as_correlation_mha": float(mf.e_tot - mc.e_tot) * 1e3,
        "n_pauli_terms": len(terms),
        "jw_dev_mha": jw_dev,
        "hamiltonian_json": str(ham_path),
        "molecule_name": record["name"],
        "e0_qsci_ha": None,
        "delta_e_qsci_mha": None,
        "qsci_note": (
            "QSCI is filled by tin_inform_e0.py --ingest-qsci. "
            "14-char H-cGQE is not used unless n_qubits==14."
        ),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "updated_at": _now_iso(),
    }
    e0_path = lig_dir / "e0.json"
    doc = empty_e0_doc(ligand)
    if e0_path.is_file():
        try:
            loaded = json.loads(e0_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                doc.update(loaded)
        except json.JSONDecodeError:
            pass
    avas_meta = dict(doc.get("avas") or empty_e0_doc(ligand)["avas"])
    avas_meta["fair_pair"] = avas_row.get("fair_pair")
    avas_meta["threshold"] = float(avas_row["threshold"])
    avas_meta["threshold_start"] = avas_row.get("threshold_start")
    avas_meta["ao_labels_rule"] = "Sn 5s/5p + first-C 2p (atom-indexed)"
    doc["avas"] = avas_meta
    doc["fair_pair"] = avas_row.get("fair_pair")
    if avas_row.get("fair_agreed"):
        doc["fair_avas"] = {
            "fair_pair": avas_row["fair_pair"],
            "charge": 0,
            "ao_labels_rule": "Sn 5s/5p + first-C 2p (atom-indexed)",
            "threshold_start": avas_row.get("threshold_start"),
            "threshold": float(avas_row["threshold"]),
            "ncas": int(avas_row["ncas"]),
            "nelecas": avas_row["nelecas"],
            "n_qubits": n_qubits,
            "n_fci": int(avas_row["n_fci"]),
            "same_threshold_for": list(
                (avas_row.get("fair_agreed") or {}).get("same_threshold_for") or []
            ),
        }
    charges = dict(doc.get("charges") or {})
    # New CASCI Hamiltonian: do not keep first-wave QSCI ΔE on a different space.
    charges[str(charge)] = charge_block
    doc["charges"] = charges
    doc["ligand"] = ligand
    synced = sync_top_level(doc)
    dump_json(e0_path, synced)
    if avas_row.get("fair_pair"):
        dump_json(lig_dir / "e0_fair.json", synced)
    print(
        f"  CASCI {float(mc.e_tot):.8f} Ha  HF {float(mf.e_tot):.8f} Ha  "
        f"wrote {ham_path} and {e0_path}"
    )
    return charge_block


def _pick_primary_qsci(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for mol in results:
        sampler = str(mol.get("sampler") or "")
        if sampler.startswith("pool-native"):
            return mol
    return results[0]


def _best_sweep(mol: Mapping[str, Any]) -> dict[str, Any]:
    sweeps = [
        row
        for row in (mol.get("sweep_results") or [])
        if row.get("qsci_energy") is not None
    ]
    if not sweeps:
        raise ValueError("QSCI sweeps empty/failed")
    return min(
        sweeps,
        key=lambda row: (
            float(row["error_vs_casci_mha"])
            if row.get("error_vs_casci_mha") is not None
            else 1e99
        ),
    )


def ingest_qsci(e0_path: Path, qsci_path: Path, charge: int) -> dict[str, Any]:
    doc = json.loads(e0_path.read_text(encoding="utf-8"))
    qsci = json.loads(qsci_path.read_text(encoding="utf-8"))
    results = qsci.get("results") or []
    if not results:
        raise ValueError(f"No QSCI results in {qsci_path}")
    mol = _pick_primary_qsci(results)
    best = _best_sweep(mol)
    charges = dict(doc.get("charges") or {})
    block = dict(charges.get(str(charge)) or {})
    block["e0_qsci_ha"] = float(best["qsci_energy"])
    block["delta_e_qsci_mha"] = (
        None
        if best.get("error_vs_casci_mha") is None
        else float(best["error_vs_casci_mha"])
    )
    block["qsci_n_samples"] = best.get("n_samples_used")
    block["qsci_n_unique"] = best.get("n_unique_bitstrings")
    block["qsci_json"] = str(qsci_path)
    block["qsci_sampler"] = mol.get("sampler") or qsci.get("primary_sampler")
    block["used_hcgqe_operators"] = False
    block["used_14char_hcgqe"] = bool(qsci.get("used_14char_hcgqe"))
    block["qsci_backend"] = mol.get("backend")
    block["qsci_updated_at"] = _now_iso()
    control = next(
        (row for row in results if str(row.get("sampler") or "") == "entangled-hf"),
        None,
    )
    if control is not None:
        try:
            ctrl_best = _best_sweep(control)
            block["qsci_control_entangled_hf"] = {
                "sampler": "entangled-hf",
                "e0_qsci_ha": float(ctrl_best["qsci_energy"]),
                "delta_e_qsci_mha": ctrl_best.get("error_vs_casci_mha"),
                "n_samples_used": ctrl_best.get("n_samples_used"),
                "n_unique_bitstrings": ctrl_best.get("n_unique_bitstrings"),
            }
        except ValueError:
            block["qsci_control_entangled_hf"] = {
                "sampler": "entangled-hf",
                "error": "sweeps empty/failed",
            }
    charges[str(charge)] = block
    doc["charges"] = charges
    synced = sync_top_level(doc)
    dump_json(e0_path, synced)
    fair_path = e0_path.with_name("e0_fair.json")
    if fair_path.is_file() or doc.get("fair_pair"):
        dump_json(fair_path, synced)
    print(
        f"Ingested QSCI sampler={block.get('qsci_sampler')} "
        f"E={block['e0_qsci_ha']:.8f} Ha  "
        f"ΔE={block['delta_e_qsci_mha']} mHa into {e0_path}"
    )
    return block


def pool_native_operators(
    n_qubits: int,
    n_electrons: int,
    *,
    max_ops: int = 64,
    angle: float = 0.1,
) -> tuple[list[str], list[float], dict[str, Any]]:
    """UCCSD-like first-Pauli pool words sized to n_qubits. Never pad 14-char."""
    if n_qubits == 14:
        meta = {
            "note": "n_qubits==14 would allow 14-char H-cGQE; this sampler still uses the pool, not the checkpoint.",
        }
    else:
        meta = {}
    demo_root = str(MATGENQ_DEMO)
    if demo_root not in sys.path:
        sys.path.insert(0, demo_root)
    from demo.pool import enumerate_uccsd_excitations, excitation_pauli_words

    if n_electrons % 2 != 0 or n_qubits % 2 != 0:
        raise ValueError(
            f"Pool-native UCCSD needs even n_electrons and n_qubits; "
            f"got n_e={n_electrons} n_q={n_qubits}"
        )
    n_occ = n_electrons // 2
    n_orb = n_qubits // 2
    excitations = enumerate_uccsd_excitations(n_occ, n_orb)
    words: list[str] = []
    seen: set[str] = set()
    identity = "I" * n_qubits
    for exc in tqdm(
        excitations,
        desc="Pool-native UCCSD words",
        unit="exc",
        dynamic_ncols=True,
        disable=None,
    ):
        for _, mapping in excitation_pauli_words(exc, n_qubits):
            mapping = {q: p for q, p in mapping.items() if p != "Z"}
            if not mapping:
                continue
            word = "".join(mapping.get(q, "I") for q in range(n_qubits))
            if len(word) != n_qubits or word == identity or word in seen:
                continue
            seen.add(word)
            words.append(word)
            break
        if len(words) >= int(max_ops):
            break
    if not words:
        raise RuntimeError(
            f"Pool-native operator construction returned no words "
            f"for n_qubits={n_qubits} n_electrons={n_electrons}"
        )
    thetas = [float(angle)] * len(words)
    meta.update(
        {
            "source": "MatGen-Q demo.pool (remove-Z, first Pauli per excitation)",
            "n_excitations_enumerated": len(excitations),
            "n_unique_words": len(words),
            "max_ops": int(max_ops),
            "angle": float(angle),
            "word_lengths": sorted({len(w) for w in words}),
            "padded_14char": False,
            "used_14char_hcgqe": False,
        }
    )
    return words, thetas, meta


def run_qsci_job(
    *,
    ligand: str,
    charge: int,
    out_dir: Path,
    n_shots: int,
    n_samples: Sequence[int],
    qsci_sampler: str = "both",
    max_pool_ops: int = 64,
) -> Path:
    try:
        import cudaq
    except ImportError as exc:
        raise SystemExit("CUDA-Q is required for --qsci") from exc

    from src.gqe.eval.qsci import load_hamiltonian_records, run_qsci_for_molecule
    from src.gqe.common.hamiltonian_utils import get_active_electron_count

    lig_dir = out_dir / ligand
    e0_path = lig_dir / "e0.json"
    ham_path = lig_dir / f"hamiltonian_c{int(charge)}.json"
    if not ham_path.is_file():
        raise FileNotFoundError(f"Missing {ham_path}; run CASCI export first")
    records = load_hamiltonian_records(ham_path)
    if not records:
        raise ValueError(f"No records in {ham_path}")
    record = records[0]
    n_qubits = int(record["n_qubits"])
    n_electrons = get_active_electron_count(record)
    if n_qubits > MAX_QUBITS_L40S:
        raise SystemExit(
            f"{record.get('name')} is {n_qubits}q; L40S statevector cap is "
            f"{MAX_QUBITS_L40S}. Do not pad a 14-char policy onto this H."
        )
    if n_qubits != 14:
        print(
            f"  Skipping 14-char H-cGQE (n_qubits={n_qubits}); "
            "pool-native words are generated at this width."
        )
    option = os.environ.get("CUDAQ_NVIDIA_OPTION", "fp64").strip() or "fp64"
    if "mgpu" in option:
        raise SystemExit(
            f"Refusing CUDAQ_NVIDIA_OPTION={option!r} for ≤24q E0; use fp64 nvidia, "
            "not mgpu. Two ligands scale as two 1-GPU Slurm array tasks."
        )
    cudaq.set_target("nvidia", option=option)
    print(
        f"QSCI {record['name']} n_qubits={n_qubits} n_e={n_electrons} "
        f"terms={record.get('n_pauli_terms')} target=nvidia,{option} "
        f"sampler={qsci_sampler}"
    )

    want = {"pool-native", "entangled-hf"} if qsci_sampler == "both" else {qsci_sampler}
    results: list[dict[str, Any]] = []
    pool_meta: dict[str, Any] | None = None

    if "pool-native" in want:
        ops, thetas, pool_meta = pool_native_operators(
            n_qubits, n_electrons, max_ops=int(max_pool_ops)
        )
        print(
            f"  Primary sampler=pool-native-uccsd  n_ops={len(ops)} "
            f"word_len={pool_meta.get('word_lengths')}"
        )
        primary = run_qsci_for_molecule(
            record,
            operators=ops,
            thetas=thetas,
            n_samples_list=list(n_samples),
            bond_dims=[64],
            n_shots=int(n_shots),
            backend="nvidia",
        )
        primary["sampler"] = "pool-native-uccsd"
        primary["used_hcgqe_operators"] = False
        primary["used_14char_hcgqe"] = False
        primary["pool"] = pool_meta
        primary["n_operators"] = len(ops)
        results.append(primary)

    if "entangled-hf" in want:
        print("  Control sampler=entangled-hf")
        control = run_qsci_for_molecule(
            record,
            operators=None,
            thetas=None,
            n_samples_list=list(n_samples),
            bond_dims=[64],
            n_shots=int(n_shots),
            backend="nvidia",
        )
        control["sampler"] = "entangled-hf"
        control["used_hcgqe_operators"] = False
        control["used_14char_hcgqe"] = False
        results.append(control)

    if not results:
        raise RuntimeError(f"No QSCI results for sampler={qsci_sampler}")

    out_path = lig_dir / f"qsci_c{int(charge)}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    dump_json(
        out_path,
        {
            "experiment": "labelled_ours_avas_qsci_fair_me_cf3",
            "description": (
                "Fair-pair labelled-ours AVAS Hamiltonian. Primary sampler is "
                "pool-native UCCSD-like (MatGen-Q demo.pool, sized to n_qubits). "
                "Entangled-HF is a named control. Not 14-char H-cGQE. Not a DFT ΔBDE."
            ),
            "ligand": ligand,
            "charge": int(charge),
            "primary_sampler": "pool-native-uccsd"
            if any(str(r.get("sampler", "")).startswith("pool-native") for r in results)
            else results[0].get("sampler"),
            "control_sampler": "entangled-hf"
            if any(r.get("sampler") == "entangled-hf" for r in results)
            else None,
            "used_14char_hcgqe": False,
            "backend": f"nvidia,{option}",
            "n_shots": int(n_shots),
            "pool": pool_meta,
            "results": results,
        },
    )
    ingest_qsci(e0_path, out_path, charge)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--charge", type=int, default=0, choices=(0, 1))
    parser.add_argument("--max-qubits", type=int, default=MAX_QUBITS_L40S)
    parser.add_argument("--max-fci", type=int, default=MAX_FCI_DETS)
    parser.add_argument(
        "--skip-jw-check",
        action="store_true",
        help="Skip sparse JW vs CASCI (always skipped for n_qubits>16).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "AVAS projector threshold. Fair-pair starts here (default 0.2) and "
            "raises BOTH ligands together until n_q≤24. Without --fair-pair, "
            "this value is used exactly (no independent auto-raise)."
        ),
    )
    parser.add_argument(
        "--fair-pair",
        default=None,
        help="Share one AVAS threshold across ligands, e.g. me_cf3. Charge 0 only.",
    )
    parser.add_argument(
        "--pyscf-verbose",
        type=int,
        default=PYSCF_VERBOSE_DEFAULT,
        help="PySCF mol/CASCI verbose. Default 3; 4 dumps the Python source file.",
    )
    parser.add_argument(
        "--fair-wait-sec",
        type=int,
        default=7200,
        help="Seconds to wait for the partner ligand AVAS probe in a fair pair.",
    )
    parser.add_argument(
        "--qsci",
        action="store_true",
        help="Sample QSCI on GPU for an already-exported Hamiltonian.",
    )
    parser.add_argument(
        "--qsci-sampler",
        choices=("pool-native", "entangled-hf", "both"),
        default="both",
        help="Primary pool-native UCCSD-like sampler, entangled-HF control, or both.",
    )
    parser.add_argument(
        "--max-pool-ops",
        type=int,
        default=64,
        help="Cap on pool-native first-Pauli UCCSD words (sized to n_qubits).",
    )
    parser.add_argument("--n-shots", type=int, default=8192)
    parser.add_argument("--n-samples", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--ingest-qsci", type=Path, default=None)
    parser.add_argument(
        "--rank",
        action="store_true",
        help="Re-run tin_inform_rank.py after writing e0.json (does not change ΔBDE order).",
    )
    return parser


def maybe_rank(out_dir: Path) -> None:
    from src.gqe.eval.tin_inform_rank import main as rank_main

    rank_main(["--out-dir", str(out_dir)])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = _resolve(args.out_dir)
    ligand = str(args.ligand).lower()
    if args.ingest_qsci is not None:
        ingest_qsci(out_dir / ligand / "e0.json", _resolve(args.ingest_qsci), args.charge)
        if args.rank:
            maybe_rank(out_dir)
        return 0
    if args.qsci:
        run_qsci_job(
            ligand=ligand,
            charge=args.charge,
            out_dir=out_dir,
            n_shots=args.n_shots,
            n_samples=args.n_samples,
            qsci_sampler=args.qsci_sampler,
            max_pool_ops=args.max_pool_ops,
        )
        if args.rank:
            maybe_rank(out_dir)
        return 0
    run_casci_export(
        ligand=ligand,
        charge=args.charge,
        out_dir=out_dir,
        max_qubits=args.max_qubits,
        max_fci=args.max_fci,
        skip_jw_check=args.skip_jw_check,
        threshold=args.threshold,
        fair_pair=parse_fair_pair(args.fair_pair),
        pyscf_verbose=args.pyscf_verbose,
        fair_wait_sec=args.fair_wait_sec,
    )
    if args.rank:
        maybe_rank(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
