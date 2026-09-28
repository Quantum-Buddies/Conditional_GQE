#!/usr/bin/env python3
"""Labelled-ours R–Sn(OH)₃ B3LYP opt + MP2 single-point scorer.

NOT MatGen-Q Table 1. Writes only under results/tin_ab/methyltin_ours/ (or
--out-dir) with label=ours on every JSON.

Neutral: charge 0 singlet RKS-B3LYP, then RHF-MP2 at the DFT geometry.
Cation:  charge +1 doublet UKS-B3LYP (starts from opt neutral), then UHF-MP2.
Sn–C BDE at B3LYP:
  BDE(0)  = E(R•) + E(•Sn(OH)₃) − E(RSn(OH)₃)
  BDE(+1) = E(R•) + E(Sn(OH)₃⁺) − E(RSn(OH)₃⁺)   # charge on Sn, homolytic Sn–C
Do not write 72.6 → 21.2 kcal/mol as a GQE number.

Usage:
  PYTHONPATH=<repo> python src/gqe/eval/tin_inform_dft.py \\
    --ligands me,et,nbu,vinyl,ph \\
    --out-dir results/tin_ab/methyltin_ours
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import traceback
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from tqdm.auto import tqdm

from src.gqe.data.rsn_oh3_templates import (
    FIRST_WAVE,
    ROLE_LIGAND,
    ROLE_OH_H,
    ROLE_OH_O,
    ROLE_SN,
    UNEVALUATED_POOL,
    MoleculeGuess,
    atomic_13p5nm_proxy,
    build_rsn_oh3,
    build_sn_oh3_fragment,
    parse_ligand_list,
)

try:
    from pyscf import dft, gto, mp, scf
    from pyscf.geomopt import berny_solver, geometric_solver
    from pyscf.lib import param as pyscf_param
except ImportError as exc:  # pragma: no cover - env contract
    raise SystemExit(
        "PySCF is required in cudaq-env. "
        f"Import failed: {exc}"
    ) from exc

try:
    from pyscf.scf.addons import smearing_ as pyscf_smearing
except ImportError:  # pragma: no cover
    pyscf_smearing = None


HA_TO_EV = 27.211386245981
HA_TO_KCAL = 627.5094740631
BASIS = "def2-svp"
ECP = {"Sn": "def2-svp"}
FUNCTIONAL = "B3LYP"
FORBIDDEN_GQE_BDE = (
    "Do not write 72.6→21.2 kcal/mol as a GQE number; that is classical "
    "UCCSD(T) in arXiv:2607.23988, not this labelled-ours B3LYP ΔBDE."
)
DEFAULT_OUT = Path("results/tin_ab/methyltin_ours")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_ready(payload), indent=2, sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8")


def write_xyz(path: Path, atoms: Sequence[Sequence[Any]], comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(atoms)), comment]
    for item in atoms:
        sym = str(item[0])
        xyz = item[1]
        lines.append(f"{sym:2s} {float(xyz[0]):16.10f} {float(xyz[1]):16.10f} {float(xyz[2]):16.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mol_atoms_xyz(mol: Any) -> list[list[Any]]:
    coords = mol.atom_coords(unit="Angstrom")
    out: list[list[Any]] = []
    for i in range(mol.natm):
        out.append([mol.atom_pure_symbol(i), [float(x) for x in coords[i]]])
    return out


def _parse_mem_mb() -> int:
    raw = os.environ.get("SLURM_MEM_PER_NODE") or os.environ.get("SLURM_MEM_PER_CPU")
    if not raw:
        return 180_000
    token = str(raw).strip().upper().replace(" ", "")
    try:
        if token.endswith("G"):
            mb = int(float(token[:-1]) * 1024)
        elif token.endswith("M"):
            mb = int(float(token[:-1]))
        elif token.endswith("K"):
            mb = max(int(float(token[:-1]) / 1024), 1024)
        else:
            mb = int(float(token))
    except ValueError:
        return 180_000
    return max(mb - 8_192, 16_384)


def make_mol(
    atoms: Sequence[Sequence[Any]] | str,
    *,
    charge: int,
    spin: int,
    verbose: int,
    max_memory: int,
) -> Any:
    if isinstance(atoms, str):
        atom = atoms
        has_sn = "Sn" in atoms
    else:
        atom = [[str(sym), list(xyz)] for sym, xyz in atoms]
        has_sn = any(str(sym) == "Sn" for sym, _xyz in atoms)
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = BASIS
    mol.ecp = dict(ECP) if has_sn else {}
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.unit = "Angstrom"
    mol.verbose = int(verbose)
    mol.max_memory = int(max_memory)
    mol.symmetry = False
    mol.build()
    return mol


def make_dft(mol: Any) -> Any:
    mf = dft.UKS(mol) if mol.spin else dft.RKS(mol)
    mf.xc = FUNCTIONAL
    mf.conv_tol = 1e-8
    mf.max_cycle = 160
    mf.chkfile = None
    return mf


def _closed_to_open_dm(dm: Any) -> np.ndarray:
    arr = np.asarray(dm, dtype=float)
    if arr.ndim == 3:
        return arr
    return np.stack([0.5 * arr, 0.5 * arr], axis=0)


def run_scf(mf: Any, dm0: Any | None = None) -> Any:
    """SCF with retries for stubborn UKS cations (vinyl vertical IP)."""
    last_e: Any = None
    if dm0 is not None:
        last_e = mf.kernel(dm0)
        if mf.converged:
            return mf
    last_e = mf.kernel()
    if mf.converged:
        return mf
    mf.level_shift = 0.5
    mf.damp = 0.5
    last_e = mf.kernel(dm0) if dm0 is not None else mf.kernel()
    if mf.converged:
        mf.level_shift = 0.0
        mf.damp = 0.0
        last_e = mf.kernel(mf.make_rdm1())
        if mf.converged:
            return mf
    mf.level_shift = 0.0
    mf.damp = 0.0
    try:
        mf_n = mf.newton()
        guess = mf.make_rdm1() if getattr(mf, "e_tot", None) is not None else dm0
        last_e = mf_n.kernel(guess) if guess is not None else mf_n.kernel()
        if mf_n.converged:
            return mf_n
    except Exception:
        pass
    if pyscf_smearing is not None:
        try:
            mf_s = pyscf_smearing(mf, sigma=0.02)
            last_e = mf_s.kernel()
            if mf_s.converged:
                return mf_s
        except Exception:
            pass
    raise RuntimeError(f"SCF failed to converge: E={last_e}")


def run_mp2_at_geom(mol: Any, pbar: Any | None = None) -> dict[str, Any]:
    if pbar is not None:
        pbar.set_postfix_str("HF")
    mf = scf.UHF(mol) if mol.spin else scf.RHF(mol)
    mf.conv_tol = 1e-8
    mf.max_cycle = 128
    run_scf(mf)
    if pbar is not None:
        pbar.set_postfix_str("MP2")
    pt = mp.MP2(mf)
    e_corr = pt.kernel()[0]
    e_mp2 = float(mf.e_tot + e_corr)
    return {
        "label": "ours",
        "hf_ha": float(mf.e_tot),
        "e_corr_ha": float(e_corr),
        "mp2_ha": e_mp2,
        "converged_hf": bool(mf.converged),
        "method": "UHF-MP2" if mol.spin else "RHF-MP2",
        "note": "MP2 single-point at DFT geometry; not a second opt",
    }


def _opt_engine(name: str) -> tuple[Callable[..., Any], str]:
    key = name.lower()
    if key in {"geometric", "geometric_solver"}:
        return geometric_solver.kernel, "geometric"
    if key in {"berny", "pyberny", "berny_solver"}:
        return berny_solver.kernel, "berny"
    raise ValueError(f"unknown geomopt solver {name!r}")


def optimize_dft(
    mol: Any,
    *,
    solver: str,
    maxsteps: int,
    desc: str,
    dm0: Any | None = None,
) -> tuple[Any, Any, bool]:
    mf = run_scf(make_dft(mol), dm0=dm0)
    engine, _engine_name = _opt_engine(solver)
    pbar = tqdm(total=maxsteps, desc=desc, leave=False, unit="step")

    def _cb(_locals: dict[str, Any]) -> None:
        pbar.update(1)

    try:
        conv, mol_eq = engine(
            mf,
            callback=_cb,
            maxsteps=maxsteps,
            assert_convergence=False,
        )
    except TypeError:
        conv, mol_eq = engine(mf, callback=_cb, maxsteps=maxsteps)
    finally:
        pbar.close()
    mol_eq.verbose = mol.verbose
    mol_eq.max_memory = mol.max_memory
    mol_eq.symmetry = False
    mf_eq = run_scf(make_dft(mol_eq))
    return mol_eq, mf_eq, bool(conv)


def copy_mol_as(mol: Any, *, charge: int, spin: int) -> Any:
    atoms = mol_atoms_xyz(mol)
    return make_mol(
        atoms,
        charge=charge,
        spin=spin,
        verbose=mol.verbose,
        max_memory=mol.max_memory,
    )


def atoms_from_guess(guess: MoleculeGuess) -> list[list[Any]]:
    return [[sym, list(xyz)] for sym, xyz, _role in guess.atoms]


def split_optimized_fragments(
    mol: Any,
    guess: MoleculeGuess,
) -> tuple[list[list[Any]], list[list[Any]]]:
    coords = mol.atom_coords(unit="Angstrom")
    if mol.natm != len(guess.atoms):
        raise RuntimeError("atom count changed during opt; cannot split R / Sn(OH)₃")
    r_atoms: list[list[Any]] = []
    sn_atoms: list[list[Any]] = []
    for i, (sym, _xyz0, role) in enumerate(guess.atoms):
        xyz = [float(x) for x in coords[i]]
        if role == ROLE_LIGAND:
            r_atoms.append([sym, xyz])
        elif role in {ROLE_SN, ROLE_OH_O, ROLE_OH_H}:
            sn_atoms.append([sym, xyz])
        else:
            raise RuntimeError(f"unknown role {role}")
    return r_atoms, sn_atoms


def _lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def with_file_lock(path: Path, fn: Callable[[], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(path)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            fn()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def merge_scores(out_dir: Path) -> dict[str, Any]:
    ligands: dict[str, Any] = {}
    for score_path in sorted(out_dir.glob("*/score.json")):
        try:
            payload = json.loads(score_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        lig = str(payload.get("ligand") or score_path.parent.name)
        ligands[lig] = payload
    doc = {
        "label": "ours",
        "not_matgenq_table1": True,
        "forbidden_gqe_bde_claim": FORBIDDEN_GQE_BDE,
        "basis": BASIS,
        "ecp": dict(ECP),
        "functional": FUNCTIONAL,
        "correlated": "MP2_single_point_at_DFT_geom",
        "out_dir": str(out_dir),
        "first_wave": list(FIRST_WAVE),
        "unevaluated_pool": list(UNEVALUATED_POOL),
        "updated_at": _now_iso(),
        "ligands": ligands,
    }
    dump_json(out_dir / "scores.json", doc)
    return doc


def _ha_ev_kcal(delta_ha: float) -> dict[str, float]:
    return {
        "ha": float(delta_ha),
        "eV": float(delta_ha * HA_TO_EV),
        "kcal_mol": float(delta_ha * HA_TO_KCAL),
    }


def score_ligand(
    ligand_id: str,
    *,
    out_dir: Path,
    solver: str,
    maxsteps: int,
    verbose: int,
    skip_mp2: bool,
    write_guesses_only: bool,
) -> dict[str, Any]:
    guess = build_rsn_oh3(ligand_id)
    lig_dir = out_dir / ligand_id
    lig_dir.mkdir(parents=True, exist_ok=True)

    comment = guess.provenance
    write_xyz(lig_dir / "ours_starting_guess.xyz", atoms_from_guess(guess), comment)
    dump_json(
        lig_dir / "starting_guess.json",
        {
            "label": "ours",
            "not_matgenq_table1": True,
            "ligand": ligand_id,
            "name": guess.name,
            "smiles": guess.smiles,
            "provenance": guess.provenance,
            "basis": BASIS,
            "ecp": dict(ECP),
            "geometry": guess.cartesian(),
            "element_counts": guess.element_counts(),
            "written_at": _now_iso(),
        },
    )
    proxy = atomic_13p5nm_proxy(guess.element_counts())

    record: dict[str, Any] = {
        "label": "ours",
        "not_matgenq_table1": True,
        "ligand": ligand_id,
        "name": guess.name,
        "smiles": guess.smiles,
        "basis": BASIS,
        "ecp": dict(ECP),
        "functional": FUNCTIONAL,
        "optimizer": solver,
        "provenance": guess.provenance,
        "written_at": _now_iso(),
        "element_counts": guess.element_counts(),
        "cxro_proxy_13p5nm": proxy,
        "forbidden_gqe_bde_claim": FORBIDDEN_GQE_BDE,
        "status": "guesses_only" if write_guesses_only else "running",
    }
    dump_json(lig_dir / "score.json", record)
    if write_guesses_only:
        return record

    tmp = lig_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    pyscf_param.TMPDIR = str(tmp)

    max_memory = _parse_mem_mb()
    stages = [
        "neutral_dft_opt",
        "cation_vertical_dft",
        "cation_dft_opt",
        "fragments",
        "mp2",
    ]
    stage_bar = tqdm(stages, desc=f"{ligand_id} stages", leave=True)

    mol0 = make_mol(
        atoms_from_guess(guess),
        charge=0,
        spin=0,
        verbose=verbose,
        max_memory=max_memory,
    )
    stage_bar.set_postfix_str("opt 0")
    mol0_eq, mf0, conv0 = optimize_dft(
        mol0, solver=solver, maxsteps=maxsteps, desc=f"{ligand_id} opt0"
    )
    e0 = float(mf0.e_tot)
    write_xyz(
        lig_dir / f"{guess.name}_neutral.xyz",
        mol_atoms_xyz(mol0_eq),
        f"labelled-ours B3LYP/{BASIS} opt; charge=0 singlet; NOT MatGen-Q; ligand={ligand_id}",
    )
    stage_bar.update(1)

    mol_vert = copy_mol_as(mol0_eq, charge=1, spin=1)
    stage_bar.set_postfix_str("vert +1")
    mf_vert = run_scf(
        make_dft(mol_vert),
        dm0=_closed_to_open_dm(mf0.make_rdm1()),
    )
    e_vert = float(mf_vert.e_tot)
    stage_bar.update(1)

    mol1 = copy_mol_as(mol0_eq, charge=1, spin=1)
    stage_bar.set_postfix_str("opt +1")
    mol1_eq, mf1, conv1 = optimize_dft(
        mol1,
        solver=solver,
        maxsteps=maxsteps,
        desc=f"{ligand_id} opt+1",
        dm0=mf_vert.make_rdm1(),
    )
    e1 = float(mf1.e_tot)
    write_xyz(
        lig_dir / f"{guess.name}_cation.xyz",
        mol_atoms_xyz(mol1_eq),
        f"labelled-ours UKS-B3LYP/{BASIS} opt; charge=+1 doublet; NOT MatGen-Q; ligand={ligand_id}",
    )
    stage_bar.update(1)

    r_atoms, _sn_from_parent = split_optimized_fragments(mol0_eq, guess)
    stage_bar.set_postfix_str("R• / Sn(OH)3")
    mol_r = make_mol(r_atoms, charge=0, spin=1, verbose=verbose, max_memory=max_memory)
    mol_r_eq, mf_r, conv_r = optimize_dft(
        mol_r, solver=solver, maxsteps=maxsteps, desc=f"{ligand_id} R•"
    )
    e_r = float(mf_r.e_tot)
    write_xyz(
        lig_dir / f"{ligand_id}_R_rad_ours.xyz",
        mol_atoms_xyz(mol_r_eq),
        f"labelled-ours UKS-B3LYP R• fragment; ligand={ligand_id}",
    )

    sn_guess = build_sn_oh3_fragment()
    shared = out_dir / "_shared"
    shared.mkdir(parents=True, exist_ok=True)
    sn_rad_xyz = shared / "sn_oh3_rad_ours.xyz"
    sn_cat_xyz = shared / "sn_oh3_cation_ours.xyz"
    sn_meta = shared / "sn_oh3_fragments.json"
    sn_state: dict[str, Any] = {}

    def _ensure_sn_fragments() -> None:
        if sn_rad_xyz.is_file() and sn_cat_xyz.is_file() and sn_meta.is_file():
            sn_state.update(json.loads(sn_meta.read_text(encoding="utf-8")))
            return
        mol_sn = make_mol(
            atoms_from_guess(sn_guess),
            charge=0,
            spin=1,
            verbose=verbose,
            max_memory=max_memory,
        )
        mol_sn_eq, mf_sn, conv_rad = optimize_dft(
            mol_sn, solver=solver, maxsteps=maxsteps, desc="Sn(OH)3•"
        )
        e_rad = float(mf_sn.e_tot)
        write_xyz(
            sn_rad_xyz,
            mol_atoms_xyz(mol_sn_eq),
            "labelled-ours UKS-B3LYP •Sn(OH)3; NOT MatGen-Q",
        )
        mol_sn_p = copy_mol_as(mol_sn_eq, charge=1, spin=0)
        mol_sn_p_eq, mf_sn_p, conv_cat = optimize_dft(
            mol_sn_p, solver=solver, maxsteps=maxsteps, desc="Sn(OH)3+"
        )
        e_cat = float(mf_sn_p.e_tot)
        write_xyz(
            sn_cat_xyz,
            mol_atoms_xyz(mol_sn_p_eq),
            "labelled-ours RKS-B3LYP Sn(OH)3+; NOT MatGen-Q",
        )
        meta = {
            "label": "ours",
            "not_matgenq_table1": True,
            "e_sn_oh3_rad_ha": e_rad,
            "e_sn_oh3_cation_ha": e_cat,
            "opt_converged_rad": bool(conv_rad),
            "opt_converged_cation": bool(conv_cat),
            "basis": BASIS,
            "ecp": dict(ECP),
            "functional": FUNCTIONAL,
            "written_at": _now_iso(),
        }
        dump_json(sn_meta, meta)
        sn_state.update(meta)

    with_file_lock(sn_meta, _ensure_sn_fragments)
    e_sn_rad = float(sn_state["e_sn_oh3_rad_ha"])
    e_sn_cat = float(sn_state["e_sn_oh3_cation_ha"])
    conv_sn_rad = bool(sn_state.get("opt_converged_rad", False))
    conv_sn_cat = bool(sn_state.get("opt_converged_cation", False))
    stage_bar.update(1)

    mp2_0 = mp2_1 = mp2_vert = None
    if not skip_mp2:
        stage_bar.set_postfix_str("MP2")
        p_mp2 = tqdm(total=3, desc=f"{ligand_id} MP2", leave=False)
        mp2_0 = run_mp2_at_geom(mol0_eq, p_mp2)
        p_mp2.update(1)
        mp2_1 = run_mp2_at_geom(mol1_eq, p_mp2)
        p_mp2.update(1)
        mp2_vert = run_mp2_at_geom(copy_mol_as(mol0_eq, charge=1, spin=1), p_mp2)
        p_mp2.update(1)
        p_mp2.close()
    stage_bar.update(1)
    stage_bar.close()

    vip_dft = e_vert - e0
    aip_dft = e1 - e0
    bde0 = e_r + e_sn_rad - e0
    bde1 = e_r + e_sn_cat - e1
    delta_bde = bde0 - bde1

    ip_block: dict[str, Any] = {
        "label": "ours",
        "vertical_dft": _ha_ev_kcal(vip_dft),
        "adiabatic_dft": _ha_ev_kcal(aip_dft),
        "definitions": {
            "vertical": "E(+1 at 0-geom) − E(0 opt)",
            "adiabatic": "E(+1 opt) − E(0 opt)",
        },
    }
    if mp2_0 and mp2_1 and mp2_vert:
        ip_block["vertical_mp2"] = _ha_ev_kcal(float(mp2_vert["mp2_ha"]) - float(mp2_0["mp2_ha"]))
        ip_block["adiabatic_mp2"] = _ha_ev_kcal(float(mp2_1["mp2_ha"]) - float(mp2_0["mp2_ha"]))

    record.update(
        {
            "status": "scored",
            "opt_converged": {"neutral": bool(conv0), "cation": bool(conv1), "R_rad": bool(conv_r)},
            "energies_ha": {
                "neutral_dft_opt": e0,
                "cation_dft_opt": e1,
                "cation_vertical_dft": e_vert,
                "R_rad_dft": e_r,
                "sn_oh3_rad_dft": e_sn_rad,
                "sn_oh3_cation_dft": e_sn_cat,
                "neutral_mp2": None if mp2_0 is None else mp2_0,
                "cation_mp2": None if mp2_1 is None else mp2_1,
                "cation_vertical_mp2": None if mp2_vert is None else mp2_vert,
            },
            "ip": ip_block,
            "bde": {
                "label": "ours",
                "level": "B3LYP/def2-svp",
                "formula_neutral": "E(R•)+E(•Sn(OH)3)−E(RSn(OH)3)",
                "formula_cation": "E(R•)+E(Sn(OH)3+)−E(RSn(OH)3+)",
                "cation_channel": "homolytic Sn–C, charge remains on Sn(OH)3+",
                "bde_neutral": _ha_ev_kcal(bde0),
                "bde_cation": _ha_ev_kcal(bde1),
                "delta_bde": _ha_ev_kcal(delta_bde),
                "delta_bde_definition": "BDE(0)−BDE(+1)",
                "forbidden_gqe_bde_claim": FORBIDDEN_GQE_BDE,
                "sn_oh3_fragment_converged": {
                    "rad": bool(conv_sn_rad),
                    "cation": bool(conv_sn_cat),
                },
            },
            "paths": {
                "starting_guess": str(lig_dir / "ours_starting_guess.xyz"),
                "neutral_xyz": str(lig_dir / f"{guess.name}_neutral.xyz"),
                "cation_xyz": str(lig_dir / f"{guess.name}_cation.xyz"),
            },
            "written_at": _now_iso(),
        }
    )
    dump_json(lig_dir / "score.json", record)
    dump_json(
        lig_dir / "provenance.json",
        {
            "label": "ours",
            "not_matgenq_table1": True,
            "ligand": ligand_id,
            "smiles": guess.smiles,
            "functional": FUNCTIONAL,
            "basis": BASIS,
            "ecp": dict(ECP),
            "charge_neutral": 0,
            "multiplicity_neutral": 1,
            "charge_cation": 1,
            "multiplicity_cation": 2,
            "mp2": "single-point at DFT geom",
            "date": _now_iso(),
            "forbidden_gqe_bde_claim": FORBIDDEN_GQE_BDE,
        },
    )
    return record


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    threads = int(payload.get("omp_threads") or 1)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    try:
        rec = score_ligand(
            payload["ligand"],
            out_dir=Path(payload["out_dir"]),
            solver=str(payload["solver"]),
            maxsteps=int(payload["maxsteps"]),
            verbose=int(payload["verbose"]),
            skip_mp2=bool(payload["skip_mp2"]),
            write_guesses_only=bool(payload["write_guesses_only"]),
        )
        rec["_ok"] = True
        rec["_error"] = None
        return rec
    except Exception as exc:
        rec = {
            "label": "ours",
            "ligand": payload["ligand"],
            "status": "failed",
            "_ok": False,
            "_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "written_at": _now_iso(),
            "forbidden_gqe_bde_claim": FORBIDDEN_GQE_BDE,
        }
        lig_dir = Path(payload["out_dir"]) / str(payload["ligand"])
        dump_json(lig_dir / "score.json", rec)
        return rec


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligands", type=str, default=",".join(FIRST_WAVE))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--solver", type=str, default="geometric", choices=("geometric", "berny"))
    parser.add_argument("--max-opt-steps", type=int, default=80)
    parser.add_argument("--verbose", type=int, default=3)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--skip-mp2", action="store_true")
    parser.add_argument("--write-guesses-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    ligands = parse_ligand_list(args.ligands)
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_root = (REPO_ROOT / "results" / "tin_ab" / "methyltin_ours").resolve()
    if allowed_root not in out_dir.parents and out_dir != allowed_root:
        # Permit the canonical tree only (or a subdir of it).
        if not str(out_dir).startswith(str(allowed_root)):
            raise SystemExit(
                f"refusing out-dir {out_dir}: labelled-ours output must live under {allowed_root}"
            )

    n_workers = max(1, int(args.n_workers))
    if args.write_guesses_only:
        n_workers = 1
    cpu = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    omp = max(1, cpu // n_workers)
    os.environ.setdefault("OMP_NUM_THREADS", str(omp if n_workers == 1 else omp))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

    jobs = [
        {
            "ligand": lig,
            "out_dir": str(out_dir),
            "solver": args.solver,
            "maxsteps": int(args.max_opt_steps),
            "verbose": int(args.verbose),
            "skip_mp2": bool(args.skip_mp2),
            "write_guesses_only": bool(args.write_guesses_only),
            "omp_threads": omp,
        }
        for lig in ligands
    ]

    results: list[dict[str, Any]]
    if n_workers == 1:
        results = [_worker(job) for job in tqdm(jobs, desc="ligands")]
    else:
        ctx = get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            results = list(
                tqdm(pool.imap(_worker, jobs), total=len(jobs), desc="ligands")
            )

    with_file_lock(out_dir / "scores.json", lambda: merge_scores(out_dir))
    failed = [r for r in results if not r.get("_ok", False) and r.get("status") == "failed"]
    if failed and args.fail_fast:
        return 1
    if failed and len(failed) == len(results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
