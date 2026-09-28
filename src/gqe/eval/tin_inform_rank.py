#!/usr/bin/env python3
"""Pick half of the labelled-ours R–Sn(OH)₃ materials-informatics loop.

Reads results/tin_ab/methyltin_ours/scores.json and per-ligand */score.json
from src/gqe/eval/tin_inform_dft.py. Completes ranking + next-ligand pick.

Does **not** claim a GQE energy, does **not** copy MatGen-Q Table 1, and does
**not** treat 72.6→21.2 kcal/mol or a 13.5 nm many-body absorption spectrum
as something this script computed.

Ranking of scored ligands (lexicographic):
  1. larger ΔBDE = BDE(0) − BDE(+1)  (kcal/mol)
  2. smaller adiabatic IP            (eV)
  3. larger CXRO 13.5 nm f2 proxy    (atomic sum, not σ at 92 eV)

If ≥3 scored rows exist, a numpy-only RBF GP (no sklearn) maps
[n_C, n_F, is_aryl, is_alkenyl, adiabatic_IP, BDE0] → ΔBDE and the next
unevaluated ligand (i-Pr, n-Pr, allyl, CF3) maximises expected improvement.
If the GP is too shaky: CF3 if electron-withdrawing is untested, else allyl.

CLI (repo root)::

    PYTHONPATH=. python src/gqe/eval/tin_inform_rank.py \\
        --scores results/tin_ab/methyltin_ours/scores.json \\
        --out-dir results/tin_ab/methyltin_ours
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gqe.data.rsn_oh3_templates import (
    EXPECTED_COUNTS,
    FIRST_WAVE,
    LIGAND_SMILES,
    UNEVALUATED_POOL,
    atomic_13p5nm_proxy,
    build_rsn_oh3,
)

DEFAULT_OUT = Path("results/tin_ab/methyltin_ours")
DEFAULT_SCORES = DEFAULT_OUT / "scores.json"
FEATURE_NAMES = ("n_C", "n_F", "is_aryl", "is_alkenyl", "adiabatic_IP", "BDE0")
ARYL = frozenset({"ph"})
ALKENYL = frozenset({"vinyl", "allyl"})
WITHDRAWING = frozenset({"cf3"})
DISPLAY = {
    "me": "Me",
    "et": "Et",
    "nbu": "n-Bu",
    "vinyl": "vinyl",
    "ph": "Ph",
    "ipr": "i-Pr",
    "npr": "n-Pr",
    "allyl": "allyl",
    "cf3": "CF3",
}
PALETTE = {
    "alkyl": "#0072B2",
    "aryl": "#E69F00",
    "alkenyl": "#009E73",
    "withdrawing": "#D55E00",
    "other": "#666666",
}
FORBIDDEN_GQE_BDE = (
    "Do not write 72.6→21.2 kcal/mol as a GQE number; that is classical "
    "UCCSD(T) in arXiv:2607.23988, not this labelled-ours B3LYP ΔBDE."
)
DISCLAIMER = (
    "Labelled-ours classical DFT ranking. Not a GQE energy. "
    "Not MatGen-Q Table 1. Atomic 13.5 nm values are a Henke/CXRO f2 sum, "
    "not a many-body absorption spectrum."
)
CHEMICAL_NEXT_RULE = (
    "Chemical next-ligand rule (used when the 6-D RBF GP is not fit or is "
    "too shaky to trust expected improvement): (1) pick CF3 if it is still "
    "unevaluated and no scored ligand is electron-withdrawing (n_F>0 or "
    "id=cf3); (2) else pick allyl if unevaluated (alkenyl radical "
    "stabilization after Sn–C cleavage); (3) else i-Pr, then n-Pr. "
    "Classical chemistry heuristic, not a GQE policy."
)


def apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 1.15,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "#000000",
            "legend.framealpha": 1.0,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavuserif",
        }
    )


def bold_legend(ax: Any, **kwargs: Any) -> Any:
    leg = ax.legend(**kwargs)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    title = leg.get_title()
    if title is not None:
        title.set_fontweight("bold")
    return leg


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _nested_float(block: Any, *keys: str) -> float | None:
    cur: Any = block
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    if cur is None or isinstance(cur, Mapping):
        return None
    try:
        val = float(cur)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def display_name(ligand_id: str) -> str:
    return DISPLAY.get(ligand_id.lower(), ligand_id)


def chem_class(ligand_id: str, n_f: float) -> str:
    key = ligand_id.lower()
    if key in WITHDRAWING or n_f > 0:
        return "withdrawing"
    if key in ARYL:
        return "aryl"
    if key in ALKENYL:
        return "alkenyl"
    return "alkyl"


def structural_features(ligand_id: str, counts: Mapping[str, Any] | None = None) -> dict[str, float]:
    key = ligand_id.lower()
    src = dict(counts or EXPECTED_COUNTS.get(key) or {})
    n_c = float(src.get("C") or 0)
    n_f = float(src.get("F") or 0)
    return {
        "n_C": n_c,
        "n_F": n_f,
        "is_aryl": 1.0 if key in ARYL else 0.0,
        "is_alkenyl": 1.0 if key in ALKENYL else 0.0,
    }


E0_OVERLAY_KEYS = (
    "e0_casci_ha",
    "e0_qsci_ha",
    "delta_e_qsci_mha",
    "e0_n_qubits",
    "e0_ncas",
    "e0_nelecas",
    "qsci_sampler",
    "e0_qsci_control_entangled_hf_ha",
    "delta_e_qsci_control_entangled_hf_mha",
)


def overlay_e0_fields(payload: dict[str, Any], e0: Mapping[str, Any]) -> None:
    """Copy optional E0 onto a ligand score. Primary QSCI is pool-native UCCSD.

    Top-level e0.json keys already prefer pool-native after ingest. Nested
    ``qsci_control_entangled_hf`` is a named control, never the rank overlay.
    """
    charge0 = (e0.get("charges") or {}).get("0")
    if not isinstance(charge0, Mapping):
        charge0 = (e0.get("charges") or {}).get(0)
    src = charge0 if isinstance(charge0, Mapping) else {}
    sampler = src.get("qsci_sampler") or e0.get("qsci_sampler")
    payload["e0_casci_ha"] = e0.get("e0_casci_ha")
    if payload["e0_casci_ha"] is None:
        payload["e0_casci_ha"] = src.get("e0_casci_ha")
    payload["e0_qsci_ha"] = e0.get("e0_qsci_ha")
    if payload["e0_qsci_ha"] is None:
        payload["e0_qsci_ha"] = src.get("e0_qsci_ha")
    payload["delta_e_qsci_mha"] = e0.get("delta_e_qsci_mha")
    if payload["delta_e_qsci_mha"] is None:
        payload["delta_e_qsci_mha"] = src.get("delta_e_qsci_mha")
    payload["e0_n_qubits"] = e0.get("e0_n_qubits", src.get("n_qubits"))
    payload["e0_ncas"] = e0.get("e0_ncas", src.get("ncas"))
    payload["e0_nelecas"] = e0.get("e0_nelecas", src.get("nelecas"))
    if sampler is not None:
        payload["qsci_sampler"] = sampler
    ctrl = src.get("qsci_control_entangled_hf") if isinstance(src, Mapping) else None
    if isinstance(ctrl, Mapping):
        if ctrl.get("e0_qsci_ha") is not None:
            payload["e0_qsci_control_entangled_hf_ha"] = ctrl["e0_qsci_ha"]
        if ctrl.get("delta_e_qsci_mha") is not None:
            payload["delta_e_qsci_control_entangled_hf_mha"] = ctrl["delta_e_qsci_mha"]


def merge_ligand_scores(out_dir: Path) -> dict[str, Any]:
    ligands: dict[str, Any] = {}
    paths = sorted(out_dir.glob("*/score.json"))
    for score_path in tqdm(paths, desc="Per-ligand JSON", unit="file", disable=len(paths) < 2):
        try:
            payload = json.loads(score_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        lig = str(payload.get("ligand") or score_path.parent.name).lower()
        e0_path = score_path.parent / "e0.json"
        if e0_path.is_file():
            try:
                e0 = json.loads(e0_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                e0 = None
            if isinstance(e0, Mapping):
                overlay_e0_fields(payload, e0)
        ligands[lig] = payload
    return {
        "label": "ours",
        "not_matgenq_table1": True,
        "forbidden_gqe_bde_claim": FORBIDDEN_GQE_BDE,
        "out_dir": str(out_dir),
        "first_wave": list(FIRST_WAVE),
        "unevaluated_pool": list(UNEVALUATED_POOL),
        "updated_at": _now_iso(),
        "ligands": ligands,
        "merged_from_per_ligand": True,
    }


def load_scores(scores_path: Path, out_dir: Path) -> dict[str, Any]:
    per_ligand = merge_ligand_scores(out_dir)
    if scores_path.is_file():
        try:
            doc = json.loads(scores_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse {scores_path}: {exc}") from exc
        if not isinstance(doc, dict):
            doc = {"ligands": doc}
        ligands = doc.get("ligands")
        if isinstance(ligands, list):
            as_dict: dict[str, Any] = {}
            for item in ligands:
                if isinstance(item, Mapping) and item.get("ligand"):
                    as_dict[str(item["ligand"]).lower()] = dict(item)
            doc["ligands"] = as_dict
            ligands = as_dict
        if not isinstance(ligands, dict):
            doc["ligands"] = {}
            ligands = doc["ligands"]
        for key, rec in per_ligand.get("ligands", {}).items():
            if key not in ligands or not _is_scored(ligands[key]):
                ligands[key] = rec
                continue
            kept = ligands[key]
            if isinstance(kept, dict) and isinstance(rec, Mapping):
                for e0_key in E0_OVERLAY_KEYS:
                    if rec.get(e0_key) is not None:
                        kept[e0_key] = rec[e0_key]
        doc["ligands"] = ligands
        doc.setdefault("label", "ours")
        doc.setdefault("not_matgenq_table1", True)
        return doc
    if per_ligand.get("ligands"):
        return per_ligand
    raise FileNotFoundError(
        f"Missing scores file: {scores_path}\n"
        "The pick half needs classical ligand scores from the score half "
        "(src/gqe/eval/tin_inform_dft.py).\n"
        f"Expected JSON: {REPO_ROOT / DEFAULT_SCORES}\n"
        f"Also looked for per-ligand */score.json under {out_dir}.\n"
        "Refusing to invent ΔBDE / IP numbers. "
        + FORBIDDEN_GQE_BDE
    )


def _delta_bde_kcal(rec: Mapping[str, Any]) -> float | None:
    val = _nested_float(rec, "bde", "delta_bde", "kcal_mol")
    if val is not None:
        return val
    val = _nested_float(rec, "delta_bde_kcal")
    if val is not None:
        return val
    b0 = _bde0_kcal(rec)
    b1 = _nested_float(rec, "bde", "bde_cation", "kcal_mol")
    if b0 is not None and b1 is not None:
        return b0 - b1
    return None


def _bde0_kcal(rec: Mapping[str, Any]) -> float | None:
    val = _nested_float(rec, "bde", "bde_neutral", "kcal_mol")
    if val is not None:
        return val
    return _nested_float(rec, "bde_neutral_kcal")


def _adiabatic_ip_eV(rec: Mapping[str, Any]) -> float | None:
    val = _nested_float(rec, "ip", "adiabatic_dft", "eV")
    if val is not None:
        return val
    val = _nested_float(rec, "ip", "adiabatic_mp2", "eV")
    if val is not None:
        return val
    return _nested_float(rec, "adiabatic_ip_eV")


def _proxy(rec: Mapping[str, Any], ligand_id: str) -> float | None:
    val = _nested_float(rec, "cxro_proxy_13p5nm", "weighted_sum_f2")
    if val is not None:
        return val
    val = _nested_float(rec, "proxy_13p5nm")
    if val is not None:
        return val
    counts = rec.get("element_counts")
    if isinstance(counts, Mapping) and counts:
        return float(atomic_13p5nm_proxy(counts)["weighted_sum_f2"])
    expected = EXPECTED_COUNTS.get(ligand_id.lower())
    if expected:
        return float(atomic_13p5nm_proxy(expected)["weighted_sum_f2"])
    return None


def _is_scored(rec: Mapping[str, Any] | None) -> bool:
    if not isinstance(rec, Mapping):
        return False
    if rec.get("status") in {"failed", "running", "guesses_only"}:
        return False
    return _delta_bde_kcal(rec) is not None


def flatten_row(ligand_id: str, rec: Mapping[str, Any]) -> dict[str, Any]:
    key = ligand_id.lower()
    counts = rec.get("element_counts") if isinstance(rec.get("element_counts"), Mapping) else None
    feats = structural_features(key, counts)
    n_f = feats["n_F"]
    return {
        "ligand": key,
        "display": display_name(key),
        "smiles": rec.get("smiles") or LIGAND_SMILES.get(key),
        "chem_class": chem_class(key, n_f),
        "status": rec.get("status"),
        "n_C": feats["n_C"],
        "n_F": n_f,
        "is_aryl": feats["is_aryl"],
        "is_alkenyl": feats["is_alkenyl"],
        "delta_bde_kcal": _delta_bde_kcal(rec),
        "bde0_kcal": _bde0_kcal(rec),
        "bde_cation_kcal": _nested_float(rec, "bde", "bde_cation", "kcal_mol"),
        "adiabatic_ip_eV": _adiabatic_ip_eV(rec),
        "proxy_13p5nm": _proxy(rec, key),
        "e0_casci_ha": _nested_float(rec, "e0_casci_ha"),
        "e0_qsci_ha": _nested_float(rec, "e0_qsci_ha"),
        "delta_e_qsci_mha": _nested_float(rec, "delta_e_qsci_mha"),
        "qsci_sampler": rec.get("qsci_sampler"),
        "e0_qsci_control_entangled_hf_ha": _nested_float(
            rec, "e0_qsci_control_entangled_hf_ha"
        ),
        "delta_e_qsci_control_entangled_hf_mha": _nested_float(
            rec, "delta_e_qsci_control_entangled_hf_mha"
        ),
        "e0_n_qubits": rec.get("e0_n_qubits"),
        "e0_ncas": rec.get("e0_ncas"),
        "label": rec.get("label", "ours"),
    }


def fingerprint(row: Mapping[str, Any]) -> np.ndarray:
    return np.array(
        [
            row["n_C"],
            row["n_F"],
            row["is_aryl"],
            row["is_alkenyl"],
            np.nan if row.get("adiabatic_ip_eV") is None else row["adiabatic_ip_eV"],
            np.nan if row.get("bde0_kcal") is None else row["bde0_kcal"],
        ],
        dtype=float,
    )


def rank_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scored = [dict(r) for r in rows if r.get("delta_bde_kcal") is not None]
    scored.sort(
        key=lambda r: (
            -float(r["delta_bde_kcal"]),
            np.inf if r.get("adiabatic_ip_eV") is None else float(r["adiabatic_ip_eV"]),
            -np.inf if r.get("proxy_13p5nm") is None else -float(r["proxy_13p5nm"]),
        )
    )
    for i, row in enumerate(tqdm(scored, desc="Rank ligands", unit="ligand", disable=len(scored) < 3), start=1):
        row["rank"] = i
    return scored


def unevaluated_rows(
    scored_ids: Sequence[str],
    *,
    failed_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    have = {k.lower() for k in scored_ids}
    failed = {k.lower() for k in (failed_ids or [])}
    out: list[dict[str, Any]] = []
    for key in tqdm(list(UNEVALUATED_POOL), desc="Unevaluated pool", unit="ligand", disable=False):
        if key in have or key in failed:
            continue
        guess = build_rsn_oh3(key)
        counts = guess.element_counts()
        feats = structural_features(key, counts)
        out.append(
            {
                "ligand": key,
                "display": display_name(key),
                "smiles": guess.smiles,
                "chem_class": chem_class(key, feats["n_F"]),
                "n_C": feats["n_C"],
                "n_F": feats["n_F"],
                "is_aryl": feats["is_aryl"],
                "is_alkenyl": feats["is_alkenyl"],
                "delta_bde_kcal": None,
                "bde0_kcal": None,
                "adiabatic_ip_eV": None,
                "proxy_13p5nm": float(atomic_13p5nm_proxy(counts)["weighted_sum_f2"]),
                "ip_imputed": False,
                "imputation_quality": None,
            }
        )
    return out


def impute_ip_bde0(candidate: dict[str, Any], scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    same = [
        r
        for r in scored
        if r.get("chem_class") == candidate["chem_class"]
        and r.get("adiabatic_ip_eV") is not None
        and r.get("bde0_kcal") is not None
    ]
    any_complete = [
        r
        for r in scored
        if r.get("adiabatic_ip_eV") is not None and r.get("bde0_kcal") is not None
    ]
    if same:
        quality, src = "same_class", same
    elif any_complete:
        quality, src = "global_mean", any_complete
    else:
        candidate["imputation_quality"] = "none"
        return candidate
    candidate["adiabatic_ip_eV"] = float(np.mean([float(r["adiabatic_ip_eV"]) for r in src]))
    candidate["bde0_kcal"] = float(np.mean([float(r["bde0_kcal"]) for r in src]))
    candidate["ip_imputed"] = True
    candidate["imputation_quality"] = quality
    return candidate


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, lengthscale: float, variance: float) -> np.ndarray:
    delta = x1[:, None, :] - x2[None, :, :]
    sqdist = np.sum(delta * delta, axis=-1)
    scale = max(lengthscale, 1e-8) ** 2
    return variance * np.exp(-0.5 * sqdist / scale)


def _chol_solve(k_xx: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    jitter = 1e-10
    eye = np.eye(k_xx.shape[0])
    for _ in range(8):
        try:
            chol = np.linalg.cholesky(k_xx + jitter * eye)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        return None
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    return chol, alpha


def gp_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    lengthscale: float,
    signal_var: float,
    noise_var: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    k_xx = rbf_kernel(x_train, x_train, lengthscale, signal_var) + noise_var * np.eye(len(x_train))
    solved = _chol_solve(k_xx, y_train)
    if solved is None:
        return None
    chol, alpha = solved
    k_s = rbf_kernel(x_train, x_test, lengthscale, signal_var)
    k_ss = rbf_kernel(x_test, x_test, lengthscale, signal_var)
    mu = k_s.T @ alpha
    v = np.linalg.solve(chol, k_s)
    var = np.diag(k_ss) - np.sum(v * v, axis=0)
    return mu, np.maximum(var, 0.0)


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std, mean, std


def norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def norm_cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    scale = math.sqrt(2.0)
    flat = z.reshape(-1)
    out = np.empty(flat.shape, dtype=float)
    for i, value in enumerate(flat):
        out[i] = 0.5 * (1.0 + math.erf(float(value) / scale))
    return out.reshape(z.shape)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.05) -> np.ndarray:
    """EI for maximizing ΔBDE (kcal/mol)."""
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    ei = np.zeros_like(mu)
    tiny = sigma < 1e-12
    ei[tiny] = np.maximum(mu[tiny] - best - xi, 0.0)
    ok = ~tiny
    z = (mu[ok] - best - xi) / sigma[ok]
    ei[ok] = (mu[ok] - best - xi) * norm_cdf(z) + sigma[ok] * norm_pdf(z)
    return np.maximum(ei, 0.0)


def gp_loo_r2(
    x: np.ndarray,
    y: np.ndarray,
    lengthscale: float,
    signal_var: float,
    noise_var: float,
) -> tuple[float, float]:
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pred = gp_predict(x[mask], y[mask], x[i : i + 1], lengthscale, signal_var, noise_var)
        if pred is None:
            continue
        preds[i] = pred[0][0]
    good = np.isfinite(preds)
    if not np.any(good):
        return -np.inf, np.inf
    resid = y[good] - preds[good]
    rmse = float(np.sqrt(np.mean(resid * resid)))
    ss_tot = float(np.sum((y[good] - np.mean(y[good])) ** 2))
    r2 = 1.0 - float(np.sum(resid * resid)) / ss_tot if ss_tot > 1e-18 else 0.0
    return r2, rmse


def chemical_next(unevaluated: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]]) -> tuple[str | None, str]:
    names = [r["ligand"] for r in unevaluated]
    if not names:
        return None, "empty_unevaluated_pool"
    withdrawing_tested = any(
        float(r.get("n_F") or 0) > 0 or r.get("chem_class") == "withdrawing" for r in scored
    )
    if "cf3" in names and not withdrawing_tested:
        return "cf3", "chemical_withdrawing_untested"
    if "allyl" in names:
        return "allyl", "chemical_alkenyl_next"
    for fallback in ("ipr", "npr"):
        if fallback in names:
            return fallback, f"chemical_fallback_{fallback}"
    return names[0], "chemical_pool_order"


def fit_surrogate_and_pick(
    scored: Sequence[Mapping[str, Any]],
    unevaluated: Sequence[dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    chem_name, chem_tag = chemical_next(unevaluated, scored)
    result: dict[str, Any] = {
        "seed": int(seed),
        "fingerprint": list(FEATURE_NAMES),
        "n_scored_for_gp": 0,
        "gp_attempted": False,
        "gp_shaky": True,
        "gp_shaky_reasons": [],
        "lengthscale": None,
        "loo_r2": None,
        "loo_rmse_kcal": None,
        "candidates": [],
        "picked_ligand": chem_name,
        "picked_display": None if chem_name is None else display_name(chem_name),
        "smiles": None if chem_name is None else LIGAND_SMILES.get(chem_name),
        "pick_tag": chem_tag,
        "rule": "chemical",
        "rule_documentation": CHEMICAL_NEXT_RULE,
        "chemical_fallback": {"ligand": chem_name, "tag": chem_tag},
    }
    gp_rows = [
        r
        for r in scored
        if r.get("delta_bde_kcal") is not None
        and r.get("adiabatic_ip_eV") is not None
        and r.get("bde0_kcal") is not None
    ]
    result["n_scored_for_gp"] = len(gp_rows)
    if not unevaluated:
        result["gp_shaky_reasons"].append("no_unevaluated_ligands")
        result["picked_ligand"] = None
        result["picked_display"] = None
        result["pick_tag"] = "pool_exhausted"
        return result
    if len(gp_rows) < 3:
        result["gp_shaky_reasons"].append(f"n_scored={len(gp_rows)}<3; numpy RBF GP not fit")
        return result

    result["gp_attempted"] = True
    x_raw = np.vstack([fingerprint(r) for r in gp_rows])
    y = np.array([float(r["delta_bde_kcal"]) for r in gp_rows], dtype=float)
    if not np.all(np.isfinite(x_raw)) or not np.all(np.isfinite(y)):
        result["gp_shaky_reasons"].append("non_finite_train_features")
        return result
    if float(np.std(y)) < 1e-8:
        result["gp_shaky_reasons"].append("zero_variance_delta_bde")
        return result

    x, mean, std = standardize(x_raw)
    signal_var = float(max(np.var(y, ddof=1), 1e-4))
    noise_var = float(max(1.0, 0.05 * signal_var))
    best_ls, best_rmse, best_r2 = 2.0, np.inf, -np.inf
    for ls in tqdm((0.5, 0.8, 1.2, 2.0, 3.5), desc="GP lengthscales", unit="ls"):
        r2, rmse = gp_loo_r2(x, y, ls, signal_var, noise_var)
        if rmse < best_rmse:
            best_rmse, best_r2, best_ls = rmse, r2, ls
    result["lengthscale"] = best_ls
    result["loo_r2"] = None if not math.isfinite(best_r2) else float(best_r2)
    result["loo_rmse_kcal"] = None if not math.isfinite(best_rmse) else float(best_rmse)
    result["signal_var"] = signal_var
    result["noise_var"] = noise_var

    reasons: list[str] = []
    if best_r2 < 0.10:
        reasons.append(f"loo_r2={best_r2:.3f}<0.10")
    if best_rmse > 1.5 * float(np.std(y)):
        reasons.append(
            f"loo_rmse={best_rmse:.3g} kcal/mol > 1.5×std(ΔBDE)={1.5 * float(np.std(y)):.3g}"
        )
    if len(gp_rows) <= len(FEATURE_NAMES):
        reasons.append(f"{len(gp_rows)} points in {len(FEATURE_NAMES)}-D; interpolator is underdetermined")

    candidates = [impute_ip_bde0(dict(r), gp_rows) for r in unevaluated]
    x_c_raw = np.vstack([fingerprint(r) for r in candidates])
    if not np.all(np.isfinite(x_c_raw)):
        reasons.append("candidate_fingerprint_has_nan_after_imputation")
        result["gp_shaky_reasons"] = reasons
        return result
    pred = gp_predict(x, y, (x_c_raw - mean) / std, best_ls, signal_var, noise_var)
    if pred is None:
        reasons.append("cholesky_failed")
        result["gp_shaky_reasons"] = reasons
        return result
    mu, var = pred
    sigma = np.sqrt(np.maximum(var, 0.0))
    best_y = float(np.max(y))
    ei = expected_improvement(mu, sigma, best_y, xi=0.05)
    if float(np.mean(sigma)) > 1.5 * float(np.std(y)):
        reasons.append(
            f"mean_pred_sigma={float(np.mean(sigma)):.3g} > 1.5×std(y)={1.5 * float(np.std(y)):.3g}"
        )
    if float(np.ptp(ei)) < 0.10 * max(float(np.mean(ei)), 1e-8):
        reasons.append("expected_improvement_nearly_flat")
    if any(c.get("imputation_quality") == "global_mean" for c in candidates):
        reasons.append("IP/BDE0 imputed across chemical class (global_mean)")

    cand_rows = []
    for rec, m, s, e in zip(candidates, mu, sigma, ei):
        cand_rows.append(
            {
                "ligand": rec["ligand"],
                "display": rec["display"],
                "chem_class": rec["chem_class"],
                "mu_delta_bde_kcal": float(m),
                "sigma_delta_bde_kcal": float(s),
                "expected_improvement": float(e),
                "imputation_quality": rec.get("imputation_quality"),
                "fingerprint": fingerprint(rec).tolist(),
            }
        )
    cand_rows.sort(key=lambda d: -d["expected_improvement"])
    result["candidates"] = cand_rows
    gp_pick = cand_rows[0]["ligand"] if cand_rows else chem_name
    result["gp_pick"] = gp_pick
    result["gp_shaky"] = bool(reasons)
    result["gp_shaky_reasons"] = reasons
    if reasons:
        result["picked_ligand"] = chem_name
        result["picked_display"] = None if chem_name is None else display_name(chem_name)
        result["smiles"] = None if chem_name is None else LIGAND_SMILES.get(chem_name)
        result["pick_tag"] = chem_tag
        result["rule"] = "chemical_because_gp_shaky"
        result["rule_documentation"] = (
            CHEMICAL_NEXT_RULE
            + " The GP was fit (numpy RBF, no sklearn) but rejected: "
            + "; ".join(reasons)
            + "."
        )
    else:
        result["picked_ligand"] = gp_pick
        result["picked_display"] = display_name(str(gp_pick))
        result["smiles"] = LIGAND_SMILES.get(str(gp_pick))
        result["pick_tag"] = "expected_improvement"
        result["rule"] = "expected_improvement"
        result["rule_documentation"] = (
            "6-D numpy RBF GP maps fingerprint "
            "[n_C, n_F, is_aryl, is_alkenyl, adiabatic_IP, BDE0] → ΔBDE "
            "(kcal/mol). Next ligand maximises expected improvement over the "
            f"best scored ΔBDE ({best_y:.3g} kcal/mol), ξ=0.05 kcal/mol. "
            "Lengthscale chosen by leave-one-out RMSE. Not a GQE policy."
        )
    return result


def plot_inform_ip_bde(
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    pick: Mapping[str, Any] | None = None,
) -> Path | None:
    scored = [
        r
        for r in rows
        if r.get("delta_bde_kcal") is not None and r.get("adiabatic_ip_eV") is not None
    ]
    if not scored:
        return None
    apply_pub_style()
    proxies = np.array([float(r["proxy_13p5nm"] or 0.0) for r in scored], dtype=float)
    pmin, pmax = float(np.min(proxies)), float(np.max(proxies))
    span = max(pmax - pmin, 1e-9)
    sizes = 70.0 + 220.0 * (proxies - pmin) / span
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    used: set[str] = set()
    for rec, size in zip(scored, sizes):
        cls = str(rec["chem_class"])
        ax.scatter(
            rec["adiabatic_ip_eV"],
            rec["delta_bde_kcal"],
            s=size,
            c=PALETTE.get(cls, PALETTE["other"]),
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
            label=cls if cls not in used else None,
        )
        used.add(cls)
        ax.annotate(
            rec["display"],
            (rec["adiabatic_ip_eV"], rec["delta_bde_kcal"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
            fontweight="bold",
        )
    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE.get(cls, PALETTE["other"]),
            markeredgecolor="black",
            markersize=8,
            label=cls,
        )
        for cls in sorted(used)
    ]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#444444",
            markeredgecolor="black",
            markersize=4 + 6 * t,
            label=rf"Henke $f_2$ sum {pmin + t * span:.1f}",
        )
        for t in (0.0, 0.5, 1.0)
    ]
    bold_legend(ax, handles=class_handles + size_handles, loc="best", title="Class / 13.5 nm proxy")
    ax.set_xlabel("Adiabatic IP (eV)")
    ax.set_ylabel(r"$\Delta$BDE = BDE$_0$ − BDE$_+$ (kcal mol$^{-1}$)")
    ax.set_title("Ligand map (labelled-ours DFT)")
    pick_disp = None if pick is None else pick.get("picked_display")
    footnote = DISCLAIMER if not pick_disp else f"Next pick: {pick_disp}. {DISCLAIMER}"
    fig.text(0.02, 0.01, footnote, fontsize=7.5, va="bottom", ha="left")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_inform_rank(
    ranked: Sequence[Mapping[str, Any]],
    out_path: Path,
    pick: Mapping[str, Any] | None = None,
) -> Path | None:
    if not ranked:
        return None
    apply_pub_style()
    names = [str(r["display"]) for r in ranked]
    y = np.arange(len(names))
    dbde = np.array([np.nan if r.get("delta_bde_kcal") is None else float(r["delta_bde_kcal"]) for r in ranked])
    ip = np.array([np.nan if r.get("adiabatic_ip_eV") is None else float(r["adiabatic_ip_eV"]) for r in ranked])
    proxy = np.array([np.nan if r.get("proxy_13p5nm") is None else float(r["proxy_13p5nm"]) for r in ranked])
    colors = [PALETTE.get(str(r["chem_class"]), PALETTE["other"]) for r in ranked]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, max(3.6, 0.55 * len(names) + 1.8)), sharey=True)
    panels = (
        (axes[0], dbde, r"$\Delta$BDE (kcal mol$^{-1}$)", r"Primary: max $\Delta$BDE"),
        (axes[1], ip, "Adiabatic IP (eV)", "Then: min IP"),
        (axes[2], proxy, r"Henke $f_2$ sum at 13.5 nm", "Then: max atomic proxy"),
    )
    for ax, values, xlabel, title in panels:
        finite = np.isfinite(values)
        ax.barh(
            y[finite],
            values[finite],
            color=[c for c, ok in zip(colors, finite) if ok],
            edgecolor="black",
            linewidth=0.7,
            height=0.72,
        )
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.4)
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    if np.any(np.isfinite(ip)):
        ip_lo = float(np.nanmin(ip))
        ip_hi = float(np.nanmax(ip))
        pad = max(0.15 * (ip_hi - ip_lo), 0.15)
        axes[1].set_xlim(ip_lo - pad, ip_hi + pad)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Ligand (best at top)")
    class_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=PALETTE[cls],
            markeredgecolor="black",
            markersize=8,
            label=cls,
        )
        for cls in sorted({str(r["chem_class"]) for r in ranked})
    ]
    bold_legend(axes[2], handles=class_handles, loc="lower right", title="Class")
    pick_disp = None if pick is None else pick.get("picked_display")
    rule = "" if pick is None else str(pick.get("rule", ""))
    pretty = {
        "expected_improvement": "expected improvement",
        "chemical_because_gp_shaky": "chemical next; GP shaky",
        "chemical": "chemical next",
    }.get(rule, rule.replace("_", " "))
    title = "Ranked ligands (classical DFT)"
    if pick_disp:
        title += f"  ·  next: {pick_disp}"
        if pretty:
            title += f" ({pretty})"
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=7.5, va="bottom")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _round(value: Any, ndigits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "rank",
        "ligand",
        "display",
        "smiles",
        "chem_class",
        "n_C",
        "n_F",
        "is_aryl",
        "is_alkenyl",
        "delta_bde_kcal",
        "bde0_kcal",
        "bde_cation_kcal",
        "adiabatic_ip_eV",
        "proxy_13p5nm",
        "e0_casci_ha",
        "e0_qsci_ha",
        "delta_e_qsci_mha",
        "qsci_sampler",
        "e0_qsci_control_entangled_hf_ha",
        "delta_e_qsci_control_entangled_hf_mha",
        "e0_n_qubits",
        "e0_ncas",
        "status",
        "label",
    )
    return {k: _round(row[k]) for k in keys if k in row}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="scores.json from tin_inform_dft.py (default: <out-dir>/scores.json)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()
    if args.scores is None:
        scores = out_dir / "scores.json"
    else:
        scores = args.scores.expanduser()
        if not scores.is_absolute():
            scores = (Path.cwd() / scores).resolve()
        else:
            scores = scores.resolve()
    return scores, out_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scores_path, out_dir = resolve_paths(args)
    try:
        doc = load_scores(scores_path, out_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    ligands = dict(doc.get("ligands") or {})
    failed_ids = [
        str(key).lower()
        for key, rec in ligands.items()
        if isinstance(rec, Mapping) and rec.get("status") == "failed"
    ]
    rows = [flatten_row(key, rec) for key, rec in ligands.items()]
    ranked = rank_rows(rows)
    scored_ids = [r["ligand"] for r in ranked]
    pool = unevaluated_rows(scored_ids, failed_ids=failed_ids)
    pick = fit_surrogate_and_pick(ranked, pool, seed=args.seed)
    first_pending = [k for k in FIRST_WAVE if k not in scored_ids]

    fig_ip = out_dir / "fig_inform_ip_bde.png"
    fig_rank = out_dir / "fig_inform_rank.png"
    saved_ip = saved_rank = None
    if not args.no_plot:
        saved_ip = plot_inform_ip_bde(ranked, fig_ip, pick=pick)
        saved_rank = plot_inform_rank(ranked, fig_rank, pick=pick)
    if not ranked:
        print(
            f"{scores_path} has no scored ligand rows; plotting functions are "
            "defined but figures were not saved.",
            file=sys.stderr,
        )

    figures = {
        "fig_inform_ip_bde": None if saved_ip is None else str(saved_ip),
        "fig_inform_rank": None if saved_rank is None else str(saved_rank),
    }
    ranking = {
        "label": "ours",
        "not_matgenq_table1": True,
        "forbidden_gqe_bde_claim": FORBIDDEN_GQE_BDE,
        "disclaimer": DISCLAIMER,
        "out_dir": str(out_dir),
        "scores": str(scores_path),
        "updated_at": _now_iso(),
        "rank_keys": [
            "delta_bde_kcal descending",
            "adiabatic_ip_eV ascending",
            "proxy_13p5nm descending",
        ],
        "ranked": [public_row(r) for r in ranked],
        "first_wave_pending": first_pending,
        "failed_ids": failed_ids,
        "unevaluated_pool": [public_row(r) for r in pool],
        "next_ligand": {
            "ligand": pick.get("picked_ligand"),
            "display": pick.get("picked_display"),
            "smiles": pick.get("smiles"),
            "rule": pick.get("rule"),
            "pick_tag": pick.get("pick_tag"),
            "reason": pick.get("rule_documentation"),
            "pool": "unevaluated",
        },
        "note": (
            "ΔBDE is labelled-ours B3LYP, not UCCSD(T) 72.6→21.2 and not GQE. "
            "13.5 nm column is a CXRO atomic f2 proxy, not ab initio absorption. "
            "Optional e0_casci/e0_qsci columns are not used in this rank order. "
            "CASCI E0 is labelled-ours AVAS at our XYZ; QSCI overlay is "
            "pool-native-uccsd (not 14-char H-cGQE unless n_qubits==14). "
            "Entangled-HF is a named control column only."
        ),
    }
    pick_payload = {
        "disclaimer": DISCLAIMER,
        "label": "ours",
        "not_matgenq_table1": True,
        "forbidden_gqe_bde_claim": FORBIDDEN_GQE_BDE,
        "picked_ligand": pick.get("picked_ligand"),
        "picked_display": pick.get("picked_display"),
        "smiles": pick.get("smiles"),
        "rule": pick.get("rule"),
        "pick_tag": pick.get("pick_tag"),
        "rule_documentation": pick.get("rule_documentation"),
        "unevaluated_pool": [r["ligand"] for r in pool],
        "first_wave_pending": first_pending,
        "surrogate": pick,
        "figures": figures,
        "optional_e0_note": (
            "Optional E0 is labelled-ours AVAS+CASCI then pool-native-uccsd "
            "QSCI on our XYZ (entangled-HF is a control). It is not ΔBDE, "
            "not IP, not a 13.5 nm spectrum, and not 14-char H-cGQE unless "
            "that Hamiltonian is 14 qubits."
        ),
        "updated_at": _now_iso(),
    }
    dump_json(out_dir / "ranking.json", ranking)
    dump_json(out_dir / "rank_table.json", ranking)
    dump_json(out_dir / "pick.json", pick_payload)

    print(f"scored_ligands\t{len(ranked)}")
    print(f"picked_ligand\t{pick.get('picked_ligand')}")
    print(f"rule\t{pick.get('rule')}")
    print(f"pick_json\t{out_dir / 'pick.json'}")
    print(f"rank_table\t{out_dir / 'rank_table.json'}")
    print(f"fig_inform_ip_bde\t{figures['fig_inform_ip_bde']}")
    print(f"fig_inform_rank\t{figures['fig_inform_rank']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
