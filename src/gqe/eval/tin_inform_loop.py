#!/usr/bin/env python3
"""Closed-loop driver for labelled-ours R–Sn(OH)₃ informatics.

Reads pick.json / rank_table.json written by tin_inform_rank.py and prints
the next ligand to score. Does **not** invent ΔBDE/IP, does **not** run GQE
as a molecule generator, and does **not** copy XYZ into tin_resist.yaml.

  propose  = remaining unevaluated ligand from pick.json
  score    = tin_inform_dft.py (B3LYP ΔBDE / IP) — launched by sbatch
  pick     = tin_inform_rank.py

Optional H-cGQE E0 is a later column on the same ranked table, not this loop's
acquisition function.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = Path("results/tin_ab/methyltin_ours")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_out(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


BUSY_STATUS = frozenset({"scored", "failed", "running"})


def _ligand_busy(out_dir: Path, name: str) -> bool:
    score_path = out_dir / str(name) / "score.json"
    if not score_path.is_file():
        return False
    rec = load_json(score_path)
    if rec.get("status") == "scored" and rec.get("bde"):
        return True
    return rec.get("status") in BUSY_STATUS


def next_ligand_id(out_dir: Path) -> str | None:
    pick_path = out_dir / "pick.json"
    if not pick_path.is_file():
        raise FileNotFoundError(
            f"Missing {pick_path}. Run tin_inform_rank.py after scoring."
        )
    pick = load_json(pick_path)
    name = pick.get("picked_ligand")
    if not name:
        return None
    if _ligand_busy(out_dir, str(name)):
        return None
    return str(name)


def remaining_ligand_ids(out_dir: Path) -> list[str]:
    """Unevaluated pool members that are not already scored, failed, or running."""
    pick_path = out_dir / "pick.json"
    if not pick_path.is_file():
        return []
    pick = load_json(pick_path)
    pool = [str(x) for x in (pick.get("unevaluated_pool") or [])]
    picked = pick.get("picked_ligand")
    if picked and str(picked) not in pool:
        pool = [str(picked), *pool]
    out: list[str] = []
    for name in pool:
        if not _ligand_busy(out_dir, name):
            out.append(name)
    return out


def record_step(out_dir: Path, ligand: str, *, job_id: str | None) -> Path:
    hist_path = out_dir / "loop_history.json"
    hist: dict[str, Any]
    if hist_path.is_file():
        hist = load_json(hist_path)
    else:
        hist = {
            "label": "ours",
            "not_matgenq_table1": True,
            "role": "propose_score_pick",
            "gqe_proposes_molecules": False,
            "iterations": [],
        }
    pick = load_json(out_dir / "pick.json") if (out_dir / "pick.json").is_file() else {}
    rank = load_json(out_dir / "rank_table.json") if (out_dir / "rank_table.json").is_file() else {}
    ranked = rank.get("ranked") or []
    hist["iterations"].append(
        {
            "scored_ligand": ligand,
            "job_id": job_id,
            "picked_next": pick.get("picked_ligand"),
            "rule": pick.get("rule"),
            "n_scored": len(ranked),
            "rank_leader": None if not ranked else ranked[0].get("ligand"),
            "leader_delta_bde_kcal": None if not ranked else ranked[0].get("delta_bde_kcal"),
            "remaining": pick.get("unevaluated_pool"),
            "updated_at": _now_iso(),
        }
    )
    hist["updated_at"] = _now_iso()
    hist_path.write_text(json.dumps(hist, indent=2) + "\n", encoding="utf-8")
    return hist_path


def print_status(out_dir: Path) -> int:
    rank_path = out_dir / "rank_table.json"
    pick_path = out_dir / "pick.json"
    if not rank_path.is_file() or not pick_path.is_file():
        print(f"Need {rank_path} and {pick_path}", file=sys.stderr)
        return 1
    rank = load_json(rank_path)
    pick = load_json(pick_path)
    nxt = next_ligand_id(out_dir)
    print("closed_loop\tlabelled-ours DFT propose→score→pick")
    print("gqe_proposes_molecules\tfalse")
    print(f"scored\t{len(rank.get('ranked') or [])}")
    for row in tqdm(rank.get("ranked") or [], desc="Ranked", unit="ligand"):
        print(
            f"rank{row.get('rank')}\t{row.get('ligand')}\t"
            f"dBDE={row.get('delta_bde_kcal')}\tIP={row.get('adiabatic_ip_eV')}"
        )
    print(f"picked_next\t{pick.get('picked_ligand')}")
    print(f"rule\t{pick.get('rule')}")
    print(f"remaining\t{pick.get('unevaluated_pool')}")
    print(f"submit\t{nxt or 'none'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--next-id",
        action="store_true",
        help="Print the next ligand id only (empty if the pool is done).",
    )
    parser.add_argument(
        "--remaining-ids",
        action="store_true",
        help="Print one remaining (not running/scored/failed) ligand id per line.",
    )
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--record-step", metavar="LIGAND")
    parser.add_argument("--job-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = _resolve_out(args.out_dir)
    if args.next_id:
        name = next_ligand_id(out_dir)
        if name:
            print(name)
        return 0
    if args.remaining_ids:
        for name in remaining_ligand_ids(out_dir):
            print(name)
        return 0
    if args.record_step:
        path = record_step(out_dir, args.record_step, job_id=args.job_id)
        print(path)
        return 0
    return print_status(out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
