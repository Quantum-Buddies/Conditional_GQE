"""Persistent circuit→energy cache for fast DAPO RL.

Stores fixed-θ CUDA-Q expectation values keyed by
(molecule, n_qubits, n_electrons, theta, operator sequence) so repeated
circuits across epochs/runs skip expensive observe() calls.

Backend: SQLite (append-safe, multi-process friendly for precompute).
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence


def circuit_energy_cache_key(
    operators: Sequence[str],
    molecule_id: str,
    n_qubits: int,
    n_electrons: int,
    theta: float,
) -> str:
    """Deterministic MD5 key for a circuit energy evaluation."""
    ops = "|".join(operators)
    ctx = f"{molecule_id}:{int(n_qubits)}q:{int(n_electrons)}e:th={float(theta):.8f}"
    return hashlib.md5(f"{ops}#{ctx}".encode()).hexdigest()


class PersistentEnergyCache:
    """SQLite-backed cache mapping circuit context → energy.

    Thread-safe within a process (one connection + lock). Safe for concurrent
    writers across processes via SQLite WAL mode.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._conn = sqlite3.connect(
            str(self.path),
            timeout=60.0,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS energies (
                key TEXT PRIMARY KEY,
                molecule TEXT NOT NULL,
                n_qubits INTEGER NOT NULL,
                n_electrons INTEGER NOT NULL,
                theta REAL NOT NULL,
                n_ops INTEGER NOT NULL,
                energy REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_energies_mol ON energies(molecule)"
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "PersistentEnergyCache":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _key(
        self,
        operators: Sequence[str],
        molecule_id: str,
        n_qubits: int,
        n_electrons: int,
        theta: float,
    ) -> str:
        return circuit_energy_cache_key(
            operators, molecule_id, n_qubits, n_electrons, theta
        )

    def get(
        self,
        operators: Sequence[str],
        molecule_id: str,
        n_qubits: int,
        n_electrons: int,
        theta: float,
    ) -> float | None:
        key = self._key(operators, molecule_id, n_qubits, n_electrons, theta)
        with self._lock:
            cur = self._conn.execute(
                "SELECT energy FROM energies WHERE key = ?", (key,)
            )
            row = cur.fetchone()
            if row is None:
                self._misses += 1
                return None
            self._hits += 1
            self._conn.execute(
                "UPDATE energies SET hit_count = hit_count + 1 WHERE key = ?",
                (key,),
            )
            self._conn.commit()
            return float(row[0])

    def put(
        self,
        operators: Sequence[str],
        energy: float,
        molecule_id: str,
        n_qubits: int,
        n_electrons: int,
        theta: float,
    ) -> None:
        key = self._key(operators, molecule_id, n_qubits, n_electrons, theta)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO energies
                    (key, molecule, n_qubits, n_electrons, theta, n_ops, energy, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(key) DO UPDATE SET energy = excluded.energy
                """,
                (
                    key,
                    molecule_id,
                    int(n_qubits),
                    int(n_electrons),
                    float(theta),
                    len(operators),
                    float(energy),
                ),
            )
            self._conn.commit()

    def get_many(
        self,
        operators_batch: Sequence[Sequence[str]],
        molecule_id: str,
        n_qubits: int,
        n_electrons: int,
        theta: float,
    ) -> tuple[list[float | None], list[int]]:
        """Lookup a batch. Returns (energies_or_None, miss_indices)."""
        energies: list[float | None] = [None] * len(operators_batch)
        miss_indices: list[int] = []
        if not operators_batch:
            return energies, miss_indices

        keys = [
            self._key(ops, molecule_id, n_qubits, n_electrons, theta)
            for ops in operators_batch
        ]
        unique_keys = list(dict.fromkeys(keys))
        found: dict[str, float] = {}
        with self._lock:
            # Chunk IN queries to stay under SQLite variable limits
            chunk = 500
            for start in range(0, len(unique_keys), chunk):
                batch_keys = unique_keys[start:start + chunk]
                placeholders = ",".join("?" * len(batch_keys))
                cur = self._conn.execute(
                    f"SELECT key, energy FROM energies WHERE key IN ({placeholders})",
                    batch_keys,
                )
                for key, energy in cur.fetchall():
                    found[key] = float(energy)

            hit_keys = [k for k in keys if k in found]
            if hit_keys:
                # bump hit_count in one pass
                for start in range(0, len(hit_keys), chunk):
                    batch_keys = hit_keys[start:start + chunk]
                    placeholders = ",".join("?" * len(batch_keys))
                    self._conn.execute(
                        f"UPDATE energies SET hit_count = hit_count + 1 "
                        f"WHERE key IN ({placeholders})",
                        batch_keys,
                    )
                self._conn.commit()

        for i, key in enumerate(keys):
            if key in found:
                energies[i] = found[key]
                self._hits += 1
            else:
                miss_indices.append(i)
                self._misses += 1
        return energies, miss_indices

    def put_many(
        self,
        operators_batch: Sequence[Sequence[str]],
        energy_list: Sequence[float],
        molecule_id: str,
        n_qubits: int,
        n_electrons: int,
        theta: float,
    ) -> None:
        if not operators_batch:
            return
        rows = []
        for ops, energy in zip(operators_batch, energy_list):
            key = self._key(ops, molecule_id, n_qubits, n_electrons, theta)
            rows.append(
                (
                    key,
                    molecule_id,
                    int(n_qubits),
                    int(n_electrons),
                    float(theta),
                    len(ops),
                    float(energy),
                )
            )
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO energies
                    (key, molecule, n_qubits, n_electrons, theta, n_ops, energy, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(key) DO UPDATE SET energy = excluded.energy
                """,
                rows,
            )
            self._conn.commit()

    def has_key(
        self,
        operators: Sequence[str],
        molecule_id: str,
        n_qubits: int,
        n_electrons: int,
        theta: float,
    ) -> bool:
        key = self._key(operators, molecule_id, n_qubits, n_electrons, theta)
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM energies WHERE key = ? LIMIT 1", (key,)
            )
            return cur.fetchone() is not None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM energies")
            n_entries = int(cur.fetchone()[0])
            cur = self._conn.execute(
                "SELECT molecule, COUNT(*) FROM energies GROUP BY molecule"
            )
            per_mol = {row[0]: int(row[1]) for row in cur.fetchall()}
        total = self._hits + self._misses
        return {
            "path": str(self.path),
            "n_entries": n_entries,
            "per_molecule": per_mol,
            "session_hits": self._hits,
            "session_misses": self._misses,
            "session_hit_rate": self._hits / max(total, 1),
        }

    def reset_session_counters(self) -> None:
        self._hits = 0
        self._misses = 0


def resolve_energies_with_cache(
    operators_batch: Sequence[Sequence[str]],
    *,
    molecule_id: str,
    n_qubits: int,
    n_electrons: int,
    theta: float,
    eval_fn,
    cache: PersistentEnergyCache | None = None,
    cache_only: bool = False,
) -> tuple[list[float], dict[str, int]]:
    """Resolve energies via persistent cache with optional CUDA-Q fallback.

    Args:
        operators_batch: circuits to evaluate
        eval_fn: callable(list[list[str]]) -> list[float] for cache misses
        cache: PersistentEnergyCache or None (always call eval_fn)
        cache_only: if True, misses get energy 0.0 and are counted as skipped
                    (caller may filter); eval_fn is not called

    Returns:
        (energies, stats) with hits/misses/skipped
    """
    n = len(operators_batch)
    if n == 0:
        return [], {"hits": 0, "misses": 0, "skipped": 0}

    if cache is None:
        energies = list(eval_fn(list(operators_batch)))
        return energies, {"hits": 0, "misses": n, "skipped": 0}

    partial, miss_idx = cache.get_many(
        operators_batch, molecule_id, n_qubits, n_electrons, theta
    )
    hits = n - len(miss_idx)
    skipped = 0

    if not miss_idx:
        return [float(e) for e in partial], {"hits": hits, "misses": 0, "skipped": 0}

    if cache_only:
        energies = [float(e) if e is not None else 0.0 for e in partial]
        skipped = len(miss_idx)
        return energies, {"hits": hits, "misses": len(miss_idx), "skipped": skipped}

    miss_ops = [list(operators_batch[i]) for i in miss_idx]
    miss_energies = list(eval_fn(miss_ops))
    cache.put_many(
        miss_ops, miss_energies, molecule_id, n_qubits, n_electrons, theta
    )
    energies = [0.0] * n
    for i, e in enumerate(partial):
        if e is not None:
            energies[i] = float(e)
    for i, e in zip(miss_idx, miss_energies):
        energies[i] = float(e)
    return energies, {"hits": hits, "misses": len(miss_idx), "skipped": 0}
