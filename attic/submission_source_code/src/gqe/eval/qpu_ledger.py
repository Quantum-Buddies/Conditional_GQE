"""Durable SQLite-backed QPU job ledger with idempotency and cost accounting.

Features:
- Idempotent submission: same circuit_hash + device + shots + measurement_basis
  returns existing job without creating a duplicate
- Error classification: transient (rate limit, timeout) vs permanent (auth, circuit)
- Retry/backoff: exponential backoff with max retries before permanent failure
- Cost accounting: estimated cost at submission, actual cost at completion
- Budget enforcement: refuses submission when estimated cost exceeds remaining budget
- Job lifecycle tracking: submitted -> queued -> running -> completed/failed
- Persistence: SQLite database survives process restarts
- Split-aware: tracks pipeline_stage (sqd_pilot vs qwc_diagnostics) separately

No live QPU calls are made by this module. It only tracks metadata.
"""
from __future__ import annotations

import argparse
import enum
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Pricing models
# ---------------------------------------------------------------------------

AWS_RIGETTI_PRICING: Dict[str, float] = {
    "per_task": 30.0,
    "per_shot": 0.0425,
}

DIRECT_RIGETTI_PRICING: Dict[str, float] = {
    "per_minute": 12000.0,
}

KNOWN_PRICING: Dict[str, Dict[str, float]] = {
    # Rigetti (AWS Braket)
    "aws:rigetti:qpu:cepheus-1-108q": AWS_RIGETTI_PRICING,
    # Rigetti (Direct QCS)
    "rigetti:rigetti:qpu:cepheus-1-108q": DIRECT_RIGETTI_PRICING,
    # Rigetti (Azure) - 36q variant
    "azure:rigetti:qpu:cepheus-1-36q": AWS_RIGETTI_PRICING,
    # IonQ (AWS Braket)
    "aws:ionq:qpu:forte-1": {"per_task": 30.0, "per_shot": 8.0},
    "aws:ionq:qpu:forte-enterprise-1": {"per_task": 30.0, "per_shot": 8.0},
    # IonQ (Azure)
    "azure:ionq:qpu:forte-1": {"per_task": 30.0, "per_shot": 8.0},
    "azure:ionq:qpu:forte-enterprise-1": {"per_task": 30.0, "per_shot": 8.0},
    # IQM (AWS Braket)
    "aws:iqm:qpu:garnet": {"per_task": 30.0, "per_shot": 0.145},
    "aws:iqm:qpu:emerald": {"per_task": 30.0, "per_shot": 0.16},
    # AQT (AWS Braket)
    "aws:aqt:qpu:ibex-q1": {"per_task": 30.0, "per_shot": 2.35},
    # QuEra (AWS Braket)
    "aws:quera:qpu:aquila": {"per_task": 30.0, "per_shot": 1.0},
    # Pasqal (Azure)
    "azure:pasqal:qpu:fresnel": {"per_minute": 500.0},
    # Simulators (free or low-cost)
    "ionq:ionq:sim:simulator": {"per_task": 0.0, "per_shot": 0.0, "per_minute": 0.0},
    "rigetti:rigetti:sim:qvm": {"per_task": 0.0, "per_shot": 0.0, "per_minute": 0.0},
    "qbraid:qbraid:sim:qir-sv": {"per_task": 0.0, "per_shot": 0.0, "per_minute": 0.0},
    "aws:aws:sim:sv1": {"per_task": 0.0, "per_shot": 0.0, "per_minute": 7.5},
    "aws:aws:sim:dm1": {"per_task": 0.0, "per_shot": 0.0, "per_minute": 7.5},
    "aws:aws:sim:tn1": {"per_task": 0.0, "per_shot": 0.0, "per_minute": 7.5},
    "azure:quantinuum:sim:h2-1e": {"per_task": 0.0, "per_shot": 0.0, "per_minute": 0.0},
    "azure:quantinuum:sim:h2-1sc": {"per_task": 0.0, "per_shot": 0.0, "per_minute": 0.0},
    "azure:pasqal:sim:emu-tn": {"per_task": 0.0, "per_shot": 0.0, "per_minute": 25.0},
}


def estimate_cost(
    device_id: str,
    shots: int,
    n_circuits: int = 1,
) -> float:
    """Estimate the credit cost for a QPU submission.

    Args:
        device_id: qBraid device ID.
        shots: Number of measurement shots per circuit.
        n_circuits: Number of circuits in the batch.

    Returns:
        Estimated cost in qBraid credits.
    """
    pricing = KNOWN_PRICING.get(device_id)
    if pricing is None:
        return 0.0  # Unknown pricing, don't block

    if "per_task" in pricing and "per_shot" in pricing:
        # AWS-style: per-task + per-shot
        return n_circuits * pricing["per_task"] + n_circuits * shots * pricing["per_shot"]
    elif "per_minute" in pricing:
        # Direct-style: per-minute (estimate ~1 min per circuit)
        return n_circuits * pricing["per_minute"] / 60.0
    return 0.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, enum.Enum):
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED_TRANSIENT = "failed_transient"
    FAILED_PERMANENT = "failed_permanent"
    RETRY_SCHEDULED = "retry_scheduled"
    CANCELLED = "cancelled"


class ErrorClass(str, enum.Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"

    @classmethod
    def classify(cls, error_message: str) -> "ErrorClass":
        """Classify an error message as transient or permanent."""
        msg = error_message.lower()
        # Transient errors
        transient_keywords = [
            "rate limit", "too many requests", "429",
            "timeout", "timed out", "connection reset",
            "temporarily unavailable", "service unavailable",
            "gateway", "503", "502", "500",
        ]
        for kw in transient_keywords:
            if kw in msg:
                return cls.TRANSIENT

        # Permanent errors
        permanent_keywords = [
            "unauthorized", "invalid api key", "forbidden",
            "access denied", "permission denied",
            "circuit", "unsupported gate", "too many qubits",
            "invalid", "malformed", "compilation failed",
        ]
        for kw in permanent_keywords:
            if kw in msg:
                return cls.PERMANENT

        return cls.UNKNOWN


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    """A single QPU job ledger entry."""
    circuit_hash: str
    molecule: str
    device_id: str
    shots: int
    n_qubits: int
    n_circuits: int
    measurement_basis: str  # "computational_basis" for SQD, "qwc_group_N" for QWC
    pipeline_stage: str  # "sqd_pilot" or "qwc_diagnostics"
    metadata: Dict[str, Any] = field(default_factory=dict)
    job_id: Optional[str] = None
    status: JobStatus = JobStatus.SUBMITTED
    submitted_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    estimated_cost_credits: Optional[float] = None
    actual_cost_credits: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: Optional[float] = None
    error_class: Optional[ErrorClass] = None
    error_message: Optional[str] = None
    counts: Optional[Dict[str, int]] = None
    energy: Optional[float] = None

    def can_retry(self) -> bool:
        """Check if this job can be retried."""
        if self.status == JobStatus.FAILED_PERMANENT:
            return False
        if self.error_class == ErrorClass.PERMANENT:
            return False
        return self.retry_count < self.max_retries

    def idempotency_key(self) -> str:
        """Generate a unique key for idempotency checking."""
        return f"{self.circuit_hash}:{self.device_id}:{self.shots}:{self.measurement_basis}"


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class QpuLedger:
    """SQLite-backed durable QPU job ledger.

    All operations are thread-safe via SQLite's serialized mode.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._budget: Optional[float] = None
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS qpu_jobs (
                job_id TEXT PRIMARY KEY,
                circuit_hash TEXT NOT NULL,
                molecule TEXT NOT NULL,
                device_id TEXT NOT NULL,
                shots INTEGER NOT NULL,
                n_qubits INTEGER NOT NULL,
                n_circuits INTEGER NOT NULL,
                measurement_basis TEXT NOT NULL,
                pipeline_stage TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'submitted',
                submitted_at REAL NOT NULL,
                completed_at REAL,
                estimated_cost_credits REAL,
                actual_cost_credits REAL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                next_retry_at REAL,
                error_class TEXT,
                error_message TEXT,
                counts TEXT,
                energy REAL,
                idempotency_key TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_idempotency ON qpu_jobs(idempotency_key);
            CREATE INDEX IF NOT EXISTS idx_status ON qpu_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_molecule ON qpu_jobs(molecule);
            CREATE INDEX IF NOT EXISTS idx_pipeline_stage ON qpu_jobs(pipeline_stage);
        """)
        self._conn.commit()

    def _entry_from_row(self, row: sqlite3.Row) -> LedgerEntry:
        return LedgerEntry(
            circuit_hash=row["circuit_hash"],
            molecule=row["molecule"],
            device_id=row["device_id"],
            shots=row["shots"],
            n_qubits=row["n_qubits"],
            n_circuits=row["n_circuits"],
            measurement_basis=row["measurement_basis"],
            pipeline_stage=row["pipeline_stage"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            submitted_at=row["submitted_at"],
            completed_at=row["completed_at"],
            estimated_cost_credits=row["estimated_cost_credits"],
            actual_cost_credits=row["actual_cost_credits"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            next_retry_at=row["next_retry_at"],
            error_class=ErrorClass(row["error_class"]) if row["error_class"] else None,
            error_message=row["error_message"],
            counts=json.loads(row["counts"]) if row["counts"] else None,
            energy=row["energy"],
        )

    def submit(self, entry: LedgerEntry, job_id: str) -> LedgerEntry:
        """Submit a job to the ledger with idempotency check.

        If an entry with the same idempotency key already exists, returns
        the existing entry without creating a duplicate.

        Args:
            entry: Ledger entry for the job.
            job_id: QPU job ID to assign.

        Returns:
            The ledger entry (existing if duplicate, new otherwise).

        Raises:
            ValueError: If estimated cost exceeds remaining budget.
        """
        idem_key = entry.idempotency_key()

        # Check for existing entry
        existing = self._conn.execute(
            "SELECT * FROM qpu_jobs WHERE idempotency_key = ?",
            (idem_key,),
        ).fetchone()
        if existing is not None:
            return self._entry_from_row(existing)

        # Estimate cost
        est_cost = estimate_cost(entry.device_id, entry.shots, entry.n_circuits)
        # Create a copy to avoid mutating the caller's entry
        entry = LedgerEntry(
            circuit_hash=entry.circuit_hash,
            molecule=entry.molecule,
            device_id=entry.device_id,
            shots=entry.shots,
            n_qubits=entry.n_qubits,
            n_circuits=entry.n_circuits,
            measurement_basis=entry.measurement_basis,
            pipeline_stage=entry.pipeline_stage,
            metadata=entry.metadata,
            job_id=job_id,
            status=JobStatus.SUBMITTED,
            submitted_at=entry.submitted_at,
            estimated_cost_credits=est_cost,
            retry_count=entry.retry_count,
            max_retries=entry.max_retries,
        )

        # Budget check
        if self._budget is not None:
            total_spent = self.total_credits_spent()
            total_estimated = total_spent + sum(
                e.estimated_cost_credits or 0
                for e in self.list_jobs()
                if e.status not in (JobStatus.COMPLETED, JobStatus.CANCELLED)
                and e.estimated_cost_credits is not None
            )
            if total_estimated + est_cost > self._budget:
                raise ValueError(
                    f"Submission would exceed budget: "
                    f"estimated {est_cost:.2f} credits, "
                    f"remaining budget {self._budget - total_estimated:.2f} credits"
                )

        # Insert
        self._conn.execute(
            """
            INSERT INTO qpu_jobs (
                job_id, circuit_hash, molecule, device_id, shots, n_qubits,
                n_circuits, measurement_basis, pipeline_stage, metadata,
                status, submitted_at, estimated_cost_credits,
                retry_count, max_retries, idempotency_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id, entry.circuit_hash, entry.molecule, entry.device_id,
                entry.shots, entry.n_qubits, entry.n_circuits,
                entry.measurement_basis, entry.pipeline_stage,
                json.dumps(entry.metadata), entry.status.value,
                entry.submitted_at, entry.estimated_cost_credits,
                entry.retry_count, entry.max_retries, idem_key,
            ),
        )
        self._conn.commit()
        return entry

    def get(self, job_id: str) -> LedgerEntry:
        """Retrieve a ledger entry by job ID.

        Raises:
            KeyError: If job_id is not found.
        """
        row = self._conn.execute(
            "SELECT * FROM qpu_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Job {job_id} not found in ledger")
        return self._entry_from_row(row)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        actual_cost: Optional[float] = None,
    ) -> None:
        """Update job status and optionally record actual cost."""
        completed_at = time.time() if status == JobStatus.COMPLETED else None
        if actual_cost is not None:
            self._conn.execute(
                "UPDATE qpu_jobs SET status = ?, completed_at = ?, actual_cost_credits = ? WHERE job_id = ?",
                (status.value, completed_at, actual_cost, job_id),
            )
        else:
            self._conn.execute(
                "UPDATE qpu_jobs SET status = ?, completed_at = ? WHERE job_id = ?",
                (status.value, completed_at, job_id),
            )
        self._conn.commit()

    def record_error(
        self,
        job_id: str,
        error_class: ErrorClass,
        error_message: str,
    ) -> None:
        """Record an error for a job."""
        entry = self.get(job_id)

        # If max retries exhausted, mark as permanent
        if entry.retry_count >= entry.max_retries:
            status = JobStatus.FAILED_PERMANENT
            error_class = ErrorClass.PERMANENT
        elif error_class == ErrorClass.PERMANENT:
            status = JobStatus.FAILED_PERMANENT
        else:
            status = JobStatus.FAILED_TRANSIENT

        self._conn.execute(
            "UPDATE qpu_jobs SET status = ?, error_class = ?, error_message = ? WHERE job_id = ?",
            (status.value, error_class.value, error_message, job_id),
        )
        self._conn.commit()

    def schedule_retry(self, job_id: str) -> float:
        """Schedule a retry for a failed transient job.

        Returns the backoff delay in seconds.
        """
        entry = self.get(job_id)
        if not entry.can_retry():
            raise ValueError(f"Job {job_id} cannot be retried")

        retry_count = entry.retry_count + 1
        # Exponential backoff: base_delay * 2^retry_count
        base_delay = 5.0
        delay = base_delay * (2 ** (retry_count - 1))
        next_retry = time.time() + delay

        # Check if max retries reached after this one
        if retry_count >= entry.max_retries:
            new_status = JobStatus.RETRY_SCHEDULED
        else:
            new_status = JobStatus.RETRY_SCHEDULED

        self._conn.execute(
            "UPDATE qpu_jobs SET retry_count = ?, status = ?, next_retry_at = ? WHERE job_id = ?",
            (retry_count, new_status.value, next_retry, job_id),
        )
        self._conn.commit()
        return delay

    def record_results(
        self,
        job_id: str,
        counts: Dict[str, int],
        energy: float,
        actual_cost: Optional[float] = None,
    ) -> None:
        """Record job results (counts, energy) and mark as completed."""
        self._conn.execute(
            """
            UPDATE qpu_jobs
            SET status = ?, counts = ?, energy = ?, completed_at = ?,
                actual_cost_credits = ?
            WHERE job_id = ?
            """,
            (
                JobStatus.COMPLETED.value,
                json.dumps(counts),
                energy,
                time.time(),
                actual_cost,
                job_id,
            ),
        )
        self._conn.commit()

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        molecule: Optional[str] = None,
        pipeline_stage: Optional[str] = None,
    ) -> List[LedgerEntry]:
        """List jobs with optional filters."""
        query = "SELECT * FROM qpu_jobs WHERE 1=1"
        params: List[Any] = []

        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        if molecule is not None:
            query += " AND molecule = ?"
            params.append(molecule)
        if pipeline_stage is not None:
            query += " AND pipeline_stage = ?"
            params.append(pipeline_stage)

        query += " ORDER BY submitted_at"
        rows = self._conn.execute(query, params).fetchall()
        return [self._entry_from_row(row) for row in rows]

    def total_credits_spent(self) -> float:
        """Total credits actually spent on completed jobs."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(actual_cost_credits), 0) as total FROM qpu_jobs WHERE status = ?",
            (JobStatus.COMPLETED.value,),
        ).fetchone()
        return float(row["total"])

    def set_budget(self, credits: float) -> None:
        """Set the credit budget for budget enforcement."""
        self._budget = credits

    def estimate_cost(
        self,
        device_id: str,
        shots: int,
        n_circuits: int = 1,
    ) -> float:
        """Estimate cost for a potential submission."""
        return estimate_cost(device_id, shots, n_circuits)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def import_metadata_json(self, path: Path) -> int:
        """Import a single qbraid_job_metadata_*.json or *_submission_meta.json file.

        Returns 1 if imported, 0 if skipped (already exists or no useful data).
        """
        data = json.loads(Path(path).read_text())
        molecule = data.get("molecule", "unknown")
        device_id = data.get("device", data.get("device_id", "unknown"))
        shots = int(data.get("shots", 0))
        n_qubits = int(data.get("n_qubits", 0))
        n_circuits = int(data.get("n_circuits", data.get("num_circuits", 1)))
        job_id = data.get("job_id", data.get("qbraid_job_id", f"imported_{path.stem}"))
        circuit_hash = data.get("circuit_hash", data.get("hash", "imported"))
        measurement_basis = data.get("measurement_basis", "computational_basis")
        pipeline_stage = data.get("pipeline_stage", data.get("mode", "sqd_pilot"))
        status_str = data.get("status", "completed")
        try:
            status = JobStatus(status_str)
        except ValueError:
            status = JobStatus.COMPLETED
        estimated_cost = data.get("estimated_cost_credits")
        actual_cost = data.get("actual_cost_credits")
        counts = data.get("counts")
        energy = data.get("energy")

        idem_key = f"{circuit_hash}:{device_id}:{shots}:{measurement_basis}"

        existing = self._conn.execute(
            "SELECT * FROM qpu_jobs WHERE idempotency_key = ?", (idem_key,)
        ).fetchone()
        if existing is not None:
            return 0

        self._conn.execute(
            """INSERT INTO qpu_jobs (
                job_id, circuit_hash, molecule, device_id, shots, n_qubits,
                n_circuits, measurement_basis, pipeline_stage, metadata,
                status, submitted_at, completed_at, estimated_cost_credits,
                actual_cost_credits, counts, energy, idempotency_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id, circuit_hash, molecule, device_id, shots, n_qubits,
                n_circuits, measurement_basis, pipeline_stage,
                json.dumps({"imported_from": str(path)}),
                status.value, data.get("submitted_at", time.time()),
                data.get("completed_at"), estimated_cost, actual_cost,
                json.dumps(counts) if counts else None, energy, idem_key,
            ),
        )
        self._conn.commit()
        return 1

    def import_metadata_dir(self, directory: Path, pattern: str = "qbraid_job_metadata_*.json") -> int:
        """Import all matching metadata JSON files from a directory.

        Also tries `*_submission_meta.json` if the primary pattern yields nothing.
        Returns the number of files imported.
        """
        directory = Path(directory)
        files = sorted(directory.glob(pattern))
        if not files:
            files = sorted(directory.glob("*_submission_meta.json"))
        if not files:
            files = sorted(directory.glob("*metadata*.json"))

        imported = 0
        for f in files:
            try:
                imported += self.import_metadata_json(f)
            except Exception:
                pass
        return imported

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_init(args: argparse.Namespace) -> None:
    ledger = QpuLedger(args.db)
    print(f"Ledger initialized at {args.db}")
    ledger.close()


def _cli_import(args: argparse.Namespace) -> None:
    ledger = QpuLedger(args.db)
    if args.path.is_dir():
        n = ledger.import_metadata_dir(args.path)
    else:
        n = ledger.import_metadata_json(args.path)
    print(f"Imported {n} job(s) from {args.path}")
    ledger.close()


def _cli_status(args: argparse.Namespace) -> None:
    ledger = QpuLedger(args.db)
    jobs = ledger.list_jobs()
    if not jobs:
        print("No jobs in ledger.")
        ledger.close()
        return
    print(f"{'Job ID':30s} {'Molecule':15s} {'Device':30s} {'Status':20s} {'Shots':>6s} {'Cost':>8s}")
    print("-" * 115)
    for j in jobs:
        cost = f"{j.estimated_cost_credits:.1f}" if j.estimated_cost_credits else "-"
        print(f"{j.job_id:30s} {j.molecule:15s} {j.device_id:30s} {j.status.value:20s} {j.shots:6d} {cost:>8s}")
    ledger.close()


def _cli_cost(args: argparse.Namespace) -> None:
    ledger = QpuLedger(args.db)
    spent = ledger.total_credits_spent()
    jobs = ledger.list_jobs()
    estimated_total = sum(j.estimated_cost_credits or 0 for j in jobs)
    completed = [j for j in jobs if j.status == JobStatus.COMPLETED]
    failed = [j for j in jobs if j.status in (JobStatus.FAILED_PERMANENT, JobStatus.FAILED_TRANSIENT)]
    pending = [j for j in jobs if j.status not in (JobStatus.COMPLETED, JobStatus.FAILED_PERMANENT, JobStatus.FAILED_TRANSIENT, JobStatus.CANCELLED)]
    print(f"=== Ledger Cost Report ===")
    print(f"  Total jobs:       {len(jobs)}")
    print(f"  Completed:        {len(completed)}")
    print(f"  Failed:           {len(failed)}")
    print(f"  Pending/active:   {len(pending)}")
    print(f"  Credits spent:    {spent:.2f}")
    print(f"  Credits estimated (all): {estimated_total:.2f}")
    if args.budget is not None:
        remaining = args.budget - spent
        print(f"  Budget:           {args.budget:.2f}")
        print(f"  Remaining:        {remaining:.2f}")
    ledger.close()


def _cli_poll(args: argparse.Namespace) -> None:
    ledger = QpuLedger(args.db)
    jobs = ledger.list_jobs(status=JobStatus.RETRY_SCHEDULED)
    jobs += ledger.list_jobs(status=JobStatus.FAILED_TRANSIENT)
    if not jobs:
        print("No jobs pending retry.")
        ledger.close()
        return
    now = time.time()
    for j in jobs:
        if j.next_retry_at and j.next_retry_at <= now:
            if j.can_retry():
                delay = ledger.schedule_retry(j.job_id)
                print(f"Scheduled retry for {j.job_id} (delay {delay:.0f}s)")
            else:
                print(f"Job {j.job_id} cannot be retried (max retries reached)")
        else:
            eta = (j.next_retry_at - now) if j.next_retry_at else 0
            print(f"Job {j.job_id} waiting for retry (eta {eta:.0f}s)")
    ledger.close()


def _cli_retrieve(args: argparse.Namespace) -> None:
    ledger = QpuLedger(args.db)
    entry = ledger.get(args.job_id)
    print(f"Job:     {entry.job_id}")
    print(f"Molecule: {entry.molecule}")
    print(f"Device:   {entry.device_id}")
    print(f"Status:   {entry.status.value}")
    if entry.counts:
        print(f"Counts:   {json.dumps(entry.counts, indent=2)[:500]}...")
    if entry.energy is not None:
        print(f"Energy:   {entry.energy:.6f} Ha")
    if entry.actual_cost_credits is not None:
        print(f"Cost:     {entry.actual_cost_credits:.2f} credits")
    ledger.close()


def main() -> None:
    import sys
    # Parse --db before subcommand
    parser = argparse.ArgumentParser(
        description="Durable QPU job ledger — init, import, inspect, and manage job state.",
    )
    parser.add_argument("--db", type=Path, default=Path("results/eval/qpu_jobs.sqlite"),
                        help="SQLite ledger database path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize a new ledger database")
    p_init.set_defaults(func=_cli_init)

    p_import = sub.add_parser("import", help="Import existing metadata JSON files")
    p_import.add_argument("path", type=Path, help="JSON file or directory to import")
    p_import.set_defaults(func=_cli_import)

    p_status = sub.add_parser("status", help="Show all jobs in the ledger")
    p_status.set_defaults(func=_cli_status)

    p_cost = sub.add_parser("cost", help="Show cost accounting summary")
    p_cost.add_argument("--budget", type=float, default=None, help="Total credit budget for remaining calculation")
    p_cost.set_defaults(func=_cli_cost)

    p_poll = sub.add_parser("poll", help="Check and schedule retries for failed transient jobs")
    p_poll.set_defaults(func=_cli_poll)

    p_retrieve = sub.add_parser("retrieve", help="Show details for a specific job")
    p_retrieve.add_argument("job_id", type=str, help="Job ID to retrieve")
    p_retrieve.set_defaults(func=_cli_retrieve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
