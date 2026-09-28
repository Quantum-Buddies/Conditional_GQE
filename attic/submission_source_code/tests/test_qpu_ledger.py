"""Tests for the durable SQLite-backed QPU job ledger.

Tests cover:
- Idempotency: same circuit+device+shots -> no duplicate submission
- Error classification: transient vs permanent errors
- Retry/backoff state tracking
- Cost accounting: credits debited match pricing model
- Job lifecycle: submitted -> queued -> running -> completed/failed
- No live QPU calls required
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.gqe.eval.qpu_ledger import (
    QpuLedger,
    JobStatus,
    ErrorClass,
    LedgerEntry,
    AWS_RIGETTI_PRICING,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger(tmp_path):
    """Create a fresh in-memory or temp-file ledger for each test."""
    db_path = tmp_path / "test_ledger.sqlite"
    return QpuLedger(db_path)


@pytest.fixture
def sample_circuit_hash():
    """A deterministic hash representing a bound circuit."""
    return "sha256:abcdef1234567890"


@pytest.fixture
def sample_entry(sample_circuit_hash):
    """A sample ledger entry for testing."""
    return LedgerEntry(
        circuit_hash=sample_circuit_hash,
        molecule="h2",
        device_id="aws:rigetti:qpu:cepheus-1-108q",
        shots=4096,
        n_qubits=4,
        n_circuits=5,
        measurement_basis="computational_basis",
        pipeline_stage="sqd_pilot",
        metadata={"operators": ["XIXI", "ZIZI"], "thetas": [0.1, 0.2]},
    )


# ---------------------------------------------------------------------------
# Idempotency tests
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Same circuit + device + shots should not create duplicate submissions."""

    def test_duplicate_submission_returns_existing(self, ledger, sample_entry):
        """First submit creates a job; second submit with same hash returns same job_id."""
        # First submission
        entry1 = ledger.submit(sample_entry, job_id="job-001")
        assert entry1.job_id == "job-001"
        assert entry1.status == JobStatus.SUBMITTED

        # Second submission with same circuit hash + device + shots
        entry2 = ledger.submit(sample_entry, job_id="job-002")
        # Should return the existing entry, not create a new one
        assert entry2.job_id == "job-001"
        assert entry2.status == JobStatus.SUBMITTED

    def test_different_shots_creates_new_entry(self, ledger, sample_entry):
        """Same circuit but different shots should create a new entry."""
        entry1 = ledger.submit(sample_entry, job_id="job-001")
        sample_entry.shots = 8192  # Different shots
        entry2 = ledger.submit(sample_entry, job_id="job-002")
        assert entry2.job_id == "job-002"
        assert entry1.job_id != entry2.job_id

    def test_different_device_creates_new_entry(self, ledger, sample_entry):
        """Same circuit but different device should create a new entry."""
        entry1 = ledger.submit(sample_entry, job_id="job-001")
        sample_entry.device_id = "aws:ionq:qpu:forte-1"
        entry2 = ledger.submit(sample_entry, job_id="job-002")
        assert entry2.job_id == "job-002"

    def test_different_circuit_hash_creates_new_entry(self, ledger, sample_entry):
        """Different circuit hash should create a new entry."""
        entry1 = ledger.submit(sample_entry, job_id="job-001")
        sample_entry.circuit_hash = "sha256:different_hash"
        entry2 = ledger.submit(sample_entry, job_id="job-002")
        assert entry2.job_id == "job-002"

    def test_different_measurement_basis_creates_new_entry(self, ledger, sample_entry):
        """Same circuit but different measurement basis (SQD vs QWC) creates new entry."""
        entry1 = ledger.submit(sample_entry, job_id="job-001")
        sample_entry.measurement_basis = "qwc_group_3"
        entry2 = ledger.submit(sample_entry, job_id="job-002")
        assert entry2.job_id == "job-002"


# ---------------------------------------------------------------------------
# Error classification tests
# ---------------------------------------------------------------------------

class TestErrorClassification:
    """Test transient vs permanent error classification."""

    def test_transient_error_allows_retry(self, ledger, sample_entry):
        """A transient error (rate limit, timeout) should allow retry."""
        entry = ledger.submit(sample_entry, job_id="job-001")
        ledger.record_error(
            job_id="job-001",
            error_class=ErrorClass.TRANSIENT,
            error_message="Rate limited (429)",
        )
        updated = ledger.get("job-001")
        assert updated.status == JobStatus.FAILED_TRANSIENT
        assert updated.retry_count == 0
        assert updated.error_class == ErrorClass.TRANSIENT
        assert updated.can_retry()

    def test_permanent_error_blocks_retry(self, ledger, sample_entry):
        """A permanent error (invalid circuit, auth failure) should block retry."""
        entry = ledger.submit(sample_entry, job_id="job-001")
        ledger.record_error(
            job_id="job-001",
            error_class=ErrorClass.PERMANENT,
            error_message="Circuit too deep for device",
        )
        updated = ledger.get("job-001")
        assert updated.status == JobStatus.FAILED_PERMANENT
        assert updated.error_class == ErrorClass.PERMANENT
        assert not updated.can_retry()

    def test_classify_rate_limit_as_transient(self):
        """Rate limit errors should be classified as transient."""
        assert ErrorClass.classify("Too many requests (429)") == ErrorClass.TRANSIENT
        assert ErrorClass.classify("Rate limit exceeded") == ErrorClass.TRANSIENT

    def test_classify_auth_error_as_permanent(self):
        """Authentication errors should be classified as permanent."""
        assert ErrorClass.classify("Unauthorized: invalid API key") == ErrorClass.PERMANENT
        assert ErrorClass.classify("Forbidden: access denied") == ErrorClass.PERMANENT

    def test_classify_timeout_as_transient(self):
        """Timeouts should be classified as transient."""
        assert ErrorClass.classify("Request timed out") == ErrorClass.TRANSIENT
        assert ErrorClass.classify("Connection reset by peer") == ErrorClass.TRANSIENT

    def test_classify_circuit_error_as_permanent(self):
        """Circuit-level errors should be classified as permanent."""
        assert ErrorClass.classify("Circuit contains unsupported gate") == ErrorClass.PERMANENT
        assert ErrorClass.classify("Too many qubits for device") == ErrorClass.PERMANENT


# ---------------------------------------------------------------------------
# Retry/backoff tests
# ---------------------------------------------------------------------------

class TestRetryBackoff:
    """Test retry state tracking and backoff logic."""

    def test_retry_increments_count(self, ledger, sample_entry):
        """Each retry should increment the retry count."""
        entry = ledger.submit(sample_entry, job_id="job-001")
        ledger.record_error("job-001", ErrorClass.TRANSIENT, "Timeout")
        assert ledger.get("job-001").retry_count == 0

        ledger.schedule_retry("job-001")
        assert ledger.get("job-001").retry_count == 1
        assert ledger.get("job-001").status == JobStatus.RETRY_SCHEDULED

        ledger.schedule_retry("job-001")
        assert ledger.get("job-001").retry_count == 2

    def test_max_retries_exhausted(self, ledger, sample_entry):
        """After max retries, status should become FAILED_PERMANENT."""
        entry = ledger.submit(sample_entry, job_id="job-001")
        for i in range(3):
            ledger.record_error("job-001", ErrorClass.TRANSIENT, f"Timeout #{i}")
            ledger.schedule_retry("job-001")

        # After 3 retries, next failure should be permanent
        ledger.record_error("job-001", ErrorClass.TRANSIENT, "Timeout #3")
        updated = ledger.get("job-001")
        assert updated.status == JobStatus.FAILED_PERMANENT
        assert not updated.can_retry()

    def test_backoff_delay_increases(self, ledger, sample_entry):
        """Backoff delay should increase with each retry."""
        entry = ledger.submit(sample_entry, job_id="job-001")
        delays = []
        for i in range(3):
            ledger.record_error("job-001", ErrorClass.TRANSIENT, f"Error #{i}")
            delay = ledger.schedule_retry("job-001")
            delays.append(delay)

        # Each delay should be >= previous (exponential backoff)
        assert delays[1] >= delays[0]
        assert delays[2] >= delays[1]

    def test_next_retry_time_set(self, ledger, sample_entry):
        """schedule_retry should set next_retry_at timestamp."""
        entry = ledger.submit(sample_entry, job_id="job-001")
        ledger.record_error("job-001", ErrorClass.TRANSIENT, "Error")
        before = time.time()
        ledger.schedule_retry("job-001")
        after = time.time()
        updated = ledger.get("job-001")
        assert updated.next_retry_at is not None
        assert before <= updated.next_retry_at <= after + 300  # within 5 min


# ---------------------------------------------------------------------------
# Cost accounting tests
# ---------------------------------------------------------------------------

class TestCostAccounting:
    """Test credit cost tracking and budget enforcement."""

    def test_cost_recorded_on_submission(self, ledger, sample_entry):
        """Submission should record estimated cost."""
        entry = ledger.submit(sample_entry, job_id="job-001")
        # AWS Rigetti: 30 cr/task + 0.0425 cr/shot, 5 circuits
        # = 5 * 30 + 5 * 4096 * 0.0425 = 150 + 870.4 = 1020.4
        # But estimated cost is per-task, not per-circuit in batch mode
        # For batch: 30 + 4096 * 0.0425 = 30 + 174.08 = 204.08
        assert entry.estimated_cost_credits > 0
        assert entry.estimated_cost_credits is not None

    def test_cost_not_debited_until_completion(self, ledger, sample_entry):
        """Credits should only be debited after job completion, not at submission."""
        entry = ledger.submit(sample_entry, job_id="job-001")
        assert entry.actual_cost_credits is None

        ledger.update_status("job-001", JobStatus.COMPLETED, actual_cost=204.08)
        updated = ledger.get("job-001")
        assert updated.actual_cost_credits == 204.08

    def test_total_credits_spent(self, ledger, sample_entry):
        """Total credits spent should sum completed jobs only."""
        sample_entry.job_id_override = "job-001"
        e1 = ledger.submit(sample_entry, job_id="job-001")
        sample_entry.circuit_hash = "sha256:different"
        e2 = ledger.submit(sample_entry, job_id="job-002")

        ledger.update_status("job-001", JobStatus.COMPLETED, actual_cost=204.08)
        ledger.update_status("job-002", JobStatus.COMPLETED, actual_cost=204.08)

        total = ledger.total_credits_spent()
        assert abs(total - 408.16) < 0.01

    def test_budget_check_blocks_submission(self, ledger, sample_entry):
        """Budget check should block submission when estimated cost exceeds remaining budget."""
        # Set a very low budget
        ledger.set_budget(100.0)
        # AWS Rigetti cost for 4096 shots = ~204 credits > 100
        with pytest.raises(ValueError, match="budget"):
            ledger.submit(sample_entry, job_id="job-001")

    def test_budget_check_allows_within_limit(self, ledger, sample_entry):
        """Budget check should allow submission within budget."""
        ledger.set_budget(2000.0)
        entry = ledger.submit(sample_entry, job_id="job-001")
        assert entry.estimated_cost_credits <= 2000.0

    def test_aws_rigetti_pricing(self):
        """Verify AWS Rigetti pricing model."""
        pricing = AWS_RIGETTI_PRICING
        assert "per_task" in pricing
        assert "per_shot" in pricing
        assert pricing["per_task"] == 30.0
        assert pricing["per_shot"] == 0.0425

    def test_estimate_cost(self, ledger, sample_entry):
        """Cost estimation should match pricing model."""
        cost = ledger.estimate_cost(
            device_id="aws:rigetti:qpu:cepheus-1-108q",
            shots=4096,
            n_circuits=1,
        )
        expected = 30.0 + 4096 * 0.0425  # 204.08
        assert abs(cost - expected) < 0.01


# ---------------------------------------------------------------------------
# Job lifecycle tests
# ---------------------------------------------------------------------------

class TestJobLifecycle:
    """Test job status transitions and lifecycle tracking."""

    def test_status_transitions(self, ledger, sample_entry):
        """Job should transition through expected statuses."""
        entry = ledger.submit(sample_entry, job_id="job-001")
        assert entry.status == JobStatus.SUBMITTED

        ledger.update_status("job-001", JobStatus.QUEUED)
        assert ledger.get("job-001").status == JobStatus.QUEUED

        ledger.update_status("job-001", JobStatus.RUNNING)
        assert ledger.get("job-001").status == JobStatus.RUNNING

        ledger.update_status("job-001", JobStatus.COMPLETED, actual_cost=204.08)
        assert ledger.get("job-001").status == JobStatus.COMPLETED

    def test_get_nonexistent_job_raises(self, ledger):
        """Getting a non-existent job should raise KeyError."""
        with pytest.raises(KeyError):
            ledger.get("nonexistent-job-id")

    def test_list_jobs_by_status(self, ledger, sample_entry):
        """List jobs filtered by status."""
        e1 = ledger.submit(sample_entry, job_id="job-001")
        sample_entry.circuit_hash = "sha256:hash2"
        e2 = ledger.submit(sample_entry, job_id="job-002")
        sample_entry.circuit_hash = "sha256:hash3"
        e3 = ledger.submit(sample_entry, job_id="job-003")

        ledger.update_status("job-001", JobStatus.COMPLETED, actual_cost=204.08)
        ledger.update_status("job-002", JobStatus.QUEUED)

        completed = ledger.list_jobs(status=JobStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].job_id == "job-001"

        submitted = ledger.list_jobs(status=JobStatus.SUBMITTED)
        assert len(submitted) == 1
        assert submitted[0].job_id == "job-003"

    def test_list_jobs_by_molecule(self, ledger, sample_entry):
        """List jobs filtered by molecule."""
        e1 = ledger.submit(sample_entry, job_id="job-001")
        sample_entry.molecule = "lih"
        sample_entry.circuit_hash = "sha256:hash2"
        e2 = ledger.submit(sample_entry, job_id="job-002")

        h2_jobs = ledger.list_jobs(molecule="h2")
        assert len(h2_jobs) == 1
        assert h2_jobs[0].molecule == "h2"

    def test_list_jobs_by_pipeline_stage(self, ledger, sample_entry):
        """List jobs filtered by pipeline stage (sqd_pilot vs qwc_diagnostics)."""
        e1 = ledger.submit(sample_entry, job_id="job-001")
        sample_entry.pipeline_stage = "qwc_diagnostics"
        sample_entry.circuit_hash = "sha256:hash2"
        e2 = ledger.submit(sample_entry, job_id="job-002")

        sqd_jobs = ledger.list_jobs(pipeline_stage="sqd_pilot")
        assert len(sqd_jobs) == 1
        assert sqd_jobs[0].pipeline_stage == "sqd_pilot"

    def test_record_results(self, ledger, sample_entry):
        """Recording results should store counts and energy."""
        entry = ledger.submit(sample_entry, job_id="job-001")
        counts = {"0011": 2048, "1100": 1024, "0101": 512, "1010": 512}
        ledger.record_results("job-001", counts=counts, energy=-1.13727)
        updated = ledger.get("job-001")
        assert updated.status == JobStatus.COMPLETED
        assert updated.counts == counts
        assert abs(updated.energy - (-1.13727)) < 1e-10


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:
    """Test that the ledger survives process restarts."""

    def test_ledger_survives_reopen(self, tmp_path, sample_entry):
        """Reopening the ledger should preserve all entries."""
        db_path = tmp_path / "test_ledger.sqlite"
        ledger1 = QpuLedger(db_path)
        entry = ledger1.submit(sample_entry, job_id="job-001")
        ledger1.update_status("job-001", JobStatus.COMPLETED, actual_cost=204.08)
        del ledger1

        ledger2 = QpuLedger(db_path)
        entry2 = ledger2.get("job-001")
        assert entry2.job_id == "job-001"
        assert entry2.status == JobStatus.COMPLETED
        assert entry2.actual_cost_credits == 204.08

    def test_concurrent_ledger_access(self, tmp_path, sample_entry):
        """Two ledger instances on same DB should see each other's writes."""
        db_path = tmp_path / "test_ledger.sqlite"
        ledger1 = QpuLedger(db_path)
        ledger1.submit(sample_entry, job_id="job-001")

        ledger2 = QpuLedger(db_path)
        entry = ledger2.get("job-001")
        assert entry.job_id == "job-001"
