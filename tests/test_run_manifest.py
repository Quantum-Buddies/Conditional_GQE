"""Tests for run_manifest module."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gqe.common.run_manifest import (
    create_run_manifest,
    save_run_manifest,
    attach_manifest_to_result,
)


def test_create_run_manifest_basic():
    m = create_run_manifest(command="pytest tests/test_run_manifest.py")
    assert m["schema_version"] == "1.0"
    assert "run_id" in m
    assert len(m["run_id"]) == 12
    assert "timestamp_utc" in m
    assert m["command"] == "pytest tests/test_run_manifest.py"
    assert "git_commit" in m
    assert "git_dirty" in m
    assert "versions" in m
    assert "pip_freeze" in m


def test_create_run_manifest_with_seed_and_files(tmp_path):
    f1 = tmp_path / "input1.txt"
    f1.write_text("hello world")
    f2 = tmp_path / "input2.yaml"
    f2.write_text("key: value")

    m = create_run_manifest(
        command="python run.py",
        seed=42,
        input_files=[str(f1), str(f2)],
    )
    assert m["seed"] == 42
    assert str(f1) in m["input_hashes"]
    assert str(f2) in m["input_hashes"]
    assert len(m["input_hashes"][str(f1)]) == 64  # SHA-256 hex


def test_save_run_manifest_no_overwrite(tmp_path):
    m = create_run_manifest(command="test")
    out = tmp_path / "manifest.json"
    p1 = save_run_manifest(m, out)
    assert p1 == out
    assert p1.exists()

    m2 = create_run_manifest(command="test2")
    p2 = save_run_manifest(m2, out)
    assert p2 != out
    assert p2.exists()
    assert out.exists()  # original preserved

    data1 = json.loads(out.read_text())
    data2 = json.loads(p2.read_text())
    assert data1["run_id"] != data2["run_id"]


def test_attach_manifest_to_result():
    result = {"energy": -1.137, "molecule": "H2"}
    m = create_run_manifest(command="test")
    attached = attach_manifest_to_result(result, m)
    assert "run_manifest" in attached
    assert attached["run_manifest"]["run_id"] == m["run_id"]
    assert attached["energy"] == -1.137


def test_no_api_key_in_manifest():
    m = create_run_manifest(command="test")
    s = json.dumps(m)
    assert "QBRAID_API_KEY" not in s
    assert "api_key" not in s.lower()
