"""Tests for CUDA-Q qBraid targeting and GQE JSON loading. No QPU, no API key."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.gqe.eval.cudaq_qbraid_target import (
    CEPHEUS_BRAKET_MACHINE,
    CEPHEUS_DIRECT_MACHINE,
    QIR_SV_MACHINE,
    is_direct_rigetti,
    is_hardware_qbraid_machine,
    is_simulator_machine,
    resolve_machine,
    set_qbraid_machine,
)
from src.gqe.eval.gqe_qsci_sample import (
    count_two_qubit_gates_qasm,
    estimate_two_qubit_from_pauli_words,
    evaluate_delta_mha,
    load_gqe_operators,
)
from src.gqe.eval.sqd import (
    repair_bitstring_to_particle_number,
    self_consistent_config_recovery,
)


class _FakeCudaq:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set_target(self, name: str, **kwargs):
        self.calls.append((name, kwargs))


class TestMachineIds:
    def test_aliases(self):
        assert resolve_machine("qir-sv") == QIR_SV_MACHINE
        assert resolve_machine("cepheus") == CEPHEUS_BRAKET_MACHINE
        assert resolve_machine("cepheus-direct") == CEPHEUS_DIRECT_MACHINE

    def test_simulator_not_hardware(self):
        assert is_simulator_machine(QIR_SV_MACHINE)
        assert not is_hardware_qbraid_machine(QIR_SV_MACHINE)
        assert is_hardware_qbraid_machine(CEPHEUS_BRAKET_MACHINE)
        assert is_direct_rigetti(CEPHEUS_DIRECT_MACHINE)

    def test_set_qir_sv_no_allow_flag(self, monkeypatch):
        monkeypatch.delenv("CUDAQ_ALLOW_QPU", raising=False)
        fake = _FakeCudaq()
        resolved = set_qbraid_machine("qir-sv", cudaq_module=fake)
        assert resolved == QIR_SV_MACHINE
        assert fake.calls == [("qbraid", {"machine": QIR_SV_MACHINE})]
        assert "api_key" not in fake.calls[0][1]

    def test_refuse_cepheus_without_allow(self, monkeypatch):
        monkeypatch.delenv("CUDAQ_ALLOW_QPU", raising=False)
        fake = _FakeCudaq()
        with pytest.raises(RuntimeError, match="CUDAQ_ALLOW_QPU"):
            set_qbraid_machine("cepheus", cudaq_module=fake)
        assert fake.calls == []

    def test_allow_cepheus_braket(self, monkeypatch):
        monkeypatch.setenv("CUDAQ_ALLOW_QPU", "1")
        monkeypatch.delenv("CUDAQ_ALLOW_DIRECT_RIGETTI", raising=False)
        fake = _FakeCudaq()
        resolved = set_qbraid_machine("cepheus", cudaq_module=fake)
        assert resolved == CEPHEUS_BRAKET_MACHINE
        assert fake.calls[0] == ("qbraid", {"machine": CEPHEUS_BRAKET_MACHINE})

    def test_refuse_direct_rigetti(self, monkeypatch):
        monkeypatch.setenv("CUDAQ_ALLOW_QPU", "1")
        monkeypatch.delenv("CUDAQ_ALLOW_DIRECT_RIGETTI", raising=False)
        fake = _FakeCudaq()
        with pytest.raises(RuntimeError, match="DIRECT_RIGETTI"):
            set_qbraid_machine("cepheus-direct", cudaq_module=fake)

    def test_refuse_inline_api_key(self, monkeypatch):
        monkeypatch.setenv("CUDAQ_QBRAID_API_KEY_INLINE", "1")
        monkeypatch.delenv("CUDAQ_ALLOW_QPU", raising=False)
        fake = _FakeCudaq()
        with pytest.raises(RuntimeError, match="inline"):
            set_qbraid_machine("qir-sv", cudaq_module=fake)


class TestScoreCore:
    def test_repair_drops_extra_electron(self):
        occupancy = [0.9, 0.8, 0.1, 0.05]
        # 1110 → weight 3, want 2; drop lowest occupancy occupied (qubit 2)
        repaired = repair_bitstring_to_particle_number(0b0111, 4, 2, occupancy)
        assert bin(repaired).count("1") == 2

    def test_recovery_maps_wrong_n_into_sector(self):
        counts = {"0011": 80, "0111": 20}  # 0111 has N=3
        recovered = self_consistent_config_recovery(counts, n_qubits=4, n_electrons=2)
        assert all(bs.count("1") == 2 for bs in recovered)
        assert recovered["0011"] >= 80


class TestGqeJson:
    def test_load_operators(self, tmp_path: Path):
        payload = {
            "results": [
                {
                    "system": "sno_14q",
                    "gqe_selected_operators": [
                        {"index": 0, "coefficient_real": 0.05, "pauli_word": "XX" + "I" * 12},
                        {"index": 1, "coefficient_real": -0.05, "pauli_word": "YY" + "I" * 12},
                    ],
                }
            ]
        }
        path = tmp_path / "gqe.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        words, thetas, row = load_gqe_operators(path, "sno_14q")
        assert len(words) == 2
        assert thetas == [0.05, -0.05]
        assert row["system"] == "sno_14q"

    def test_qasm_two_qubit_count(self):
        qasm = "OPENQASM 2.0;\nqreg q[4];\ncx q[0],q[1];\ncz q[1],q[2];\nh q[0];\n"
        assert count_two_qubit_gates_qasm(qasm) == 2

    def test_pauli_two_qubit_estimate(self):
        assert estimate_two_qubit_from_pauli_words(["XXIIIIIIIIIIII"]) == 2
        assert estimate_two_qubit_from_pauli_words(["IIIIIIIIIIIIII"]) == 0

    def test_delta_mha(self):
        # 1.6 mHa above CASCI
        d = evaluate_delta_mha(-288.10752104, -288.10912104)
        assert abs(d - 1.6) < 1e-6
