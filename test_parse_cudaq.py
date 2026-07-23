import os
import sys
from pathlib import Path

# Add src/ directory to python path
src_dir = str(Path(__file__).resolve().parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from gqe.data.hamiltonian_utils import load_hamiltonian_data, get_exact_energy


def test_parse_cudaq():
    """Explicit test function for CUDA-Q hamiltonian parsing."""
    json_path = Path(__file__).resolve().parent / "results" / "data" / "hamiltonians.json"
    data = load_hamiltonian_data(json_path, "h2")
    assert data is not None
    assert data["name"] == "h2"
    assert data["n_qubits"] == 4


def test_load_hamiltonian_data():
    """Test loading hamiltonian data for h2."""
    json_path = Path(__file__).resolve().parent / "results" / "data" / "hamiltonians.json"
    data = load_hamiltonian_data(json_path, "h2")
    assert data is not None
    assert data["name"] == "h2"
    assert data["n_qubits"] == 4


def test_get_exact_energy_h2():
    """Test exact energy calculation for h2."""
    json_path = Path(__file__).resolve().parent / "results" / "data" / "hamiltonians.json"
    data = load_hamiltonian_data(json_path, "h2")
    exact_energy = get_exact_energy(data["terms"], data["n_qubits"])
    assert isinstance(exact_energy, float)
    assert abs(exact_energy - (-1.137)) < 0.1
