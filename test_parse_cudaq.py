import sys
import os
sys.path.insert(0, 'src')
from gqe.data.hamiltonian_utils import load_hamiltonian_data, get_exact_energy

data = load_hamiltonian_data(Path("results/data/hamiltonians.json"), "h2")
print("H2 reference exact energy:", get_exact_energy(data["terms"], data["n_qubits"]))
