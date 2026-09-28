import cudaq
from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData
from openfermion.transforms import jordan_wigner

geometry = [
    ("H", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.0, 0.74)),
]

molecule = MolecularData(
    geometry=geometry,
    basis="sto-3g",
    multiplicity=1,
    charge=0,
    description="h2",
)
mf = run_pyscf(molecule, run_scf=1, run_fci=1)
qubit_ham = jordan_wigner(mf.get_molecular_hamiltonian())
print("Num Pauli terms:", len(list(qubit_ham.terms)))
# Should be ~14 terms for H2 STO-3G