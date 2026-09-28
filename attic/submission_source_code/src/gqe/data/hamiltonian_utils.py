import json
import numpy as np
from pathlib import Path
from scipy.sparse.linalg import eigsh
from openfermion.ops import QubitOperator
from openfermion.linalg import get_sparse_operator

def load_hamiltonian_data(json_path: Path, system_name: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for rec in data.get("records", []):
        if rec["name"] == system_name:
            return rec
    raise ValueError(f"System {system_name} not found in {json_path}")

def get_exact_energy(terms_list: list, n_qubits: int) -> float:
    q_op = QubitOperator()
    for term_dict in terms_list:
        term_str = term_dict["term"]
        coeff = term_dict["real"] + 1j * term_dict["imag"]
        if term_str == "I" or term_str == "":
            q_op += QubitOperator("", coeff)
        else:
            of_term = " ".join(f"{p[0]}{p[1:]}" for p in term_str.split())
            q_op += QubitOperator(of_term, coeff)
            
    sparse_op = get_sparse_operator(q_op, n_qubits)
    if sparse_op.shape[0] <= 32:
        evals = np.linalg.eigvalsh(sparse_op.toarray())
        return float(np.real(evals[0]))
    else:
        evals, _ = eigsh(sparse_op, k=1, which="SA")
        return float(np.real(evals[0]))
