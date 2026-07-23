import json
from pathlib import Path
from src.gqe.data.hamiltonian_utils import load_hamiltonian_data, get_exact_energy

def test_check_iodobenzene():
    paths = [
        Path('results/data/hamiltonians.json'),
        Path('results/data/hamiltonians_phase3.json/hamiltonians.json'),
        Path('results/data/hamiltonians_merged.json'),
        Path('results/data/hamiltonians_iodobenzene.json/hamiltonians.json'),
        Path('results/data/fragments/hamiltonians.json'),
    ]

    for p in paths:
        if p.exists():
            try:
                d = load_hamiltonian_data(p, 'iodobenzene')
                if d:
                    const_term = next((t['real'] for t in d['terms'] if t['term'] in ('I', '')), None)
                    e_exact = get_exact_energy(d['terms'], d['n_qubits'])
                    print(f'\n{p}: const={const_term}, exact={e_exact}')
            except Exception as e:
                print(f'\n{p}: error {e}')
