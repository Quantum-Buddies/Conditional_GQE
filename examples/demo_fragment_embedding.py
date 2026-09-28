import json
import numpy as np
from pathlib import Path

from src.gqe.data.generate_hamiltonians import _generate_record
from src.gqe.common.hamiltonian_utils import exact_diagonalize_hamiltonian

def main():
    print("=== Hierarchical FMO + cGQE Demonstration ===")
    
    data = json.loads(Path('results/data/fragments/hamiltonians.json').read_text())
    parent_record = data['records'][0]
    parent_name = parent_record['name']
    
    print(f"Parent Molecule: {parent_name}")
    
    print("Diagonalizing Parent (8 qubits)...")
    parent_e, parent_gap = exact_diagonalize_hamiltonian(parent_record)
    print(f"Parent Ground State Energy: {parent_e:.6f} Ha\n")
    
    fragments = parent_record.get('fragments', [])
    print(f"Found {len(fragments)} fragments in the FMO plan.")
    
    fragment_energies = []
    
    for i, frag in enumerate(fragments):
        frag_name = frag['name']
        print(f"\n--- Fragment {i+1}: {frag_name} ---")
        
        mol_cfg = {
            'name': frag_name,
            'geometry': frag['geometry'],
            'charge': frag.get('charge', 0),
            'multiplicity': frag.get('multiplicity', 1),
            'active_space': frag['active_space']
        }
        
        defaults = {
            'basis': frag['basis'],
            'split': 'test'
        }
        
        try:
            frag_record = _generate_record(
                molecule=mol_cfg,
                dataset_defaults=defaults,
                fragment_plan=None
            )
            print(f"Fragment Qubits: {frag_record['n_qubits']}, Pauli Terms: {len(frag_record['terms'])}")
            
            e_frag, _ = exact_diagonalize_hamiltonian(frag_record)
            print(f"Fragment Ground State: {e_frag:.6f} Ha")
            fragment_energies.append(e_frag)
            
        except Exception as e:
            print(f"Error: {e}")

    e_sum = sum(fragment_energies)
    print("\n=== Recombination Results ===")
    print(f"Sum of Fragment Energies: {e_sum:.6f} Ha")
    print(f"Full Unfragmented Energy: {parent_e:.6f} Ha")
    print(f"Difference (Simple Sum Error): {(e_sum - parent_e)*1000:.2f} mHa")

if __name__ == '__main__':
    main()
