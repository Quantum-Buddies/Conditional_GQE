import json
from pathlib import Path

def main():
    print("=== Final L40S Multi-GPU Benchmark Summary ===")
    
    # 1. Load FMO Fragment Results (3-GPU)
    fmo_path = Path('results/baselines/cudaq_gqe_fmo_3gpu.json')
    if fmo_path.exists():
        fmo_data = json.loads(fmo_path.read_text())
        print(f"\n[FMO Fragments - CUDA-Q 3x L40S]")
        fragment_energies = []
        for r in fmo_data['results']:
            name = r['system']
            energy = r['baseline_energy']
            ref = r['reference_energy']
            delta = (energy - ref) * 1000
            print(f"  - {name:20}: {energy:.6f} Ha (Delta: {delta:.2f} mHa)")
            fragment_energies.append(energy)
        
        e_sum = sum(fragment_energies)
        print(f"  => Recombined Energy (Simple Sum): {e_sum:.6f} Ha")
    
    # 2. Load Parent Iodobenzene Results (if available on L40S)
    parent_path = Path('results/baselines/cudaq_gqe_l40s_3gpu.json')
    if parent_path.exists():
        parent_data = json.loads(parent_path.read_text())
        print(f"\n[Parent Molecule - CUDA-Q 3x L40S]")
        for r in parent_data['results']:
            if r['system'] == 'iodobenzene':
                energy = r['baseline_energy']
                ref = r['reference_energy']
                delta = (energy - ref) * 1000
                print(f"  - Iodobenzene (8q)  : {energy:.6f} Ha (Delta: {delta:.2f} mHa)")
                
                if 'e_sum' in locals():
                    emb_error = (e_sum - ref) * 1000
                    print(f"  => Embedding Error (FMO vs Parent Exact): {emb_error:.2f} mHa")
                    gqe_scaling_gap = (e_sum - energy) * 1000
                    print(f"  => Fragment Recombination Gap (FMO vs Parent GQE): {gqe_scaling_gap:.2f} mHa")

    print("\n[Scaling Evidence]")
    print("  Hardware: 3x NVIDIA L40S (48GB each)")
    print("  Backend:  CUDA-Q MQPU (PMIx Distributed)")
    print("  Parallel: Multi-QPU Operator Evaluation (Rank 0, 1, 2)")

if __name__ == '__main__':
    main()
