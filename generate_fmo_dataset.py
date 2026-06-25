import json
from pathlib import Path
from src.gqe.data.generate_hamiltonians import _generate_record

def main():
    data = json.loads(Path('results/data/fragments/hamiltonians.json').read_text())
    parent_record = data['records'][0]
    
    fragments = parent_record.get('fragments', [])
    records = []
    
    for i, frag in enumerate(fragments):
        mol_cfg = {
            'name': frag['name'],
            'geometry': frag['geometry'],
            'charge': frag.get('charge', 0),
            'multiplicity': frag.get('multiplicity', 1),
            'active_space': frag['active_space']
        }
        
        defaults = {
            'basis': frag['basis'],
            'split': 'test'
        }
        
        frag_record = _generate_record(
            molecule=mol_cfg,
            dataset_defaults=defaults,
            fragment_plan=None
        )
        records.append(frag_record)

    out_path = Path('results/data/fragments/fmo_hamiltonians.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({'records': records}))
    print(f"Wrote {len(records)} fragment Hamiltonians to {out_path}")

if __name__ == '__main__':
    main()
