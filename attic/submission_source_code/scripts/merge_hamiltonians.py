#!/usr/bin/env python3
"""Merge multiple Hamiltonian JSON files into one."""
import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: merge_hamiltonians.py output.json input1.json input2.json ...")
        sys.exit(1)
    
    out_path = Path(sys.argv[1])
    all_records = []
    seen_names = set()
    
    for in_path in sys.argv[2:]:
        with open(in_path) as f:
            data = json.load(f)
        records = data.get("records", data) if isinstance(data, dict) else data
        for r in records:
            name = r.get("name", "")
            if name not in seen_names:
                all_records.append(r)
                seen_names.add(name)
                print(f"  {name}: {r.get('n_qubits', '?')} qubits")
            else:
                print(f"  {name}: already present, skipping")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"records": all_records}, f, indent=2)
    print(f"\nMerged {len(all_records)} molecules into {out_path}")

if __name__ == "__main__":
    main()
