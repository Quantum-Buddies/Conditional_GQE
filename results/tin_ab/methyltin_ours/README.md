# Labelled-ours methyltin / R–Sn(OH)₃ outputs

This directory is the **only** allowed landing zone for DFT/MP2 geometries of RSn(OH)₃ produced by `configs/methyltin_ours_opt.yaml` and `src/gqe/eval/tin_inform_dft.py`.

Every JSON here must have `"label": "ours"`. Coordinates are **not** MatGen-Q Table 1.

| Path | What |
|---|---|
| `<ligand>/ours_starting_guess.xyz` | Hand Cartesian template (provenance in comment) |
| `<ligand>/<name>_ours_neutral.xyz` | B3LYP opt, charge 0 (after the nodes job) |
| `<ligand>/<name>_ours_cation.xyz` | UKS-B3LYP opt, charge +1 (after the nodes job) |
| `<ligand>/score.json` | IP, BDE, CXRO 13.5 nm **proxy** |
| `scores.json` | Merged scores |
| `ranking.json` | `tin_inform_rank.py` |

Do not copy anything from here into `configs/tin_resist.yaml`. Those rungs stay `geometry: null` until MatGen-Q SI coordinates exist.

Do not write 72.6 → 21.2 kcal/mol as a GQE number.
