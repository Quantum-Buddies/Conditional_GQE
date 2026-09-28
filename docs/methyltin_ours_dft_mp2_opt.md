# Labelled-ours R–Sn(OH)₃ DFT/MP2 opt (not MatGen-Q Table 1)

**Status:** pipeline implemented (21 Aug 2026). Starting guesses are hand Cartesian templates. No DFT opt has been run until `jobs/tin_inform_dft.sbatch` is submitted.

This path exists so we can build **our** RSn(OH)₃ 0/+1 geometries without pretending they are the unpublished MatGen-Q SI.

## Hard split

| File | Role |
|---|---|
| `configs/tin_resist.yaml` | MatGen-Q rungs. `geometry: null` stays until SI XYZ exists. |
| `configs/methyltin_ours_opt.yaml` | Ours only. Same `geometry: null` until an opt finishes. |
| `results/tin_ab/methyltin_ours/` | Only place an ours XYZ may land. |
| `src/gqe/data/rsn_oh3_templates.py` | Hand Cartesian builders (no RDKit). |
| `src/gqe/eval/tin_inform_dft.py` | Propose→score CLI. |
| `src/gqe/eval/tin_inform_rank.py` | Rank ΔBDE + CXRO proxy; next-ligand picker. |
| `jobs/tin_inform_dft.sbatch` | AIRE `nodes` array 0–4 (Me, Et, n-Bu, vinyl, Ph). |

Do **not** copy ours coordinates into `tin_resist.yaml`. Do **not** score 72.6 → 21.2 kcal/mol as a GQE number (classical UCCSD(T) in [arXiv:2607.23988](https://arxiv.org/abs/2607.23988)).

## Chemistry

- Basis `def2-svp`, `ecp={"Sn": "def2-svp"}` (ECP28MDF on Sn; same key as companion `SnOMolecule`).
- Neutral: charge 0, singlet, **RKS-B3LYP** opt, then **RHF-MP2** single-point at the DFT geom (not a second opt).
- Cation: charge +1, doublet, **UKS-B3LYP** opt (may start from optimized neutral), then **UHF-MP2** single-point.
- Vertical IP: \(E(+1\) at 0-geom\() - E(0)\). Adiabatic IP: \(E(+1\) opt\() - E(0\) opt\). Hartree and eV.
- Sn–C BDE at B3LYP: \(\mathrm{BDE}(0)=E(\mathrm{R}\bullet)+E(\bullet\mathrm{Sn(OH)_3})-E(\mathrm{RSn(OH)_3})\). Cation channel keeps charge on Sn: \(\mathrm{BDE}(+1)=E(\mathrm{R}\bullet)+E(\mathrm{Sn(OH)_3}^+)-E(\mathrm{RSn(OH)_3}^+)\). \(\Delta\mathrm{BDE}=\mathrm{BDE}(0)-\mathrm{BDE}(+1)\).
- 13.5 nm column is a CXRO-style atomic \(f_2\) **proxy** (Henke tables at 91.84 eV), **not** ab initio \(\sigma(92\,\mathrm{eV})\).

First-wave ligands: Me, Et, n-Bu, vinyl, Ph.
Unevaluated pool (picker only): i-Pr, n-Pr, allyl, CF3.

## How to run (do not invent MatGen-Q XYZ)

```bash
source /scratch/kcwp264/.aire_scratch_env.sh
conda activate /scratch/kcwp264/.conda_envs/cudaq-env
cd /scratch/kcwp264/Conditional-GQE_materials
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Starting guesses only (login node OK):
python src/gqe/eval/tin_inform_dft.py --ligands me,et,nbu,vinyl,ph \
  --out-dir results/tin_ab/methyltin_ours --write-guesses-only

# Full DFT: not login-node A2, not GPU
sbatch jobs/tin_inform_dft.sbatch
```

Author request for their SI: `docs/methyltin_xyz_author_request.md` (draft, unsent).
