# Tin ligand loop: propose → score → pick

One page for the **labelled-ours** organotin materials-informatics cycle on AIRE. Classical DFT ranks ligands; optional \(E_0\) is AVAS+CASCI (and GPU QSCI) on **our** XYZ, not the acquisition function.

GQE does **not** propose molecules. The closed loop is:

```
                    ┌──────────────────────────────────────────┐
  templates/pool    │                                          │
  R on RSn(OH)3 ──► propose ──► score (B3LYP ΔBDE, IP) ──► pick
                    │            tin_inform_dft.py              │
                    │            tin_inform_rank.py             │
                    └──────── remaining pool (n-Pr last) ──────┘
```

Driver: `src/gqe/eval/tin_inform_loop.py` + `jobs/tin_inform_loop_step.sbatch` (one ligand per job, then `sbatch` the next pick). Optional H-cGQE \(E_0\) is a **later column** on the ranked table, not the acquisition function.

## Loop

| Stage | What | Not |
|---|---|---|
| **Propose** | Ligand `R` on `RSn(OH)3`. Original unevaluated pool: **i-Pr, n-Pr, allyl, CF3**. | Not a GQE decoder emitting molecules. |
| **Score** | `src/gqe/eval/tin_inform_dft.py` → `results/tin_ab/methyltin_ours/scores.json` (plus optional per-ligand JSON). **ΔBDE**, adiabatic **IP**, Henke **13.5 nm proxy**. | Not MatGen-Q Table 1. Not 72.6→21.2 as a GQE number (that collapse is classical UCCSD(T) in [arXiv:2607.23988](https://arxiv.org/abs/2607.23988)). |
| **Pick** | `src/gqe/eval/tin_inform_rank.py` ranks scored rows and chooses the next pool member. | Not a GQE policy. |

CLI (repo root, `cudaq-env` not required for pick):

```bash
source .aire_scratch_env.sh
cd <repo-root>
PYTHONPATH=. python src/gqe/eval/tin_inform_rank.py \
  --scores results/tin_ab/methyltin_ours/scores.json \
  --out-dir results/tin_ab/methyltin_ours
```

Ligand ids in JSON match the score half (`me`, `et`, `nbu`, `vinyl`, `ph`, `ipr`, `npr`, `allyl`, `cf3`).

| Job | What |
|---|---|
| `jobs/tin_inform_dft.sbatch` | First wave array `me,et,nbu,vinyl,ph` |
| `jobs/tin_inform_loop_step.sbatch` | Score **pick.json** next ligand, re-rank, chain the following pick |
| `jobs/tin_inform_loop_remaining.sbatch` | Parallel DFT of remaining pool ids (`TIN_INFORM_CHAIN` off) |
| `jobs/tin_inform_e0.sbatch` | AVAS+CASCI+JW for `me`, `cf3`, `vinyl` (charge 0) |
| `jobs/tin_inform_e0_qsci.sbatch` | 1× L40S QSCI per CASCI Hamiltonian |

```bash
python src/gqe/eval/tin_inform_loop.py --status
sbatch jobs/tin_inform_loop_step.sbatch   # scores current pick (n-Pr as of 22 Aug 2026, job 7439561)
```

Missing `scores.json` is a hard error (no invented numbers). Empty ligand list: pick still runs the chemical rule; figures are skipped.

## Rank and pick rule

Lexicographic order on **scored** ligands:

1. **ΔBDE** = BDE(neutral) − BDE(cation) (kcal/mol), larger first (stronger ionisation-induced Sn–C collapse).
2. **Adiabatic IP** (eV), smaller first (easier to ionise).
3. **Atomic 13.5 nm proxy**: Henke imaginary form factor \(f_2\) sum at 91.84 eV (CXRO tables). Marker size / third bar only. **Not** a many-body absorption spectrum and **not** an IMePh 92 eV resource estimate.

If **≥ 3** scored rows have ΔBDE, IP, and BDE\(_0\), fit a **numpy-only** RBF GP (no sklearn) on fingerprint  
`[n_C, n_F, is_aryl, is_alkenyl, adiabatic_IP, BDE0] → ΔBDE`.  
Next ligand = max **expected improvement**. If the GP is shaky (LOO \(R^2<0.1\), underdetermined 6-D, class-wide IP/BDE\(_0\) imputation, flat EI, failed Cholesky), **do not trust EI**:

- **CF3** if electron-withdrawing is untested (`n_F=0` on all scored rows);
- else **allyl**;
- else i-Pr, then n-Pr.

Documented in `pick.json` (`rule`, `gp_shaky_reasons`).

## Optional E0 column (not the score)

After **our** XYZ exists, `src/gqe/eval/tin_inform_e0.py` builds an AVAS active space
(Sn 5s/5p + alpha-C 2p, Sayfutyarova arXiv:1701.07862; same *shells* as
MatGen-Q arXiv:2607.23988, **not** their unpublished XYZ), runs CASCI, and
exports a Jordan–Wigner JSON. QSCI primary sampler is **pool-native-uccsd**
(MatGen-Q `demo.pool` sized to `n_qubits`) on one L40S (`n_qubits ≤ 24`);
entangled-HF is a named control. Fair Me/CF3 pair: jobs **7441848** (CASCI)
and **7441849** (QSCI), shared AVAS `threshold=0.4` (`threshold_start=0.2`),
shells Sn 5s/5p + first-C 2p. Pool-native QSCI−CASCI from `e0.json`: Me
**15.841** mHa (22q / 11o / 14e), CF3 **5.868** mHa (18q / 9o / 12e). Not
trained 14-char H-cGQE (`n_q ≠ 14`). Vinyl first-wave E0 used thresh 0.2 /
24q and is a different experiment. Columns `e0_casci_ha` / `e0_qsci_ha` /
`delta_e_qsci_mha` land on the rank table **without** changing ΔBDE order.

The 14-character H-cGQE checkpoint is **not** applied unless that Hamiltonian
is 14 qubits. Do not pad 12/14-char words onto a 22q resist space.

```bash
# CASCI on nodes (array: me, cf3, vinyl; charge 0)
sbatch jobs/tin_inform_e0.sbatch
# QSCI: one GPU per ligand, waits on the matching CASCI array index
sbatch --dependency=aftercorr:<CASCI_JOBID> jobs/tin_inform_e0_qsci.sbatch
```

gpu4pyscf can accelerate **new** DFT on L40S (CUDA 12x, multi-GPU DF/direct SCF)
but must **not** be mixed into the already-scored B3LYP table.

Geometries live under `results/tin_ab/methyltin_ours/` with `label: ours`. Do not copy XYZ into `configs/tin_resist.yaml` (MatGen-Q rungs stay `geometry: null`).

## AIRE: 3× L40S split

*Paths/partition names shown are for the University of Leeds AIRE cluster — adapt for your site.*

Max **3 GPUs / node** (L40S 48 GB, PCIe, no NVLink). Never train or DFT on login-node A2 GPUs. Source `.aire_scratch_env.sh`; env `cudaq-env` for GPU QSCI.

Independent Hamiltonians scale as a **Slurm array of 1 GPU**, not as `nvidia,mgpu`
(CUDA-Q only distributes a statevector across GPUs above ~25 qubits). DFT remaining
ligands scale as a **nodes array** (`jobs/tin_inform_loop_remaining.sbatch`).

| Work | Partition | Typical flags |
|---|---|---|
| DFT / MP2 BDE + IP (**score**) | `nodes` | `--cpus-per-task=32 --mem=256G --time=08:00:00` |
| AVAS + CASCI + JW export | `nodes` | `--cpus-per-task=32 --mem=256G --array=0-2` |
| QSCI optional \(E_0\) | `gpu` | `--gres=gpu:1 --cpus-per-task=8 --mem=85G --array=0-2` |
| **Pick** (this script) | login or `nodes` | CPU, seconds; numpy + matplotlib |

Outputs: `pick.json`, `rank_table.json`, `fig_inform_ip_bde.png`, `fig_inform_rank.png`, per-ligand `e0.json`.
