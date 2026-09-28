# H-cGQE graph-conditioned zero-shot: factual brief

**Date:** 21 August 2026  
**Verdict:** **No.** The Hub/README claim that a chemistry GNN conditions the decoder for zero-shot organotin circuits is **not supported** by checkpoints, infer code, or logged energies. What exists is Hamiltonian-conditioned generation on small organics, plus a frozen 12-char dialect failure on SnO 14q.

Canonical tin contract (unchanged): [`tin_euv_execution_plan.md`](tin_euv_execution_plan.md). Executable holdout protocol: [`hcgqe_generalization_heldout_organotin.md`](hcgqe_generalization_heldout_organotin.md). Industrial plan file was not edited.

Plots / JSON from this audit: `results/generalization/`.

---

## What generalization evidence exists (numbers + paths)

### 1. Training sets — no Sn in SFT or DAPO

| Set | Names / graphs | Sequences | Sn? | Path |
|---|---|---|---|---|
| UCCSD SFT (public ckpt) | **15 names**, **9 unique connectivities** (H₂, H₂O, LiH, BeH₂, N₂, iodobenzene, methyl iodide, IMePh, phenol) | 275 (25 original + 10× coeff-noise) | **No** | `results/train/gqe_dataset_summary.json`, `results/train/gqe_supervised_dataset.pt` |
| UCCSD teacher subset | 11 names, vocab 149 | 121 | **No** | `results/train/uccsd_dataset/gqe_dataset_summary.json` |
| Vanilla DAPO | `h2`, `lih`, `beh2`, `n2` | 200 epochs | **No** | `results/train/h_cgqe_rl_ablation_vanilla_dapo_rl_metrics.json` |
| qBraid DAPO/QD-GRPO | **32 names** (adds HF, NH₃, CH₄, CO, benzene, toluene, anisole, o-cresol, diarylethene, IMePh, iodobenzene, …) | 24 epochs; mean reward 2.047 → 1.043 | **No Sn** | `results/train/h_cgqe_model_qbraid_rl_rl_metrics.json` |
| 14-char SnO SFT | **1 graph** (`[Sn]=O`), 2097 teacher sequences from the MatGen-Q 14-char pool | Job **7431349**, 26/80 epochs, early stop | In-distribution SFT, **not** zero-shot | `results/tin_ab/sft_pool14/sft_pool14_summary.json` |

`configs/tin_resist.yaml` marks `sno_14q` / `methyltin*` / `butyltin_22q` as `split: test`. Methyltin/butyltin still have `geometry: null`. Sn SMILES + `PERIODIC_TABLE["Sn"]=50` were added for wiring **after** the GIC vocab was frozen; they do not appear in SFT `names` or DAPO molecule lists.

IMePh: SFT/eval used the **wrong isomer** (2-iodo-6-methylphenol Cartesian / old SMILES). YAML was corrected **21 Aug 2026** to 4-iodo-2-methylphenol `Oc1ccc(I)cc1C`. Those eval numbers are not the Mitsubishi isomer and are **not** 92 eV absorption ([arXiv:2602.20234](https://arxiv.org/abs/2602.20234), ~200 logical qubits — out of scope).

### 2. GNN — exists as a separate regressor; **not used at infer**

| Item | Fact | Path |
|---|---|---|
| Architecture | `ChemistryEncoder`: edge-aware MPNN + `to_prefix_token`. **Saved ckpt is 2 layers, dim 32**, not the Hub’s “3-layer” cartoon. | `src/gqe/models/chemistry_encoder.py`, `results/train/chemistry_encoder.pt` |
| What it was trained to predict | Hamiltonian **properties** (n_qubits, Pauli counts, …), 5 samples | `results/tables/conditioning_ablation_main_summary.json` (val MAE ~275–309, Hartree-scale units) |
| Decoder at generate time | `HamiltonianEncoder` over Pauli strings + coefficients → cross-attention | `src/gqe/models/h_cgqe_transformer.py` `HcGQEModel.generate` |
| Infer | **Zero** matches for `ChemistryEncoder` / `prefix` / `graph` | `src/gqe/models/infer_h_cgqe.py` |
| Public / SnO ckpts | Prefixes `encoder.*` + `decoder.*` only. **0 GNN-like tensors.** Vocab has **0** `Sn` operator words. | `results/train/h_cgqe_uccsd_model.pt` (7.85 M params, vocab 149); Hub [`Ryukijano/h-cgqe-gic2026`](https://huggingface.co/Ryukijano/h-cgqe-gic2026) ships `h_cgqe_rl_gic2026.pt` of the same family |

There is **no** GNN-on vs GNN-off circuit ablation. The `graph` vs `flat` table is property regression, not ΔE.

### 3. Metrics that actually exist

**Public GIC SFT** (`h_cgqe_uccsd_model.pt`): `best_val_loss` **1.183**, 278 logged epochs, final val acc **0.992**. Word lengths: 4×8, **12×59**, 14×20, 16×19, 20×19, 22×20 (`results/tin_ab/sft_vocab_word_lengths.json`). Length mask `<= n_qubits` licenses the 12-char cluster on 14q.

**Seen-molecule energies** (`results/eval/h_cgqe_evaluation_gic2026.json`, 100 samples; ΔE mHa vs file `reference_energy`):

| Molecule | H-cGQE ΔE | HF gap | ΔE − HF | Reading |
|---|---:|---:|---:|---|
| h2_0.74 | 20.54 | 20.38 | +0.16 | At HF |
| h2_2.0 | 164.77 | 164.81 | −0.04 | At HF |
| lih_1.6 | 1.88 | 1.81 | +0.07 | Near HF |
| beh2_1.3 | 56.94 | 33.76 | **+23.18** | Worse than HF |
| n2_1.1 | 128.83 | 126.55 | +2.28 | Near HF |
| iodobenzene | 3.53 | 1.96 | +1.57 | Worse than HF |
| methyl_iodide | 1.51 | 4.71 | −3.20 | Beats this file’s HF; **not** the 0.63 mHa L-BFGS number |
| imeph (old isomer) | 25.39 | 19.02 | +6.38 | Worse than HF |
| phenol | 45.30 | 44.81 | +0.50 | At HF |

**CH₃I 8q L-BFGS** (seen toy, not tin): **0.63 mHa** vs CASCI(4e,4o); CUDA-Q GQE 2.65 mHa; HEA-VQE 988 mHa. `results/phase3_final/benchmark_ch3i_consolidated.json`. Hub card quotes 0.63 mHa and already states zero-shot “remains an open evaluation item.”

**SFT vs DAPO ablation** (`results/phase3_final/ablation_sft_vs_rl.json`): RL helps H₂ (20.52 → 0.15 mHa) and does **not** rescue N₂ (~127 mHa), BeH₂ (~34 mHa), phenol (~45 mHa). No Sn.

**SnO 14q** — only public tin Hamiltonian. CASCI **−288.10912104 Ha**; AS correlation **6.120 mHa**; \(N_\mathrm{FCI}=49\). Dataset dump: [`Ryukijano/sno14q-gqe-qsci-s4`](https://huggingface.co/datasets/Ryukijano/sno14q-gqe-qsci-s4).

| Column | ΔE vs CASCI (mHa) | \(N_\mathrm{dets}\) | Source |
|---|---:|---:|---|
| HF | **+6.120** | 1 | `qsci_controls_cpu_numpy.json` |
| Frozen H-cGQE 12-char (job 7386634, wrong endianness) | +6.022 | 13 | `hcgqe_qsci_7386634.json`; 16 samples → **3 unique** sequences; **318/318 ops length 12** |
| Frozen H-cGQE 12-char padded (mapping fixed) | **+6.120** | 1 | NumPy S4 and CUDA-Q 7431345 |
| MatGen-Q smoke-best (NumPy) | +2.882 | 17 | vs Demo A +2.869 |
| Same circuit (CUDA-Q 7431345) | +2.152 | 17 | shot noise |
| Best pool-14 single (NumPy) | +3.934 | 11 | `pool14q_random_2` — not a trained policy |
| Pool union greedy (NumPy) | **+0.953** | **21** | S4 pass; not 49/49 |
| Pool union greedy (CUDA-Q) | +1.422 | 21 | `gate1_pool_native_pass: true` |
| UCCSD-VQE (per-instance) | **+1.2×10⁻⁵** | 48 params / 490 evals | MatGen-Q `results_vqe.json` |
| ADAPT-VQE (per-instance) | **+8.9×10⁻⁵** | 20 ops / 3112 evals | `results_adapt_vqe.json` |
| **14-char H-cGQE SFT best single (job 7434600)** | **+2.597** | **21** | `hcgqe_qsci_pool14_7434600.json`; native 14-char; GNN unused; **s4_pass false** (misses 1.6) |
| 14-char H-cGQE SFT union cap 32 | **+2.149** | **32** | same file; not 49/49 |

**14-char SFT (in-distribution SnO, GNN unused):** `best_val_loss` **3.948**, final val acc **0.101**, early-stopped **26/80**. `results/tin_ab/h_cgqe_sft_pool14_7431349_metrics.json`. Infer used that ckpt; this is **not** zero-shot.

Closest “train A, infer B” **energy** proxy: organic mixed-length SFT → SnO 14q. It failed at **HF** because of **12-char dialect**, not because VQE is weak (VQE solves this H). Identity-padded `transfer_from_*` rows in `results/eval/h_cgqe_operators_for_qsci.json` are labelled **not** converged H-cGQE. `transfer_learning_dataset.json` is SMILES tokens, not energies.

---

## What does **not** exist

- Held-out organotin (methyltin / n-Bu / aromatic Sn) ΔE table  
- Any ΔE vs number of unique training graphs / epochs **generalization series** (do not invent one)  
- GNN prefix inside `generate()` for tin or organics  
- Sn in SFT names or DAPO molecule lists  
- 14-char H-cGQE QSCI on SnO is **done** (job 7434600: best +2.597 mHa / 21 dets; union32 +2.149 mHa; misses 1.6 mHa)  
- Methyltin **MatGen-Q SI** XYZ (blocked: `results/tin_ab/methyltin_xyz_blocker.json`). Labelled-ours DFT XYZ exists under `results/tin_ab/methyltin_ours/` — do not copy into `tin_resist.yaml`.  
- Trackio project (CLI broken on login Python 3.9; no trackio files in this repo). Scratch W&B runs are `dino_foresight.train`  
- GIC 2026 public score sheet (competition closed; independent paper experiment)  
- 72.6 → 21.2 kcal/mol as a GQE number ([arXiv:2607.23988](https://arxiv.org/abs/2607.23988) **classical UCCSD(T)**)  
- 92 eV IMePh on L40S ([arXiv:2602.20234](https://arxiv.org/abs/2602.20234))

Published “GNN conditions GQE” result is **Ising combinatorial optimisation**, ≤10 qubits ([arXiv:2501.16986](https://arxiv.org/abs/2501.16986) / Digital Discovery), **not** organotin \(E_0\).

---

## Is the GNN used at infer for tin?

**No.** Frozen SnO decode loads `h_cgqe_uccsd_model.pt`, encodes the **Pauli Hamiltonian**, and emits **12-character** UCCSD words that are trailing-I padded to 14q. `ChemistryEncoder` is not imported. After the endianness fix, QSCI is exactly HF (+6.120 mHa, 1 det).

---

## Recommended metric protocol (molecules we can run on AIRE)

1. **Dialect is measured:** 14-char infer on SnO is job **7434600**. In-distribution: every word length 14; \(n_\mathrm{samples}=64\); best single **+2.597 mHa / 21 dets**; union cap 32 **+2.149 mHa**. **Misses 1.6 mHa.** This is **not** zero-shot. **Do not DAPO.**  
2. **Wire GNN only after that.** Four columns on the **same** H: GNN+HEnc+dec / HEnc+dec / decoder-only / HF.  
3. **Train diversity vs held-out ΔE** (the plot that does not exist today): x = number of unique train connectivities (exact-length vocab); y = zero-shot ΔE vs CASCI on held-out topologies.  
   - **Train (seen):** GIC light set + SnO 14q. Optional labelled-ours methyltin 0 after documented DFT (`docs/methyltin_ours_dft_mp2_opt.md`).  
   - **Hold out:** aromatic Sn only after XYZ exists; n-Bu vs Me only if **both** have ours or SI XYZ. **Do not invent MatGen-Q XYZ.**  
   - **AIRE sizes:** statevector **≤24q**; QSCI/MPS **28–32q**.  
   - Every row: ΔE vs CASCI, vs HF, vs per-instance VQE/ADAPT/GPT-2; \(N_\mathrm{dets}\); word length \(= n_q\); Tanimoto + atom-type histogram vs nearest train graph; split label `seen` / `held-out` / `in-distribution-SFT`.  
4. **Do not** sbatch DAPO, paid QPU, or 92 eV. Do not plot 72.6→21.2 as model accuracy.

---

## Artefacts written this audit

| Path | Contents |
|---|---|
| `results/generalization/hcgqe_generalization_inventory.json` | Machine-readable inventory |
| `results/generalization/factual_metrics_table.csv` | Compact table including explicit `DOES_NOT_EXIST` rows |
| `results/generalization/fig1_sno14q_de_vs_casci.{png,pdf}` | SnO 14q ΔE columns (existing only) |
| `results/generalization/fig2_seen_molecule_de_vs_hf.{png,pdf}` | Seen GIC eval vs HF |
| `results/generalization/fig3_sft_pool14_loss.{png,pdf}` | 14-char SFT loss (in-distribution) |
| `results/generalization/fig4_vocab_and_diversity.{png,pdf}` | Vocab lengths + diversity counts (zeros are real) |
| `results/generalization/plot_factual_inventory.py` | Regenerator |

No ΔE-vs-n_graphs curve was drawn: that series is not in the logs.
