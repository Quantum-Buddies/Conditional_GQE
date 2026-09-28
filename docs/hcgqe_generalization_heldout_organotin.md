# H-cGQE held-out organotin generalization — executable plan

**Status date:** 21 August 2026
**Verdict:** generalization is a **story ahead of the data**, not a measured capability.
**Factual audit + plots (21 Aug evening):** [`hcgqe_generalization_factual_brief.md`](hcgqe_generalization_factual_brief.md) and `results/generalization/` (inventory JSON, four figures). Tin contract Phases 0–7 unchanged; **§17 informatics loop appended**.
**Canonical tin contract (do not replace):** [`docs/tin_euv_execution_plan.md`](tin_euv_execution_plan.md)
**Materials-informatics loop (not GQE inverse design):** labelled-ours DFT 0/+1 **IP / Sn–C BDE ranking** on `RSn(OH)3` — [`tin_euv_execution_plan.md` §17](tin_euv_execution_plan.md); [`tin_inform_loop.md`](tin_inform_loop.md); opt scaffold [`methyltin_ours_dft_mp2_opt.md`](methyltin_ours_dft_mp2_opt.md).
**Do not edit:** the local plan file `industrial_tin_euv_gqe_f3714a97.plan.md` (outside the repo, in the author's Cursor plans directory)

One-sentence claim to test (not a result): a graph-conditioned H-cGQE decoder should emit a native-length UCCSD circuit for an unseen organotin topology in one forward pass, with ΔE vs CASCI beating per-instance VQE/GPT-2-GQE that otherwise sit at Hartree–Fock.

That claim is currently **unsupported**. The public checkpoint is a Hamiltonian-encoder + decoder (no GNN tensors). Frozen SnO 14q infer used the **12-char** dialect and sat at HF. There is **no** held-out aromatic-tin energy table. No Trackio project exists for this campaign. Scratch W&B runs on this account are `dino_foresight`, not H-cGQE.

---

## 0. Recovery note

The previous investigation agent died after ingesting the prompt (transcript is a single user turn; no files written). This document is the completed A/B/C deliverable. Numbers below are from on-disk JSON/checkpoints, not from that failed run.

---

## 1. Three scientific layers (keep them separate)

| Layer | What it is | In scope for H-cGQE on AIRE? | Honest metric |
|---|---|---|---|
| **(i) Ground-state \(E_0\)** | Variational / QSCI energy of an exported JW Hamiltonian vs CASCI/FCI in the **same** active space | **Yes** — this is what the transformer can emit | ΔE mHa vs CASCI; \(N_\mathrm{dets} < N_\mathrm{FCI}\); word length == \(n_q\) |
| **(ii) Neutral vs cation Sn–C** | Ionisation-induced change in Sn–C dissociation | **Only** after layer (i) is ≤ 1.6 mHa on **both** charges, on geometries we actually have | Sign/magnitude vs **our** CASCI (or published SI XYZ). **72.6 → 21.2 kcal/mol is classical UCCSD(T)** in [arXiv:2607.23988](https://arxiv.org/abs/2607.23988), never a GQE number |
| **(iii) 13.5 nm / Auger / 92 eV** | Photoionization, secondary-electron cascade, IMePh iodine-edge absorption | **Out of scope** on 1–3× L40S. ~200 logical qubits for IMePh 92 eV is [arXiv:2602.20234](https://arxiv.org/abs/2602.20234), not this repo | Do not run; do not score |

Industrial narrative (solubility switch / sub-7 nm pattern) is chemistry storytelling. Ground-state GQE does not simulate EUV photons.

---

## 2. What was actually trained and evaluated (Part A)

### 2.1 Checkpoints and dialects

| Artifact | Path | What it is |
|---|---|---|
| Public GIC generator | `results/train/h_cgqe_uccsd_model.pt` (31 MB, 27 Jun 2026); Hub [`Ryukijano/h-cgqe-gic2026`](https://huggingface.co/Ryukijano/h-cgqe-gic2026) | **7.85 M** params; vocab **149** (145 words + 4 special). Word lengths: 4×8, **12×59**, **14×20**, 16×19, 20×19, 22×20. SFT `best_val_loss` **1.183**. **Zero GNN / ChemistryEncoder tensors** |
| Mixed-length SFT (qBraid) | `results/train/h_cgqe_model_b200_sft.pt` | Vocab **317**; val acc **0.962**; **not** the SnO infer checkpoint |
| 14-char SnO SFT | `results/tin_ab/h_cgqe_sft_pool14_7431349.pt` (job **7431349**, COMPLETED 21 Aug 01:36) | Vocab **53**; 2 097 teacher sequences; **all 49 unique words length 14**; `best_val_loss` **3.948**; final val acc **0.101**; early-stopped at **26 / 80** epochs. **In-distribution on SnO**, not zero-shot |
| GNN (separate) | `results/train/chemistry_encoder.pt`, `graph_conditioning.pt` | Property **regression** heads (n_qubits, Pauli counts, …). **Not loaded by** `infer_h_cgqe.py` |

Hub card prose says “Chemical Graph Neural Network + Transformer”. The **weights used for SnO infer do not contain a GNN**. Conditioning at generate-time is `HamiltonianEncoder` over Pauli terms + coefficients (`src/gqe/models/h_cgqe_transformer.py`). `ChemistryEncoder` exists (`src/gqe/models/chemistry_encoder.py`) and `PERIODIC_TABLE["Sn"]=50` is in `graph_dataset.py`, but **infer does not call them**.

`build_length_token_mask(..., exact=False)` keeps `len(word) <= n_qubits`. On 14q that **licenses the 59-token 12-char cluster**. `--exact-word-length` now exists; frozen job 7386634 did **not** use it.

### 2.2 Training-set diversity (no Sn)

Supervised dataset summary (`results/train/gqe_dataset_summary.json`): **15 names**, 275 sequences (25 original + 10× coeff-noise augment):

`h2`, `h2_0.74`, `h2o_1.0_631g_cas8`, `lih`, `lih_1.6_*`, `beh2`, `beh2_1.3_full`, `n2`, `n2_1.1_*`, `iodobenzene_cas12`, `methyl_iodide_cas12`, `imeph_cas12`, `phenol_cas12`.

`configs/tin_resist.yaml` marks `sno_14q` / `methyltin*` / `butyltin_22q` as **`split: test`**. Methyltin/butyltin still have **`geometry: null`**. Sn SMILES tokens were added **after** the GIC SFT vocab was frozen.

There is **no organotin**, **no aromatic tin**, and **no held-out topology energy table** in the training JSON.

### 2.3 Logging

| Logger | Finding |
|---|---|
| Trackio | `trackio list projects` is empty. No tin/H-cGQE Trackio Space. |
| W&B on `$SCRATCH/wandb` | Recent runs are `dino_foresight.train`, not Conditional-GQE. |
| JSON | Real numbers live under `results/eval/`, `results/tin_ab/`, `results/phase3_final/`. |

### 2.4 Plot-ready table — GIC generator on **seen** small molecules

Source: `results/eval/h_cgqe_evaluation_gic2026.json` (100 samples; `energy_error` is mHa vs the file’s `reference_energy`). **Seen / in-family**, not held-out tin.

| Molecule | n_q (typical) | Split | ΔE vs ref (mHa) | HF-like baseline (mHa) | ΔE − HF (mHa) | Note |
|---|---|---|---:|---:|---:|---|
| h2_0.74 | 4 | seen | 20.54 | 20.38 | +0.16 | At HF |
| h2_2.0 | 4 | seen | 164.77 | 164.81 | −0.04 | At HF |
| lih_1.6 | 12 | seen | 1.88 | 1.81 | +0.07 | Near HF |
| beh2_1.3 | 14 | seen | 56.94 | 33.76 | **+23.18** | **Worse than HF** |
| n2_1.1 | 20 | seen | 128.83 | 126.55 | +2.28 | Near HF |
| iodobenzene | 8 | seen | 3.53 | 1.96 | +1.57 | Worse than HF |
| methyl_iodide | 8 | seen | 1.51 | 4.71 | **−3.20** | Beats this file’s HF; **not** the 0.63 mHa L-BFGS number |
| imeph | 8 | seen (wrong-then-fixed isomer) | 25.39 | 19.02 | +6.38 | Worse than HF |
| phenol | 12 | seen | 45.30 | 44.81 | +0.50 | At HF |

**CH₃I 8q chemical-accuracy number** (separate, after L-BFGS-B): **0.63 mHa** vs CASCI(4e,4o) in `results/phase3_final/benchmark_ch3i_consolidated.json` (CUDA-Q GQE 2.65 mHa; HEA-VQE 988 mHa). That is a **seen iodine toy**, not organotin zero-shot. Hub card reports this 0.63 mHa; it also states “Held-out / zero-shot molecule generalization remains an open evaluation item.”

`results/phase3_final/transfer_learning_dataset.json` is SMILES tokenisation of 10 molecules (vocab 53) — **not** an energy transfer table. `h_cgqe_operators_for_qsci.json` “transfer_from_*” entries pad operators with identity and self-label as **not** a converged H-cGQE result.

### 2.5 Plot-ready table — SnO 14q (the only public tin Hamiltonian)

CASCI = **−288.10912104 Ha**. Active-space correlation (RHF − CASCI) = **6.120 mHa**. \(N_\mathrm{FCI}\) = **49**. Identity: Sn–O = 1.8325 Å, def2-SVP + ECP28MDF, (2e,7o).

| Column | Backend | Seen vs held-out | ΔE vs CASCI (mHa) | \(N_\mathrm{dets}\) | Notes |
|---|---|---|---:|---:|---|
| HF / RHF | NumPy S4 & CUDA-Q 7431345 | n/a | **+6.120** | 1 | Floor if the circuit does not excite |
| Frozen H-cGQE (`h_cgqe_uccsd_model.pt`) | Job 7386634 (wrong endianness) | **unseen Sn**, **wrong dialect** | +6.022 | 13 | 16 samples → **3 unique** sequences; **318/318 operators length 12** |
| Frozen H-cGQE 12-char padded | NumPy S4 & CUDA-Q 7431345 | same | **+6.120** | 1 | Mapping-fixed; sits **exactly at HF** |
| MatGen-Q GPT-2 smoke best | Demo A JSON | instance-trained GPT-2 | **+2.869** | — | Teacher target |
| Same circuit, our NumPy S4 | `qsci_controls_cpu_numpy.json` | control | **+2.882** | 17 | +0.013 vs Demo A |
| Same circuit, CUDA-Q 7431345 | `qsci_controls_7431345.json` | control | **+2.152** | 17 | Shot-noise; still ≫ 1.6 |
| Best native 14-char single (NumPy) | `pool14q_random_2` | pool control | **+3.934** | 11 | Not a trained policy |
| Best native 14-char single (CUDA-Q) | `pool14q_single_157` | pool control | **+4.382** | 2 | |
| Pool union greedy (NumPy) | 5 circuits | pool control | **+0.953** | **21** | S4 pass; **not** 49/49 |
| Pool union greedy (CUDA-Q 7431345) | 5 circuits | pool control | **+1.422** | **21** | S4 still true (`gate1_pool_native_pass`) |
| UCCSD-VQE (MatGen-Q) | `results_vqe.json` | **per-instance** | **+0.000012** | full ansatz | 48 params, 3824 CX, 490 energy evals |
| ADAPT-VQE (MatGen-Q) | `results_adapt_vqe.json` | **per-instance** | **+0.000089** | 20 ops | 3112 energy evals |
| Demo A union | smoke GPT-2 | instance | **+0.000** | **49/49** | Tautological FCI; **forbidden as a GQE win** |
| Demo B Emerald replay | counts JSON | hardware protocol | **+0.575** | 31 | **0 cr**; live wallet **~677 cr** |
| Paper Emerald (published) | [arXiv:2607.23988](https://arxiv.org/abs/2607.23988) | their GPT-2 | **+0.330** | — | Not our circuit |
| 14-char H-cGQE infer | Job **7434600** | in-distribution SFT | **PENDING — no ΔE yet** | — | `tin_infer_pool14`; do not invent scores |

**GNN at SnO 14q infer: unused.** Frozen decode is Hamiltonian-conditioned, 12-char padded, HF energy.

**User thesis vs this table.** On the only public tin H, per-instance UCCSD-VQE and ADAPT-VQE **do not** fall back to HF — they solve the 49-determinant space. Frozen H-cGQE **does** fall back to HF, because of dialect, not because VQE is weak. Zero-shot GNN emission is **not measured**.

### 2.6 Job status (21 Aug 2026 afternoon)

| Job | Name | State | Implication |
|---|---|---|---|
| 7431345 | `tin_qsci_ctrl` | **COMPLETED** 0:0, 35 s | GPU S4 confirm; union +1.422 mHa / 21 dets |
| 7431349 | `tin_sft_pool14` | **COMPLETED** 0:0, 1 m 31 s | 14-char ckpt on disk; val acc 10 % |
| 7434600 | `tin_infer_pool14` | **PENDING (Priority)** 21 Aug evening | No logs/JSON; **do not invent** ΔE / \(N_\mathrm{dets}\) / word lengths |
| 7431324 DAPO | — | cancelled earlier | Stay cancelled |

---

## 3. Literature (Part B) — what GQE can and cannot claim

`parallel-cli` is installed but **does not run on this login node** (`GLIBC_2.35` missing). Citations below are from Hugging Face Papers API, arXiv abs/html, and web search. **No GIC score sheet is public**; do not invent one.

### 3.1 GQE / conditional / graph-conditioned generation

- **GPT-QE / GQE (Nakaji et al., not “Nakagawa”):** [arXiv:2401.09253](https://arxiv.org/abs/2401.09253), [HF paper](https://huggingface.co/papers/2401.09253). Decoder-only transformer emits operator tokens for **a given Hamiltonian**; proof-of-concept on electronic-structure H, including N₂ dissociation. Instance / pretrain-then-finetune, **not** a molecular-graph zero-shot GNN.
- **Conditional-GQE (Minami, Nakaji, Suzuki, Aspuru-Guzik, Kadowaki):** [arXiv:2501.16986](https://arxiv.org/abs/2501.16986), journal [Digital Discovery, DOI:10.1039/D5DD00138B](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00138b). Encoder–decoder + **GNN on Ising graphs** for **combinatorial optimisation**, ≤ 10 qubits, ~99 % on **new CO problems**. This is the closest published “GNN conditions a GQE decoder” result — **Ising CO, not organotin \(E_0\)**.
- **SpinGQE:** [arXiv:2603.24298](https://arxiv.org/abs/2603.24298). Decoder-only GQE for **spin** Hamiltonians.
- **ADAPT-GQE:** [arXiv:2607.22468](https://arxiv.org/abs/2607.22468) (*Learning to Prepare Molecular Ground States with Transformer Models*). Train on ADAPT-VQE circuits, then **one forward pass** for a new Hamiltonian; RL can exceed the teacher; demo on **imipramine**; Quantinuum Helios-1. Adjacent “learn the circuit” lane — **not** a tin-resist substitute ([tin plan §10](tin_euv_execution_plan.md)).

### 3.2 MatGen-Q tin paper (baseline, not ours)

- [arXiv:2607.23988](https://arxiv.org/abs/2607.23988) / [HF](https://huggingface.co/papers/2607.23988); code [KarimElgammal/gqe-qsci-euv-photoresists](https://github.com/KarimElgammal/gqe-qsci-euv-photoresists). GPT-2 GQE + QSCI on monoalkyltin oxo-hydroxides. **72.6 → 21.2 kcal/mol is classical UCCSD(T)** on methyltin ionisation, quoted in the abstract as the solubility switch — **not** a GQE energy. Public companion ships **SnO 14q only**; **no methyltin XYZ**.
- Hardware: SnO 14q **+0.330 mHa** on IQM Emerald; n-butyl 22q **+3.92 mHa** (depth-truncated) then recovery **+0.18–0.21 mHa**. Our replay of their counts: **+0.575 mHa / 0 cr**.

### 3.3 GIC 2026 (no score sheets)

- Challenge text: [pqic.org/challenge](https://www.pqic.org/challenge) (Mitsubishi Chemical & AIST Advanced Materials / GQE / EUV semiconductor materials). Announcement: [Connected DMV](https://www.connecteddmv.org/news/gic-2026-mitsubishi-aist). Ceremony: [Quantum World Congress 2026](https://www.quantumworldcongress.com/2026). **Public pages do not publish a held-out organotin topology rubric or numerical score sheet.** Competition is closed; this experiment is independent.

### 3.4 EUV tin-resist chemistry vs ground-state GQE

These papers describe **Sn–C homolysis driven by 13.5 nm photons and secondary electrons**. They are **not** CASCI/QSCI observables.

- Brainard-style mechanism slides: Sn–C ~2.0 eV vs C–C ~3.6 eV; secondary electrons ~20–80 eV; organotin → tin oxide solubility switch — [euvlitho.com/2019/P35.pdf](https://www.euvlitho.com/2019/P35.pdf).
- n-Butyltin–oxo cages under EUV: volatile butane/butene/octane; Φ ≈ 5 Sn–C cleavages per absorbed photon (initial) — [J. Mater. Chem. C, DOI:10.1039/D5TC04167H](https://pubs.rsc.org/en/content/articlehtml/2026/tc/d5tc04167h).
- Review of metal-based EUV resists / Sn12 cages (Inpria lineage) — [J. Photopolym. Sci. Technol. 35, 81 (2022)](https://www.jstage.jst.go.jp/article/photopolymer/35/1/35_81/_pdf/-char/ja).
- Sn 4d photoemission peaking near 92 eV — [ARCNL / J. Micro/Nanolith. (OA PDF)](https://ir.arcnl.nl/pub/176/00162OA.pdf).
- Low-energy-electron patterning of TinOH — [arXiv:1910.02511](https://arxiv.org/abs/1910.02511).
- **92 eV IMePh absorption resource estimate (~200 logical qubits)** — [arXiv:2602.20234](https://arxiv.org/abs/2602.20234). Out of scope here. YAML isomer is already **4-iodo-2-methylphenol** `Oc1ccc(I)cc1C`.

Layer (i) GQE can compute **ground-state \(E_0\)** of a chosen active space. It does **not** compute Auger cascades, secondary-electron yields, or 13.5 nm cross-sections.

### 3.5 Held-out graph generalization in ML for chemistry

These are **classical GNN property** papers. They motivate **leave-one-topology-out** tests; they are **not** GQE energy numbers.

- OOD molecular graphs / invariant subgraphs — [J. Cheminformatics 2025, 10.1186/s13321-025-01142-w](https://link.springer.com/article/10.1186/s13321-025-01142-w).
- Dual-axis coverage of organic topologies (H/C/N/O/F; **not Sn**) — [arXiv:2512.14418](https://arxiv.org/abs/2512.14418).
- Soft-causal OOD on molecule graphs — [arXiv:2505.06283](https://arxiv.org/abs/2505.06283).
- Multi-fidelity GNN transfer — [PMC11258334](https://pmc.ncbi.nlm.nih.gov/articles/PMC11258334/).

**Implication:** even mature GNNs treat scaffold/topology shift as a hard OOD problem. A 15-molecule H/C/N/O/I SFT set plus an unwired Sn token does **not** establish aromatic-tin zero-shot circuits.

---

## 4. Held-out protocol (no invented MatGen-Q XYZ)

### 4.1 Geometry rules

1. **Never** invent Cartesian coordinates for `CH3Sn(OH)3` / `n-BuSn(OH)3` and write them into `configs/tin_resist.yaml`. Those rungs stay `geometry: null` until SI XYZ exists (`docs/methyltin_xyz_author_request.md` is an **unsent** draft).
2. **Labelled-ours** opts only under `results/tin_ab/methyltin_ours/` with provenance JSON (`docs/methyltin_ours_dft_mp2_opt.md`). Recipe: PySCF def2-SVP + ECP on Sn, B3LYP then MP2, charge 0/+1. Score vs **our** CASCI, not Table 1 of 2607.23988.
3. Public **SMILES / CID** are allowed as **graph identities** for the GNN and as opt starting points **after** a documented conformer (PubChem 3D if present). SMILES ≠ Hamiltonian.

Public organotin **graphs** (SMILES only; no XYZ claimed):

| Role | Identity | SMILES | Provenance |
|---|---|---|---|
| Train-simple Sn (already exported) | SnO 14q | `[Sn]=O` | MatGen-Q companion geometry 1.8325 Å |
| Train-simple alkyl (ours opt, later) | methyltin trihydroxide | `C[Sn](O)(O)O` | connectivity only until ours XYZ |
| Chain-length holdout (ours opt, later) | n-butyltin | `CCCC[Sn](O)(O)O` | connectivity; CAS 2273-43-0 / [PubChem CID 16767](https://pubchem.ncbi.nlm.nih.gov/compound/16767) is the **oxide** `CCCC[Sn](O)=O`, not the trihydroxide — do not conflate |
| Aromatic holdout | tetraphenyltin | `c1ccc([Sn](c2ccccc2)(c3ccccc3)c4ccccc4)cc1` | [PubChem CID 61146](https://pubchem.ncbi.nlm.nih.gov/compound/61146) |
| Aromatic holdout | triphenyltin hydroxide | `O[Sn](c1ccccc1)(c2ccccc2)c3ccccc3` | [PubChem CID 9907219](https://pubchem.ncbi.nlm.nih.gov/compound/9907219) |

If PubChem 3D exists, use it as **ours starting guess** and re-opt. If it does not, skip that molecule. **No MatGen-Q Table 1 coordinates from SMILES.**

### 4.2 Train / test split (once geometries exist)

**Train (seen):** GIC light set (H₂, LiH, BeH₂, H₂O, N₂, phenol, CH₃I) **plus** at least one **simple Sn** (SnO 14q native 14-char pool; optionally ours-opt methyltin 0 if the opt finishes). Operator vocab must be **exact-length** for each \(n_q\).

**Hold out (unseen topologies):**

- Aromatic Sn (tetraphenyltin / Ph₃SnOH) — topology shift the GIC-judge story actually cares about.
- Chain-length: n-Bu vs Me **only if both have ours or SI XYZ**.
- Charge: methyltin +1 **only** as layer (ii), after layer (i) ≤ 1.6 mHa.

**Do not** train on the holdout SMILES, then call infer “zero-shot”.

### 4.3 Metrics (every row)

For each molecule × charge × method:

| Field | Requirement |
|---|---|
| ΔE vs **CASCI** (mHa) | Same H, same active space |
| ΔE vs **HF** | Detect HF-fallback |
| ΔE vs per-instance **UCCSD-VQE** / **ADAPT-VQE** / MatGen-Q **GPT-2** | Fair; report energy **and** n_eval / wall time |
| \(N_\mathrm{dets}\), union cap | At 14q cap **32**; if \(N_\mathrm{dets}=N_\mathrm{FCI}\), report **best single circuit** only |
| Word length | Must equal \(n_q\); 12-char pad is a **fail** |
| Graph distance | Tanimoto (Morgan r=2) and atom-type histogram vs nearest train graph; report both |
| Split label | `seen` / `held-out` / `in-distribution-SFT` (SnO pool14 is the last) |

Chemical accuracy: **≤ 1.6 mHa vs CASCI**.

### 4.4 GNN ablation (mandatory; currently impossible on the public ckpt)

Wire `ChemistryEncoder.to_prefix_token` into `HcGQEModel.generate` **after** 14-char infer works. Then four columns on the **same** held-out H:

1. **GNN + HamiltonianEncoder + decoder** (the thesis).
2. **HamiltonianEncoder + decoder** (today’s architecture).
3. **Decoder-only** (zero the encoder memory / learned molecule ID).
4. **HF** (no circuit).

If (1) ≉ (2) ≉ (3) ≈ HF on unseen Sn, the GNN is not doing the work. The existing `graph` vs `flat` tables (`results/tables/conditioning_ablation_main_summary.json`, **5 samples**, Hamiltonian **property** MAE, val MAE hundreds of Hartree-scale units) are **not** this ablation.

### 4.5 Dependency: 14-char SFT infer before any generalization claim on SnO

Exact-length mask is implemented. Job **7431349** wrote a real ckpt. **Do not DAPO. Do not iodine-DAPO onto SnO. Do not paid QPU.**

```bash
source .aire_scratch_env.sh
cd <repo-root>
export TIN_SFT_CKPT=<repo-root>/results/tin_ab/h_cgqe_sft_pool14_7431349.pt
sbatch jobs/tin_infer_pool14.sbatch
```

Pass bar for **in-distribution** SnO (still not zero-shot): all emitted words length 14 and in `pool_sno14q.json`; `n_samples≥64`; unique sequences ≫ 3; best single **or** capped union (\(N_\mathrm{dets}≤32\)) **< 1.6 mHa** vs CASCI. Expectation: val acc 10 % and 26-epoch early stop make this **likely to miss**; then rebuild teacher data / train longer **on 14-char vocab**, still **not** DAPO-from-12-char.

Only after that bar: consider GNN wiring + multi-molecule 14-char SFT. Held-out aromatic tin remains **gated on XYZ**.

---

## 5. Unblocked now vs gated

### Unblocked now (1× L40S, `cudaq-env`, source scratch env, no login A2)

1. **Wait** for job **7434600** (`tin_infer_pool14`, already queued). When COMPLETED, log ΔE vs CASCI, \(N_\mathrm{dets}\), word-length histogram from JSON. Do not invent scores while PENDING.
2. Dump infer JSON next to `results/tin_ab/hcgqe_infer_sno14q_7386634.json` for an honest 12-char vs 14-char pair.
3. Optionally Trackio-init a **private** project when that job starts (none exists today).
4. Send or drop the methyltin SI request (`docs/methyltin_xyz_author_request.md`); start **ours** DFT/MP2 on `--partition=nodes` when a labelled starting guess exists.

### Gated

| Gate | Until |
|---|---|
| Zero-shot organotin table | Held-out XYZ (SI or ours opt) **and** GNN actually in `generate()` **and** 14-char policy |
| Layer (ii) 0 vs +1 Sn–C | Both charges ≤ 1.6 mHa QSCI; still vs CASCI, not 72.6→21.2 |
| `tin_dapo.sbatch` | S5: 14-char frozen/SFT ≤ 1.6 mHa on SnO **only**; molecule list SnO only |
| Paid Emerald | Live ~**677 cr** ≪ 6360; replay already +0.575 mHa / 0 cr |
| 92 eV / ~200q IMePh | Never on 3× L40S |
| “GNN zero-shot” talk | Ablation table in §4.4 |

---

## 6. AIRE job shapes (copy, do not improvise)

*Paths/partition names shown are for the University of Leeds AIRE cluster — adapt for your site.*

```
# infer / QSCI (unblocked)
#SBATCH --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=85G --time=02:00:00

# ours DFT/MP2 (when starting guess exists)
#SBATCH --partition=nodes --cpus-per-task=32 --mem=256G --time=08:00:00

# DAPO (gated)
#SBATCH --partition=gpu --gres=gpu:3 --cpus-per-task=24 --mem=200G --time=1-00:00:00
# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET=Socket
```

Env: `source .aire_scratch_env.sh`; `conda activate cudaq-env`; modules `cuda/12.6.2` + `miniforge/24.7.1` in GPU jobs. Checkpoints/logs on `$SCRATCH`. No API keys in sbatch.

---

## 7. Answer to the parent agent

**(a) Metrics.** See §2.4–2.5. Headline: frozen H-cGQE on SnO 14q = **+6.120 mHa / 1 det (HF)** with **12-char** words; pool-native S4 union = **+0.953 mHa / 21 dets** (NumPy) and **+1.422 mHa / 21 dets** (CUDA-Q job 7431345); MatGen-Q VQE/ADAPT on the same H are **~0.000 mHa**; CH₃I 8q L-BFGS = **0.63 mHa** (seen); 14-char SFT ckpt exists; infer job **7434600** **PENDING** (no ΔE yet); GNN **not used** at infer. Informatics loop = labelled-ours DFT IP / Sn–C BDE ranking, not GQE proposing molecules (§8).

**(b) Literature.** GPT-QE [2401.09253](https://arxiv.org/abs/2401.09253); conditional-GQE GNN is **Ising CO** [2501.16986](https://arxiv.org/abs/2501.16986); tin baseline [2607.23988](https://arxiv.org/abs/2607.23988); ADAPT-GQE [2607.22468](https://arxiv.org/abs/2607.22468); EUV Sn–C chemistry is photon/electron-driven, not GQE. No public GIC score sheet.

**(c) This file:** `<repo-root>/docs/hcgqe_generalization_heldout_organotin.md`

**(d) Capability vs story:** **story ahead of the data.**

---

## 8. Materials-informatics loop (pointer)

Industrial ranking **without** MatGen-Q SI XYZ is **labelled-ours DFT 0/+1 IP / Sn–C BDE** on `RSn(OH)3`, **not** GQE proposing molecules. GQE stays optional native-length \(E_0\) vs **our** CASCI after ours geometries exist. **92 eV / Auger remains out of scope.** Never copy ours XYZ into `configs/tin_resist.yaml` (MatGen-Q rungs stay `geometry: null`).

- Contract note: [`tin_euv_execution_plan.md` §17](tin_euv_execution_plan.md)
- Opt scaffold: [`methyltin_ours_dft_mp2_opt.md`](methyltin_ours_dft_mp2_opt.md)
- Dedicated loop file: [`tin_inform_loop.md`](tin_inform_loop.md)

Do **not** DAPO. Do **not** `DEMOB_REAL`. Job **7434600** is left in the queue.

---

## Sources

- [The generative quantum eigensolver (GQE) … (Nakaji et al.)](https://arxiv.org/abs/2401.09253) (Jan 2024)
- [HF paper 2401.09253](https://huggingface.co/papers/2401.09253)
- [Conditional-GQE / GQCO (Minami et al.)](https://arxiv.org/abs/2501.16986) (Jan 2025)
- [Digital Discovery conditional-GQE](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00138b)
- [SpinGQE](https://arxiv.org/abs/2603.24298)
- [ADAPT-GQE / Learning to Prepare Molecular Ground States](https://arxiv.org/abs/2607.22468) (2026)
- [MatGen-Q tin GQE+QSCI](https://arxiv.org/abs/2607.23988) (2026)
- [MatGen-Q GitHub companion](https://github.com/KarimElgammal/gqe-qsci-euv-photoresists)
- [HF paper 2607.23988](https://huggingface.co/papers/2607.23988)
- [IMePh 92 eV resource estimate](https://arxiv.org/abs/2602.20234)
- [GIC 2026 challenge](https://www.pqic.org/challenge)
- [GIC Mitsubishi/AIST announcement](https://www.connecteddmv.org/news/gic-2026-mitsubishi-aist)
- [Quantum World Congress 2026](https://www.quantumworldcongress.com/2026)
- [Organotin EUV ambient/Sn–C slides](https://www.euvlitho.com/2019/P35.pdf)
- [EUV reactions of tin–oxo cages](https://pubs.rsc.org/en/content/articlehtml/2026/tc/d5tc04167h)
- [Chemical mechanisms of metal-based EUV resists](https://www.jstage.jst.go.jp/article/photopolymer/35/1/35_81/_pdf/-char/ja)
- [EUV photoemission of a tin photoresist](https://ir.arcnl.nl/pub/176/00162OA.pdf)
- [Patterning Sn-based EUV resists with low-energy electrons](https://arxiv.org/abs/1910.02511)
- [OOD molecular graph GNNs](https://link.springer.com/article/10.1186/s13321-025-01142-w)
- [Dual-axis organic topology coverage](https://arxiv.org/abs/2512.14418)
- [Soft causal molecular OOD](https://arxiv.org/abs/2505.06283)
- [Multi-fidelity GNN transfer](https://pmc.ncbi.nlm.nih.gov/articles/PMC11258334/)
- [PubChem tetraphenyltin CID 61146](https://pubchem.ncbi.nlm.nih.gov/compound/61146)
- [PubChem triphenyltin hydroxide CID 9907219](https://pubchem.ncbi.nlm.nih.gov/compound/9907219)
- [PubChem butyltin oxide CID 16767](https://pubchem.ncbi.nlm.nih.gov/compound/16767)
- [H-cGQE model card](https://huggingface.co/Ryukijano/h-cgqe-gic2026)
- [S4 dump dataset](https://huggingface.co/datasets/Ryukijano/sno14q-gqe-qsci-s4)
