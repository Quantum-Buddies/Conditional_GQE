# Tin-resist GQE+QSCI execution plan

**Status date:** 20 August 2026
**Teams:** Quantum-Buddies / Gyanateet (Ryukijano) on University of Leeds AIRE
**Use case:** Mitsubishi Chemical & AIST GIC 2026 EUV photoresist chemistry
**Competition posture:** Phase 3 completed; not 1st and not runner-up (private notification). Public awards remain at [Quantum World Congress 2026, 23–25 September, College Park](https://www.quantumworldcongress.com/2026). A 50% QWC ticket is a private-email benefit, not a scientific deliverable.
**This document:** how to *execute* the industrial tin-resist GQE+QSCI experiment on leftover QPU credits and AIRE GPUs. It is not a consolation write-up.

Canonical stacks:

| Role | Path | Upstream |
|---|---|---|
| H-cGQE (ours) | `<repo-root>` | [Quantum-Buddies/Conditional_GQE](https://github.com/Quantum-Buddies/Conditional_GQE) |
| MatGen-Q baseline | `$GQE_MATGENQ_DIR` (clone of the gqe-qsci-euv-photoresists baselines repo) | [KarimElgammal/gqe-qsci-euv-photoresists](https://github.com/KarimElgammal/gqe-qsci-euv-photoresists) |
| Preprint | arXiv:[2607.23988](https://arxiv.org/abs/2607.23988) | Elgammal & Maußner |
| Checkpoint | `results/train/h_cgqe_uccsd_model.pt` (31 MB, 27 Jun 2026) | Hub: [Ryukijano/h-cgqe-gic2026](https://huggingface.co/Ryukijano/h-cgqe-gic2026) — **ground-state circuit generator** |

Scratch-root `$SCRATCH` has **no git remote**. Cloud orchestrate kickoff is unavailable here (`CURSOR_API_KEY` unset; `bun` missing). This plan is the local execution contract.

---

## 1. What “doing it” means

The Mitsubishi/AIST brief, scored the way the published tin paper scores itself:

1. **Industrial resist chemistry** on `CH3Sn(OH)3`, `n-BuSn(OH)3`, and the SnO fragment; **charge 0 and +1**; Sn–C ionisation collapse reported against **classical** references (UCCSD(T)/DFT), never as a GQE trophy.
2. **Same Hamiltonian recipe as published:** def2-SVP + ECP28MDF on Sn; SnO by valence energy window; resists by AVAS (Sn 5s/5p + C 2p, then +O 2p).
3. **GQE circuits + QSCI**, chemical accuracy **≤ 1.6 mHa vs CASCI/FCI** on rungs where the published ladder supports that claim (SnO through 32q; methyltin through 30q; n-butyl 22q both charges).
4. **Classical baselines on AIRE `nodes`** (there is no `cpu` partition).
5. **Optional QPU:** SnO 14q **QSCI energy vs CASCI**, matching the public hardware number **+0.330 mHa on IQM Emerald** ([arXiv:2607.23988](https://arxiv.org/abs/2607.23988); Demo B companion **+0.575 mHa**). Not 8q bitstring-fidelity demos.
6. **Judge-reproducible smoke + artefact dump** (GATE 1 assert, JSON, logs, `seff` when present).

**Forbidden score:** 72.6 → 21.2 kcal/mol Sn–C collapse is **classical UCCSD(T)** in the preprint. Do not write it as a GQE win. GQE+QSCI is always vs CASCI/FCI on the same active space.

**Out of scope on L40S:** 92 eV / ~200-logical-qubit IMePh absorption. Track B (correct IMePh isomer) is not a tin blocker.

---

## 2. Scoring card

| Quantity | Truth | Pass |
|---|---|---|
| ΔE vs CASCI (mHa) | Identical H as MatGen-Q | ≤ 1.6 mHa at SnO 14q **without** filling all 49 determinants; then SnO 16–24q on L40S SV; 28–32q MPS |
| Best single-circuit ΔE | One sampled circuit | Report separately from GEVP/union |
| Union / GEVP ΔE | Cross-circuit subspace | Valid only if \(N_\mathrm{dets} < N_\mathrm{FCI}\). At 14q, 49/49 is tautological CASCI |
| Depth, \(N_\mathrm{ops}\), CX | Emitted words | Compare to GPT-2 GQE at matched depth |
| GPU-hours | `seff` after the job | Same error, less or equal wall-clock |
| Sn–C collapse | Neutral vs cation **at accurate QSCI rungs only** | Sign/magnitude vs their chemistry story; energies still vs CASCI |
| QPU | QSCI energy vs CASCI | IQM Emerald 14q; reject 8q fidelity |

Active-space correlation on published SnO 14q: **6.12 mHa** (RHF −288.10300087 Ha, CASCI −288.10912104 Ha). A solver sitting at HF is a **+6.02 mHa** failure, not a near-miss.

---

## 3. AIRE contract (do not improvise)

*Paths/partition names shown are for the University of Leeds AIRE cluster — adapt for your site.*

Source first, always:

```bash
source .aire_scratch_env.sh
```

Env: `conda activate cudaq-env`.
Modules in GPU jobs: `cuda/12.6.2` + `miniforge/24.7.1`.
**Never train on login-node A2 GPUs.** Probe CUDA-Q only inside `srun -p gpu --gres=gpu:1`.

Verified `sinfo` partitions (20 Aug 2026): `nodes`, `gpu`, `himem`, `teachingnodes`. **No `cpu`.** Wall-clock cap on `nodes` / `gpu` / `himem`: **2 days**.

| Job class | Partition | Flags | Why |
|---|---|---|---|
| Classical ladder, CASCI/CCSD | `nodes` | `--cpus-per-task=32 --mem=256G --time=08:00:00` | 168-core / 768 GB class. Not GPU. |
| SnO 14–24q QSCI / controls | `gpu` | `--gres=gpu:1 --cpus-per-task=8 --mem=85G --time=12:00:00` | 1× L40S 48 GB. Default without CPU/mem is 1 CPU / 1 GB. |
| DAPO (only if Phase 4 opens) | `gpu` | `--gres=gpu:3 --cpus-per-task=24 --mem=200G --time=1-00:00:00` | Max 3 GPUs/node. PCIe, no NVLink. |
| CASCI blow-up | `himem` | 2.3 TB | Only if 256G OOM (32q FCI 166M dets). |

L40S NCCL (3-GPU jobs only):

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Statevector cap: **24q** on PCIe L40S (`cuStateVec` distributed mode segfaults at 25q). Above 24q: `tensornet-mps`, not `nvidia-mqpu` MPI.

`seff` is **not** on compute-node PATH. Every tin `jobs/*.sbatch` already uses:

```bash
command -v seff >/dev/null && seff "${SLURM_JOB_ID}" || echo "seff not on PATH"
```

Job **7386634** science finished; Slurm state **FAILED 127** solely because the submitted script called `seff` unguarded. Do not re-run GATE 1 to “fix” that.

Optional: `aire-agent validate jobs/tin_qsci_controls.sbatch` from `$HOME/.aire-agent` before submit. Submit with raw `sbatch`, not MCP.

---

## 4. Inventory already on disk (verified)

### 4.1 MatGen-Q companion (public)

Demo A / Demo B / classical / VQE all share **one** instance: SnO (2e,7o) = 14 qubits, Sn–O = 1.8325 Å, def2-SVP + ECP28MDF, GATE 1 CASCI **−288.10912104 Ha**.

No `.xyz` for `CH3Sn(OH)3` or `n-BuSn(OH)3` exists in the public repo. **Do not invent coordinates.**

`SnOMolecule` in `code/demo/molecule.py` already accepts `n_active_electrons` / `n_active_orbitals`. SnO **ladder** rungs can be rebuilt from this class (energy window). Resist rungs cannot, until SI geometries exist.

### 4.2 H-cGQE tin wiring (already added)

| Item | Location | Note |
|---|---|---|
| Sn SMILES token | `src/gqe/data/smiles_encoder.py` | `MULTI_CHAR_TOKENS` includes `Sn`; maps `sno_14q`, `methyltin`, cations, `butyltin_22q` |
| Z=50 GNN | `src/gqe/data/graph_dataset.py` | `PERIODIC_TABLE["Sn"] = 50` |
| Config | `configs/tin_resist.yaml` | `geometry: null` on all non-SnO rungs |
| Exporter | `src/gqe/data/export_matgenq_hamiltonian.py` | **Must** call their `SnOMolecule`. `generate_hamiltonians.py` / openfermionpyscf STO-3G is a different molecule |

### 4.3 Job 7386634 (`tin_qsci`, 20 Aug 2026, gpu020, 1× L40S, 3m57s)

| Gate | Result |
|---|---|
| Classical ladder | RHF −288.10300087 / CASCI −288.10912104 / CCSD −288.42351063 / CCSD(T) −288.44244265 Ha |
| GATE 1 | **PASSED** (\|dev\| = 2.12×10⁻⁹ Ha) |
| JW cross-check | 5.68×10⁻¹¹ mHa vs CASCI |
| Demo A SMOKE union | **+0.000 mHa, 49/49 dets** — tautological full CI |
| Demo A SMOKE best single | **+2.869 mHa** |
| Exported H | `sno_14q`, **14 qubits**, **1366 Pauli terms**, CASCI −288.109121037883 Ha |
| Frozen infer | 16 samples → **3 unique sequences**; **318/318 operators are 12-char** |
| Frozen QSCI | **+6.022 mHa** vs CASCI, 13 unique bitstrings — HF-like (AS correlation is 6.12 mHa) |

Artefacts: `results/tin_ab/{hamiltonians_tin/hamiltonians.json,pool_sno14q.json,hcgqe_infer_sno14q_7386634.json,hcgqe_qsci_7386634.json,sft_vocab_word_lengths.json}`.

MatGen-Q 14-char pool is already exported: **289 tokens, 49 unique words, all length 14** (`pool_sno14q.json`, from their `demo/pool.py::build_pool`).

### 4.4 SFT checkpoint dialect

`h_cgqe_uccsd_model.pt`: 7,705,397 parameters; vocab 149 (145 operator words + 4 special). Word-length histogram: 4×8, **12×59**, **14×20**, 16×19, 20×19, 22×20.

`build_length_token_mask` allows `len(word) <= n_qubits`. On 14q that **licenses the 12-char majority**. `qsci._make_hcgqe_kernel` then **trailing-I pads** to 14, leaving qubits 12–13 unexcited. Prior Cepheus dry-runs already rejected “malformed/padded Pauli words (all operators must be explicit 14-character IXYZ strings)”.

**Do not DAPO this checkpoint onto SnO.** The file `results/tin_ab/sft_vocab_word_lengths.json` already records that STOP.

### 4.5 IMePh YAML (Track B, not a blocker)

`configs/gic2026_molecules.yaml` `imeph`: OH on ring C1, iodine ortho, methyl on the other ortho → **2-iodo-6-methylphenol**. Mitsubishi/AIST IMePh is **4-iodo-2-methylphenol**. Wrong isomer. Do not spend tin cycles on it.

---

## 5. Operator-dialect STOP (must pass before DAPO)

Root cause of +6.022 mHa, in order:

1. SFT vocab is a **mixed-size** UCCSD dialect from GIC light molecules (4–22q), not MatGen-Q’s 14-char (2e,7o) pool.
2. Length mask is `<= 14`, so the policy emits the 59-token 12-char cluster.
3. Trailing-I pad ≠ a 14q excitation. Occupied JW qubits are 0–1; the last two virtuals never see X/Y.
4. 16 samples collapsed to 3 unique sequences. QSCI support stays near HF (13 dets, +6.022 mHa).

**Correct fix is native 14-character words**, not a smarter pad. Leading-I pad would skip occupied orbitals; trailing-I pad skips virtuals. Either is a hole in the (2e,7o) generator.

### STOP gates (execute in order; halt on red)

| ID | Gate | Pass | Fail → |
|---|---|---|---|
| S0 | GATE 1 CASCI | −288.10912104 Ha to 1e−6 | Already passed (7386634). Do not rebuild STO-3G. |
| S1 | Exported H | `n_qubits==14`, 1366 terms, CASCI matches GATE 1 | Re-run exporter only; never `generate_hamiltonians.py` |
| S2 | Emitted words | Every operator `len(w)==14` and in the MatGen-Q 14-char pool (or an explicit 14q UCCSD rebuild) | **STOP. No DAPO.** Fix mask/vocab. |
| S3 | Sequence diversity | Unique sequences ≫ 3 at `n_samples≥16` | Raise temperature / rebuild vocab; do not reward-hack |
| S4 | Honest QSCI | Best single-circuit ΔE < 1.6 mHa **and** \(N_\mathrm{dets}<49\), **or** pool-native control column passes `tin_qsci_controls.py` | If pool-native also fails, the sampler/QSCI stack is broken — debug `qsci.py`, do not DAPO |
| S5 | Frozen H-cGQE | After S2, frozen (or 14q-SFT) policy ≤ 1.6 mHa vs CASCI without 49/49 fill | Only then open Phase 4 DAPO, **SnO 14q only** |
| S6 | Methyltin / n-Bu | Public XYZ or author SI | **Blocked.** No invented geometry. SnO ladder (energy window) may proceed. |
| S7 | QPU | Simulator QSCI ≤ 1.6 mHa on the **same** circuits; then IQM Emerald energy vs CASCI | No 8q fidelity; no Cepheus-as-tin-A/B |

`jobs/tin_dapo.sbatch` currently lists `sno_14q methyltin_14q methyltin_cation_14q`. The last two **are not in** `hamiltonians_tin/hamiltonians.json`. Submitting it now would train on missing molecules and a 12-char dialect. **Do not submit.**

---

## 6. Ordered phases

### Phase 0 — Identity of the Hamiltonian (done)

Clone MatGen-Q; run `classical_baseline.py` and `SMOKE=1 python run_demo.py` on 1× L40S; export H via `export_matgenq_hamiltonian.py`.

**STOP if GATE 1 fails.** It passed.

Do **not** resubmit `jobs/tin_qsci.sbatch`. That job already proved the H and the 12-char failure mode.

### Phase 1 — Wire Sn into H-cGQE (done)

SMILES token, Z=50, `configs/tin_resist.yaml`, exporter. Methyltin/butyltin entries are **stubs** (`geometry: null`).

### Phase 1.5 — Honest four-column QSCI control (NEXT; no training)

Existing files:

- `src/gqe/eval/tin_qsci_controls.py`
- `jobs/tin_qsci_controls.sbatch`

Columns on the **identical** exported H:

1. HF only (expect ~+6.12 mHa).
2. MatGen-Q smoke best circuit (tokens from `code/results_demo.json`; reported +2.869 mHa).
3. Frozen 12-char H-cGQE (7386634 infer; expect ~+6.02 mHa).
4. Native 14-char pool sequences (random + short singles from `pool_sno14q.json`).

Pass criterion in the script: pool-native ΔE < 1.6 mHa **without** filling 49/49. If a union hits 49 dets, the honest number is the **best single circuit**.

**SBATCH shape (already in file):**

```
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=12:00:00
```

**STOP:** if column 4 cannot beat HF by a non-tautological margin, debug sampling/QSCI before touching the transformer.

### Phase 2 — Fix the 14q operator dialect (no DAPO)

Do this on 1× L40S after Phase 1.5, not on the login node.

**2A. Exact-length decode (cheap probe).**
`build_length_token_mask` currently keeps `len(word) <= n_qubits`. Add an **exact** mask (`len(word) == n_qubits`) and pass it from `infer_h_cgqe.py` when `--exact-word-length` is set. That restricts the SFT head to the **20 native 14-char words** already in the checkpoint.

Re-infer `n_samples≥64`, temperature ≥ 1.0. Require S2 (all words length 14). Then QSCI.

Expected: still weak (20 tokens vs MatGen-Q’s 49 unique / 289 angled tokens). Useful only as an ablation.

**2B. Replace the vocabulary (the real fix).**
Build the H-cGQE vocab from `pool_sno14q.json` (289 tokens, all 14-char), **not** from the mixed SFT vocab. Options, in preference order:

1. Supervised warm-start **on that pool** using MatGen-Q smoke/full Demo A sequences as teacher (1× L40S, `jobs/` new `tin_sft_pool14.sbatch` when written; until then a 1-GPU `train_supervised.py` / `train_h_cgqe.py` run).
2. If no teacher sequences beyond the smoke best circuit: SFT on synthetic UCCSD sequences from `src/gqe/common/operator_pool.py` **restricted to 14q words**, then freeze-eval.
3. `train_rl_dapo.py --from-scratch` on `sno_14q` **only after** a 14-char vocab exists. That is Phase 4, not 2B.

Do **not** trailing-I pad 12-char SFT words into this vocab.

**2C. Frozen eval.**
`infer_h_cgqe.py` + `src/gqe/eval/qsci.py` on the 14-char policy. Metric: best single-circuit and a **capped** union (\(N_\mathrm{dets} \le 32\) at 14q so it cannot silently become FCI).

**STOP:** if 2C still sits at HF, inspect whether `exp_pauli` words match pool endianness (MatGen-Q interleaved α/β, HF on qubits 0–1). Do not open 3-GPU DAPO to paper over a mapping bug.

### Phase 3 — Frozen H-cGQE + QSCI vs GPT-2 GQE + QSCI

On the **same** H:

| Column | How |
|---|---|
| MatGen-Q GPT-2 GQE+QSCI | `run_demo.py` full (40 iters), not only SMOKE |
| H-cGQE frozen / 14q-SFT | Phase 2C |
| UCCSD-VQE | `vqe_baseline.py` (rubric quantum baseline) |
| ADAPT-VQE | `adapt_vqe_baseline.py` |
| Classical | already on disk for 14q; extend on `nodes` for larger SnO rungs |

Reuse `jobs/tin_qsci.sbatch` **only after** infer uses 14-char words. Patch the infer invocation: `--n-samples 64` (not 16), `--exact-word-length` once implemented.

**SnO ladder while methyltin XYZ is blocked.** `SnOMolecule(n_active_electrons=…, n_active_orbitals=…)` plus the paper’s energy window:

| Rung | Space | Qubits | \(N_\mathrm{dets}\) | Backend on AIRE | Paper GQE+QSCI (mHa) |
|---|---|---|---|---|---|
| GATE 1 | (2e,7o) | 14 | 49 | 1× L40S SV | 0.000 (union) |
| | (2e,9o) | 18 | 81 | 1× L40S SV | 0.000 |
| | (10e,8o) | 16 | 3.1k | 1× L40S SV | 0.12 / 0.10 |
| | (12e,10o) | 20 | 44.1k | 1× L40S SV | 0.73 |
| | (14e,12o) | 24 | 627k | 1× L40S SV | 0.31 (depth 80) |
| | (14e,14o) | 28 | 11.8M | 1× L40S MPS | 0.98 |
| | (16e,16o) | 32 | 166M | 1× L40S MPS; CASCI on `himem` if needed | 1.45 |

Only 14q has a **published asserted** GATE 1 constant in the companion. Higher rungs: compute CASCI ourselves with the same class, then compare ΔE to Table 1 as a **cross-check**, not as a hardcoded assert. Extend `export_matgenq_hamiltonian.py` with `--n-active-electrons` / `--n-active-orbitals`; do not copy paper CASCI digits by hand.

L40S SV stops at 24q. 28–32q: MPS (`--bond-dims 64 128`) and report bond-dimension sensitivity. Multi-node 34–40q exact SV (Leonardo 256 GPU) is **not** an AIRE capability (max 3 GPUs/node).

### Phase 4 — DAPO only if Phase 3 misses 1.6 mHa

File: `jobs/tin_dapo.sbatch` (already written; **edit before submit**).

Required edits:

1. `--molecules sno_14q` only until XYZ exists.
2. `--hamiltonians` pointing at a JSON whose operators/vocab are **14-char**.
3. Checkpoint: 14q-SFT weights from Phase 2B, **not** raw `h_cgqe_uccsd_model.pt`.
4. `--use-cuda --use-bf16`; keep NCCL Socket flags; `--max-qubits 24`.
5. `--gate-auxiliary-rewards` remains on.

**SBATCH shape (already in file):**

```
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
```

Do not launch from scratch on 3 GPUs until 1-GPU SFT on the 14-char pool has a non-collapsed prior (workspace rule: SFT → DAPO, not RL-from-scratch on unseen 14q tin).

### Phase 5 — Classical ladder on `nodes`

File: `jobs/tin_casci_refs.sbatch`.

```
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
```

14q classical is already done on GPU (8.9 s). Re-running on `nodes` is optional provenance. **Required** once SnO 20q+ or methyltin H exist: HF / MP2 / CCSD / CASCI into the Hamiltonian JSON. OOM → `himem`.

Methyltin ionisation collapse table (when XYZ exists): QSCI single points vs CASCI on each charge; **UCCSD(T) 72.6 → 21.2 kcal/mol stays in the classical column**.

### Phase 6 — Optional QPU (cheapest valid scientific shot)

Scientific target: **SnO 14q QSCI energy vs CASCI**, the published hardware protocol — not IonQ/IQM 8q bitstring fidelity (already 87.5% on methyl iodide; that number is not this paper).

**Do not put H-cGQE circuits on a QPU until Phase 2C/3 simulator QSCI is ≤ 1.6 mHa with native 14-char words.** Hardware cannot repair a 12-char dialect.

Ordered spend (stop at the first row that yields a citable energy):

| Priority | Shot | Device | Credits | Why this is valid |
|---|---|---|---|---|
| 0 | Replay shipped Emerald counts | local | **0** | `DEMOB_REPLAY=demo_b_counts_emerald.json python run_demo_b.py` → +0.575 mHa, 31/49 dets. Judge-reproducible, no key. |
| 1 | Local protocol | CPU/GPU | **0** | `DEMOB_DRY=1 python run_demo_b.py` |
| 2 | qBraid QIR simulator | `qbraid:qbraid:sim:qir-sv` | **0** | Same 20-job protocol; published sim +1.120 mHa |
| 3 | Open Quantum IQM Emerald | `openquantum:iqm:qpu:emerald` | **0 qBraid**; billed to OQ Spark (~$50 / 90 days) | Same 54q QPU as the paper. MatGen-Q notes `DEMOB_SHOTS=2000 DEMOB_CAL_SHOTS=500` fits the free allowance. Jul 2026 preflight listed this route ONLINE at 0.00 qBraid credits/task. |
| 4 | qBraid AWS IQM Emerald, reduced | `aws:iqm:qpu:emerald` `DEMOB_REAL=1` `DEMOB_SHOTS=1000` | **~3800 cr** estimated in companion | Valid energy-vs-CASCI protocol; 20 jobs with calibration |
| 5 | Full published Demo B | same, 4×5000 + 16×1000 | **6360 cr** (measured 2026-07-14) | Direct A/B to +0.575 mHa / paper +0.330 mHa |
| — | H-cGQE 14-char circuits, same calibration | Emerald only | budget after 3–5 | Only if sim QSCI ≤ 1.6 mHa |
| Forbidden | 8q fidelity on Forte/Emerald | any | any | Wrong metric |
| Forbidden | Rigetti Cepheus as tin A/B | Cepheus | cheap shots, **wrong device** | Keep Cepheus for H2/FMO provenance only |
| Forbidden | IonQ Forte energy on 14q | Forte | 8 cr/shot | Orders of magnitude too expensive |

qBraid list prices ([docs.qbraid.com/v2/home/pricing](https://docs.qbraid.com/v2/home/pricing), verified 20 Aug 2026): IQM Emerald **30 cr/task + 0.16 cr/shot**; Cepheus-1-108Q **30 + 0.0425/shot**; IonQ Forte-1 **30 + 8/shot**. Open Quantum devices cost **0 qBraid credits** and spend the linked OQ wallet ([Open Quantum × qBraid](https://docs.qbraid.com/v2/account/integrations/openquantum)).

Do **not** buy qBraid GPU hours for this. AIRE already has L40S.

### Phase 7 — Artefact dump (judge-reproducible)

Dump under `$SCRATCH` (not `$HOME`, not `$TMP_SHARED`):

```
results/tin_ab/
  hamiltonians_tin/hamiltonians.json
  pool_sno14q.json
  qsci_controls_<jobid>.json
  hcgqe_infer_*.json
  hcgqe_qsci_*.json
  logs + seff text
```

Then a Hugging Face **dataset** (not a second copy of the ground-state model card) with Hamiltonians, circuits, eval JSON, and a README that states: GATE 1 constant, 12-char failure, 14-char fix, ΔE vs CASCI. Hub model [Ryukijano/h-cgqe-gic2026](https://huggingface.co/Ryukijano/h-cgqe-gic2026) stays a **circuit generator**, not a tin-energy oracle.

IMePh isomer fix is Track B after tin A/B, not part of this dump.

---

## 7. Existing sbatch files (exact)

All live in `<repo-root>/jobs/`. All source `.aire_scratch_env.sh`, activate `cudaq-env`, and make `seff` optional.

| File | When to submit | Do not submit if |
|---|---|---|
| `tin_qsci_controls.sbatch` | **Now** (Phase 1.5) | Inputs missing (they are not) |
| `tin_qsci.sbatch` | After Phase 2 infer is 14-char | 12-char checkpoint still wired |
| `tin_casci_refs.sbatch` | SnO 20q+ export, or methyltin XYZ arrives | Only 14q (already have the ladder) |
| `tin_dapo.sbatch` | Phase 4 STOP S5 failed **and** 14-char vocab exists | methyltin names still on the CLI; raw SFT ckpt |

`tin_qsci.sbatch` still runs GATE 1 + Demo A smoke + export + 16-sample infer. After Phase 2, drop the GATE 1/smoke block (proven) and raise `--n-samples`.

---

## 8. QPU credit ledger (no secrets)

Inventory from repo docs, Phase 3 manifests, and env **names** only. Live wallet is not in git; re-read on [account.qbraid.com](https://account.qbraid.com) before any paid job. `QBRAID_API_KEY` is present in the AIRE environment; `~/.netrc` has a W&B machine entry. Do not print values. Do not commit keys. Do not put keys in sbatch.

| Source | What it records | Implication |
|---|---|---|
| `docs/QBRAID_STRATEGY.md` / credit-usage notes | Planning budget **11,000** qBraid credits (mid-challenge) | Historical planning figure |
| `submission/.../efficiency_metrics.json` | **13,400** available, **612** used (4.6%) at Phase 3 close | Documented leftover **~12,788** qBraid credits |
| Same file | Rigetti Cepheus 3 tasks × 4096 shots; IQM Emerald 1×1024 shots (**193.84 cr**, 8q **fidelity**) | That Emerald spend is **not** a tin energy result |
| `results/qpu/iqm_ionq_rigetti_preflight_20260726_v3/` | Open Quantum Garnet/Emerald/Forte listed **0.00** qBraid cr; AWS Cepheus pilot **204.08 cr** | OQ is the first live hardware path |
| MatGen-Q README | Full Demo B Emerald = **6360 cr**; 1000-shot variant ≈ **3800 cr** | Fits inside the leftover ledger several times over |
| GIC leftover providers | qBraid + IQM + IonQ + Rigetti from prior work | Spend IQM Emerald on **14q QSCI**; leave IonQ unused for tin |

**Budget rule:** one scientific Emerald energy (priority 3 or 4) ≫ ten 8q fidelity screens. Cap any single paid job with `DEMOB_MAX_CREDITS` (companion default 7000). If the live wallet is an org wallet showing `qbraidCredits: 0`, transfer or use Open Quantum; do not debug by resubmitting.

---

## 9. What is blocked until methyltin XYZ / SI exists

Blocked:

- Neutral vs cation **A/B on CH3Sn(OH)3**
- n-BuSn(OH)3 22q industrial ligand
- AVAS Sn–C vs Sn–O crosslinking rungs (22/26/30q+)
- Scoring the ionisation-induced Sn–C collapse as a QSCI-verified curve (CASCI differences at those geometries)
- `tin_dapo.sbatch` molecule list items `methyltin_*`

Not blocked:

- SnO 14q operator-dialect fix and honest QSCI
- SnO 16–24q SV ladder from `SnOMolecule` (same public geometry)
- SnO 28–32q MPS
- Classical 14q (done)
- Demo B Emerald energy vs CASCI
- Track B IMePh isomer (later, separate)

**Do not** guess Cartesian coordinates from SMILES. Ask MatGen-Q authors for SI geometries or wait for a public dump. Until then the executable industrial fragment is **SnO**, which is the paper’s validation ladder and the only public Hamiltonian.

---

## 10. Track B (explicitly later)

- Replace `imeph` YAML with 4-iodo-2-methylphenol.
- 92 eV iodine-edge / ~200 logical qubits: **out of scope** on 3× L40S.
- ADAPT-GQE ([arXiv:2607.22468](https://arxiv.org/abs/2607.22468)) is a crowded “learn the circuit” lane; do not chase it as a tin substitute.

---

## 11. Git / secrets hygiene

- Tin code (`configs/tin_resist.yaml`, `jobs/`, `src/gqe/data/export_matgenq_hamiltonian.py`, `src/gqe/eval/tin_qsci_controls.py`, `results/tin_ab/`) is **untracked** on `main` tracking `origin/main` (`Quantum-Buddies/Conditional_GQE`).
- Do not commit API keys, `.netrc`, or credential-bearing remote URLs.
- Checkpoints (`*.pt`) stay on `$SCRATCH`; do not add them to git.
- Hugging Face uploads use the Hub CLI / stored credential helper, not tokens pasted into sbatch.

---

## 12. Notion

This plan is mirrored onto the existing page [Implementation Plan: H-cGQE vs MatGen-Q tin-resist A/B](https://app.notion.com/p/3c2868e41fc08100a534dea8758d899e) (appended 20 Aug 2026 evening). Canonical file remains this markdown on scratch.

---

## 13. Single next command

Inputs for Phase 1.5 are on disk. The job does **not** train and does **not** DAPO. It is the S4 existence proof that 14-char pool circuits QSCI-sample this H.

```bash
source .aire_scratch_env.sh
cd <repo-root>
sbatch jobs/tin_qsci_controls.sbatch
```

Watch: `squeue --me` and `tail -f logs/tin_qsci_ctrl_*.out`.
After it finishes: read `results/tin_ab/qsci_controls_<jobid>.json`. If `gate1_pool_native_pass` is true, start Phase 2B (14-char vocab). If false, debug QSCI/sampling. **Do not** `sbatch jobs/tin_dapo.sbatch`.

---

## Sources

- [arXiv:2607.23988](https://arxiv.org/abs/2607.23988) — MatGen-Q tin-resist GQE+QSCI (Aug 2026)
- [KarimElgammal/gqe-qsci-euv-photoresists](https://github.com/KarimElgammal/gqe-qsci-euv-photoresists) — public SnO 14q companion
- [Ryukijano/h-cgqe-gic2026](https://huggingface.co/Ryukijano/h-cgqe-gic2026) — H-cGQE ground-state generator
- [Quantum World Congress 2026, 23–25 Sep](https://www.quantumworldcongress.com/2026)
- [GIC 2026 timeline (pqic.org)](https://www.pqic.org/challenge) — Phase 3 winners notified by 12 Aug 2026; ceremony at QWC
- [qBraid v2 pricing](https://docs.qbraid.com/v2/home/pricing) — Emerald 0.16 cr/shot; Cepheus 0.0425; Forte 8
- [Open Quantum × qBraid](https://docs.qbraid.com/v2/account/integrations/openquantum) — 0 qBraid credits; Spark allowance
- [AIRE docs](https://arcdocs.leeds.ac.uk/aire/) — partitions, storage, GPU nodes
- [arXiv:2607.22468](https://arxiv.org/abs/2607.22468) — ADAPT-GQE (adjacent, not this experiment)

---

## 14. Execution log (21 Aug 2026, ~00:40 BST) — STOP gate + independent tracks

- **STOP:** job **7431320** S4 failed (`gate1_pool_native_pass` false). Mapping debug continues. **No DAPO. No QPU spend.** Pending DAPO job **7431324** was cancelled (`scancel`) before it left `PD`.
- **imeph-trackb:** `configs/gic2026_molecules.yaml` `imeph` is **4-iodo-2-methylphenol** `Oc1ccc(I)cc1C` (CAS 60577-30-2); SMILES map corrected (was benzyl iodide `ICc1ccccc1`). Geometry = PubChem CID 143713 3D conformer. Stated CAS **(4e,4o)**. **92 eV not run** (out of scope).
- **qpu-emerald (zero-cost):** `DEMOB_REPLAY=demo_b_counts_emerald.json python run_demo_b.py`; `DEMOB_REAL` unset. Union QSCI **−288.10854577 Ha** vs CASCI **−288.10912104 Ha** = **+0.575 mHa**, 31/49 dets, this-run cost **0.00 cr**. Live qBraid wallet ~**677 cr**, AWS **0**, auto-recharge off (SDK; no keys logged). Browser UI for account.qbraid.com / Open Quantum was not available in this session.
- **methyltin-xyz:** author-request draft in `docs/methyltin_xyz_author_request.md` (**unsent**). Labelled-ours DFT/MP2 scaffold: `configs/methyltin_ours_opt.yaml` + `docs/methyltin_ours_dft_mp2_opt.md`. MatGen-Q rungs stay `geometry: null`.

---

## 15. Execution log (21 Aug 2026, ~01:00 BST) — S4 PASSED (NumPy); dump-notion-hf

Verified on disk: `results/tin_ab/qsci_controls_cpu_numpy.json` (`sampler_backend=numpy`, `bitstring_convention=cudaq_char_q_equals_qubit_q`).

- **S4 PASSED:** `gate1_pool_native_pass` **true**. Smoke-best **+2.882 mHa** (17 dets) vs Demo A **+2.869 mHa**. Greedy pool union **+0.953 mHa / 21 dets** (not 49/49). HF and frozen 12-char padded both **+6.120 mHa**. Best native 14-char single: `pool14q_random_2` **+3.934 mHa**, 11 dets.
- **Mapping:** Pauli scorer had used `int(bitstring,2)` LSB-right; CUDA-Q/MatGen-Q leftmost char = qubit 0; HF is `11000000000000`.
- **Jobs:** GPU confirm **7431345** and 14-char SFT **7431349** still **PENDING (Priority)**. No 14-char ckpt → **infer not submitted**. **No DAPO. No paid QPU.**
- Next STOP is 14-char infer (`n_samples=64`, union cap 32) after 7431349 writes a real ckpt — not DAPO.
- Live qBraid wallet ~**677 cr**; full Emerald ~**6360 cr** does not fit. Demo B replay already **+0.575 mHa / 0 cr**.
- Notion: [Implementation Plan](https://app.notion.com/p/3c2868e41fc08100a534dea8758d899e). HF dataset: [Ryukijano/sno14q-gqe-qsci-s4](https://huggingface.co/datasets/Ryukijano/sno14q-gqe-qsci-s4) (circuit generator dump; SnO 14q H pointer; no industrial Sn–C / 92 eV claim).
- **STOP unchanged:** no iodine DAPO, no invented methyltin XYZ, no 49/49 as a GQE win, no 72.6→21.2 as GQE.

---

## 16. Held-out organotin generalization (pointer only)

Independent experiment, not a rewrite of this contract: [`docs/hcgqe_generalization_heldout_organotin.md`](hcgqe_generalization_heldout_organotin.md).

**Verdict (21 Aug 2026):** zero-shot GNN emission on unseen organotin topologies is **not a measured capability**. Frozen SnO 14q infer used the 12-char dialect and sat at HF (+6.120 mHa). The public checkpoint has no GNN tensors at generate-time. Jobs **7431345** (GPU S4) and **7431349** (14-char SFT) **COMPLETED**; 14-char infer **7434600** is **PENDING (Priority)** — leave it; do not invent scores. Not DAPO.

Do not treat 13.5 nm / Auger / 92 eV or 72.6→21.2 kcal/mol as H-cGQE results. Layer (i) is ground-state \(E_0\) vs CASCI; layer (ii) 0 vs +1 Sn–C waits on honest QSCI rungs and real XYZ.

---

## 17. Materials-informatics loop (labelled-ours DFT ranking; not GQE inverse design)

**Appended 21 Aug 2026 evening.** This section does **not** rewrite Phases 0–7 or the STOP gates.

**Infer status:** job **7434600** (`tin_infer_pool14`) is **PENDING (Priority)**. No logs/JSON yet. **Do not invent** ΔE vs CASCI, \(N_\mathrm{dets}\), or word lengths. **Do not** `sbatch jobs/tin_dapo.sbatch`. **Do not** `DEMOB_REAL`.

The industrial ranking we can run **without** unpublished MatGen-Q SI XYZ is a **labelled-ours DFT 0/+1 IP / Sn–C BDE ranking** on `RSn(OH)3`. GQE does **not** propose molecules.

| Step | What it is | What it is not |
|---|---|---|
| Ligand set | Public SMILES / CID as **graph identities**; labelled-ours opt only under `results/tin_ab/methyltin_ours/` with provenance JSON | Invented MatGen-Q Table 1 Cartesian coordinates |
| DFT 0 / +1 | B3LYP then MP2, def2-SVP + ECP on Sn (`configs/methyltin_ours_opt.yaml`) | A GQE / transformer inverse-design loop |
| Rank | **Ionisation potential** and **Sn–C BDE** from **our** 0 vs +1 DFT (later **our** CASCI) | Scoring **72.6 → 21.2 kcal/mol** (classical UCCSD(T) in [arXiv:2607.23988](https://arxiv.org/abs/2607.23988)) as H-cGQE |
| Optional GQE | Native-length \(E_0\) vs **our** CASCI, after ours XYZ exists **and** 14-char infer passes S2 | 12-char padded HF; 49/49 tautology; molecule generation |
| 92 eV / Auger / 13.5 nm | **Out of scope** on 1–3× L40S | IMePh iodine-edge absorption (~200 logical qubits, [arXiv:2602.20234](https://arxiv.org/abs/2602.20234)) |

**Hard split:** never copy ours XYZ into `configs/tin_resist.yaml`. MatGen-Q methyltin / butyltin rungs stay `geometry: null` until SI exists. Ours geometries live only under `results/tin_ab/methyltin_ours/`.

Scaffold: [`docs/methyltin_ours_dft_mp2_opt.md`](methyltin_ours_dft_mp2_opt.md). Dedicated loop notes: [`docs/tin_inform_loop.md`](tin_inform_loop.md). Holdout protocol: [`docs/hcgqe_generalization_heldout_organotin.md`](hcgqe_generalization_heldout_organotin.md).

H-cGQE **ChemistryEncoder / GNN is not loaded** at tin infer (`src/gqe/models/infer_h_cgqe.py`). Generate-time conditioning is `HamiltonianEncoder` over Pauli terms.
