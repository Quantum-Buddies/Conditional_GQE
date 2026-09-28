# Operator dialect limits Hamiltonian-conditioned GQE on SnO (2e,7o), with a labelled-ours DFT loop for \(\mathrm{RSn(OH)_3}\)

**Draft for writing (not a submitted manuscript).** Numbers are copied from on-disk JSON in `docs/paper/results_ledger.json`. Do not insert GIC placements, 20–40q official rubrics, or 72.6→21.2 kcal/mol as a GQE score.

Authors (fill): Gyanateet Dutta and collaborators. Affiliation: University of Leeds (AIRE); Quantum-Buddies.

## Abstract

Generative quantum eigensolvers (GQE) amortize ansatz design by sampling Pauli-word sequences for a given electronic Hamiltonian and recovering a subspace energy with selected configuration interaction (QSCI). We report what a Hamiltonian-conditioned transformer (H-cGQE) actually computes on the only public organotin fragment Hamiltonian, SnO at 1.8325 Å in a (2e,7o) active space (14 qubits, 1366 Pauli terms, CASCI \(-288.10912104\) Ha). Mixed-length supervised pretraining emits 12-character operator words that pad onto 14 qubits and, after a bitstring-endianness correction, recover **exactly Hartree–Fock** (\(+6.120\) mHa, one determinant). Supervised fine-tuning on the native 14-character pool, then 64-sample inference, yields a best single-circuit QSCI error of **\(+2.597\) mHa** (21 determinants) and a 32-determinant union of **\(+2.149\) mHa**. Both miss chemical accuracy (\(1.6\) mHa) and neither fills the 49-determinant FCI space. A pool-native greedy union of the same tokens, *without* the trained policy, reaches **\(+0.953\) mHa** at 21 determinants, so the correlation is in the pool. The chemistry GNN shipped with the repository is a five-sample property regressor and is **not** loaded at generation time. Separately, we rank labelled-ours \(\mathrm{RSn(OH)_3}\) ligands at B3LYP/def2-SVP by the ionisation-induced Sn–C bond-dissociation collapse \(\Delta\mathrm{BDE}\). Methyl is largest (\(\Delta\mathrm{BDE}=38.36\) kcal/mol), then ethyl, \(n\)-butyl, and phenyl. That ranking is classical DFT, not GQE, and is not the UCCSD(T) 72.6→21.2 kcal/mol figure from the MatGen-Q preprint. GQE here is a valence ground-state solver; 92 eV core-hole spectroscopy is out of scope.

## 1. Introduction

GQE replaces on-chip variational optimisation with an autoregressive generator of operator sequences, followed by classical angle refinement and (optionally) QSCI [Nakaji *et al.*, arXiv:2401.09253]. Conditional-GQE attaches a graph encoder for *Ising combinatorial* problems, not organotin \(E_0\) [Minami *et al.*, arXiv:2501.16986]. MatGen-Q applies GPT-2 GQE+QSCI to monoalkyltin oxo-hydroxides and quotes a classical UCCSD(T) Sn–C collapse of 72.6→21.2 kcal/mol as the solubility-switch story; the public companion ships **SnO 14q only** [Elgammal & Maußner, arXiv:2607.23988].

Industrial EUV resists motivate the chemistry. They do not license scoring GQE as a 13.5 nm absorption or Auger solver. Iodine-edge resource estimates of order 200 logical qubits [arXiv:2602.20234] are outside 1–3× L40S.

This paper therefore asks three measurable questions:

1. On the **same** public SnO Hamiltonian, does H-cGQE beat Hartree–Fock without filling FCI?
2. Is generation graph-conditioned?
3. Can a **labelled-ours** DFT loop rank \(\mathrm{RSn(OH)_3}\) ligands without using unpublished MatGen-Q Cartesian coordinates?

## 2. Methods

### 2.1 Hamiltonian

SnO, \(r(\mathrm{Sn–O})=1.8325\) Å, def2-SVP with ECP28MDF on Sn, valence window (2e,7o), 14 Jordan–Wigner qubits, 1366 Pauli terms. Export **must** use the MatGen-Q `SnOMolecule` class (`src/gqe/data/export_matgenq_hamiltonian.py`). STO-3G `generate_hamiltonians.py` is a different molecule. CASCI reference: \(-288.10912104\) Ha. Active-space correlation versus RHF: \(6.120\) mHa. \(N_\mathrm{FCI}=49\).

QSCI bitstrings follow CUDA-Q: leftmost character is qubit 0. The Hartree–Fock occupation is `11000000000000`. Scoring `int(bitstring, 2)` as LSB-right is incorrect and was fixed before the numbers below.

Unions are reported only with \(N_\mathrm{dets}<49\). A 49/49 union is tautological CASCI.

### 2.2 H-cGQE

The public checkpoint `h_cgqe_uccsd_model.pt` (7.85 M parameters, vocab 149) is a Hamiltonian encoder plus decoder. `infer_h_cgqe.py` does not import `ChemistryEncoder`. Word-length histogram of the UCCSD SFT vocab: 12-character words dominate (59 of 149). The length mask `len(word) ≤ n_qubits` therefore licenses 12-character tokens on 14q SnO.

A second checkpoint, `h_cgqe_sft_pool14_7431349.pt`, is supervised on MatGen-Q 14-character pool sequences (2097 teachers, 49 unique 14-character words, early-stopped at epoch 26, validation accuracy \(0.101\)). This is **in-distribution** SFT on SnO, not zero-shot.

Inference job 7434600: 64 samples, exact 14-character words, QSCI with union cap 32.

### 2.3 Labelled-ours DFT loop

Hand Cartesian templates for \(\mathrm{RSn(OH)_3}\) (`src/gqe/data/rsn_oh3_templates.py`). B3LYP/def2-SVP, ECP on Sn, geometric optimiser. Neutral singlet RKS; cation doublet UKS; Sn–C BDE with charge remaining on \(\mathrm{Sn(OH)_3^+}\). MP2 is a single point at the DFT geometry. Outputs live only under `results/tin_ab/methyltin_ours/` with `label: ours`. Coordinates are **not** copied into `configs/tin_resist.yaml`.

Rank: larger \(\Delta\mathrm{BDE}=\mathrm{BDE}(0)-\mathrm{BDE}(+1)\), then smaller adiabatic IP, then a Henke/CXRO atomic \(f_2\) sum at 91.84 eV as a **proxy**, not \(\sigma(92\,\mathrm{eV})\).

## 3. Results

### 3.1 Seen organics (context, not tin)

On the GIC evaluation JSON, most in-family molecules sit at or above the file Hartree–Fock gap (H\(_2\), N\(_2\), phenol, BeH\(_2\)). CH\(_3\)I at 8q with L-BFGS is \(0.63\) mHa versus CAS(4e,4o); that is a seen iodine toy, not an EUV resist. IMePh in that JSON used a non-industrial isomer (\(+25.4\) mHa, worse than HF). YAML was later corrected to 4-iodo-2-methylphenol `Oc1ccc(I)cc1C`; those old energies are not re-claimed.

### 3.2 SnO 14q

| Column | \(\Delta E\) vs CASCI (mHa) | \(N_\mathrm{dets}\) |
|---|---:|---:|
| Hartree–Fock | 6.120 | 1 |
| H-cGQE 12-char, mapping-fixed | 6.120 | 1 |
| H-cGQE 14-char SFT, best of 64 (job 7434600) | **2.597** | **21** |
| Same policy, union cap 32 | **2.149** | **32** |
| MatGen-Q smoke-best (CUDA-Q, job 7431345) | 2.152 | 17 |
| Pool-native greedy union (NumPy) | **0.953** | **21** |
| UCCSD-VQE, per-instance | \(1.2\times10^{-5}\) | ansatz |

The trained 14-character policy **beats Hartree–Fock** and matches the MatGen-Q smoke circuit to \(\sim 0.4\) mHa on the best sample, but **does not reach 1.6 mHa**. The untuned pool union does. Per-instance VQE solves the Hamiltonian; the gap is amortised generation, not missing correlation in the active space.

Figure: `results/generalization/fig1_sno14q_de_vs_casci.pdf`.

### 3.3 Labelled-ours ligand ranking

B3LYP \(\Delta\mathrm{BDE}\) (kcal/mol): CF\(_3\) **40.15**, Me 38.36, vinyl 33.42, Et 25.51, \(n\)-Bu 22.00, Ph 21.72, \(i\)-Pr 16.04, allyl 7.82. Adiabatic IPs (eV): CF\(_3\) 9.78, Me 9.71, vinyl 9.49, Et 9.15, \(n\)-Bu 9.00, Ph 8.98, \(i\)-Pr 8.74, allyl 8.38. Vinyl vertical-cation SCF failed on the first array and succeeded after initialising UKS from the neutral density (job 7437565_0). \(n\)-Pr is the last pool member (job 7439561). The six-dimensional RBF GP remains shaky; picks after CF\(_3\) followed the documented chemical rule, not expected improvement.

Do not quote \(n\)-Bu UHF-MP2 adiabatic IP (12.98 eV): the MP2 energy at the DFT-optimised cation lies **above** the vertical MP2 energy.

Figures: `results/tin_ab/methyltin_ours/fig_inform_ip_bde.png`, `fig_inform_rank.png`.

## 4. Discussion

H-cGQE as implemented is a **Hamiltonian-conditioned circuit generator**. Claims of GNN-conditioned zero-shot organotin topologies are not supported by checkpoints or `infer_h_cgqe.py`.

The 12-character collapse is a **tokenizer / length-mask** failure. Fixing endianness without fixing word length leaves the energy at HF. Fixing word length recovers most of the MatGen-Q smoke error but not chemical accuracy. Because a 21-determinant pool union already sits at \(0.95\) mHa, further RL (DAPO) is not the first lever and is **not run** for this paper.

The DFT loop is the honest “propose → score → pick” half: ligands are enumerated classically; GQE is an optional \(E_0\) column only after native-length inference versus **our** CASCI. That column is still empty (`e0_casci_ha` is null in `rank_table.json`).

## 5. Limitations

- No public MatGen-Q methyltin / \(n\)-butyl XYZ; no held-out organotin \(\Delta E\) table.
- No GNN-on vs GNN-off circuit ablation.
- Vinyl and CF\(_3\) DFT incomplete at draft time.
- No 92 eV, Auger, or QPU claim beyond the existing Emerald replay of published counts (\(+0.575\) mHa, 0 credits).
- GIC 2026 outcome is not a scientific deliverable (no public score sheet; not a win or placement).

## 6. Data and code

- Model hub: `https://huggingface.co/Ryukijano/h-cgqe-gic2026`
- SnO S4 dump: `https://huggingface.co/datasets/Ryukijano/sno14q-gqe-qsci-s4`
- Code: `https://github.com/Quantum-Buddies/Conditional_GQE`
- Ledger: `docs/paper/results_ledger.json`

## References (seed)

1. Nakaji *et al.*, *The generative quantum eigensolver (GQE)*, arXiv:2401.09253.
2. Minami *et al.*, *Conditional generative quantum eigensolver*, arXiv:2501.16986; *Digital Discovery*.
3. Elgammal & Maußner, *Generative quantum eigensolver for EUV photoresists*, arXiv:2607.23988.
4. Kanno *et al.* / related 92 eV resource paper, arXiv:2602.20234 (out of scope here).
5. Henke, Gullikson, Davis, *At. Data Nucl. Data Tables* **54**, 181 (1993).
