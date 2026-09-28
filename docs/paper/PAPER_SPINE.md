# Paper spine — H-cGQE on SnO and labelled-ours tin ligands

**Date:** 22 August 2026
**Repo:** `Conditional-GQE_materials`
**This is an independent paper experiment.** GIC 2026 is closed; C-GQE did not win or place (50% QWC discount only). Do not invent a Mitsubishi/AIST 20–40q rubric or a public score sheet.

Writing draft: [`manuscript.md`](manuscript.md). Locked numbers: [`results_ledger.json`](results_ledger.json). Figures: `results/generalization/fig1_sno14q_de_vs_casci.{png,pdf}` and `results/tin_ab/methyltin_ours/fig_inform_*.png`.

## Claim (one sentence)

A Hamiltonian-conditioned GQE decoder on the **public** SnO (2e,7o) 14-qubit Hamiltonian is limited by **operator dialect**, not by missing VQE correlation: mixed-length SFT sits at Hartree–Fock; native 14-character SFT recovers **+2.597 mHa / 21 dets** and still **misses 1.6 mHa**, while a **pool-native QSCI union** of the same tokens reaches **+0.953 mHa / 21 dets**. A separate **labelled-ours B3LYP** loop ranks `RSn(OH)₃` ligands by ΔBDE/IP; that ranking is **not** a GQE energy.

## Allowed claims

1. Generate-time H-cGQE is `HamiltonianEncoder` + decoder. The chemistry GNN is a 5-sample property regressor and is **not** loaded at infer.
2. Public UCCSD SFT vocab is mixed-length (majority 12-char). On 14q SnO, `len(word) ≤ n_qubits` licensed 12-char words; mapping-fixed QSCI is **+6.120 mHa = HF**, 1 det.
3. QSCI endianness: CUDA-Q leftmost char = qubit 0; HF bitstring `11000000000000`.
4. 14-char pool SFT (job 7431349, val acc ~10%, early stop 26/80) then infer/QSCI (job 7434600, 64 samples): best **+2.597 mHa**, 21 dets, `filled_full_ci: false`; union cap 32 **+2.149 mHa**. `s4_pass: false`.
5. Pool-native greedy union (NumPy S4): **+0.953 mHa**, 21 dets, not 49/49. Per-instance UCCSD-VQE on the same H is ~**10⁻⁵ mHa**.
6. Labelled-ours B3LYP/def2-SVP (+ECP on Sn), geometries under `results/tin_ab/methyltin_ours/` only. Ranked ΔBDE (kcal/mol): **CF3 40.15 > Me 38.36 > vinyl 33.42 > Et 25.51 > n-Bu 22.00 > Ph 21.72 > i-Pr 16.04 > allyl 7.82**. n-Pr DFT is the last pool member (job 7439561). Vinyl SCF retry **succeeded** (job 7437565_0).
7. GQE here is **valence \(E_0\)** vs CASCI on a clamped-nucleus Hamiltonian. 13.5 nm column in the ligand table is a **Henke/CXRO atomic \(f_2\) proxy**, not \(\sigma(92\,\mathrm{eV})\).

## Forbidden claims

- Graph-conditioned zero-shot organotin circuits.
- 72.6 → 21.2 kcal/mol as a GQE or H-cGQE number (classical UCCSD(T) in arXiv:2607.23988).
- 92 eV / Auger / IMePh iodine-edge absorption (~200 logical qubits, arXiv:2602.20234).
- MatGen-Q Table 1 / SI methyltin or n-butyl XYZ (not public). Do not copy ours XYZ into `configs/tin_resist.yaml`.
- Scoring 49/49 union as a GQE win.
- GIC win, placement, or a 20–40 qubit official primary filter.
- DAPO on tin (do not `sbatch jobs/tin_dapo.sbatch`).
- n-Bu **MP2** adiabatic IP (12.98 eV) as physical: UHF-MP2 at the DFT cation geometry is **above** the vertical MP2 energy. Report **B3LYP IPs** as the primary column.

## Venue / framing

Natural homes: *Digital Discovery* (conditional GQE lineage) or arXiv-first then JCTC methods. Title should name **dialect / QSCI / labelled-ours DFT**, not “GNN zero-shot EUV resists.”

## Remaining compute (paper, not GIC)

| Item | Why | Status |
|---|---|---|
| Vinyl SCF retry | Complete first-wave ΔBDE | **done** (job 7437565_0, ΔBDE 33.42 kcal/mol) |
| CF3 DFT | Chemical next ligand | **done** (job 7437565_1, **rank 1**, ΔBDE 40.15 kcal/mol) |
| Allyl / i-Pr DFT | Chemical pool | **done** (allyl 7.82; i-Pr 16.04 kcal/mol) |
| n-Pr DFT | Last pool member | **running** job 7439561 |
| AVAS/CASCI + QSCI on labelled-ours Me/CF3/vinyl (charge 0) | Optional \(E_0\) vs **our** CASCI | `jobs/tin_inform_e0.sbatch` then GPU array `tin_inform_e0_qsci.sbatch`; not 14-char H-cGQE unless \(n_q=14\) |
| GNN-on vs off on same H | Honest ablation | not started |
| SnO 16–24q ladder | Scaling SI | not started |
| DAPO | — | **do not run** |
