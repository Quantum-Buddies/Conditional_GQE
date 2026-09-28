---
license: mit
pretty_name: SnO 14q GQE+QSCI S4 controls
language:
- en
tags:
- quantum-computing
- quantum-chemistry
- qsci
- gqe
- cuda-q
- hamiltonian
size_categories:
- n<1K
task_categories:
- other
configs:
- config_name: default
  data_files:
  - split: train
    path: data/s4_rows.jsonl
---

# SnO 14q GQE+QSCI S4 controls

Reproducible **circuit-generator** artefacts for H-cGQE vs MatGen-Q on the **published SnO (2e,7o) 14-qubit**
Hamiltonian. This is **not** an industrial Sn–C resist result, **not** a 92 eV iodine-edge absorption run, and
**not** a claim that 49/49 determinants or the classical 72.6→21.2 kcal/mol Sn–C collapse are GQE wins.

Companion model card remains a ground-state circuit generator: [Ryukijano/h-cgqe-gic2026](https://huggingface.co/Ryukijano/h-cgqe-gic2026).

## Verified S4 (CPU NumPy / MatGen-Q-equivalent sampler)

Source: `data/qsci_controls_cpu_numpy.json` (written 21 Aug 2026 00:55 BST).

| Quantity | Value |
|---|---|
| `gate1_pool_native_pass` | **true** |
| Sampler | `numpy`, `qsci_backend=matgenq_subspace` |
| Bitstring convention | leftmost char = qubit 0 (`11000000000000` HF) |
| CASCI | −288.10912104 Ha |
| Smoke-best ΔE | **+2.882 mHa** (17 dets) vs Demo A **+2.869 mHa** |
| Greedy pool union ΔE | **+0.953 mHa**, **21 dets** (< 49 FCI) |
| Best native 14-char single | `pool14q_random_2` **+3.934 mHa**, 11 dets |
| HF / frozen 12-char | **+6.120 mHa** (HF-like; AS correlation) |

Mapping fix: Pauli scorer previously used `int(bitstring, 2)` (LSB-right). CUDA-Q / MatGen-Q use **leftmost = qubit 0**.

GPU CUDA-Q confirm job **7431345** and 14-char SFT job **7431349** were still **PENDING** at dump time. No DAPO. No paid QPU.

## Files

- `hamiltonians/hamiltonian_pointer.json` — CASCI, geometry, term count, source repo
- `hamiltonians/hamiltonians.json` — exported SnO 14q Pauli Hamiltonian (1366 terms)
- `data/s4_summary.json` — compact S4 + mapping-fix ΔE vs CASCI
- `data/qsci_controls_cpu_numpy.json` — full CPU S4 JSON
- `data/s4_rows.jsonl` / `data/s4_rows.csv` — viewer table (Dataset Viewer config reads jsonl only)
- `circuits/s4_circuit_samples.json` — smoke-best / union / 12-char operator samples
- `circuits/pool_sno14q.json` — 14-char UCCSD pool
- `notes/12char_frozen_failure.md` — why the mixed 12-char SFT checkpoint fails on this H

## Provenance

- Hamiltonian: [KarimElgammal/gqe-qsci-euv-photoresists](https://github.com/KarimElgammal/gqe-qsci-euv-photoresists) `SnOMolecule` (arXiv:[2607.23988](https://arxiv.org/abs/2607.23988))
- H-cGQE stack: [Quantum-Buddies/Conditional_GQE](https://github.com/Quantum-Buddies/Conditional_GQE)
- Code license: MIT (this dump). Hamiltonian terms remain those of the MatGen-Q companion.

## STOP

No iodine DAPO. No invented methyltin XYZ. Do not score 49/49 dets as a GQE win. Do not write 72.6→21.2 kcal/mol as GQE.
