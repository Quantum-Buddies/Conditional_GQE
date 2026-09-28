# Experiment: cudaq-gqe-cepheus-sno14

Protocol ID: `cudaq-gqe-cepheus-sno14`
Date: 2026-08-23
Author: Gyanateet Dutta (Ryukijano)

## Hypothesis & Prediction

**H1.** NVIDIA CUDA-Q Solvers GQE (`cudaq_solvers.gqe`) can propose a short
HF + `exp_pauli` circuit on the public SnO 14q Hamiltonian such that
computational-basis samples, classically QSCI-diagonalized, lie within
chemical accuracy of GATE 1 CASCI.

**Prediction.** \(E_\mathrm{QSCI} - E_\mathrm{CASCI} \le 1.6\,\mathrm{mHa}\)
with \(N_\mathrm{dets} < N_\mathrm{FCI} = 49\).

**Falsification.** GPU (noiseless) QSCI of the GQE circuit stays at HF
(\(+6.12\,\mathrm{mHa}\)) or exceeds \(1.6\,\mathrm{mHa}\). If that happens,
do **not** submit Cepheus.

This is **not** an advantage-over-SHCI claim. GQE does **not** propose
molecules and does **not** re-rank labelled-ours DFT ΔBDE.

## Design

| Item | Specification |
|------|-----------------|
| Instance | SnO (2e, 7o) = 14 qubits, public MatGen-Q GATE 1 H |
| Hamiltonian | `results/tin_ab/hamiltonians_tin/hamiltonians.json` record `sno_14q` |
| CASCI referee | \(-288.10912104\) Ha |
| Circuit generator | `cudaq_solvers.gqe` via `src/gqe/baselines/run_cudaq_gqe.py` |
| Ansatz | HF `X` on occupied qubits, then \(\le 10\) `exp_pauli` words |
| QSCI | `cudaq.sample` / `sample_async` then classical subspace diag |
| QPU | Rigetti Cepheus-1-108Q through `cudaq.set_target("qbraid", machine=...)` only after GPU and qBraid QIR-SV gates pass |

## Primary Outcome

**One metric:** \(\Delta E = (E_\mathrm{QSCI} - E_\mathrm{CASCI})\) in mHa
from the GQE-sampled subspace (union of unique bitstrings, particle-number
filtered, \(N_\mathrm{dets} < 49\)).

Pass: \(\Delta E \le 1.6\) mHa.

## Secondary outcomes (exploratory)

- HF-only QSCI (should be \(+6.120\) mHa).
- Entangled-HF / occupancy-recovery / S-CORE-repaired counts vs raw.
- Routed two-qubit gate count after OpenQASM translation.
- qBraid QIR-SV \(\Delta E\) vs GPU \(\Delta E\).
- Cepheus \(\Delta E\) (only if prior gates pass). Labelled hardware, not DFT.

## Conditions

| ID | Description | Seeds |
|----|-------------|-------|
| C0 | HF kernel, GPU `nvidia` fp64 sample | 42, 43, 44 |
| C1 | CUDA-Q Solvers GQE train on GPU (`--ngates 6`, `--max-iters 25`) | 3047 |
| C2 | Sample C1 circuit on GPU, QSCI | 42, 43, 44 |
| C3 | Sample C1 circuit on `qbraid:qbraid:sim:qir-sv` (≤2000 shots) | 42 |
| C4 | Sample C1 on `aws:rigetti:qpu:cepheus-1-108q` iff C2 passes and `CUDAQ_ALLOW_QPU=1` | 42 |

## Controls & Ablations

| Control | Specification |
|---------|----------------|
| Baseline | HF determinant QSCI (known \(+6.120\) mHa vs CASCI) |
| Ablation | GQE circuit without S-CORE repair vs with repair (exploratory on QPU counts) |
| Seeds | GQE `cfg.seed=3047`; sampling `{42,43,44}` on GPU |
| Hyperparameters | `ngates=6`, `max_iters=25`, `num_samples` = library default (5), pool scales as `run_cudaq_gqe.py` |
| Hardware | *Paths/partition names shown are for the University of Leeds AIRE cluster — adapt for your site.* Train/observe: 1× L40S, `cudaq.set_target("nvidia", option="fp64")`. Sample dry: qBraid QIR-SV. Hardware sample: Cepheus via qBraid CUDA-Q target, **not** qBraid Python SDK |
| Leakage | Ligand DFT rank table is frozen; this experiment does not write `rank_table.json` |

## Data

- Hamiltonian snapshot: `results/tin_ab/hamiltonians_tin/hamiltonians.json` (`sno_14q`).
- GQE JSON: `experiments/cudaq-gqe-cepheus-sno14/outputs/cudaq_gqe_sno14_<jobid>.json` (symlink `cudaq_gqe_sno14_latest.json`).
- QSCI JSON: `experiments/cudaq-gqe-cepheus-sno14/outputs/`.
- Forbidden inputs: labelled-ours Me/CF3 18–22q H, `configs/tin_resist.yaml` XYZ overwrite, 14-char padded dialect from the frozen 12-char checkpoint.

## Analysis Plan

- Report \(\Delta E\) vs GATE 1 CASCI; variational check \(E_\mathrm{QSCI} \ge E_\mathrm{CASCI} - 10^{-8}\).
- Require \(N_\mathrm{dets} < 49\) (49/49 is tautological CASCI).
- No p-hacking: do not add extra operators after seeing \(\Delta E\).
- Negative results are reported (HF-like GQE is a valid fail).

## Stopping Rules

1. If C2 \(\Delta E > 1.6\) mHa: stop; no C3 spend beyond the free QIR-SV budget if C2 already failed; **never C4**.
2. If translated two-qubit count \(> 400\): abort C4.
3. If `CUDAQ_ALLOW_QPU` is unset: C4 must refuse.
4. Direct `rigetti:rigetti:qpu:cepheus-1-108q` (per-minute) is refused unless `CUDAQ_ALLOW_DIRECT_RIGETTI=1`.
5. Do not `sbatch jobs/tin_dapo.sbatch`. Do not set `DEMOB_REAL`.

## Pre-registration Status

Checklist completed: 2026-08-23

- [x] Primary hypothesis ID referenced (H1)
- [x] Pre-specified prediction (≤ 1.6 mHa)
- [x] Falsification criterion documented
- [x] Primary outcome metric defined (one only)
- [x] Secondary metrics labeled exploratory
- [x] Unit of analysis: one GQE circuit × sampling seed
- [x] Inclusion: `sno_14q` only
- [x] No train/test split (single published H)
- [x] Baseline = HF QSCI
- [x] Ablation pre-specified (recovery on/off)
- [x] Seeds fixed
- [x] Hyperparameters fixed
- [x] Hardware named
- [x] Comparison method: ΔE vs CASCI
- [x] Stopping rules defined
- [x] Failed runs logged, not discarded
- [x] Output directory defined
- [x] No DFT re-rank
- [x] Negative results will be reported

```
Protocol ID: cudaq-gqe-cepheus-sno14
Checklist completed: 2026-08-23
Author: Gyanateet Dutta
Amendments: two CUDA-Q installs; C1 H1 falsified (job 7443318); no Cepheus
```

## Amendments

2026-08-23: Two CUDA-Q installs. `cudaq-solvers` 0.6 requires CUDA-Q 0.14
(`libcudaq-solvers.so` undefined `should_log` on 0.15). Native
`cudaq.set_target("qbraid")` ships in 0.15.0+. Train/GPU QSCI use
`cudaq-env` (0.14.2). qBraid QIR-SV and
Cepheus sample use `cudaq15-qbraid` (0.15.1).
Hypothesis, pass metric, and QPU gates unchanged.

2026-08-23: Job 7443216 ran library defaults (`ngates=20`, `max_iters=100`)
because `solvers.gqe` ignores kwargs when `config=` is passed. That JSON is
kept as a miswired control (observe ΔE = 6.25 mHa, QSCI ΔE = 6.120 mHa, 2
dets). C1 is re-run after setting `cfg.ngates` / `cfg.max_iters` on the
config object (`jobs/cudaq_gqe_sno14.sbatch`). No hyperparameter search
after seeing ΔE.

2026-08-23: Protocol C1 job **7443318** (`ngates=6`, `max_iters=25`, seed
3047): GQE observe ΔE = 6.135 mHa; GPU QSCI ΔE = **6.120 mHa**, \(N_\mathrm{dets}=1\)
(HF bitstring only). **H1 falsified.** Stopping rule 1: no Cepheus.
qBraid `POST /jobs` to `qbraid:qbraid:sim:qir-sv` returns Runtime API 500
even for a 2-qubit Bell (auth and `GET /devices` succeed; last completed
Cepheus jobs on this account are 2026-07-26). `aws:rigetti:qpu:cepheus-1-108q`
is UNAVAILABLE. Direct `rigetti:rigetti:...` stays refused.
