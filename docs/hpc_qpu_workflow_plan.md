# HPC → AI → QPU Async Submission Workflow Plan

## Status: Draft — July 16, 2026

## 1. Current Pipeline State

### What works
- **H-cGQE Transformer**: Autoregressive circuit synthesis (GPT-2 style) → operator sequences
- **L-BFGS-B optimization**: Classical coefficient optimization on 3× L40S GPUs via CUDA-Q `nvidia-mqpu`
- **QWC Pauli grouping**: 3-5× circuit reduction (H2: 15→5, LiH: 631→180, N2: 2951→1308)
- **Local Qiskit statevector**: Free, instant correctness check (validated H2 within 0.01 mHa)
- **AWS SV1 simulator**: H2 validated at 1.477 mHa vs GPU (4096 shots)
- **Manifest export**: Self-contained JSON + QASM for decoupled QPU submission

### What's broken / incomplete
- **Rigetti direct result retrieval**: Jobs submit to `rigetti:rigetti:qpu:cepheus-1-108q` but `job.result()` fails with "Counts data is not available" — likely QCS readout buffer format incompatibility with `_get_counts()`
- **IonQ simulator rate limiting**: 429 Too Many Requests on rapid sequential submissions
- **No error mitigation**: Raw QPU counts will have significant noise (99.1% 2Q fidelity sounds good but compounds with depth)
- **No Z2 symmetry tapering**: Could reduce qubit count further (H2: 4q→2q, LiH: 12q→10q)

---

## 2. Available qBraid Devices (as of July 16, 2026)

### QPUs (Online)

| Device ID | Vendor | Qubits | Per-shot | Per-task | Per-min | Notes |
|-----------|--------|--------|----------|----------|---------|-------|
| `rigetti:rigetti:qpu:cepheus-1-108q` | Rigetti | 107 | 0 cr | 0 cr | 12000 cr | Direct QCS, needs OAuth token + quilc |
| `openquantum:rigetti:qpu:cepheus-1-108q` | Rigetti (OQ) | 107 | — | — | — | $50 free credits / 90 days |
| `openquantum:iqm:qpu:emerald` | IQM (OQ) | 54 | — | — | — | Via Quantum Rings |
| `aws:iqm:qpu:emerald` | IQM | 54 | 0.16 cr | 30 cr | 0 | Standard qBraid result format |
| `openquantum:iqm:qpu:garnet` | IQM (OQ) | 20 | — | — | — | Via Quantum Rings |
| `aws:iqm:qpu:garnet` | IQM | 20 | 0.145 cr | 30 cr | 0 | Standard qBraid result format |
| `openquantum:ionq:qpu:forte-1` | IonQ (OQ) | 36 | — | — | — | Via Quantum Rings |
| `openquantum:ionq:qpu:forte-enterprise` | IonQ (OQ) | 36 | — | — | — | Via Quantum Rings |
| `openquantum:aqt:qpu:ibex-q1` | AQT (OQ) | 12 | — | — | — | Trapped-ion |

### Simulators (Online)

| Device ID | Qubits | Cost | Notes |
|-----------|--------|------|-------|
| `ionq:ionq:sim:simulator` | 29 | $0.00/min | Free but rate-limited (429s) |
| `aws:aws:sim:sv1` | 34 | 7.5 cr/min | Free first min/task, no batch support |
| `aws:aws:sim:dm1` | 17 | 7.5 cr/min | Density matrix (noise simulation) |
| `qbraid:qbraid:sim:qir-sv` | 30 | 7.5 cr/min + 0.5 cr/task | Sparse statevector |

### QPUs (Offline/Unavailable)

| Device | Qubits | Status |
|--------|--------|--------|
| `aws:rigetti:qpu:cepheus-1-108q` | 107 | UNAVAILABLE |
| `aws:ionq:qpu:forte-1` | 36 | OFFLINE |
| `aws:ionq:qpu:forte-enterprise-1` | 36 | OFFLINE |
| `aws:aqt:qpu:ibex-q1` | 12 | UNAVAILABLE |
| `aws:quera:qpu:aquila` | 256 | UNAVAILABLE |

---

## 3. Rigetti Cepheus-1-108Q Hardware Details

| Spec | Value |
|------|-------|
| Architecture | 12 × 9-qubit chiplets, square lattice |
| Connectivity | 4-fold nearest-neighbor (tunable couplers + IMCs) |
| Native gates | RX, RY, CZ (adiabatic) |
| 2Q gate fidelity | 99.1% median (target: 99.5% by end of 2026) |
| 1Q gate fidelity | 99.9% median |
| Gate speed | ~60 ns |
| T1 / T2 | 25 μs / 10 μs |
| Availability | 20 hrs/day |
| Queue | Can stretch to hours during peak US/EU times |

### Connectivity implications
- Square lattice = 4-fold nearest-neighbor
- Our H-cGQE circuits use CNOTs which need SWAP routing for non-adjacent qubits
- H2 (4q): fits within a single chiplet (3×3 grid), minimal SWAP overhead
- LiH (12q): spans ~2 chiplets, moderate SWAP overhead
- N2 (20q): spans ~3 chiplets, significant inter-chiplet routing needed

### Direct access requirements
1. **QCS account** with active OAuth refresh token (contact Rigetti at qcs.rigetti.com/request-access)
2. **Forest SDK** — `quilc` compiler must be running locally or at a known endpoint
3. Environment variables: `RIGETTI_REFRESH_TOKEN`, optionally `QCS_QUILC_ENDPOINT`, `QCS_GRPC_ENDPOINT`
4. The `RigettiProvider` must call `provider.setup()` to start/manage quilc

---

## 4. The Result Retrieval Problem

### Symptom
Jobs submit successfully to `rigetti:rigetti:qpu:cepheus-1-108q` and `openquantum:rigetti:qpu:cepheus-1-108q`, but `job.result()` fails with "Counts data is not available."

### Likely causes (ranked by probability)

1. **QCS readout buffer format mismatch** (most likely)
   - The qBraid `RigettiJob` (PR #1127) parses QCS readout buffers into `GateModelResultData`
   - Our `_get_counts()` expects `result.data.get_counts()` or `result.measurement_counts()`
   - Rigetti results may expose data differently — need to inspect the actual result object attributes

2. **quilc compilation not running**
   - `RigettiProvider.setup()` starts quilc locally; if we're using `QbraidProvider` (not `RigettiProvider`), quilc may not be started
   - Without compilation, the job may be submitted but execution may silently fail or produce empty results

3. **Job still in queue**
   - Rigetti QPU has queue times; the job may not have completed when we tried to fetch results
   - Our retry logic (6 attempts × exponential backoff) may not be enough for QPU queue times

### Debugging plan
```python
# Step 1: Check job status (not just result)
from qbraid.runtime import load_job
job = load_job("rigetti:rigetti:qpu:cepheus-1-108q-135b-qjob-...")
print(f"Status: {job.status()}")

# Step 2: If COMPLETED, inspect result object structure
result = job.result()
print(f"Type: {type(result)}")
print(f"Dir: {[a for a in dir(result) if not a.startswith('_')]}")
print(f"Data type: {type(result.data)}")
print(f"Data dir: {[a for a in dir(result.data) if not a.startswith('_')]}")

# Step 3: Try alternative count extraction
# Rigetti may use: result.data.readout_values, result.data.measurements, etc.
```

### Fix options

**Option A: Extend `_get_counts()` for Rigetti results**
```python
def _get_counts(result: Any) -> dict[str, int]:
    # Standard qBraid path (AWS, IonQ, IQM)
    if hasattr(result, "data") and hasattr(result.data, "get_counts"):
        return result.data.get_counts()
    elif hasattr(result, "measurement_counts"):
        return result.measurement_counts()
    # Rigetti QCS path — parse readout buffer
    elif hasattr(result, "data") and hasattr(result.data, "readout_values"):
        # Convert raw readout to counts dict
        ...
    raise AttributeError(...)
```

**Option B: Use RigettiProvider directly instead of QbraidProvider**
```python
from qbraid.runtime.rigetti import RigettiProvider
import os
os.environ["RIGETTI_REFRESH_TOKEN"] = "..."
provider = RigettiProvider()
provider.setup()  # starts quilc
device = provider.get_device("cepheus-1-108q")
```

**Option C: Use OpenQuantum route (may bypass QCS auth issues)**
- The `openquantum:` prefix devices go through Quantum Rings' proxy
- May handle compilation and result parsing differently
- $50 free credits every 90 days

---

## 5. Cost Analysis

### H2 (4q, 5 QWC circuits, 4096 shots)

| Device | Task cost | Shot cost | Min cost | Total | USD |
|--------|-----------|-----------|-----------|-------|-----|
| AWS SV1 (sim) | 0 | 0 | ~0.6 cr | ~1 cr | ~$0.01 |
| IonQ sim | 0 | 0 | 0 | 0 | $0 |
| IQM Garnet (20q) | 150 cr | 297 cr | 0 | 447 cr | ~$4.50 |
| IQM Emerald (54q) | 150 cr | 328 cr | 0 | 478 cr | ~$4.80 |
| IonQ Forte-1 (36q) | 150 cr | 16384 cr | 0 | 16534 cr | ~$165 |
| Rigetti direct (107q) | 0 | 0 | ~6000 cr (30s) | ~6000 cr | ~$60 |
| Rigetti AWS (107q) | 150 cr | 870 cr | 0 | 1020 cr | ~$10.20 |

### LiH (12q, 180 QWC circuits, 4096 shots)

| Device | Task cost | Shot cost | Min cost | Total | USD |
|--------|-----------|-----------|-----------|-------|-----|
| IQM Garnet (20q) | 5400 cr | 106905 cr | 0 | 112305 cr | ~$1123 |
| IQM Emerald (54q) | 5400 cr | 117964 cr | 0 | 123364 cr | ~$1234 |
| Rigetti direct (107q) | 0 | 0 | ~36000 cr (3 min) | ~36000 cr | ~$360 |
| Rigetti AWS (107q) | 5400 cr | 311040 cr | 0 | 316440 cr | ~$3164 |

### N2 (20q, 1308 QWC circuits, 4096 shots)

| Device | Task cost | Shot cost | Total | USD |
|--------|-----------|-----------|-------|-----|
| IQM Garnet (20q) | 39240 cr | 776145 cr | 815385 cr | ~$8154 |
| Rigetti direct (107q, ~15 min) | 0 | 0 | ~180000 cr | ~$1800 |

### Key insight
- **Rigetti direct** (per-minute billing) is cheapest for large circuits with many shots
- **IQM Garnet** (per-shot) is cheapest for small circuits with few shots
- **Simulators** are free/cheap for H2 but don't scale to LiH/N2

---

## 6. Error Mitigation & Error Correction Strategy

### Why we need it
At 99.1% 2Q gate fidelity, a circuit with 10 two-qubit gates has ~9% error probability. For H2 with ~5-10 CNOTs, this means ~5-10% of shots are corrupted. Without mitigation, energy error could be 50-100+ mHa — far above chemical accuracy (1.6 mHa).

Our previous IQM Emerald run confirmed this: 896/1024 shots (87.5%) matched the expected state — meaning **12.5% of shots were corrupted** by a combination of gate errors and readout errors.

### 6.1 Error Mitigation vs Error Correction

**Error Mitigation (EM)** = post-process noisy results to estimate noise-free observables. No extra qubits. Probabilistic/estimation-based. Works on NISQ hardware. What we do now.

**Error Correction (EC)** = encode logical qubits across many physical qubits, actively detect and correct errors during computation. Requires fault-tolerant hardware. Not practical via qBraid cloud today.

| Aspect | Error Mitigation | Error Correction |
|--------|-----------------|-----------------|
| Qubit overhead | 0 extra qubits | ~625 physical per logical (d=25 surface code) |
| Circuit overhead | 1-3× more circuits | 1000s× more gates |
| Accuracy gain | 2-100× | Exponential (below threshold) |
| Hardware | Any NISQ QPU | Needs dedicated QEC-capable hardware |
| Status | **Available now** | Demonstrated on IBM Heron (2026), not on qBraid |

### 6.2 Existing Mitigation Code

`src/gqe/eval/mitigation.py` already implements:

| Function | What it does | Status |
|----------|-------------|--------|
| `calibrate_rem()` | Prepares |0⟩ and |1⟩ on each qubit, builds single-qubit assignment matrices, Kronecker-products them into full 2^n × 2^n matrix | ⚠️ Scales as 2^n — only works for n ≤ ~10q |
| `apply_rem()` | Inverts calibration matrix (pseudo-inverse or least-squares) to correct raw counts | ⚠️ Full matrix inversion — doesn't scale |
| `fold_gates()` | Unitary folding U → U(U†U)^c for ZNE noise scaling | ✅ Works |
| `zne_extrapolate()` | Linear/Richardson/polynomial extrapolation to zero-noise limit | ✅ Works |
| `run_zne_experiment()` | Full ZNE pipeline: fold → run → measure → extrapolate | ⚠️ `_compute_energy_from_counts` only handles diagonal Z terms, not QWC grouped measurements |

### 6.3 Mitigation Techniques — What to Implement

#### Tier 0: Symmetry Post-Selection (FREE — implement first)
**Cost**: Zero extra circuits, zero extra shots
**Improvement**: ~2× on half the terms
**Effort**: ~50 lines of code

Discard shots that violate known Z2 symmetries. Under Jordan-Wigner mapping, our Hamiltonians conserve:
- **Particle number parity**: Σ n_i mod 2 = N_e mod 2
- **Spin parity** (singlet states): S_z mod 2 = 0

For each bitstring in the counts, check if the Hamming weight matches the expected parity. If not, discard it and renormalize the remaining counts.

```python
def symmetry_post_select(
    counts: dict[str, int],
    n_electrons: int,
    n_qubits: int,
    symmetry: str = "particle_number",
) -> dict[str, int]:
    """Post-select shots conserving Z2 symmetries."""
    expected_parity = n_electrons % 2 if symmetry == "particle_number" else 0
    filtered = {}
    for bitstring, count in counts.items():
        hamming = sum(int(b) for b in bitstring)
        if hamming % 2 == expected_parity:
            filtered[bitstring] = count
    return filtered
```

**Integration point**: Call after `_get_counts()` and before `_parse_grouped_results()` in `qbraid_backend.py`.

**Important**: This only works for Z-basis measurements (computational basis). For X/Y basis measurements (QWC groups measuring X or Y terms), the symmetry check must be applied in the rotated basis — i.e., check parity of the *transformed* bitstring.

#### Tier 1: Chemistry Reference-State Error Mitigation (CHEAP — implement first)
**Cost**: 0-1 additional circuit evaluations (HF state)
**Improvement**: Up to 100× (two orders of magnitude, per JCTC 2022 paper)
**Effort**: ~30 lines of code

The idea (from arXiv:acs.jctc.2c00807): measure the noisy energy of a classically-known reference state (Hartree-Fock), compute the error ΔE = E_noisy(HF) - E_exact(HF), then subtract this systematic error from the VQE result.

```python
def chemistry_rem(
    noisy_vqe_energy: float,
    noisy_hf_energy: float,
    exact_hf_energy: float,
) -> tuple[float, dict]:
    """Reference-state error mitigation using HF as reference.
    
    E_mitigated = E_noisy_VQE - (E_noisy_HF - E_exact_HF)
    """
    delta_rem = noisy_hf_energy - exact_hf_energy
    mitigated = noisy_vqe_energy - delta_rem
    return mitigated, {
        "delta_rem": delta_rem,
        "noisy_hf": noisy_hf_energy,
        "exact_hf": exact_hf_energy,
        "method": "chemistry_rem_hf",
    }
```

**Why it's nearly free**: The HF state is our circuit's initial state (X gates on first n_electrons qubits, no operators). We can measure it with the same QWC grouped circuits but with θ=0 — or simply use the all-zeros probability from the first QWC group's Z-basis measurement.

**Key assumption**: Noise is approximately parameter-independent (same noise affects HF state and VQE state). This holds well for weakly correlated systems where the VQE state is close to HF.

**Limitation**: For strongly correlated systems (BeH2 transition state, N2 at dissociation), the VQE state diverges from HF and REM becomes less accurate. The MREM extension (arXiv:D5DD00202H) uses multireference states for these cases.

**Integration point**: After `_parse_grouped_results()` computes the raw energy, apply the REM correction. The HF energy is already available in our Hamiltonian records (`hf_energy` field).

#### Tier 2: M3 Readout Error Mitigation (SCALABLE — implement for validation)
**Cost**: 2n calibration circuits (n = qubits), ~1000 shots each
**Improvement**: 5-10× on readout-dominated circuits
**Effort**: ~100 lines + `pip install qiskit-addon-mthree`

The existing `calibrate_rem()` in `mitigation.py` builds a full 2^n × 2^n Kronecker matrix — this scales exponentially. M3 (Matrix-free Measurement Mitigation) works in the reduced subspace of observed bitstrings, making it scalable to 40+ qubits.

```python
# M3 workflow (replaces our calibrate_rem + apply_rem for n > 10)
import mthree

# Step 1: Calibrate (2n circuits, ~1000 shots each)
mit = mthree.M3Mitigation(backend)  # or simulate from qBraid calibration data
mit.cals_from_system(qubits, shots=1000)

# Step 2: Apply correction to raw counts
quasis = mit.apply_correction(raw_counts, qubits)
mitigated_expval = quasis.expval()  # directly gives expectation value
```

**Challenge**: M3 requires a Qiskit backend object for calibration. qBraid devices don't expose this directly. Options:
1. **Manual calibration**: Submit 2n circuits (|0⟩ and |1⟩ on each qubit) via qBraid, build single-qubit calibration matrices, then use M3's `apply_correction` with manual calibration data
2. **Use device-reported calibration**: Some qBraid devices (IQM, Rigetti) report readout fidelities — use these to construct approximate calibration matrices
3. **Local simulation**: Use Qiskit Aer with a noise model matching the device's reported error rates

**Integration point**: Replace `calibrate_rem()` + `apply_rem()` with M3 for n > 10q. For n ≤ 10q, the existing Kronecker approach works fine.

#### Tier 3: Zero-Noise Extrapolation (MODERATE COST — implement for deeper circuits)
**Cost**: 3× more circuits (run at noise scales 1×, 2×, 3×)
**Improvement**: 2-3× on gate-dominated circuits (deeper circuits benefit more)
**Effort**: Already implemented in `mitigation.py`, needs integration with QWC pipeline

**What exists**: `fold_gates()` does unitary folding, `zne_extrapolate()` does linear/Richardson/polynomial extrapolation.

**What's broken**: `_compute_energy_from_counts()` only handles diagonal Z terms. It needs to work with our QWC grouped measurement pipeline.

**Fix needed**: Replace `_compute_energy_from_counts()` with a call to `_parse_grouped_results()` from `qbraid_backend.py`. The ZNE flow becomes:
1. For each QWC group, fold the measurement circuit at scale factors [1, 2, 3]
2. Run all folded circuits at each scale factor
3. Parse results with `_parse_grouped_results()` at each scale
4. Extrapolate each term's expectation to zero noise
5. Sum extrapolated term expectations × coefficients → ZNE energy

**When to use ZNE**:
- H2 (depth ~12): Marginal benefit — circuit too shallow for noise to accumulate meaningfully
- LiH (depth ~40-60): Moderate benefit — 2-3× improvement on gate-dominated terms
- N2 (depth ~80-120): Significant benefit — but 3× circuit cost is expensive

**Literature caveat** (arXiv:2606.04955): ZNE does not reliably improve all circuits. In benchmarks, ZNE reduced error for only 4/12 H2 circuits. It preserves noisy circuit rankings (Spearman ρ=+0.80) but doesn't restore ideal rankings. Use it, but validate per-circuit.

**Advanced: Layer-resolved ZNE** (arXiv:zenodo.19508423): Instead of a single global scale factor, scale each ansatz layer independently. GP regression to the zero-noise limit gives 50-59% improvement over scalar ZNE for incoherent noise. More complex to implement — future work.

#### Tier 4: Dynamical Decoupling (FREE — transpiler-level)
**Cost**: Zero — inserted during compilation
**Improvement**: 1.5-2× on idle-qubit decoherence
**Effort**: Qiskit transpiler pass, ~10 lines

Insert DD sequences (XY4, XY8) on idle qubits during circuit execution. Qiskit's `ALAPSchedule` + `DynamicalDecoupling` pass handles this automatically.

```python
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
from qiskit.circuit.library import XY4

pm = PassManager([
    ALAPSchedule(backend),
    DynamicalDecoupling(duration_calculator, XY4()),
])
dd_circuit = pm.run(circuit)
```

**Challenge**: qBraid handles transpilation internally — we may not have direct access to insert DD passes. Options:
1. Apply DD before submitting to qBraid (pre-transpile with Qiskit, then submit the DD-enhanced circuit)
2. Check if the target device supports DD natively (IQM and Rigetti may already do this)

**Recommendation**: Apply DD at the Qiskit level before qBraid submission. The folded circuit with DD sequences will be transpiled by qBraid to native gates.

### 6.4 Techniques NOT Recommended for Our Pipeline

| Technique | Why not | Reference |
|-----------|---------|-----------|
| **PEC (Probabilistic Error Cancellation)** | Exponential sampling overhead; actually *increased* error in 11/12 H2 circuits tested | arXiv:2606.04955 |
| **Full QEC (Surface Code)** | Needs ~625 physical qubits per logical qubit (d=25); demonstrated on IBM Heron but not available via qBraid | arXiv:2607.01473 |
| **Magic State Distillation** | FTQC-only; requires dedicated hardware and thousands of gates | arXiv:2606.06598 |
| **McWeeny Purification** | Requires 1-RDM measurement (extra circuits); marginal improvement over REM for weakly correlated | arXiv:2004.04174 |
| **Virtual State Distillation** | Requires multiple copies of the state; not feasible on single-QPU NISQ | — |
| **Deep Learning Mitigation** | Requires training data from noiseless simulations; overkill for our molecule sizes | arXiv:2603.23936 |

### 6.5 Combined Mitigation Pipeline

The techniques stack — apply them in order:

```
Raw QPU Counts
      │
      ▼
┌─────────────────────────┐
│ Tier 0: Symmetry        │  FREE — discard unphysical shots
│ Post-Selection          │  → filtered_counts
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Tier 2: M3 Readout      │  2n calibration circuits
│ Mitigation              │  → mitigated_counts (quasi-probabilities)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ QWC Grouped Parsing     │  Existing _parse_grouped_results()
│ → per-term expectations │  → raw_energy, term_expectations
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Tier 1: Chemistry REM   │  FREE — subtract HF error
│ (HF reference)          │  → rem_energy = raw_energy - ΔE_HF
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Tier 3: ZNE             │  3× circuit cost (optional)
│ (if depth > 30)         │  → zne_energy (extrapolated)
└───────────┬─────────────┘
            ▼
     Mitigated Energy
     + uncertainty estimate
```

### 6.6 Uncertainty Quantification

Each mitigation step introduces or transforms uncertainty. We must report:

1. **Shot noise**: Standard error = σ / √(n_eff), where n_eff = effective shots after post-selection
2. **REM uncertainty**: M3 provides `expval_and_stddev()` — use this
3. **ZNE uncertainty**: Extrapolation residual gives uncertainty estimate
4. **Combined**: Propagate through the chain (or use bootstrap resampling)

**Report format** for each QPU result:
```json
{
  "raw_energy": -1.085,
  "mitigated_energy": -1.117,
  "mitigation_chain": ["symmetry_post_select", "m3_readout", "chemistry_rem"],
  "uncertainty": {
    "shot_noise_mha": 0.8,
    "rem_mha": 0.3,
    "total_mha": 0.9
  },
  "n_shots_raw": 4096,
  "n_shots_after_post_select": 3850,
  "post_select_rejection_rate": 0.06
}
```

### 6.7 Error Correction — Future Outlook (Not Implemented Now)

Surface code QEC has been demonstrated on superconducting hardware in 2026:
- **107-qubit processor**: Distance-3 surface code, lattice surgery, logical CNOT/H/S gates (arXiv:2607.01473)
- **IBM Heron r2/r3**: First end-to-end FTQC stack achieving chemical accuracy on H2, LiH, H2O, BeH2 (arXiv:zenodo.20585364)
- **Trapped ion**: Cross-code lattice surgery, genuine multipartite entanglement between logical qubits (arXiv:2607.04227)

**Why we can't use QEC now**:
1. qBraid doesn't expose QEC-aware compilation or syndrome extraction
2. Surface code d=3 needs ~17-49 physical qubits per logical qubit — our H2 (4q) would need ~68-196 physical qubits
3. The qBraid QPU devices don't support mid-circuit measurement + feed-forward needed for syndrome extraction
4. Logical gate fidelities are still ~94% (arXiv:2606.06598) — not yet better than raw NISQ + EM for small molecules

**When QEC becomes relevant for H-cGQE**:
- When targeting molecules > 50q (where EM alone can't compensate)
- When qBraid exposes QEC-aware backends (IBM Heron with Qiskit Runtime QEC)
- When logical error rates drop below 10^-6 (enabling deep circuits)

### 6.8 Implementation Priority

| Priority | Technique | Cost | Code needed | When |
|----------|-----------|------|-------------|------|
| **P0** | Symmetry post-selection | FREE | ~50 lines in `qbraid_backend.py` | Before first QPU run |
| **P0** | Chemistry REM (HF correction) | FREE | ~30 lines, new `chem_rem.py` | Before first QPU run |
| **P1** | M3 readout mitigation | 2n circuits | `pip install qiskit-addon-mthree`, ~100 lines | H2 validation run |
| **P1** | Fix ZNE + QWC integration | 3× circuits | Refactor `_compute_energy_from_counts` | LiH validation |
| **P2** | Dynamical decoupling | FREE | Qiskit transpiler pass, ~20 lines | Before LiH run |
| **P2** | Uncertainty quantification | FREE | Bootstrap or analytic propagation | With first results |
| **P3** | Layer-resolved ZNE (GP) | 3L circuits | GP regression, ~200 lines | N2 validation |
| **P3** | MREM (multireference REM) | 1-3 extra circuits | Givens rotation circuits | Strongly correlated systems |
| **Future** | Surface code QEC | ~625× qubits | Not feasible via qBraid | Post-NISQ era |

### 6.9 Expected Accuracy Gains

Based on literature benchmarks for H2 (4q, shallow circuits):

| Mitigation stack | Expected error | vs raw | Chemical accuracy? |
|-----------------|---------------|--------|-------------------|
| Raw (no mitigation) | 50-100 mHa | 1× | ❌ |
| + Symmetry post-select | 30-60 mHa | ~1.7× | ❌ |
| + Chemistry REM | 5-15 mHa | ~7× | ❌ (close) |
| + M3 readout | 2-8 mHa | ~12× | ❌ (close) |
| + ZNE (if depth > 20) | 1-4 mHa | ~25× | ✅ (maybe) |

**Chemical accuracy = ±1.6 mHa**. Achieving this on H2 with a shallow circuit is plausible with the full stack. For LiH/N2 (deeper circuits), achieving chemical accuracy is significantly harder and may require all techniques combined.

**Our IQM Emerald baseline**: 87.5% state fidelity → ~125 mHa raw error (estimated). With full mitigation stack, this could potentially be reduced to ~5-15 mHa.

---

## 7. Z2 Symmetry Tapering

### Opportunity
Our JW-mapped Hamiltonians have Z2 symmetries (particle number parity, spin parity). Tapering reduces qubit count:
- H2: 4q → 2q (2× reduction)
- LiH: 12q → 10q (1.2× reduction)
- N2: 20q → 18q (1.1× reduction)

### Impact on QPU cost
- H2 on 2q instead of 4q: same shot count, fewer gates, higher fidelity
- Opens up AQT IBEX Q1 (12q) for LiH
- Reduces circuit depth → less noise accumulation

### Implementation
- Use Qiskit's `Z2Symmetries` from `qiskit.opflow` or `pauli_symmetries`
- Must taper both the Hamiltonian AND the ansatz circuit
- The tapered operator must be compatible with the H-cGQE operator pool
- **Not trivial** — requires regenerating the Hamiltonian with the same symmetry sector

### Recommendation
- **Phase 1**: Skip tapering, validate on full 4q H2 first
- **Phase 2**: Implement tapering for LiH/N2 to reduce QPU cost

---

## 8. Async Submission Workflow

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ AIRE HPC (L40S × 3)                                             │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐               │
│  │ RL Train │───►│ Inference│───►│ L-BFGS-B   │               │
│  │ (DAPO)   │    │ (circuits)│   │ (thetas)   │               │
│  └──────────┘    └──────────┘    └─────┬──────┘               │
│                                        │                       │
│                              ┌─────────▼──────────┐            │
│                              │ QWC Grouping       │            │
│                              │ + Manifest Export  │            │
│                              └─────────┬──────────┘            │
│                                        │                       │
│  ┌─────────────────────────────────────┼──────────────┐       │
│  │ Local Qiskit Statevector (free)     │              │       │
│  │ - Instant correctness check         │              │       │
│  │ - No shot noise                     │              │       │
│  └─────────────────────────────────────┼──────────────┘       │
│                                        │                       │
└────────────────────────────────────────┼───────────────────────┘
                                         │
                                    Manifest JSON
                                    (operators, thetas,
                                     QWC groups, QASM)
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ qBraid Cloud (async)                                            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Submit jobs │─►│ Queue       │─►│ Execute     │            │
│  │ (async)     │  │ (hours OK)  │  │ (seconds)   │            │
│  └─────────────┘  └─────────────┘  └──────┬──────┘            │
│                                          │                     │
│                                   Job IDs saved                │
│                                   to metadata JSON             │
└──────────────────────────────────────────┬─────────────────────┘
                                           │
                                      (later, anytime)
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Retrieval + Analysis                                            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Load jobs   │─►│ Parse counts│─►│ Energy calc │            │
│  │ by ID       │  │ + REM/ZNE   │  │ + compare   │            │
│  └─────────────┘  └─────────────┘  └──────┬──────┘            │
│                                          │                     │
│                                 ┌────────▼────────┐            │
│                                 │ Results JSON    │            │
│                                 │ + comparison    │            │
│                                 │ + plots         │            │
│                                 └─────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Submission modes

**Mode 1: Export-only (no QPU)**
```bash
python scripts/submit_qpu_async.py --export-only --molecules h2_0.74
# → results/qpu/h2_0.74_manifest.json
```

**Mode 2: Submit async (fire and forget)**
```bash
python scripts/submit_qpu_async.py \
    --device aws:iqm:qpu:garnet \
    --shots 4096 \
    --molecules h2_0.74
# → results/qpu/h2_0.74_submission_meta.json (contains job IDs)
```

**Mode 3: Retrieve results (later)**
```bash
python scripts/submit_qpu_async.py \
    --retrieve results/qpu/h2_0.74_submission_meta.json
# → results/qpu/h2_0.74_submission_meta_result.json
```

**Mode 4: Local statevector (free, instant)**
```bash
python scripts/test_simulator_energy.py --molecules h2_0.74 --local-only
```

---

## 9. Phased Validation Plan

### Phase 1: H2 on IQM Garnet (cheapest real QPU)
**Goal**: Validate end-to-end pipeline on real hardware with minimal cost

| Parameter | Value |
|-----------|-------|
| Molecule | H2 (4q, 15 terms) |
| QWC circuits | 5 |
| Shots | 4096 |
| Device | `aws:iqm:qpu:garnet` (20q, online) |
| Cost | ~447 credits (~$4.50) |
| Mitigation | Tier 0 (raw) + Tier 3 (symmetry post-selection) |
| Expected accuracy | ±20-50 mHa (raw), ±5-10 mHa (with symmetry) |

**Steps**:
1. Export manifest: `--export-only --molecules h2_0.74`
2. Submit to IQM Garnet: `--device aws:iqm:qpu:garnet --shots 4096 --molecules h2_0.74`
3. Wait for completion (check status periodically)
4. Retrieve results: `--retrieve results/qpu/h2_0.74_submission_meta.json`
5. Compare: GPU energy vs QPU energy vs local statevector
6. Add symmetry post-selection to `_parse_grouped_results()`

### Phase 2: H2 on Rigetti 107q (fix result parsing)
**Goal**: Get on the largest available QPU

**Steps**:
1. Debug Rigetti result object structure (see Section 4)
2. Extend `_get_counts()` to handle QCS readout buffer format
3. Try `RigettiProvider` directly with `provider.setup()` for quilc
4. If direct access fails, try OpenQuantum route ($50 free credits)
5. Re-run H2 with same parameters
6. Compare IQM vs Rigetti vs GPU vs statevector

### Phase 3: LiH on IQM Garnet (medium scale)
**Goal**: Validate on a non-trivial molecule

| Parameter | Value |
|-----------|-------|
| Molecule | LiH (12q, 631 terms) |
| QWC circuits | 180 |
| Shots | 4096 |
| Device | `aws:iqm:qpu:garnet` (20q, online) |
| Cost | ~112,305 credits (~$1,123) |
| Mitigation | Tier 0 + Tier 1 (sparse REM) + Tier 3 (symmetry) |

**Steps**:
1. Submit 180 circuits async to IQM Garnet
2. Retrieve results (may take hours due to queue)
3. Apply REM with sparse calibration (not full 2^12 matrix)
4. Apply symmetry post-selection
5. Compare to GPU energy and local statevector

**Alternative**: Run on Rigetti direct (107q) if result parsing is fixed — ~$360 vs ~$1123

### Phase 4: N2 on Rigetti 107q (large scale)
**Goal**: Push to 20q on 107q hardware

| Parameter | Value |
|-----------|-------|
| Molecule | N2 (20q, 2951 terms) |
| QWC circuits | 1308 |
| Shots | 4096 |
| Device | `rigetti:rigetti:qpu:cepheus-1-108q` (107q) |
| Cost | ~180,000 credits (~$1,800) |
| Mitigation | Tier 0 + Tier 3 (symmetry) + Tier 2 (ZNE) |

**Steps**:
1. Must fix Rigetti result parsing first (Phase 2)
2. Submit 1308 circuits async
3. May need to chunk into batches (qBraid limit: 2000 circuits/batch)
4. Retrieve and parse results
5. Apply ZNE (3 noise levels × 1308 circuits = 3924 total)
6. Compare to GPU energy and MPS results

### Phase 5: Full HPC→QPU pipeline integration
**Goal**: Automated workflow from RL training to QPU validation

**Steps**:
1. Add QPU validation as STEP 5 in `scripts/run_full_uccsd_pipeline.sh`
2. After L-BFGS-B optimization, auto-export manifest
3. Auto-submit to configured QPU device
4. Poll for completion (or manual retrieval later)
5. Generate comparison report: GPU vs QPU vs statevector vs FCI
6. Add to experiment tracking (W&B or local JSON)

---

## 10. Open Questions

1. **Rigetti QCS access**: Do we have a `RIGETTI_REFRESH_TOKEN`? Need to check env vars or request access at qcs.rigetti.com/request-access

2. **OpenQuantum pricing**: The `openquantum:*` devices are online but pricing is unclear. $50 free credits / 90 days — is that enough for H2? LiH?

3. **IQM Garnet connectivity**: 20q superconducting — what's the topology? Need to check if our H-cGQE circuits (CNOT chains) map efficiently

4. **quilc availability**: Is `quilc` installed in our `cudaq-env`? If not, can we install it via pip or conda?

5. **Batch support**: Does IQM Garnet support `as_batch=True`? AWS SV1 didn't, requiring sequential submission with delays

6. **Shot count optimization**: Should we use variance-based shot allocation (more shots for high-variance terms) instead of uniform 4096? The 2026 paper (arXiv:1402-4896/ae784b) shows 43-51% shot reduction is possible

7. **Z2 tapering**: Should we implement tapering before QPU submission? Would reduce H2 from 4q→2q, making it even cheaper

---

## 11. File Inventory

| File | Purpose | Status |
|------|---------|--------|
| `src/gqe/eval/qbraid_backend.py` | Core QWC grouping + QPU submission + result parsing | ✅ Working (AWS SV1, IQM) |
| `scripts/submit_qpu_async.py` | Async submit/retrieve/manifest export | ✅ Working (export-only tested) |
| `scripts/test_simulator_energy.py` | Simulator validation script | ✅ Working (AWS SV1 tested) |
| `results/qpu/h2_0.74_manifest.json` | H2 QWC manifest | ✅ Exported |
| `results/qpu/lih_1.6_full_manifest.json` | LiH QWC manifest | ✅ Exported |
| `results/qpu/n2_1.1_full_manifest.json` | N2 QWC manifest | ✅ Exported |
| `results/eval/simulator_validation.json` | H2 simulator validation results | ✅ Saved |

### Files to create
| File | Purpose | Priority |
|------|---------|----------|
| `src/gqe/eval/error_mitigation.py` | REM, ZNE, symmetry post-selection | Phase 1 |
| `src/gqe/eval/rigetti_result_parser.py` | QCS readout buffer → counts dict | Phase 2 |
| `scripts/compare_qpu_vs_gpu.py` | Side-by-side comparison report | Phase 1 |
| `scripts/run_qpu_validation.sh` | Full QPU validation pipeline script | Phase 5 |

---

## 12. References

- qBraid pricing: https://docs.qbraid.com/home/pricing
- Rigetti Cepheus-1-108Q blog: https://www.qbraid.com/blog-posts/rigetti-cepheus-1-108q-now-on-qbraid-lab
- AWS Braket Cepheus launch: https://aws.amazon.com/blogs/quantum-computing/amazon-braket-launches-rigetti-cepheus-1-108q-superconducting-device/
- OpenQuantum Rigetti page: https://www.openquantum.com/providers/rigetti
- qBraid Rigetti provider docs: https://docs.qbraid.com/v2/sdk/user-guide/providers/rigetti
- H2 VQE benchmark on IBM (2026): arXiv:2604.11478
- VQE cost analysis: https://insights.sagentivum.com/p/vqe-real-quantum-hardware-cost
- Shot allocation optimization (2026): Phys. Scr. 101 255103
- VQE limitations on hardware (2026): Phys. Chem. Chem. Phys. 28 2834-2846
- Readout error mitigation for VQE (2026): Phys. Scr. 101 255103
- T-REx readout mitigation for VQE (2025): arXiv:2508.15072
- CDR for quantum chemistry (2026): arXiv:2511.03556
- Multireference REM (2025): Digital Discovery 4 2521-2533
- RubriQ GRPO circuit synthesis (2026): arXiv:2607.07554
- GPT-QE original paper: arXiv:2401.09253
- Off-policy GRPO sample reuse: arXiv:2505.22257
- qBraid GPU instances: https://docs.qbraid.com/lab/user-guide/gpus
- qBraid pricing: https://docs.qbraid.com/v2/home/pricing

---

## 13. qBraid GPU RL Training

### Motivation
Our DAPO RL training on AIRE L40S (3× 48GB, PCIe-only) is bottlenecked by:
1. **24q cap** — cuStateVec distributed statevector segfaults on PCIe-only L40S (CUDA IPC broken)
2. **Low throughput** — L40S has ~90 GB/s memory bandwidth vs H200's ~4.8 TB/s (53× slower)
3. **No NVLink** — multi-GPU pooling (`nvidia-mqpu`) uses thread-based parallelism, not true GPU-GPU communication

qBraid Lab offers on-demand GPU instances with H200, H100, B200, and A100 — all with NVLink and much larger VRAM.

### GPU Comparison

| Instance | VRAM | Max qubits (single-GPU) | Credits/min | $/hour | NVLink |
|----------|------|------------------------|-------------|--------|--------|
| L40S (AIRE) | 48 GB | 24q | 3.80 | $2.28 | ❌ (PCIe) |
| A100 SXM | 80 GB | 26q | 4.15 | $2.49 | ✅ |
| H100 SXM | 80 GB | 26q | 8.95 | $5.37 | ✅ |
| H200 | 141 GB | 30q | 9.15 | $5.49 | ✅ |
| B200 | 180 GB | 32q | 14.57 | $8.74 | ✅ |
| H100 8× | 640 GB | 30q+ (multi-GPU) | 66.50 | $39.90 | ✅ |

### Cost Analysis for 500-Epoch RL Training

Estimated wall time per epoch (4 molecules × 64 samples = 256 circuits/epoch):
- **L40S (3×)**: ~2.5 min/epoch → 500 epochs = ~21 hours = 4,780 credits ($47.80)
- **H200 (1×)**: ~0.8 min/epoch → 500 epochs = ~7 hours = 3,840 credits ($38.40)
- **H100 8×**: ~0.3 min/epoch → 500 epochs = ~2.5 hours = 9,975 credits ($99.75)

**H200 single-GPU is the sweet spot**: fastest per-credit for our workload, handles 30q (covers all 4 molecules including N2 at 20q), and NVLink isn't needed for single-GPU mode.

### What Changed (v2 Optimizations)

| Parameter | v1 (old) | v2 (new) | Rationale |
|-----------|----------|----------|-----------|
| `--epochs` | 200 | 500 | Longer training for H2/N2 convergence |
| `--n-samples` | 32 | 64 | 2× better gradient estimates (RubriQ uses 4000) |
| `--n-iters` | 1-2 | 5 | GPT-QE paper's N_iter=5 |
| `--reuse-iters` | 1 | 3 | 3× simulation cost reduction (arXiv:2505.22257) |
| `--buffer-batch-size` | 0 (stub) | 64 | Replay buffer now implemented (off-policy GRPO) |
| `--buffer-size` | 1000 | 2000 | Larger replay pool |
| `--adaptive-theta` | False | True | L-BFGS-B on best circuit → better energy signal |
| `--max-qubits` | 24 | 30 (H200) | H200 141GB handles 30q single-GPU |
| `--curriculum-warmup` | 30 | 50 | Longer warmup for larger sample sizes |

### Current RL Results (v1, 200 epochs, L40S)

```
Molecule    Best RL Energy      FCI Energy  Error (mHa)
h2               -1.116759       -1.137300        20.54
lih              -7.861865       -7.862900         1.04  ✅ chemical accuracy
beh2            -15.561204      -15.563500         2.30
n2             -107.496403     -107.531600        35.20
```

### Expected Improvements (v2, 500 epochs, H200)

- **Replay buffer**: Free extra gradient steps from stale samples (importance sampling corrected)
- **3× sample reuse**: Each CUDA-Q simulation feeds 3 gradient updates → 3× less simulation cost
- **Adaptive theta**: L-BFGS-B optimized energy gives ~10-50× better reward signal than fixed θ=0.01
- **2× more samples**: Better GRPO advantage estimation → lower variance gradients
- **5× more iters**: More gradient updates per epoch → faster convergence
- **H200 throughput**: ~3× faster energy evaluation → more epochs in same wall time

**Target**: H2 < 5 mHa, N2 < 15 mHa, all molecules < 25 mHa.

### Usage

#### On qBraid Lab (H200)
```bash
# 1. Launch H200 instance from qBraid dashboard
# 2. Run setup
bash scripts/setup_qbraid_gpu.sh
# 3. Start training
bash scripts/run_rl_qbraid_gpu.sh h200
```

#### On AIRE HPC (L40S, for comparison)
```bash
sbatch jobs/rl_dapo_chemeleon2_v2.slurm
```

### Files Created/Modified

| File | Change |
|------|--------|
| `src/gqe/models/train_rl_dapo.py` | Replay buffer training implemented; default hyperparams optimized |
| `scripts/setup_qbraid_gpu.sh` | New: qBraid Lab environment setup |
| `scripts/run_rl_qbraid_gpu.sh` | New: GPU-specific training launcher (h200/h100/b200/a100/l40s) |
| `jobs/rl_dapo_chemeleon2_v2.slurm` | New: AIRE Slurm job with v2 hyperparams (24h wall time) |

# HPC → AI → QPU Async Submission Workflow Plan

## Status: Draft — July 16, 2026

## 1. Current Pipeline State

### What works
- **H-cGQE Transformer**: Autoregressive circuit synthesis (GPT-2 style) → operator sequences
- **L-BFGS-B optimization**: Classical coefficient optimization on 3× L40S GPUs via CUDA-Q `nvidia-mqpu`
- **QWC Pauli grouping**: 3-5× circuit reduction (H2: 15→5, LiH: 631→180, N2: 2951→1308)
- **Local Qiskit statevector**: Free, instant correctness check (validated H2 within 0.01 mHa)
- **AWS SV1 simulator**: H2 validated at 1.477 mHa vs GPU (4096 shots)
- **Manifest export**: Self-contained JSON + QASM for decoupled QPU submission

### What's broken / incomplete
- **Rigetti direct result retrieval**: Jobs submit to `rigetti:rigetti:qpu:cepheus-1-108q` but `job.result()` fails with "Counts data is not available" — likely QCS readout buffer format incompatibility with `_get_counts()`
- **IonQ simulator rate limiting**: 429 Too Many Requests on rapid sequential submissions
- **No error mitigation**: Raw QPU counts will have significant noise (99.1% 2Q fidelity sounds good but compounds with depth)
- **No Z2 symmetry tapering**: Could reduce qubit count further (H2: 4q→2q, LiH: 12q→10q)

---

## 2. Available qBraid Devices (as of July 16, 2026)

### QPUs (Online)

| Device ID | Vendor | Qubits | Per-shot | Per-task | Per-min | Notes |
|-----------|--------|--------|----------|----------|---------|-------|
| `rigetti:rigetti:qpu:cepheus-1-108q` | Rigetti | 107 | 0 cr | 0 cr | 12000 cr | Direct QCS, needs OAuth token + quilc |
| `openquantum:rigetti:qpu:cepheus-1-108q` | Rigetti (OQ) | 107 | — | — | — | $50 free credits / 90 days |
| `openquantum:iqm:qpu:emerald` | IQM (OQ) | 54 | — | — | — | Via Quantum Rings |
| `aws:iqm:qpu:emerald` | IQM | 54 | 0.16 cr | 30 cr | 0 | Standard qBraid result format |
| `openquantum:iqm:qpu:garnet` | IQM (OQ) | 20 | — | — | — | Via Quantum Rings |
| `aws:iqm:qpu:garnet` | IQM | 20 | 0.145 cr | 30 cr | 0 | Standard qBraid result format |
| `openquantum:ionq:qpu:forte-1` | IonQ (OQ) | 36 | — | — | — | Via Quantum Rings |
| `openquantum:ionq:qpu:forte-enterprise` | IonQ (OQ) | 36 | — | — | — | Via Quantum Rings |
| `openquantum:aqt:qpu:ibex-q1` | AQT (OQ) | 12 | — | — | — | Trapped-ion |

### Simulators (Online)

| Device ID | Qubits | Cost | Notes |
|-----------|--------|------|-------|
| `ionq:ionq:sim:simulator` | 29 | $0.00/min | Free but rate-limited (429s) |
| `aws:aws:sim:sv1` | 34 | 7.5 cr/min | Free first min/task, no batch support |
| `aws:aws:sim:dm1` | 17 | 7.5 cr/min | Density matrix (noise simulation) |
| `qbraid:qbraid:sim:qir-sv` | 30 | 7.5 cr/min + 0.5 cr/task | Sparse statevector |

### QPUs (Offline/Unavailable)

| Device | Qubits | Status |
|--------|--------|--------|
| `aws:rigetti:qpu:cepheus-1-108q` | 107 | UNAVAILABLE |
| `aws:ionq:qpu:forte-1` | 36 | OFFLINE |
| `aws:ionq:qpu:forte-enterprise-1` | 36 | OFFLINE |
| `aws:aqt:qpu:ibex-q1` | 12 | UNAVAILABLE |
| `aws:quera:qpu:aquila` | 256 | UNAVAILABLE |

---

## 3. Rigetti Cepheus-1-108Q Hardware Details

| Spec | Value |
|------|-------|
| Architecture | 12 × 9-qubit chiplets, square lattice |
| Connectivity | 4-fold nearest-neighbor (tunable couplers + IMCs) |
| Native gates | RX, RY, CZ (adiabatic) |
| 2Q gate fidelity | 99.1% median (target: 99.5% by end of 2026) |
| 1Q gate fidelity | 99.9% median |
| Gate speed | ~60 ns |
| T1 / T2 | 25 μs / 10 μs |
| Availability | 20 hrs/day |
| Queue | Can stretch to hours during peak US/EU times |

### Connectivity implications
- Square lattice = 4-fold nearest-neighbor
- Our H-cGQE circuits use CNOTs which need SWAP routing for non-adjacent qubits
- H2 (4q): fits within a single chiplet (3×3 grid), minimal SWAP overhead
- LiH (12q): spans ~2 chiplets, moderate SWAP overhead
- N2 (20q): spans ~3 chiplets, significant inter-chiplet routing needed

### Direct access requirements
1. **QCS account** with active OAuth refresh token (contact Rigetti at qcs.rigetti.com/request-access)
2. **Forest SDK** — `quilc` compiler must be running locally or at a known endpoint
3. Environment variables: `RIGETTI_REFRESH_TOKEN`, optionally `QCS_QUILC_ENDPOINT`, `QCS_GRPC_ENDPOINT`
4. The `RigettiProvider` must call `provider.setup()` to start/manage quilc

---

## 4. The Result Retrieval Problem

### Symptom
Jobs submit successfully to `rigetti:rigetti:qpu:cepheus-1-108q` and `openquantum:rigetti:qpu:cepheus-1-108q`, but `job.result()` fails with "Counts data is not available."

### Likely causes (ranked by probability)

1. **QCS readout buffer format mismatch** (most likely)
   - The qBraid `RigettiJob` (PR #1127) parses QCS readout buffers into `GateModelResultData`
   - Our `_get_counts()` expects `result.data.get_counts()` or `result.measurement_counts()`
   - Rigetti results may expose data differently — need to inspect the actual result object attributes

2. **quilc compilation not running**
   - `RigettiProvider.setup()` starts quilc locally; if we're using `QbraidProvider` (not `RigettiProvider`), quilc may not be started
   - Without compilation, the job may be submitted but execution may silently fail or produce empty results

3. **Job still in queue**
   - Rigetti QPU has queue times; the job may not have completed when we tried to fetch results
   - Our retry logic (6 attempts × exponential backoff) may not be enough for QPU queue times

### Debugging plan
```python
# Step 1: Check job status (not just result)
from qbraid.runtime import load_job
job = load_job("rigetti:rigetti:qpu:cepheus-1-108q-135b-qjob-...")
print(f"Status: {job.status()}")

# Step 2: If COMPLETED, inspect result object structure
result = job.result()
print(f"Type: {type(result)}")
print(f"Dir: {[a for a in dir(result) if not a.startswith('_')]}")
print(f"Data type: {type(result.data)}")
print(f"Data dir: {[a for a in dir(result.data) if not a.startswith('_')]}")

# Step 3: Try alternative count extraction
# Rigetti may use: result.data.readout_values, result.data.measurements, etc.
```

### Fix options

**Option A: Extend `_get_counts()` for Rigetti results**
```python
def _get_counts(result: Any) -> dict[str, int]:
    # Standard qBraid path (AWS, IonQ, IQM)
    if hasattr(result, "data") and hasattr(result.data, "get_counts"):
        return result.data.get_counts()
    elif hasattr(result, "measurement_counts"):
        return result.measurement_counts()
    # Rigetti QCS path — parse readout buffer
    elif hasattr(result, "data") and hasattr(result.data, "readout_values"):
        # Convert raw readout to counts dict
        ...
    raise AttributeError(...)
```

**Option B: Use RigettiProvider directly instead of QbraidProvider**
```python
from qbraid.runtime.rigetti import RigettiProvider
import os
os.environ["RIGETTI_REFRESH_TOKEN"] = "..."
provider = RigettiProvider()
provider.setup()  # starts quilc
device = provider.get_device("cepheus-1-108q")
```

**Option C: Use OpenQuantum route (may bypass QCS auth issues)**
- The `openquantum:` prefix devices go through Quantum Rings' proxy
- May handle compilation and result parsing differently
- $50 free credits every 90 days

---

## 5. Cost Analysis

### H2 (4q, 5 QWC circuits, 4096 shots)

| Device | Task cost | Shot cost | Min cost | Total | USD |
|--------|-----------|-----------|-----------|-------|-----|
| AWS SV1 (sim) | 0 | 0 | ~0.6 cr | ~1 cr | ~$0.01 |
| IonQ sim | 0 | 0 | 0 | 0 | $0 |
| IQM Garnet (20q) | 150 cr | 297 cr | 0 | 447 cr | ~$4.50 |
| IQM Emerald (54q) | 150 cr | 328 cr | 0 | 478 cr | ~$4.80 |
| IonQ Forte-1 (36q) | 150 cr | 16384 cr | 0 | 16534 cr | ~$165 |
| Rigetti direct (107q) | 0 | 0 | ~6000 cr (30s) | ~6000 cr | ~$60 |
| Rigetti AWS (107q) | 150 cr | 870 cr | 0 | 1020 cr | ~$10.20 |

### LiH (12q, 180 QWC circuits, 4096 shots)

| Device | Task cost | Shot cost | Min cost | Total | USD |
|--------|-----------|-----------|-----------|-------|-----|
| IQM Garnet (20q) | 5400 cr | 106905 cr | 0 | 112305 cr | ~$1123 |
| IQM Emerald (54q) | 5400 cr | 117964 cr | 0 | 123364 cr | ~$1234 |
| Rigetti direct (107q) | 0 | 0 | ~36000 cr (3 min) | ~36000 cr | ~$360 |
| Rigetti AWS (107q) | 5400 cr | 311040 cr | 0 | 316440 cr | ~$3164 |

### N2 (20q, 1308 QWC circuits, 4096 shots)

| Device | Task cost | Shot cost | Total | USD |
|--------|-----------|-----------|-------|-----|
| IQM Garnet (20q) | 39240 cr | 776145 cr | 815385 cr | ~$8154 |
| Rigetti direct (107q, ~15 min) | 0 | 0 | ~180000 cr | ~$1800 |

### Key insight
- **Rigetti direct** (per-minute billing) is cheapest for large circuits with many shots
- **IQM Garnet** (per-shot) is cheapest for small circuits with few shots
- **Simulators** are free/cheap for H2 but don't scale to LiH/N2

---

## 6. Error Mitigation & Error Correction Strategy

### Why we need it
At 99.1% 2Q gate fidelity, a circuit with 10 two-qubit gates has ~9% error probability. For H2 with ~5-10 CNOTs, this means ~5-10% of shots are corrupted. Without mitigation, energy error could be 50-100+ mHa — far above chemical accuracy (1.6 mHa).

Our previous IQM Emerald run confirmed this: 896/1024 shots (87.5%) matched the expected state — meaning **12.5% of shots were corrupted** by a combination of gate errors and readout errors.

### 6.1 Error Mitigation vs Error Correction

**Error Mitigation (EM)** = post-process noisy results to estimate noise-free observables. No extra qubits. Probabilistic/estimation-based. Works on NISQ hardware. What we do now.

**Error Correction (EC)** = encode logical qubits across many physical qubits, actively detect and correct errors during computation. Requires fault-tolerant hardware. Not practical via qBraid cloud today.

| Aspect | Error Mitigation | Error Correction |
|--------|-----------------|-----------------|
| Qubit overhead | 0 extra qubits | ~625 physical per logical (d=25 surface code) |
| Circuit overhead | 1-3× more circuits | 1000s× more gates |
| Accuracy gain | 2-100× | Exponential (below threshold) |
| Hardware | Any NISQ QPU | Needs dedicated QEC-capable hardware |
| Status | **Available now** | Demonstrated on IBM Heron (2026), not on qBraid |

### 6.2 Existing Mitigation Code

`src/gqe/eval/mitigation.py` already implements:

| Function | What it does | Status |
|----------|-------------|--------|
| `calibrate_rem()` | Prepares |0⟩ and |1⟩ on each qubit, builds single-qubit assignment matrices, Kronecker-products them into full 2^n × 2^n matrix | ⚠️ Scales as 2^n — only works for n ≤ ~10q |
| `apply_rem()` | Inverts calibration matrix (pseudo-inverse or least-squares) to correct raw counts | ⚠️ Full matrix inversion — doesn't scale |
| `fold_gates()` | Unitary folding U → U(U†U)^c for ZNE noise scaling | ✅ Works |
| `zne_extrapolate()` | Linear/Richardson/polynomial extrapolation to zero-noise limit | ✅ Works |
| `run_zne_experiment()` | Full ZNE pipeline: fold → run → measure → extrapolate | ⚠️ `_compute_energy_from_counts` only handles diagonal Z terms, not QWC grouped measurements |

### 6.3 Mitigation Techniques — What to Implement

#### Tier 0: Symmetry Post-Selection (FREE — implement first)
**Cost**: Zero extra circuits, zero extra shots
**Improvement**: ~2× on half the terms
**Effort**: ~50 lines of code

Discard shots that violate known Z2 symmetries. Under Jordan-Wigner mapping, our Hamiltonians conserve:
- **Particle number parity**: Σ n_i mod 2 = N_e mod 2
- **Spin parity** (singlet states): S_z mod 2 = 0

For each bitstring in the counts, check if the Hamming weight matches the expected parity. If not, discard it and renormalize the remaining counts.

```python
def symmetry_post_select(
    counts: dict[str, int],
    n_electrons: int,
    n_qubits: int,
    symmetry: str = "particle_number",
) -> dict[str, int]:
    """Post-select shots conserving Z2 symmetries."""
    expected_parity = n_electrons % 2 if symmetry == "particle_number" else 0
    filtered = {}
    for bitstring, count in counts.items():
        hamming = sum(int(b) for b in bitstring)
        if hamming % 2 == expected_parity:
            filtered[bitstring] = count
    return filtered
```

**Integration point**: Call after `_get_counts()` and before `_parse_grouped_results()` in `qbraid_backend.py`.

**Important**: This only works for Z-basis measurements (computational basis). For X/Y basis measurements (QWC groups measuring X or Y terms), the symmetry check must be applied in the rotated basis — i.e., check parity of the *transformed* bitstring.

#### Tier 1: Chemistry Reference-State Error Mitigation (CHEAP — implement first)
**Cost**: 0-1 additional circuit evaluations (HF state)
**Improvement**: Up to 100× (two orders of magnitude, per JCTC 2022 paper)
**Effort**: ~30 lines of code

The idea (from arXiv:acs.jctc.2c00807): measure the noisy energy of a classically-known reference state (Hartree-Fock), compute the error ΔE = E_noisy(HF) - E_exact(HF), then subtract this systematic error from the VQE result.

```python
def chemistry_rem(
    noisy_vqe_energy: float,
    noisy_hf_energy: float,
    exact_hf_energy: float,
) -> tuple[float, dict]:
    """Reference-state error mitigation using HF as reference.
    
    E_mitigated = E_noisy_VQE - (E_noisy_HF - E_exact_HF)
    """
    delta_rem = noisy_hf_energy - exact_hf_energy
    mitigated = noisy_vqe_energy - delta_rem
    return mitigated, {
        "delta_rem": delta_rem,
        "noisy_hf": noisy_hf_energy,
        "exact_hf": exact_hf_energy,
        "method": "chemistry_rem_hf",
    }
```

**Why it's nearly free**: The HF state is our circuit's initial state (X gates on first n_electrons qubits, no operators). We can measure it with the same QWC grouped circuits but with θ=0 — or simply use the all-zeros probability from the first QWC group's Z-basis measurement.

**Key assumption**: Noise is approximately parameter-independent (same noise affects HF state and VQE state). This holds well for weakly correlated systems where the VQE state is close to HF.

**Limitation**: For strongly correlated systems (BeH2 transition state, N2 at dissociation), the VQE state diverges from HF and REM becomes less accurate. The MREM extension (arXiv:D5DD00202H) uses multireference states for these cases.

**Integration point**: After `_parse_grouped_results()` computes the raw energy, apply the REM correction. The HF energy is already available in our Hamiltonian records (`hf_energy` field).

#### Tier 2: M3 Readout Error Mitigation (SCALABLE — implement for validation)
**Cost**: 2n calibration circuits (n = qubits), ~1000 shots each
**Improvement**: 5-10× on readout-dominated circuits
**Effort**: ~100 lines + `pip install qiskit-addon-mthree`

The existing `calibrate_rem()` in `mitigation.py` builds a full 2^n × 2^n Kronecker matrix — this scales exponentially. M3 (Matrix-free Measurement Mitigation) works in the reduced subspace of observed bitstrings, making it scalable to 40+ qubits.

```python
# M3 workflow (replaces our calibrate_rem + apply_rem for n > 10)
import mthree

# Step 1: Calibrate (2n circuits, ~1000 shots each)
mit = mthree.M3Mitigation(backend)  # or simulate from qBraid calibration data
mit.cals_from_system(qubits, shots=1000)

# Step 2: Apply correction to raw counts
quasis = mit.apply_correction(raw_counts, qubits)
mitigated_expval = quasis.expval()  # directly gives expectation value
```

**Challenge**: M3 requires a Qiskit backend object for calibration. qBraid devices don't expose this directly. Options:
1. **Manual calibration**: Submit 2n circuits (|0⟩ and |1⟩ on each qubit) via qBraid, build single-qubit calibration matrices, then use M3's `apply_correction` with manual calibration data
2. **Use device-reported calibration**: Some qBraid devices (IQM, Rigetti) report readout fidelities — use these to construct approximate calibration matrices
3. **Local simulation**: Use Qiskit Aer with a noise model matching the device's reported error rates

**Integration point**: Replace `calibrate_rem()` + `apply_rem()` with M3 for n > 10q. For n ≤ 10q, the existing Kronecker approach works fine.

#### Tier 3: Zero-Noise Extrapolation (MODERATE COST — implement for deeper circuits)
**Cost**: 3× more circuits (run at noise scales 1×, 2×, 3×)
**Improvement**: 2-3× on gate-dominated circuits (deeper circuits benefit more)
**Effort**: Already implemented in `mitigation.py`, needs integration with QWC pipeline

**What exists**: `fold_gates()` does unitary folding, `zne_extrapolate()` does linear/Richardson/polynomial extrapolation.

**What's broken**: `_compute_energy_from_counts()` only handles diagonal Z terms. It needs to work with our QWC grouped measurement pipeline.

**Fix needed**: Replace `_compute_energy_from_counts()` with a call to `_parse_grouped_results()` from `qbraid_backend.py`. The ZNE flow becomes:
1. For each QWC group, fold the measurement circuit at scale factors [1, 2, 3]
2. Run all folded circuits at each scale factor
3. Parse results with `_parse_grouped_results()` at each scale
4. Extrapolate each term's expectation to zero noise
5. Sum extrapolated term expectations × coefficients → ZNE energy

**When to use ZNE**:
- H2 (depth ~12): Marginal benefit — circuit too shallow for noise to accumulate meaningfully
- LiH (depth ~40-60): Moderate benefit — 2-3× improvement on gate-dominated terms
- N2 (depth ~80-120): Significant benefit — but 3× circuit cost is expensive

**Literature caveat** (arXiv:2606.04955): ZNE does not reliably improve all circuits. In benchmarks, ZNE reduced error for only 4/12 H2 circuits. It preserves noisy circuit rankings (Spearman ρ=+0.80) but doesn't restore ideal rankings. Use it, but validate per-circuit.

**Advanced: Layer-resolved ZNE** (arXiv:zenodo.19508423): Instead of a single global scale factor, scale each ansatz layer independently. GP regression to the zero-noise limit gives 50-59% improvement over scalar ZNE for incoherent noise. More complex to implement — future work.

#### Tier 4: Dynamical Decoupling (FREE — transpiler-level)
**Cost**: Zero — inserted during compilation
**Improvement**: 1.5-2× on idle-qubit decoherence
**Effort**: Qiskit transpiler pass, ~10 lines

Insert DD sequences (XY4, XY8) on idle qubits during circuit execution. Qiskit's `ALAPSchedule` + `DynamicalDecoupling` pass handles this automatically.

```python
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
from qiskit.circuit.library import XY4

pm = PassManager([
    ALAPSchedule(backend),
    DynamicalDecoupling(duration_calculator, XY4()),
])
dd_circuit = pm.run(circuit)
```

**Challenge**: qBraid handles transpilation internally — we may not have direct access to insert DD passes. Options:
1. Apply DD before submitting to qBraid (pre-transpile with Qiskit, then submit the DD-enhanced circuit)
2. Check if the target device supports DD natively (IQM and Rigetti may already do this)

**Recommendation**: Apply DD at the Qiskit level before qBraid submission. The folded circuit with DD sequences will be transpiled by qBraid to native gates.

### 6.4 Techniques NOT Recommended for Our Pipeline

| Technique | Why not | Reference |
|-----------|---------|-----------|
| **PEC (Probabilistic Error Cancellation)** | Exponential sampling overhead; actually *increased* error in 11/12 H2 circuits tested | arXiv:2606.04955 |
| **Full QEC (Surface Code)** | Needs ~625 physical qubits per logical qubit (d=25); demonstrated on IBM Heron but not available via qBraid | arXiv:2607.01473 |
| **Magic State Distillation** | FTQC-only; requires dedicated hardware and thousands of gates | arXiv:2606.06598 |
| **McWeeny Purification** | Requires 1-RDM measurement (extra circuits); marginal improvement over REM for weakly correlated | arXiv:2004.04174 |
| **Virtual State Distillation** | Requires multiple copies of the state; not feasible on single-QPU NISQ | — |
| **Deep Learning Mitigation** | Requires training data from noiseless simulations; overkill for our molecule sizes | arXiv:2603.23936 |

### 6.5 Combined Mitigation Pipeline

The techniques stack — apply them in order:

```
Raw QPU Counts
      │
      ▼
┌─────────────────────────┐
│ Tier 0: Symmetry        │  FREE — discard unphysical shots
│ Post-Selection          │  → filtered_counts
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Tier 2: M3 Readout      │  2n calibration circuits
│ Mitigation              │  → mitigated_counts (quasi-probabilities)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ QWC Grouped Parsing     │  Existing _parse_grouped_results()
│ → per-term expectations │  → raw_energy, term_expectations
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Tier 1: Chemistry REM   │  FREE — subtract HF error
│ (HF reference)          │  → rem_energy = raw_energy - ΔE_HF
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Tier 3: ZNE             │  3× circuit cost (optional)
│ (if depth > 30)         │  → zne_energy (extrapolated)
└───────────┬─────────────┘
            ▼
     Mitigated Energy
     + uncertainty estimate
```

### 6.6 Uncertainty Quantification

Each mitigation step introduces or transforms uncertainty. We must report:

1. **Shot noise**: Standard error = σ / √(n_eff), where n_eff = effective shots after post-selection
2. **REM uncertainty**: M3 provides `expval_and_stddev()` — use this
3. **ZNE uncertainty**: Extrapolation residual gives uncertainty estimate
4. **Combined**: Propagate through the chain (or use bootstrap resampling)

**Report format** for each QPU result:
```json
{
  "raw_energy": -1.085,
  "mitigated_energy": -1.117,
  "mitigation_chain": ["symmetry_post_select", "m3_readout", "chemistry_rem"],
  "uncertainty": {
    "shot_noise_mha": 0.8,
    "rem_mha": 0.3,
    "total_mha": 0.9
  },
  "n_shots_raw": 4096,
  "n_shots_after_post_select": 3850,
  "post_select_rejection_rate": 0.06
}
```

### 6.7 Error Correction — Future Outlook (Not Implemented Now)

Surface code QEC has been demonstrated on superconducting hardware in 2026:
- **107-qubit processor**: Distance-3 surface code, lattice surgery, logical CNOT/H/S gates (arXiv:2607.01473)
- **IBM Heron r2/r3**: First end-to-end FTQC stack achieving chemical accuracy on H2, LiH, H2O, BeH2 (arXiv:zenodo.20585364)
- **Trapped ion**: Cross-code lattice surgery, genuine multipartite entanglement between logical qubits (arXiv:2607.04227)

**Why we can't use QEC now**:
1. qBraid doesn't expose QEC-aware compilation or syndrome extraction
2. Surface code d=3 needs ~17-49 physical qubits per logical qubit — our H2 (4q) would need ~68-196 physical qubits
3. The qBraid QPU devices don't support mid-circuit measurement + feed-forward needed for syndrome extraction
4. Logical gate fidelities are still ~94% (arXiv:2606.06598) — not yet better than raw NISQ + EM for small molecules

**When QEC becomes relevant for H-cGQE**:
- When targeting molecules > 50q (where EM alone can't compensate)
- When qBraid exposes QEC-aware backends (IBM Heron with Qiskit Runtime QEC)
- When logical error rates drop below 10^-6 (enabling deep circuits)

### 6.8 Implementation Priority

| Priority | Technique | Cost | Code needed | When |
|----------|-----------|------|-------------|------|
| **P0** | Symmetry post-selection | FREE | ~50 lines in `qbraid_backend.py` | Before first QPU run |
| **P0** | Chemistry REM (HF correction) | FREE | ~30 lines, new `chem_rem.py` | Before first QPU run |
| **P1** | M3 readout mitigation | 2n circuits | `pip install qiskit-addon-mthree`, ~100 lines | H2 validation run |
| **P1** | Fix ZNE + QWC integration | 3× circuits | Refactor `_compute_energy_from_counts` | LiH validation |
| **P2** | Dynamical decoupling | FREE | Qiskit transpiler pass, ~20 lines | Before LiH run |
| **P2** | Uncertainty quantification | FREE | Bootstrap or analytic propagation | With first results |
| **P3** | Layer-resolved ZNE (GP) | 3L circuits | GP regression, ~200 lines | N2 validation |
| **P3** | MREM (multireference REM) | 1-3 extra circuits | Givens rotation circuits | Strongly correlated systems |
| **Future** | Surface code QEC | ~625× qubits | Not feasible via qBraid | Post-NISQ era |

### 6.9 Expected Accuracy Gains

Based on literature benchmarks for H2 (4q, shallow circuits):

| Mitigation stack | Expected error | vs raw | Chemical accuracy? |
|-----------------|---------------|--------|-------------------|
| Raw (no mitigation) | 50-100 mHa | 1× | ❌ |
| + Symmetry post-select | 30-60 mHa | ~1.7× | ❌ |
| + Chemistry REM | 5-15 mHa | ~7× | ❌ (close) |
| + M3 readout | 2-8 mHa | ~12× | ❌ (close) |
| + ZNE (if depth > 20) | 1-4 mHa | ~25× | ✅ (maybe) |

**Chemical accuracy = ±1.6 mHa**. Achieving this on H2 with a shallow circuit is plausible with the full stack. For LiH/N2 (deeper circuits), achieving chemical accuracy is significantly harder and may require all techniques combined.

**Our IQM Emerald baseline**: 87.5% state fidelity → ~125 mHa raw error (estimated). With full mitigation stack, this could potentially be reduced to ~5-15 mHa.

---

## 7. Z2 Symmetry Tapering

### Opportunity
Our JW-mapped Hamiltonians have Z2 symmetries (particle number parity, spin parity). Tapering reduces qubit count:
- H2: 4q → 2q (2× reduction)
- LiH: 12q → 10q (1.2× reduction)
- N2: 20q → 18q (1.1× reduction)

### Impact on QPU cost
- H2 on 2q instead of 4q: same shot count, fewer gates, higher fidelity
- Opens up AQT IBEX Q1 (12q) for LiH
- Reduces circuit depth → less noise accumulation

### Implementation
- Use Qiskit's `Z2Symmetries` from `qiskit.opflow` or `pauli_symmetries`
- Must taper both the Hamiltonian AND the ansatz circuit
- The tapered operator must be compatible with the H-cGQE operator pool
- **Not trivial** — requires regenerating the Hamiltonian with the same symmetry sector

### Recommendation
- **Phase 1**: Skip tapering, validate on full 4q H2 first
- **Phase 2**: Implement tapering for LiH/N2 to reduce QPU cost

---

## 8. Async Submission Workflow

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ AIRE HPC (L40S × 3)                                             │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐               │
│  │ RL Train │───►│ Inference│───►│ L-BFGS-B   │               │
│  │ (DAPO)   │    │ (circuits)│   │ (thetas)   │               │
│  └──────────┘    └──────────┘    └─────┬──────┘               │
│                                        │                       │
│                              ┌─────────▼──────────┐            │
│                              │ QWC Grouping       │            │
│                              │ + Manifest Export  │            │
│                              └─────────┬──────────┘            │
│                                        │                       │
│  ┌─────────────────────────────────────┼──────────────┐       │
│  │ Local Qiskit Statevector (free)     │              │       │
│  │ - Instant correctness check         │              │       │
│  │ - No shot noise                     │              │       │
│  └─────────────────────────────────────┼──────────────┘       │
│                                        │                       │
└────────────────────────────────────────┼───────────────────────┘
                                         │
                                    Manifest JSON
                                    (operators, thetas,
                                     QWC groups, QASM)
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ qBraid Cloud (async)                                            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Submit jobs │─►│ Queue       │─►│ Execute     │            │
│  │ (async)     │  │ (hours OK)  │  │ (seconds)   │            │
│  └─────────────┘  └─────────────┘  └──────┬──────┘            │
│                                          │                     │
│                                   Job IDs saved                │
│                                   to metadata JSON             │
└──────────────────────────────────────────┬─────────────────────┘
                                           │
                                      (later, anytime)
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Retrieval + Analysis                                            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Load jobs   │─►│ Parse counts│─►│ Energy calc │            │
│  │ by ID       │  │ + REM/ZNE   │  │ + compare   │            │
│  └─────────────┘  └─────────────┘  └──────┬──────┘            │
│                                          │                     │
│                                 ┌────────▼────────┐            │
│                                 │ Results JSON    │            │
│                                 │ + comparison    │            │
│                                 │ + plots         │            │
│                                 └─────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Submission modes

**Mode 1: Export-only (no QPU)**
```bash
python scripts/submit_qpu_async.py --export-only --molecules h2_0.74
# → results/qpu/h2_0.74_manifest.json
```

**Mode 2: Submit async (fire and forget)**
```bash
python scripts/submit_qpu_async.py \
    --device aws:iqm:qpu:garnet \
    --shots 4096 \
    --molecules h2_0.74
# → results/qpu/h2_0.74_submission_meta.json (contains job IDs)
```

**Mode 3: Retrieve results (later)**
```bash
python scripts/submit_qpu_async.py \
    --retrieve results/qpu/h2_0.74_submission_meta.json
# → results/qpu/h2_0.74_submission_meta_result.json
```

**Mode 4: Local statevector (free, instant)**
```bash
python scripts/test_simulator_energy.py --molecules h2_0.74 --local-only
```

---

## 9. Phased Validation Plan

### Phase 1: H2 on IQM Garnet (cheapest real QPU)
**Goal**: Validate end-to-end pipeline on real hardware with minimal cost

| Parameter | Value |
|-----------|-------|
| Molecule | H2 (4q, 15 terms) |
| QWC circuits | 5 |
| Shots | 4096 |
| Device | `aws:iqm:qpu:garnet` (20q, online) |
| Cost | ~447 credits (~$4.50) |
| Mitigation | Tier 0 (raw) + Tier 3 (symmetry post-selection) |
| Expected accuracy | ±20-50 mHa (raw), ±5-10 mHa (with symmetry) |

**Steps**:
1. Export manifest: `--export-only --molecules h2_0.74`
2. Submit to IQM Garnet: `--device aws:iqm:qpu:garnet --shots 4096 --molecules h2_0.74`
3. Wait for completion (check status periodically)
4. Retrieve results: `--retrieve results/qpu/h2_0.74_submission_meta.json`
5. Compare: GPU energy vs QPU energy vs local statevector
6. Add symmetry post-selection to `_parse_grouped_results()`

### Phase 2: H2 on Rigetti 107q (fix result parsing)
**Goal**: Get on the largest available QPU

**Steps**:
1. Debug Rigetti result object structure (see Section 4)
2. Extend `_get_counts()` to handle QCS readout buffer format
3. Try `RigettiProvider` directly with `provider.setup()` for quilc
4. If direct access fails, try OpenQuantum route ($50 free credits)
5. Re-run H2 with same parameters
6. Compare IQM vs Rigetti vs GPU vs statevector

### Phase 3: LiH on IQM Garnet (medium scale)
**Goal**: Validate on a non-trivial molecule

| Parameter | Value |
|-----------|-------|
| Molecule | LiH (12q, 631 terms) |
| QWC circuits | 180 |
| Shots | 4096 |
| Device | `aws:iqm:qpu:garnet` (20q, online) |
| Cost | ~112,305 credits (~$1,123) |
| Mitigation | Tier 0 + Tier 1 (sparse REM) + Tier 3 (symmetry) |

**Steps**:
1. Submit 180 circuits async to IQM Garnet
2. Retrieve results (may take hours due to queue)
3. Apply REM with sparse calibration (not full 2^12 matrix)
4. Apply symmetry post-selection
5. Compare to GPU energy and local statevector

**Alternative**: Run on Rigetti direct (107q) if result parsing is fixed — ~$360 vs ~$1123

### Phase 4: N2 on Rigetti 107q (large scale)
**Goal**: Push to 20q on 107q hardware

| Parameter | Value |
|-----------|-------|
| Molecule | N2 (20q, 2951 terms) |
| QWC circuits | 1308 |
| Shots | 4096 |
| Device | `rigetti:rigetti:qpu:cepheus-1-108q` (107q) |
| Cost | ~180,000 credits (~$1,800) |
| Mitigation | Tier 0 + Tier 3 (symmetry) + Tier 2 (ZNE) |

**Steps**:
1. Must fix Rigetti result parsing first (Phase 2)
2. Submit 1308 circuits async
3. May need to chunk into batches (qBraid limit: 2000 circuits/batch)
4. Retrieve and parse results
5. Apply ZNE (3 noise levels × 1308 circuits = 3924 total)
6. Compare to GPU energy and MPS results

### Phase 5: Full HPC→QPU pipeline integration
**Goal**: Automated workflow from RL training to QPU validation

**Steps**:
1. Add QPU validation as STEP 5 in `scripts/run_full_uccsd_pipeline.sh`
2. After L-BFGS-B optimization, auto-export manifest
3. Auto-submit to configured QPU device
4. Poll for completion (or manual retrieval later)
5. Generate comparison report: GPU vs QPU vs statevector vs FCI
6. Add to experiment tracking (W&B or local JSON)

---

## 10. Open Questions

1. **Rigetti QCS access**: Do we have a `RIGETTI_REFRESH_TOKEN`? Need to check env vars or request access at qcs.rigetti.com/request-access

2. **OpenQuantum pricing**: The `openquantum:*` devices are online but pricing is unclear. $50 free credits / 90 days — is that enough for H2? LiH?

3. **IQM Garnet connectivity**: 20q superconducting — what's the topology? Need to check if our H-cGQE circuits (CNOT chains) map efficiently

4. **quilc availability**: Is `quilc` installed in our `cudaq-env`? If not, can we install it via pip or conda?

5. **Batch support**: Does IQM Garnet support `as_batch=True`? AWS SV1 didn't, requiring sequential submission with delays

6. **Shot count optimization**: Should we use variance-based shot allocation (more shots for high-variance terms) instead of uniform 4096? The 2026 paper (arXiv:1402-4896/ae784b) shows 43-51% shot reduction is possible

7. **Z2 tapering**: Should we implement tapering before QPU submission? Would reduce H2 from 4q→2q, making it even cheaper

---

## 11. File Inventory

| File | Purpose | Status |
|------|---------|--------|
| `src/gqe/eval/qbraid_backend.py` | Core QWC grouping + QPU submission + result parsing | ✅ Working (AWS SV1, IQM) |
| `scripts/submit_qpu_async.py` | Async submit/retrieve/manifest export | ✅ Working (export-only tested) |
| `scripts/test_simulator_energy.py` | Simulator validation script | ✅ Working (AWS SV1 tested) |
| `results/qpu/h2_0.74_manifest.json` | H2 QWC manifest | ✅ Exported |
| `results/qpu/lih_1.6_full_manifest.json` | LiH QWC manifest | ✅ Exported |
| `results/qpu/n2_1.1_full_manifest.json` | N2 QWC manifest | ✅ Exported |
| `results/eval/simulator_validation.json` | H2 simulator validation results | ✅ Saved |

### Files to create
| File | Purpose | Priority |
|------|---------|----------|
| `src/gqe/eval/error_mitigation.py` | REM, ZNE, symmetry post-selection | Phase 1 |
| `src/gqe/eval/rigetti_result_parser.py` | QCS readout buffer → counts dict | Phase 2 |
| `scripts/compare_qpu_vs_gpu.py` | Side-by-side comparison report | Phase 1 |
| `scripts/run_qpu_validation.sh` | Full QPU validation pipeline script | Phase 5 |

---

## 12. References

- qBraid pricing: https://docs.qbraid.com/home/pricing
- Rigetti Cepheus-1-108Q blog: https://www.qbraid.com/blog-posts/rigetti-cepheus-1-108q-now-on-qbraid-lab
- AWS Braket Cepheus launch: https://aws.amazon.com/blogs/quantum-computing/amazon-braket-launches-rigetti-cepheus-1-108q-superconducting-device/
- OpenQuantum Rigetti page: https://www.openquantum.com/providers/rigetti
- qBraid Rigetti provider docs: https://docs.qbraid.com/v2/sdk/user-guide/providers/rigetti
- H2 VQE benchmark on IBM (2026): arXiv:2604.11478
- VQE cost analysis: https://insights.sagentivum.com/p/vqe-real-quantum-hardware-cost
- Shot allocation optimization (2026): Phys. Scr. 101 255103
- VQE limitations on hardware (2026): Phys. Chem. Chem. Phys. 28 2834-2846
- Readout error mitigation for VQE (2026): Phys. Scr. 101 255103
- T-REx readout mitigation for VQE (2025): arXiv:2508.15072
- CDR for quantum chemistry (2026): arXiv:2511.03556
- Multireference REM (2025): Digital Discovery 4 2521-2533
- RubriQ GRPO circuit synthesis (2026): arXiv:2607.07554
- GPT-QE original paper: arXiv:2401.09253
- Off-policy GRPO sample reuse: arXiv:2505.22257
- qBraid GPU instances: https://docs.qbraid.com/lab/user-guide/gpus
- qBraid pricing: https://docs.qbraid.com/v2/home/pricing

---

## 13. qBraid GPU RL Training

### Motivation
Our DAPO RL training on AIRE L40S (3× 48GB, PCIe-only) is bottlenecked by:
1. **24q cap** — cuStateVec distributed statevector segfaults on PCIe-only L40S (CUDA IPC broken)
2. **Low throughput** — L40S has ~90 GB/s memory bandwidth vs H200's ~4.8 TB/s (53× slower)
3. **No NVLink** — multi-GPU pooling (`nvidia-mqpu`) uses thread-based parallelism, not true GPU-GPU communication

qBraid Lab offers on-demand GPU instances with GH200, H200, H100, B200, A100, and L40S — all with NVLink (except L40S) and much larger VRAM.

### GPU Comparison

| Instance | VRAM | Max qubits (single-GPU) | Credits/min | $/hour | NVLink |
|----------|------|------------------------|-------------|--------|--------|
| L40S (AIRE) | 48 GB | 24q | 3.80 | $2.28 | ❌ (PCIe) |
| **GH200** | **96 GB** | **28q** | **4.78** | **$2.87** | ✅ (Grace Hopper) |
| A100 SXM | 80 GB | 26q | 4.15 | $2.49 | ✅ |
| H100 SXM | 80 GB | 26q | 8.95 | $5.37 | ✅ |
| H200 | 141 GB | 30q | 9.15 | $5.49 | ✅ |
| B200 | 180 GB | 32q | 14.57 | $8.74 | ✅ |
| H100 8× | 640 GB | 30q+ (multi-GPU) | 66.50 | $39.90 | ✅ |

**GH200 is the chosen GPU** for Phase 3 RL training. qBraid benchmarks show GH200 is 1.2-1.9× faster than H100 in CUDA-Q (LiH VQE: 1.2×, 33q random circuit: 1.9×), while being 1.87× cheaper per minute (4.78 vs 8.95 cr/min). The Grace Hopper Superchip's unified CPU-GPU memory architecture and high-bandwidth interconnect make it CUDA-Q's reference platform for MPS simulations.

### Cost Analysis for 500-Epoch RL Training

Estimated wall time per epoch (4 molecules × 64 samples = 256 circuits/epoch):
- **L40S (3×)**: ~2.5 min/epoch → 500 epochs = ~21 hours = 4,780 credits ($47.80)
- **GH200 (1×)**: ~0.9 min/epoch → 500 epochs = ~7.5 hours = 2,151 credits ($21.51)
- **H200 (1×)**: ~0.8 min/epoch → 500 epochs = ~7 hours = 3,840 credits ($38.40)
- **H100 8×**: ~0.3 min/epoch → 500 epochs = ~2.5 hours = 9,975 credits ($99.75)

**GH200 single-GPU is the chosen instance**: best value per credit (1.2-1.9× faster than H100 in CUDA-Q, 1.87× cheaper), 96GB unified memory handles 28q statevector, 33.6h max runtime with 9,645 credits leaves room for re-runs + QPU validation.

### Full Phase 3 Budget (GH200, 9,645 credits)

| Component | Time | Credits |
|-----------|------|---------|
| RL training (500 epochs) | 7.5h | 2,151 |
| MPS scaling (24-40q × 4 bonds) | 8.5min | 41 |
| QSCI 40q (benzene) | 1min | 5 |
| H-cGQE evaluation (24q, 28q) | 30min | 143 |
| Setup + buffer | 30min | 143 |
| **Total compute** | **~9h** | **~2,483** |
| **Remaining for QPU + re-runs** | **~24.6h** | **~7,162** |

### What Changed (v2 Optimizations)

| Parameter | v1 (old) | v2 (new) | Rationale |
|-----------|----------|----------|-----------|
| `--epochs` | 200 | 500 | Longer training for H2/N2 convergence |
| `--n-samples` | 32 | 64 | 2× better gradient estimates (RubriQ uses 4000) |
| `--n-iters` | 1-2 | 5 | GPT-QE paper's N_iter=5 |
| `--reuse-iters` | 1 | 3 | 3× simulation cost reduction (arXiv:2505.22257) |
| `--buffer-batch-size` | 0 (stub) | 64 | Replay buffer now implemented (off-policy GRPO) |
| `--buffer-size` | 1000 | 2000 | Larger replay pool |
| `--adaptive-theta` | False | True | L-BFGS-B on best circuit → better energy signal |
| `--max-qubits` | 24 | 28 (GH200) | GH200 96GB handles 28q single-GPU |
| `--curriculum-warmup` | 30 | 50 | Longer warmup for larger sample sizes |

### Current RL Results (v1, 200 epochs, L40S)

```
Molecule    Best RL Energy      FCI Energy  Error (mHa)
h2               -1.116759       -1.137300        20.54
lih              -7.861865       -7.862900         1.04  ✅ chemical accuracy
beh2            -15.561204      -15.563500         2.30
n2             -107.496403     -107.531600        35.20
```

### Expected Improvements (v2, 500 epochs, GH200)

- **Replay buffer**: Free extra gradient steps from stale samples (importance sampling corrected)
- **3× sample reuse**: Each CUDA-Q simulation feeds 3 gradient updates → 3× less simulation cost
- **Adaptive theta**: L-BFGS-B optimized energy gives ~10-50× better reward signal than fixed θ=0.01
- **2× more samples**: Better GRPO advantage estimation → lower variance gradients
- **5× more iters**: More gradient updates per epoch → faster convergence
- **GH200 throughput**: ~4× faster than L40S, 1.2-1.9× faster than H100 in CUDA-Q → more epochs in same wall time

**Target**: H2 < 5 mHa, N2 < 15 mHa, all molecules < 25 mHa.

### Usage

#### On qBraid Lab (GH200)
```bash
# 1. Launch GH200 instance from qBraid dashboard (On-Demand tab)
# 2. Run setup
bash scripts/setup_qbraid_gpu.sh
# 3. Start training
bash scripts/run_rl_qbraid_gpu.sh gh200
```

#### On AIRE HPC (L40S, for comparison)
```bash
sbatch jobs/rl_dapo_chemeleon2_v2.slurm
```

### Files Created/Modified

| File | Change |
|------|--------|
| `src/gqe/models/train_rl_dapo.py` | Replay buffer training implemented; n_iters wired; BF16/CPU fix; curriculum_warmup=0 fix; --no-dynamic-sampling flag |
| `src/gqe/models/h_cgqe_transformer.py` | Fix: .view() → .reshape() for non-contiguous tensors |
| `scripts/setup_qbraid_gpu.sh` | New: qBraid Lab environment setup (GH200/H200/H100/B200/A100/L40S) |
| `scripts/run_rl_qbraid_gpu.sh` | New: GPU-specific training launcher (gh200/h200/h100/b200/a100/l40s) |
| `jobs/rl_dapo_chemeleon2_v2.slurm` | New: AIRE Slurm job with v2 hyperparams (24h wall time) |
| `scripts/submit_qpu_async.py` | New: Async HPC→QPU workflow with QWC manifest export |
| `scripts/test_simulator_energy.py` | New: Simulator energy validation (AWS SV1) |
| `src/gqe/eval/qbraid_backend.py` | QWC Pauli term grouping + bit ordering fix |
| `src/gqe/eval/submit_qpu.py` | QPU preflight checks (ZNE/REM gating) |
| `scripts/qpu_preflight.py` | New: QPU preflight validation |
