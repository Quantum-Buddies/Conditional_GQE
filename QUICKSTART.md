# QUICKSTART — Conditional-GQE Phase 3

## Setup

```bash
# Clone
git clone -b phase3-submission https://github.com/Quantum-Buddies/Conditional_GQE.git
cd Conditional_GQE

# Environment (CUDA-Q + PyTorch + quantum chemistry)
conda env create -f environment-dgx-spark-cudaq.yml
conda activate conditional-gqe-cudaq
pip install -r requirements-qbraid.txt
```

## Smoke Test (CPU, no GPU required)

```bash
bash scripts/phase3/00_smoke_test.sh
```

## Full Reproduction (GPU required)

### Experiment 1 — AI vs ansatz benchmark (CH₃I, 8 qubits)
```bash
bash scripts/phase3/02_run_baselines.sh
bash scripts/phase3/03_run_hcgqe.sh
```

### Experiment 2 — QPU validation
```bash
# Check QPU availability first
python scripts/qpu_preflight.py --dry-run
# Submit (requires qBraid API key)
bash scripts/phase3/06_submit_qpu.sh
# Collect results (after job completes)
bash scripts/phase3/07_collect_qpu.sh <job_id>
```

### Experiment 3 — FMO2 reconstruction
```bash
bash scripts/phase3/04_run_fmo.sh
```

### Experiment 4 — MPS scaling curve
```bash
bash scripts/phase3/05_run_mps.sh
```

### Build report
```bash
bash scripts/phase3/08_build_report.sh
```

## Launch on qBraid

[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid.svg)](https://account.qbraid.com?link=https://github.com/Quantum-Buddies/Conditional_GQE)

## Expected Runtime

| Experiment | GPU | Approx. Time |
|---|---|---|
| Smoke test | None | < 1 min |
| Experiment 1 | 1× L40S | ~10 min |
| Experiment 2 | QPU | Queue-dependent |
| Experiment 3 | 1× L40S | ~30 min |
| Experiment 4 | 1× L40S | ~20 min |
