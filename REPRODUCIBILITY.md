# REPRODUCIBILITY — Conditional-GQE Phase 3

## Environment

- **Conda env**: `cudaq-env` (CUDA-Q 0.8+, PyTorch 2.6+, PySCF, OpenFermion, qBraid SDK)
- **Setup**: See `QUICKSTART.md`
- **Lock script**: `bash scripts/lock_environment.sh` captures git commit, Python/pip versions, GPU info, and qBraid API key presence (without exposing the key)

## Hardware

- **Development**: AIRE HPC, 3× NVIDIA L40S (48GB, PCIe-only, no NVLink)
- **qBraid GPU profiles**: L40S, H200, B200 available on-demand
- **QPU access**: Rigetti, IonQ, IBM via qBraid SDK
- **State-vector limit**: 24 qubits on L40S (cuStateVec distribution threshold = 25; segfaults on PCIe-only L40S due to CUDA IPC limitations)
- **MPS backend**: `tensornet-mps` for >24 qubit systems (single-GPU mode on L40S)

## Determinism

- All experiments use seed=42
- PyTorch deterministic mode enabled for inference
- CUDA-Q simulator is deterministic for fixed seed
- L-BFGS-B optimization uses deterministic initialization

## Result Manifests

Every result JSON in `results/phase3_final/` contains a manifest with:
- `git_commit`: Full SHA at time of generation
- `timestamp_utc`: ISO 8601 timestamp
- `molecule`, `geometry`, `basis`, `active_electrons`, `active_spatial_orbitals`, `qubits`
- `backend`, `device_id`, `shots`, `seed`
- `logical_depth`, `transpiled_depth`, `two_qubit_gates`
- `energy_hartree`, `reference_energy_hartree`, `error_mha`
- `wall_time_seconds`, `status`

Run `python scripts/lock_environment.sh` to capture environment metadata before experiments.

## QPU Preflight

Before any QPU execution:
```bash
python scripts/qpu_preflight.py --dry-run
```
This lists available devices, estimates credit costs, and saves a sanitized manifest. No QPU credits are spent in dry-run mode.

## Cached Results

Expensive runs (QPU, large MPS) have cached results in `results/phase3_final/`. Judges can reproduce from cached data without re-running on QPU.

## Known Limitations

- N₂ (12-20 qubits): Not converged — strongly correlated system, requires larger active space or different operator pool
- BeH₂ (14 qubits): Not converged — similar diagonal collapse issue
- IMePh (8 qubits, test set): 24.63 mHa error — unseen EUV molecule, not used in training
- MPS on L40S: Single-GPU only (pip-installed CUDA-Q does not support MPI tensornet)
