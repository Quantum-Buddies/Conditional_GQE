# H-cGQE qBraid Skill

Reproducible, agent-executable packaging for the H-cGQE Phase 3 submission on qBraid.

## Commands

From the project root:

```bash
# List available commands
python qbraid_skill/main.py list

# Install dependencies
python qbraid_skill/main.py install

# Run the full pipeline
bash qbraid_skill/run_all.sh
```

## Supported subcommands

- `install` — install the qBraid-compatible Python dependencies
- `prepare-dataset` — build the supervised GQE dataset from baseline results
- `train` — train the H-cGQE Transformer
- `train-rlqf` — fine-tune the trained model with RLQF (REINFORCE)
- `infer` — generate operator sequences with constrained decoding
- `optimize` — optimize rotation coefficients with L-BFGS-B on CUDA-Q
- `evaluate` — evaluate generated circuits against the GQE baseline
- `qbraid-eval` — run circuits on a qBraid-managed simulator or QPU
- `plot` — generate benchmark plots

## Manifest

See `skill.json` for the qBraid Skill manifest.
