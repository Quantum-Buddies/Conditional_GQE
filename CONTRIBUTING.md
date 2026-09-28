# Contributing to Conditional-GQE (H-cGQE)

Thanks for your interest! This project is maintained by the Quantum-Buddies
team. Bug reports, reproducibility notes, and pull requests are all welcome.

## Getting set up

```bash
git clone https://github.com/Quantum-Buddies/Conditional_GQE.git
cd Conditional_GQE
conda env create -f environment-dgx-spark-cudaq.yml   # or use venv + pip
conda activate conditional-gqe-cudaq
pip install -r requirements-qbraid.txt
```

- Python **3.11**, CUDA-Q **>= 0.10**, PyTorch 2.7+.
- Some checkpoints/datasets are Git LFS objects — run `git lfs pull` after
  cloning if `results/train/*.pt` look like pointer files.
- On qBraid Lab, `bash scripts/setup_env.sh` performs a one-shot setup with no
  sudo required.

## Running tests

```bash
pytest tests/
```

For a CPU-only sanity check of the pipeline (no GPU needed):

```bash
bash scripts/phase3/00_smoke_test.sh
```

## Code style

- Match the existing style in the file you're editing: standard library /
  third-party / local import grouping, `snake_case` functions, ASCII
  docstrings describing CLI flags.
- Keep CLI-facing flags backward compatible; add new options as `--flags`
  with sensible defaults.
- Heavy dependencies (CUDA-Q, PyTorch CUDA builds) must stay optional —
  modules under `src/gqe/eval/` and `src/gqe/models/` should degrade
  gracefully or raise clear errors when GPUs/CUDA-Q are unavailable.
- Don't commit large artifacts (`*.pt`, `*.sqlite`, result JSONs); add them
  via Git LFS or describe how to regenerate them.

## HPC / Slurm jobs

Scripts under `jobs/` are **Slurm templates specific to the AIRE cluster**
(L40S GPUs, `--partition=gpu --gres=gpu:l40s:N`). If you adapt them for
another site, keep the template and document your site-specific variant
rather than editing the shared paths in place.

## Opening issues and PRs

- **Issues**: include the command you ran, environment (GPU model, CUDA-Q
  version), and the full traceback or metric JSON. For benchmark questions,
  attach the relevant `results/**` artifact if possible.
- **Pull requests**: keep them small and scoped. Describe what changed and
  how you verified it (test run, smoke script, or eval JSON). One feature or
  fix per PR.
- Benchmark claims should cite the artifact they come from — see the
  hedged-results convention in `README.md` (e.g., "within the stated active
  space").
