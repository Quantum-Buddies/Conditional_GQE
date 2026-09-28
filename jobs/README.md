# Slurm job templates (AIRE HPC)

These `.sbatch` files are site-specific templates for the AIRE cluster
(L40S GPU nodes, `gpu` partition, conda envs under the submitter's scratch).
They are **not** portable — adapt the following for your site before submitting:

- `#SBATCH` directives (`--partition`, `--gres`, `--output`/`--error` paths)
- `SCRATCH_ROOT` (path prefix used in the script body; defaults to
  `/scratch/kcwp264` — export it or rewrite the paths)
- conda env paths / `source` lines (`cudaq-env`, `cudaq15-qbraid`)
- `GQE_MATGENQ_DIR`: clone of the `gqe-qsci-euv-photoresists` baselines repo
  (required by the QSCI / MatGen-Q jobs)
