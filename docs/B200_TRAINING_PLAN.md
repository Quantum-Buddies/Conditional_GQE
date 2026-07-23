# B200 training plan

## Decision

Use supervised warm-start followed by DAPO reinforcement learning as the primary experiment. Run direct RL from scratch only as an ablation.

The repository contains both paths. train_h_cgqe.py learns from existing GQE operator sequences, while train_rl_dapo.py can load that checkpoint or use --from-scratch.

The warm start is preferable because the RL reward requires CUDA-Q circuit evaluation and L-BFGS-B refinement. On larger and unseen molecules, a randomly initialized policy has high-variance rewards and is more likely to collapse into invalid, repetitive, or diagonal operator sequences. The warm start supplies a valid UCCSD-derived operator vocabulary and useful initial distribution; DAPO then moves beyond imitation using energy feedback.

## Available molecular data

| Dataset | Records | Range | Intended use |
|---|---:|---:|---|
| results/data/hamiltonians.json | 5 | 4–20q | baseline development |
| results/data/hamiltonians_phase3.json/hamiltonians.json | 18 | 4–14q | phase-3 benchmark/PES |
| results/data/hamiltonians_gic2026/hamiltonians.json | 35 | 4–28q | broad chemistry and EUV set |
| results/data/hamiltonians_merged.json | 21 | 4–40q | combined training/scaling set |
| results/data/hamiltonians_40plus/hamiltonians.json | 10 | 4–40q | B200 scaling stress set |

The 40+ set includes H2, LiH, BeH2, N2, ethylene, formaldehyde, benzene, and larger N2/BeH2 active spaces. Introduce 28–40q systems only after the policy is stable on the 4–20q core.

## Recommended stages

1. Train the real supervised H-cGQE model from results/train/gqe_supervised_dataset.pt.
2. Fine-tune with DAPO on H2, LiH, BeH2, and N2.
3. Add unseen 12–24q GIC molecules for generalization testing.
4. Evaluate 28–40q molecules with MPS/statevector safeguards.
5. Optimize coefficients, validate locally/free simulator, then submit selected shallow circuits to hardware.

Run the portable primary workflow with:

    SKIP_SUPERVISED=0 RL_EPOCHS=500 RL_SAMPLES=64 RL_ITERS=5 MAX_QUBITS=30 MAX_TERMS=256 bash scripts/run_b200_training.sh

For direct-RL ablation, use train_rl_dapo.py --from-scratch on the 4–20q core. It is useful scientifically, but should not replace the warm-start main result.
