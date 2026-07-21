# Verification of RL Proxy Energy

Per the physicist's excellent question, we evaluated whether the fixed-theta ($\theta=0.01$) proxy used during RL actually correlates with the true multi-start optimized energy $E^\star(s)$.

## Method
1. Built a verification script `scripts/phase3_eval/verify_rl_proxy.py`
2. Sampled 15 generated circuits for `iodobenzene`.
3. Computed the RL Proxy Energy $E_{\theta=0.01}$.
4. Computed the Converged Energy $E^\star$ using a 5-restart multi-start L-BFGS-B.
5. Computed the Spearman rank correlation between the proxy and the converged energies.

## Results
- **Spearman Rank Correlation**: 0.2270 (p-value: 0.416)
- **Finding**: The proxy energies for almost all circuits evaluated to exactly `-7078.008313 Ha` (Hartree-Fock baseline), with variations only at the 11th decimal place. 
- However, after full L-BFGS-B optimization, the energies varied significantly (from `-7078.001 Ha` down to `-7078.009 Ha`, a spread of ~8 mHa).

## Conclusion
The physicist was 100% correct. The fixed-$\theta$ proxy provides almost zero gradient or ranking signal for the RL policy on large molecules. The model is effectively optimizing noise during RL because the reward landscape is completely flat prior to L-BFGS-B optimization.

## Recommended Fixes for the Policy
To fix this, we must replace the fixed-$\theta$ proxy in `train_rl_dapo.py` with a better surrogate:
1. **Truncated Optimization**: Run 3-5 steps of L-BFGS-B during the RL reward calculation.
2. **First-order proxy**: Evaluate the gradient at $\theta=0$ and use that as the signal.
3. **Critic Network**: Train a separate critic to predict the final converged energy from the circuit structure.
