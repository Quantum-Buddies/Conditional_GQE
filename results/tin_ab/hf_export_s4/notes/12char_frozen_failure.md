# 12-char frozen H-cGQE failure on SnO 14q

The public H-cGQE checkpoint [`Ryukijano/h-cgqe-gic2026`](https://huggingface.co/Ryukijano/h-cgqe-gic2026)
(`results/train/h_cgqe_uccsd_model.pt`) is a **ground-state circuit generator** trained on a mixed-length
UCCSD word vocab. Operator words from that run are **12 characters**, then trailing-I padded onto 14 qubits.

On the exported MatGen-Q SnO 14q Hamiltonian (CASCI −288.10912104 Ha):

| Run | Column | ΔE vs CASCI | N_dets |
|---|---|---|---|
| job 7431320 (wrong Pauli endianness) | frozen_hcgqe_12char_padded | +6.022 mHa | 13 |
| CPU NumPy S4 (mapping fixed) | frozen_hcgqe_12char_padded | +6.120 mHa | 1 |
| CPU NumPy S4 | hf | +6.120 mHa | 1 |

+6.12 mHa is the active-space correlation (RHF vs CASCI). The frozen 12-char policy sits at Hartree–Fock.
It cannot emit MatGen-Q’s 289-token **14-char** pool (`pool_sno14q.json`). Do not DAPO this iodine/12-char
checkpoint onto SnO. Next scientific step is a **new** 14-char vocab SFT (Slurm **7431349**, PENDING), then
infer `n_samples=64` with union cap 32 — not DAPO, not paid QPU.
