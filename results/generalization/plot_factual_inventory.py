#!/usr/bin/env python3
"""Publication-quality factual plots for H-cGQE generalization audit.

All numbers are copied from on-disk JSON (no fitted curves, no invented
held-out organotin series). Run from cudaq-env after sourcing
/scratch/kcwp264/.aire_scratch_env.sh.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path("/scratch/kcwp264/Conditional-GQE_materials")
OUT = ROOT / "results/generalization"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe–Ito (colourblind-safe)
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERM = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#666666"
HF_FILL = "#BBBBBB"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.15,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": BLACK,
        "legend.framealpha": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _bold_legend(ax, **kwargs):
    leg = ax.legend(**kwargs)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    if leg.get_title() is not None:
        leg.get_title().set_fontweight("bold")
    return leg


def load_json(rel: str):
    return json.loads((ROOT / rel).read_text())


def build_inventory() -> dict:
    sft14 = load_json("results/tin_ab/h_cgqe_sft_pool14_7431349_metrics.json")
    sft14_sum = load_json("results/tin_ab/sft_pool14/sft_pool14_summary.json")
    vocab = load_json("results/tin_ab/sft_vocab_word_lengths.json")
    sft_ds = load_json("results/train/gqe_dataset_summary.json")
    uccsd_ds = load_json("results/train/uccsd_dataset/gqe_dataset_summary.json")
    eval_gic = load_json("results/eval/h_cgqe_evaluation_gic2026.json")
    s4_np = load_json("results/tin_ab/qsci_controls_cpu_numpy.json")
    s4_gpu = load_json("results/tin_ab/qsci_controls_7431345.json")
    infer = load_json("results/tin_ab/hcgqe_infer_sno14q_7386634.json")
    p14 = load_json("results/tin_ab/hcgqe_qsci_pool14_7434600.json")
    ch3i = load_json("results/phase3_final/benchmark_ch3i_consolidated.json")
    abl = load_json("results/phase3_final/ablation_sft_vs_rl.json")
    cond = load_json("results/tables/conditioning_ablation_main_summary.json")
    qbraid = load_json("results/train/h_cgqe_model_qbraid_rl_rl_metrics.json")
    vanilla = load_json("results/train/h_cgqe_rl_ablation_vanilla_dapo_rl_metrics.json")

    seqs = []
    for rec in infer:
        for s in rec.get("generated_sequences", []):
            seqs.append(tuple(s.get("operators") or []))
    op_lens = [len(w) for ops in seqs for w in ops]

    seen_rows = []
    for row in eval_gic:
        hf_mha = 1000.0 * float(row["baseline_error_vs_reference"])
        gqe_mha = float(row["energy_error"])
        seen_rows.append(
            {
                "molecule": row["molecule"],
                "split": "seen_in_family",
                "hcgqe_de_mha": gqe_mha,
                "hf_de_mha": hf_mha,
                "delta_minus_hf_mha": gqe_mha - hf_mha,
                "n_samples": row["n_samples"],
                "source": "results/eval/h_cgqe_evaluation_gic2026.json",
            }
        )

    ch3i_hcgqe = next(r for r in ch3i["results"] if r["method"] == "h_cgqe_rlqf")
    ch3i_gqe = next(r for r in ch3i["results"] if r["method"] == "cudaq_gqe")
    ch3i_hea = next(r for r in ch3i["results"] if r["method"] == "hardware_efficient_vqe")

    inventory = {
        "audit_date": "2026-08-22",
        "verdict": "no_measured_graph_conditioned_zero_shot",
        "verdict_sentence": (
            "H-cGQE does not currently show graph-conditioned zero-shot generalization. "
            "Generate-time conditioning is a HamiltonianEncoder over Pauli terms; the GNN "
            "is a separate property-regression checkpoint never loaded by infer. Mixed-length "
            "SFT on SnO 14q emitted 12-char words and sat at Hartree–Fock. In-distribution "
            "14-char pool SFT (job 7434600) recovered below HF but missed 1.6 mHa. No held-out "
            "organotin energy table exists."
        ),
        "forbidden_not_plotted": {
            "sn_c_collapse_kcal_mol": {
                "value": "72.6 → 21.2",
                "source": "arXiv:2607.23988 classical UCCSD(T)",
                "note": "Not a GQE/H-cGQE number. Never plotted as model accuracy.",
            },
            "imeph_92eV": {
                "source": "arXiv:2602.20234",
                "note": "~200 logical qubits; out of scope on 3× L40S.",
            },
            "gic2026_score_sheet": "Competition closed; no public numerical score sheet.",
        },
        "training": {
            "sft_uccsd_dataset": {
                "path": "results/train/gqe_dataset_summary.json",
                "checkpoint": "results/train/h_cgqe_uccsd_model.pt",
                "n_named_instances": len(sft_ds["molecules"]),
                "names": sft_ds["molecules"],
                "n_sequences": sft_ds["n_samples"],
                "n_original": sft_ds["n_original"],
                "n_augmented": sft_ds["n_augmented"],
                "vocab_size": sft_ds["vocab_size"],
                "unique_chemical_graphs_connectivity": [
                    "H2",
                    "H2O",
                    "LiH",
                    "BeH2",
                    "N2",
                    "iodobenzene",
                    "methyl_iodide",
                    "imeph",
                    "phenol",
                ],
                "n_unique_chemical_graphs": 9,
                "sn_or_organotin_in_sft": False,
                "imeph_note": (
                    "imeph_cas12 was in the SFT JSON. YAML isomer was the wrong "
                    "2-iodo-6-methylphenol layout until 21 Aug 2026; current YAML is "
                    "4-iodo-2-methylphenol Oc1ccc(I)cc1C. Eval numbers predate the fix."
                ),
            },
            "sft_uccsd_vocab_word_lengths": vocab["word_length_counts"],
            "sft_uccsd_metrics": {
                "n_params_state_dict": 7852853,
                "vocab_size": 149,
                "best_val_loss": 1.1826924979686737,
                "n_epochs_logged": 278,
                "final_val_acc": 0.9923809523809524,
                "gnn_tensors_in_checkpoint": False,
            },
            "sft_b200": {
                "checkpoint": "results/train/h_cgqe_model_b200_sft.pt",
                "vocab_size": 317,
                "best_val_loss": 1.0367504358291626,
                "final_val_acc": 0.9619047619047619,
                "sn_in_operator_vocab": False,
                "note": "Not the SnO infer checkpoint.",
            },
            "sft_pool14_sno": {
                "job": "7431349",
                "checkpoint": "results/tin_ab/h_cgqe_sft_pool14_7431349.pt",
                "metrics": "results/tin_ab/h_cgqe_sft_pool14_7431349_metrics.json",
                "n_teacher_sequences": sft14_sum["n_samples"],
                "vocab_size": sft14_sum["vocab_size"],
                "unique_operator_words": sft14_sum["word_length_counts"],
                "all_words_length_14": sft14_sum["all_operator_words_length_14"],
                "best_val_loss": sft14["best_val_loss"],
                "final_val_loss": sft14["final_val_loss"],
                "final_val_acc": sft14["final_val_acc"],
                "epochs_run": sft14["epochs_run"],
                "early_stopped": sft14["early_stopped"],
                "split": "in_distribution_sno_teacher_sequences",
                "zero_shot": False,
                "gnn_used": False,
            },
            "dapo": {
                "tin_dapo_submitted": False,
                "cancelled_job": "7431324",
                "vanilla_dapo_molecules": vanilla["config"]["molecules"],
                "qbraid_rl_n_molecules": len(qbraid["best_energies"]),
                "qbraid_rl_molecules": sorted(qbraid["best_energies"]),
                "qbraid_rl_n_epochs": qbraid["n_epochs_completed"],
                "qbraid_rl_mean_reward_first": 2.046667779840798,
                "qbraid_rl_mean_reward_last": 1.0432076903421703,
                "sn_or_organotin_in_dapo": False,
                "note": "DAPO/QD-GRPO was on small organics (H2/LiH/BeH2/N2 core; qBraid 32-name mix). No SnO DAPO.",
            },
        },
        "gnn": {
            "code": "src/gqe/models/chemistry_encoder.py",
            "architecture": "EdgeAwareMessageBlock MPNN; saved ckpt is 2 layers, hidden/latent/cond=32 (Hub prose says 3-layer 128).",
            "checkpoints": [
                "results/train/chemistry_encoder.pt",
                "results/train/graph_conditioning.pt",
            ],
            "training_task": "Hamiltonian property regression (n_qubits, Pauli counts, …)",
            "property_ablation": {
                "path": "results/tables/conditioning_ablation_main_summary.json",
                "n_samples": cond["graph"]["num_samples"],
                "graph_val_mae": cond["graph"]["final_val_mae"],
                "flat_val_mae": cond["flat"]["final_val_mae"],
                "note": "Property MAE on 5 samples; not a circuit-generation ablation.",
            },
            "wired_into_HcGQEModel_generate": False,
            "loaded_by_infer_h_cgqe": False,
            "sn_atomic_number_in_PERIODIC_TABLE": True,
            "sn_smiles_token_present": True,
            "used_at_tin_infer": False,
        },
        "generate_time_conditioning": {
            "module": "HamiltonianEncoder in src/gqe/models/h_cgqe_transformer.py",
            "input": "Pauli term ids + coefficients",
            "not": "molecular graph / ChemistryEncoder prefix tokens",
        },
        "seen_eval_gic2026": seen_rows,
        "ch3i_8q_lbfgs": {
            "path": "results/phase3_final/benchmark_ch3i_consolidated.json",
            "split": "seen_iodine_toy",
            "h_cgqe_rlqf_mha": ch3i_hcgqe["error_mha"],
            "cudaq_gqe_mha": ch3i_gqe["error_mha"],
            "hea_vqe_mha": ch3i_hea["error_mha"],
            "not_organotin_zero_shot": True,
        },
        "sno14q": {
            "casci_ha": -288.10912104,
            "hf_mha": 6.120,
            "n_fci": 49,
            "geometry": "Sn–O 1.8325 Å, def2-SVP + ECP28MDF, (2e,7o)",
            "frozen_infer": {
                "job": "7386634",
                "n_samples": 16,
                "n_unique_sequences": len(set(seqs)),
                "all_operator_words_length_12": all(L == 12 for L in op_lens),
                "n_operator_tokens": len(op_lens),
                "qsci_wrong_endianness_mha": 6.02181478240027,
                "qsci_mapping_fixed_mha": 6.120172029909554,
                "n_dets_fixed": 1,
                "dialect": "12-char padded; not MatGen-Q 14-char pool",
                "split": "unseen_Sn_wrong_operator_dialect",
            },
            "s4_numpy": {
                "path": "results/tin_ab/qsci_controls_cpu_numpy.json",
                "hf_mha": s4_np["columns"][0]["error_vs_casci_mha"],
                "matgenq_smoke_best_mha": 2.881918892455815,
                "frozen_12char_mha": 6.120172029909554,
                "best_pool14_mha": 3.933521771443793,
                "best_pool14_column": "pool14q_random_2",
                "best_pool14_n_dets": 11,
                "union_greedy_mha": 0.9526376310873275,
                "union_greedy_n_dets": 21,
                "gate1_pool_native_pass": True,
            },
            "s4_cudaq_7431345": {
                "path": "results/tin_ab/qsci_controls_7431345.json",
                "matgenq_smoke_best_mha": 2.151911921316696,
                "union_greedy_mha": 1.421578797589973,
                "union_greedy_n_dets": 21,
                "best_pool14_mha": 4.382000852160672,
                "best_pool14_column": "pool14q_single_157",
                "gate1_pool_native_pass": True,
            },
            "per_instance_vqe": {
                "uccsd_vqe_mha": 1.2264877113921102e-05,
                "uccsd_vqe_n_eval": 490,
                "adapt_vqe_mha": 8.906391713026096e-05,
                "adapt_vqe_n_eval": 3112,
                "adapt_vqe_n_ops": 20,
                "paths": [
                    "/scratch/kcwp264/baselines/gqe-qsci-euv-photoresists/code/results_vqe.json",
                    "/scratch/kcwp264/baselines/gqe-qsci-euv-photoresists/code/results_adapt_vqe.json",
                ],
            },
            "pool14_infer": {
                "submitted": True,
                "job": "7434600",
                "script": "jobs/tin_infer_pool14.sbatch",
                "checkpoint": "results/tin_ab/h_cgqe_sft_pool14_7431349.pt",
                "infer": "results/tin_ab/hcgqe_infer_pool14_7434600.json",
                "qsci": "results/tin_ab/hcgqe_qsci_pool14_7434600.json",
                "n_samples": int(p14["n_sequences"]),
                "n_scored": int(p14["n_scored"]),
                "best_single_mha": float(p14["best_single"]["error_vs_casci_mha"]),
                "best_single_n_dets": int(p14["best_single"]["n_dets"]),
                "best_single_sample_id": int(p14["best_single"]["sample_id"]),
                "union_capped_mha": float(p14["union_capped"]["error_vs_casci_mha"]),
                "union_capped_n_dets": int(p14["union_capped"]["n_dets"]),
                "max_union_dets": int(p14["max_union_dets"]),
                "filled_full_ci": bool(p14["best_single"]["filled_full_ci"]),
                "s4_pass": bool(p14["s4_pass"]),
                "split": "in_distribution_sno_teacher_sft",
                "zero_shot": False,
                "gnn_used": False,
                "chemical_accuracy_mha": 1.6,
                "note": (
                    "Native 14-char words. Best single +2.597 mHa / 21 dets; "
                    "union cap 32 dets +2.149 mHa. Misses 1.6 mHa; not HF; not 49/49."
                ),
            },
        },
        "zero_shot": {
            "held_out_organotin_energy_table": False,
            "train_on_A_infer_on_B_energy": False,
            "closest_proxy": (
                "Frozen mixed-length SFT (organic UCCSD vocab) inferred on unseen SnO 14q; "
                "emitted 12-char words (318/318); QSCI sat at HF (+6.120 mHa, 1 det) after "
                "mapping fix. This is dialect failure + unused GNN, not a GNN generalization score."
            ),
            "identity_pad_transfer": {
                "path": "results/eval/h_cgqe_operators_for_qsci.json",
                "note": "transfer_from_* entries pad operators with identity and self-label as not a converged H-cGQE result.",
            },
            "smiles_transfer_dataset": {
                "path": "results/phase3_final/transfer_learning_dataset.json",
                "note": "SMILES tokenisation of 10 molecules (vocab 53). Not an energy transfer table. No Sn.",
            },
        },
        "logging": {
            "trackio_in_repo": False,
            "trackio_cli": "Broken on login python3.9 (PEP604 | unions). No trackio files in this repo.",
            "wandb_on_scratch": "Recent /scratch/kcwp264/wandb runs are dino_foresight.train, not H-cGQE.",
            "primary_sources": [
                "results/eval/",
                "results/tin_ab/",
                "results/phase3_final/",
                "results/train/*_metrics.json",
            ],
        },
        "missing_eval": [
            "ΔE vs unique training graphs / topologies (no such series logged)",
            "Held-out organotin (methyltin / n-Bu / aromatic Sn) QSCI vs CASCI",
            "GNN-on vs GNN-off vs HamiltonianEncoder-only vs decoder-only on the same H",
            "Methyltin MatGen-Q SI XYZ (public SI missing; do not invent; labelled-ours DFT XYZ exists)",
            "Trackio project for this campaign",
        ],
        "recommended_protocol": {
            "train_seen": "GIC light set + SnO 14-char pool (exact-length vocab). Optional ours-opt methyltin 0 after documented DFT.",
            "holdout": "Aromatic Sn (Ph4Sn / Ph3SnOH) only after XYZ exists; n-Bu vs Me only if both have ours or SI XYZ.",
            "aire_sizes": "Statevector ≤24q; QSCI/MPS 28–32q. No 92 eV.",
            "metrics": [
                "ΔE vs CASCI (mHa) same H",
                "ΔE vs HF (detect HF-fallback)",
                "ΔE vs per-instance UCCSD-VQE / ADAPT-VQE / GPT-2 GQE",
                "N_dets with union cap 32 at 14q; never score 49/49 as a win",
                "word length == n_qubits",
                "Tanimoto + atom-type histogram vs nearest train graph",
                "split label: seen / held-out / in-distribution-SFT",
            ],
            "gnn_ablation_columns": [
                "GNN + HamiltonianEncoder + decoder",
                "HamiltonianEncoder + decoder (today)",
                "decoder-only",
                "HF",
            ],
        },
        "sources_hub": {
            "model": "https://huggingface.co/Ryukijano/h-cgqe-gic2026",
            "hub_weight_file": "h_cgqe_rl_gic2026.pt (same 149-vocab family; no GNN tensors)",
            "dataset": "https://huggingface.co/datasets/Ryukijano/sno14q-gqe-qsci-s4",
        },
        "sft_vs_rl_ablation": {
            "path": "results/phase3_final/ablation_sft_vs_rl.json",
            "note": "Seen organics; RL does not rescue N2/BeH2/phenol HF-like errors. No Sn.",
            "comparisons": abl["comparisons"],
        },
    }
    (OUT / "hcgqe_generalization_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    return inventory


def plot_sno_de(inv: dict) -> None:
    """Paper figure: dialect failure vs 14-char policy vs pool/VQE controls."""
    s4 = inv["sno14q"]["s4_numpy"]
    gpu = inv["sno14q"]["s4_cudaq_7431345"]
    p14 = inv["sno14q"]["pool14_infer"]
    vqe = inv["sno14q"]["per_instance_vqe"]

    labels = [
        "HF",
        "H-cGQE 12-char\n(mapping-fixed)",
        "H-cGQE 14-char SFT\nbest single\n(job 7434600)",
        "H-cGQE 14-char SFT\nunion cap 32 dets",
        "MatGen-Q smoke\n(CUDA-Q)",
        "Pool union greedy\n(NumPy, 21 dets)",
        "UCCSD-VQE\n(per-instance)",
    ]
    values = [
        s4["hf_mha"],
        s4["frozen_12char_mha"],
        p14["best_single_mha"],
        p14["union_capped_mha"],
        gpu["matgenq_smoke_best_mha"],
        s4["union_greedy_mha"],
        vqe["uccsd_vqe_mha"],
    ]
    colors = [GRAY, VERM, ORANGE, ORANGE, SKY, GREEN, BLUE]
    hatches = ["", "", "", "..", "", "", ""]

    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor=BLACK,
        linewidth=0.8,
        hatch=hatches,
        width=0.72,
        zorder=3,
    )
    ax.axhline(1.6, color=BLACK, linestyle="--", linewidth=1.3, zorder=4)
    ax.axhline(6.120, color=GRAY, linestyle=":", linewidth=1.2, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(r"$\Delta E$ vs CASCI (mHa)")
    ax.set_title(
        "SnO (2e,7o) 14q QSCI error vs CASCI (−288.10912104 Ha)\n"
        "14-char in-distribution SFT — not graph-conditioned zero-shot tin"
    )
    ax.set_ylim(0, 7.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{val:.3f}" if val >= 0.01 else f"{val:.1e}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    handles = [
        Line2D([0], [0], color=BLACK, linestyle="--", lw=1.3, label="Chemical accuracy 1.6 mHa"),
        Line2D([0], [0], color=GRAY, linestyle=":", lw=1.2, label="HF / AS correlation 6.120 mHa"),
        Patch(facecolor=VERM, edgecolor=BLACK, label="Frozen H-cGQE (12-char dialect)"),
        Patch(facecolor=ORANGE, edgecolor=BLACK, label="14-char H-cGQE SFT (job 7434600)"),
        Patch(facecolor=GREEN, edgecolor=BLACK, label="Pool-native union (not a trained policy)"),
        Patch(facecolor=BLUE, edgecolor=BLACK, label="Per-instance VQE (same H)"),
    ]
    _bold_legend(ax, handles=handles, loc="upper right", fontsize=8.5)
    ax.text(
        0.0,
        -0.30,
        "CASCI −288.10912104 Ha; N_FCI=49. Best 14-char circuit: 21 dets, not 49/49. "
        "Pool union +0.953 mHa / 21 dets shows the native pool contains the correlation. "
        "72.6→21.2 kcal/mol UCCSD(T) is omitted (not GQE). GNN unused at infer.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=GRAY,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig1_sno14q_de_vs_casci.png")
    fig.savefig(OUT / "fig1_sno14q_de_vs_casci.pdf")
    plt.close(fig)


def plot_seen_eval(inv: dict) -> None:
    rows = inv["seen_eval_gic2026"]
    names = [r["molecule"] for r in rows]
    gqe = np.array([r["hcgqe_de_mha"] for r in rows])
    hf = np.array([r["hf_de_mha"] for r in rows])

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, hf, width=w, color=GRAY, edgecolor=BLACK, linewidth=0.7, label="HF gap vs ref.", zorder=3)
    ax.bar(x + w / 2, gqe, width=w, color=SKY, edgecolor=BLACK, linewidth=0.7, label="H-cGQE (100 samples)", zorder=3)
    ax.axhline(1.6, color=BLACK, linestyle="--", linewidth=1.2, zorder=4, label="Chemical accuracy 1.6 mHa")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"$\Delta E$ vs file reference (mHa)")
    ax.set_title(
        "GIC generator on seen / in-family molecules\n"
        "Not held-out tin; most points sit at or above Hartree–Fock"
    )
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(gqe.max(), hf.max()) * 1.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    _bold_legend(ax, loc="upper left", fontsize=9)
    ax.text(
        0.0,
        -0.32,
        "Source: results/eval/h_cgqe_evaluation_gic2026.json. energy_error is mHa vs the file's "
        "reference_energy; HF gap = 1000×baseline_error_vs_reference. Log y-scale so CH3I/LiH "
        "are visible beside N2 stretch. CH3I 0.63 mHa (L-BFGS) is a separate seen 8q toy, not "
        "plotted here. IMePh row uses the pre-21-Aug isomer.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=GRAY,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig2_seen_molecule_de_vs_hf.png")
    fig.savefig(OUT / "fig2_seen_molecule_de_vs_hf.pdf")
    plt.close(fig)


def plot_sft14_loss(inv: dict) -> None:
    sft = load_json("results/tin_ab/h_cgqe_sft_pool14_7431349_metrics.json")
    epochs = np.arange(1, len(sft["train_losses"]) + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(epochs, sft["train_losses"], color=BLUE, lw=2.0, label="Train CE", zorder=3)
    ax.plot(epochs, sft["val_losses"], color=VERM, lw=2.0, label="Val CE", zorder=3)
    best_ep = int(np.argmin(sft["val_losses"])) + 1
    ax.axvline(best_ep, color=GREEN, linestyle="--", lw=1.2, label=f"Best val (epoch {best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(
        "14-char SnO pool SFT (job 7431349)\n"
        "In-distribution teacher sequences — not zero-shot generalization"
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbehind = True
    ax.set_xlim(1, len(epochs))
    _bold_legend(ax, loc="upper right", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(epochs, np.array(sft["val_accs"]) * 100.0, color=ORANGE, lw=1.6, linestyle=":", label="Val acc")
    ax2.set_ylabel("Validation token accuracy (%)", fontweight="bold")
    ax2.set_ylim(0, 20)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    ax.text(
        0.0,
        -0.18,
        "Source: results/tin_ab/h_cgqe_sft_pool14_7431349_metrics.json. "
        f"Early-stopped at {sft['epochs_run']}/80 epochs; final val acc "
        f"{100*sft['final_val_acc']:.1f}%. Vocab: 49 length-14 words. GNN unused.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=GRAY,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sft_pool14_loss.png")
    fig.savefig(OUT / "fig3_sft_pool14_loss.pdf")
    plt.close(fig)


def plot_vocab_and_diversity(inv: dict) -> None:
    vocab = inv["training"]["sft_uccsd_vocab_word_lengths"]
    lengths = sorted(int(k) for k in vocab)
    counts = [vocab[str(L)] if str(L) in vocab else vocab[L] for L in lengths]
    # json keys are strings after dump; original file has string keys
    counts = [int(vocab[str(L)]) for L in lengths]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))

    ax = axes[0]
    colors = [VERM if L == 12 else (GREEN if L == 14 else SKY) for L in lengths]
    ax.bar(lengths, counts, color=colors, edgecolor=BLACK, width=1.6, zorder=3)
    ax.set_xlabel("Operator word length (characters)")
    ax.set_ylabel("Number of vocabulary tokens")
    ax.set_title("Public GIC SFT vocab (149 tokens)")
    ax.set_xticks(lengths)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=VERM, edgecolor=BLACK, label="12-char (majority; used on SnO 14q)"),
        Patch(facecolor=GREEN, edgecolor=BLACK, label="14-char (20 tokens; not the MatGen-Q pool)"),
        Patch(facecolor=SKY, edgecolor=BLACK, label="Other lengths"),
    ]
    _bold_legend(ax, handles=handles, loc="upper right", fontsize=8)
    ax.text(12, 59 + 1.5, "59", ha="center", fontweight="bold", fontsize=9)

    ax = axes[1]
    cats = [
        "Unique organic\nconnectivities\nin SFT",
        "Named SFT\ninstances\n(no Sn)",
        "SFT sequences\n(coeff-noise\naugment)",
        "qBraid DAPO\nnames\n(still no Sn)",
        "Sn topologies\nin SFT or DAPO",
        "Held-out\norganotin\nΔE rows",
    ]
    vals = [
        inv["training"]["sft_uccsd_dataset"]["n_unique_chemical_graphs"],
        inv["training"]["sft_uccsd_dataset"]["n_named_instances"],
        inv["training"]["sft_uccsd_dataset"]["n_sequences"],
        inv["training"]["dapo"]["qbraid_rl_n_molecules"],
        0,
        0,
    ]
    bar_colors = [BLUE, SKY, SKY, ORANGE, VERM, VERM]
    ax.bar(np.arange(len(cats)), vals, color=bar_colors, edgecolor=BLACK, zorder=3)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Diversity that exists vs tin holdout that does not")
    ax.set_ylim(0, 320)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(i, v + 6, str(int(v)), ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.text(
        0.0,
        -0.28,
        "Left: results/tin_ab/sft_vocab_word_lengths.json. Right: unique graphs = connectivity "
        "classes in results/train/gqe_dataset_summary.json. Zero tin rows are measured absences, "
        "not truncated axes. No ΔE-vs-n_graphs series was logged.",
        transform=ax.transAxes,
        fontsize=7.2,
        color=GRAY,
        va="top",
    )
    fig.suptitle("Operator dialect and training-set diversity (factual counts)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_vocab_and_diversity.png")
    fig.savefig(OUT / "fig4_vocab_and_diversity.pdf")
    plt.close(fig)


def write_table_csv(inv: dict) -> None:
    rows = [
        [
            "column",
            "split",
            "de_mha_vs_casci_or_ref",
            "n_dets",
            "word_length",
            "gnn_at_infer",
            "source",
        ],
        [
            "HF_SnO14q",
            "n/a",
            f"{inv['sno14q']['s4_numpy']['hf_mha']:.3f}",
            "1",
            "n/a",
            "n/a",
            "results/tin_ab/qsci_controls_cpu_numpy.json",
        ],
        [
            "frozen_hcgqe_12char_7386634",
            "unseen_Sn_wrong_dialect",
            f"{inv['sno14q']['frozen_infer']['qsci_wrong_endianness_mha']:.3f}",
            "13",
            "12",
            "false",
            "results/tin_ab/hcgqe_qsci_7386634.json",
        ],
        [
            "frozen_hcgqe_12char_mapping_fixed",
            "unseen_Sn_wrong_dialect",
            f"{inv['sno14q']['s4_numpy']['frozen_12char_mha']:.3f}",
            "1",
            "12",
            "false",
            "results/tin_ab/qsci_controls_cpu_numpy.json",
        ],
        [
            "hcgqe_14char_sft_best_7434600",
            "in_distribution_sno_sft",
            f"{inv['sno14q']['pool14_infer']['best_single_mha']:.3f}",
            str(inv["sno14q"]["pool14_infer"]["best_single_n_dets"]),
            "14",
            "false",
            "results/tin_ab/hcgqe_qsci_pool14_7434600.json",
        ],
        [
            "hcgqe_14char_sft_union32_7434600",
            "in_distribution_sno_sft",
            f"{inv['sno14q']['pool14_infer']['union_capped_mha']:.3f}",
            str(inv["sno14q"]["pool14_infer"]["union_capped_n_dets"]),
            "14",
            "false",
            "results/tin_ab/hcgqe_qsci_pool14_7434600.json",
        ],
        [
            "pool_union_greedy_numpy",
            "pool_control_not_policy",
            f"{inv['sno14q']['s4_numpy']['union_greedy_mha']:.3f}",
            "21",
            "14",
            "n/a",
            "results/tin_ab/qsci_controls_cpu_numpy.json",
        ],
        [
            "uccsd_vqe_per_instance",
            "instance_trained",
            f"{inv['sno14q']['per_instance_vqe']['uccsd_vqe_mha']:.3e}",
            "full_ansatz",
            "14",
            "n/a",
            "baselines/.../results_vqe.json",
        ],
        [
            "ch3i_8q_hcgqe_lbfgs",
            "seen_iodine_toy",
            f"{inv['ch3i_8q_lbfgs']['h_cgqe_rlqf_mha']:.3f}",
            "n/a",
            "4",
            "false",
            "results/phase3_final/benchmark_ch3i_consolidated.json",
        ],
        [
            "heldout_organotin",
            "held-out",
            "",
            "",
            "",
            "",
            "DOES_NOT_EXIST",
        ],
        [
            "deltaE_vs_n_unique_graphs",
            "series",
            "",
            "",
            "",
            "",
            "DOES_NOT_EXIST",
        ],
    ]
    text = "\n".join(",".join(r) for r in rows) + "\n"
    (OUT / "factual_metrics_table.csv").write_text(text)


def main() -> None:
    inv = build_inventory()
    plot_sno_de(inv)
    plot_seen_eval(inv)
    plot_sft14_loss(inv)
    plot_vocab_and_diversity(inv)
    write_table_csv(inv)
    print("Wrote", OUT)
    for p in sorted(OUT.iterdir()):
        print(" ", p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
