#!/usr/bin/env python3
"""Download model checkpoints and datasets from Hugging Face Hub.

All checkpoints are hosted at: https://huggingface.co/Quantum-Buddies/Conditional-GQE-models

Usage:
    python scripts/download_models.py                  # download all models
    python scripts/download_models.py --check          # check which are missing
    python scripts/download_models.py --only essential # download only key checkpoints
    python scripts/download_models.py --only rl        # download only RL checkpoints
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "results" / "train"
DATA_DIR = ROOT / "results" / "data"

HF_REPO = "Quantum-Buddies/Conditional-GQE-models"

# All model checkpoints and datasets hosted on Hugging Face
# (local_path, hf_filename, description, category)
MODELS = [
    # === Essential models (needed to reproduce main results) ===
    ("results/train/h_cgqe_model_b200_sft.pt", "h_cgqe_model_b200_sft.pt",
     "SFT warm-start checkpoint (B200 trained)", "essential"),
    ("results/train/h_cgqe_model_rl_qd_scratch.pt", "h_cgqe_model_rl_qd_scratch.pt",
     "RL trained model with MAP-Elites (main checkpoint)", "essential"),
    ("results/train/h_cgqe_model.pt", "h_cgqe_model.pt",
     "Base H-cGQE model (6.1M params)", "essential"),
    ("results/train/chemistry_encoder.pt", "chemistry_encoder.pt",
     "Chemistry GNN encoder for cross-molecule conditioning", "essential"),
    ("results/train/gqe_supervised_dataset.pt", "gqe_supervised_dataset.pt",
     "Supervised training dataset (15M)", "essential"),

    # === RL checkpoints ===
    ("results/train/h_cgqe_rl_dapo_model.pt", "h_cgqe_rl_dapo_model.pt",
     "DAPO RL trained model", "rl"),
    ("results/train/h_cgqe_model_phase3.pt", "h_cgqe_model_phase3.pt",
     "Phase 3 production model", "rl"),
    ("results/train/h_cgqe_model_rlqf_phase3.pt", "h_cgqe_model_rlqf_phase3.pt",
     "RLQF Phase 3 model", "rl"),
    ("results/train/h_cgqe_rl_warmstart.pt", "h_cgqe_rl_warmstart.pt",
     "RL warm-start checkpoint", "rl"),

    # === Ablation checkpoints ===
    ("results/train/h_cgqe_rl_ablation_full.pt", "h_cgqe_rl_ablation_full.pt",
     "Ablation: full DAPO (KL + creativity + MMD)", "ablation"),
    ("results/train/h_cgqe_rl_ablation_vanilla_dapo.pt", "h_cgqe_rl_ablation_vanilla_dapo.pt",
     "Ablation: vanilla DAPO (no KL/creativity/MMD)", "ablation"),
    ("results/train/h_cgqe_rl_ablation_no_kl.pt", "h_cgqe_rl_ablation_no_kl.pt",
     "Ablation: no KL divergence penalty", "ablation"),
    ("results/train/h_cgqe_rl_ablation_kl_only.pt", "h_cgqe_rl_ablation_kl_only.pt",
     "Ablation: KL only", "ablation"),
    ("results/train/h_cgqe_rl_ablation_no_creativity.pt", "h_cgqe_rl_ablation_no_creativity.pt",
     "Ablation: no creativity reward", "ablation"),
    ("results/train/h_cgqe_rl_ablation_no_mmd.pt", "h_cgqe_rl_ablation_no_mmd.pt",
     "Ablation: no MMD penalty", "ablation"),
    ("results/train/h_cgqe_rl_beh2_boosted.pt", "h_cgqe_rl_beh2_boosted.pt",
     "Ablation: BeH2 boosted RL", "ablation"),

    # === Conditioning encoders ===
    ("results/train/graph_conditioning.pt", "graph_conditioning.pt",
     "Graph conditioning model (114K)", "conditioning"),
    ("results/train/flat_conditioning.pt", "flat_conditioning.pt",
     "Flat conditioning model (17K)", "conditioning"),
    ("results/train/ddp_graph_conditioning.pt", "ddp_graph_conditioning.pt",
     "DDP graph conditioning model (1.9M)", "conditioning"),

    # === Other models ===
    ("results/train/h_cgqe_model_augmented.pt", "h_cgqe_model_augmented.pt",
     "Augmented training model (6.1M)", "other"),
    ("results/train/h_cgqe_small.pt", "h_cgqe_small.pt",
     "Small model for quick tests (1.3M)", "other"),
    ("results/train/h_cgqe_uccsd_model.pt", "h_cgqe_uccsd_model.pt",
     "UCCSD-trained model", "other"),
    ("results/train/h_cgqe_rl_chemeleon2_1gpu.pt", "h_cgqe_rl_chemeleon2_1gpu.pt",
     "Chemeleon2 RL model (1 GPU)", "other"),
    ("results/train/uccsd_dataset/gqe_supervised_dataset.pt", "uccsd_dataset/gqe_supervised_dataset.pt",
     "UCCSD supervised dataset (6.2M)", "other"),
]


def check_models() -> list[tuple[str, bool]]:
    """Check which models exist locally. Returns list of (local_path, exists)."""
    results = []
    for local_path, _, _, _ in MODELS:
        p = ROOT / local_path
        results.append((local_path, p.exists()))
    return results


def download_models(category: str | None = None, dry_run: bool = False) -> None:
    """Download models from Hugging Face Hub.

    Args:
        category: If specified, only download models in that category.
        dry_run: If True, only print what would be downloaded.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    to_download = [
        (lp, hf, desc, cat) for lp, hf, desc, cat in MODELS
        if category is None or cat == category
    ]

    missing = [(lp, hf, desc) for lp, hf, desc, _ in to_download if not (ROOT / lp).exists()]
    if not missing:
        print("All requested models already exist locally.")
        return

    print(f"Downloading {len(missing)} model(s) from {HF_REPO}...")
    for local_path, hf_filename, desc in missing:
        target = ROOT / local_path
        target.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            print(f"  [DRY RUN] {hf_filename} -> {local_path} ({desc})")
            continue

        print(f"  Downloading {hf_filename} ({desc})...")
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename=hf_filename,
                local_dir=str(ROOT),
            )
            # hf_hub_download saves to local_dir/filename; ensure it's at the right place
            dl_path = Path(downloaded)
            if dl_path != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                dl_path.rename(target)
            print(f"    -> {local_path} ({target.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"    FAILED: {e}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Download model checkpoints from Hugging Face.")
    parser.add_argument("--check", action="store_true",
                        help="Check which models exist locally without downloading.")
    parser.add_argument("--only", type=str, default=None,
                        choices=["essential", "rl", "ablation", "conditioning", "other"],
                        help="Only download models in this category.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be downloaded without actually downloading.")
    args = parser.parse_args()

    if args.check:
        results = check_models()
        print(f"\nModel checkpoint status ({len(results)} total):\n")
        for local_path, exists in results:
            status = "OK" if exists else "MISSING"
            print(f"  [{status:7s}] {local_path}")
        missing = sum(1 for _, e in results if not e)
        print(f"\n{missing} missing, {len(results) - missing} present.")
        if missing > 0:
            print(f"\nTo download missing models, run:")
            print(f"  python scripts/download_models.py")
        return

    download_models(category=args.only, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
