#!/usr/bin/env python3
"""Utility for ensuring model checkpoints exist locally.

If a checkpoint is missing, automatically downloads it from Hugging Face Hub.

Usage in scripts:
    from src.gqe.common.ensure_checkpoint import ensure_checkpoint

    ckpt_path = ensure_checkpoint("results/train/h_cgqe_model.pt")
    model = torch.load(ckpt_path)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]  # src/gqe/common -> project root

HF_REPO = "Quantum-Buddies/Conditional-GQE-models"


def ensure_checkpoint(local_path: str | Path, hf_filename: Optional[str] = None,
                      auto_download: bool = True) -> Path:
    """Ensure a checkpoint file exists locally, downloading from HF if needed.

    Args:
        local_path: Relative path from project root (e.g. "results/train/h_cgqe_model.pt").
        hf_filename: Filename on Hugging Face repo. If None, uses the basename of local_path.
        auto_download: If True and file is missing, download from HF. If False, raise FileNotFoundError.

    Returns:
        Absolute Path to the checkpoint file.

    Raises:
        FileNotFoundError: If auto_download is False and file doesn't exist.
        ImportError: If huggingface_hub is not installed and download is needed.
    """
    target = ROOT / local_path if not Path(local_path).is_absolute() else Path(local_path)

    if target.exists():
        return target

    if not auto_download:
        raise FileNotFoundError(
            f"Checkpoint not found: {target}\n"
            f"Run: python scripts/download_models.py"
        )

    # Auto-download from Hugging Face
    if hf_filename is None:
        hf_filename = target.name

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            f"Checkpoint {target} not found and huggingface_hub is not installed.\n"
            f"Install with: pip install huggingface_hub\n"
            f"Then run: python scripts/download_models.py"
        )

    print(f"[ensure_checkpoint] Downloading {hf_filename} from {HF_REPO}...")
    target.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=HF_REPO,
        filename=hf_filename,
        local_dir=str(ROOT),
    )
    dl_path = Path(downloaded)
    if dl_path != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        dl_path.rename(target)

    print(f"[ensure_checkpoint] Saved to {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def ensure_essential_models() -> None:
    """Download all essential model checkpoints if missing."""
    essential = [
        "results/train/h_cgqe_model_b200_sft.pt",
        "results/train/h_cgqe_model_rl_qd_scratch.pt",
        "results/train/h_cgqe_model.pt",
        "results/train/chemistry_encoder.pt",
        "results/train/gqe_supervised_dataset.pt",
    ]
    for path in essential:
        ensure_checkpoint(path)
