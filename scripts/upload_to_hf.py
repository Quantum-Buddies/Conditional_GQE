#!/usr/bin/env python
"""Upload H-cGQE model to Hugging Face Hub.

Usage:
    HF_TOKEN=hf_xxxxx python scripts/upload_to_hf.py
"""
import os
import sys
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "Ryukijano/h-cgqe-gic2026"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_ROOT / "results" / "train" / "h_cgqe_rl_gic2026.pt"
UPLOAD_DIR = PROJECT_ROOT / "huggingface_upload"

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable.")
        print("  export HF_TOKEN=hf_xxxxx  (create at https://huggingface.co/settings/tokens)")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo
    url = create_repo(REPO_ID, token=token, exist_ok=True)
    print(f"Repo ready: {url}")

    # Copy checkpoint into upload dir
    dest_ckpt = UPLOAD_DIR / "h_cgqe_rl_gic2026.pt"
    if not dest_ckpt.exists():
        print(f"Copying checkpoint ({CKPT_PATH.name})...")
        shutil.copy2(CKPT_PATH, dest_ckpt)

    # Upload everything
    print(f"Uploading {UPLOAD_DIR} to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(UPLOAD_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload H-cGQE Transformer (GIC 2026) with model card and config",
    )
    print(f"\nDone! Model page: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
