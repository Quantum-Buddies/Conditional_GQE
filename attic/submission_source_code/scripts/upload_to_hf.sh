#!/bin/bash
# Upload model checkpoints to Hugging Face Hub
# 
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login  # needs a HF token with write access
#
# This creates the repo if it doesn't exist and uploads all .pt files.

set -e

HF_REPO="Quantum-Buddies/Conditional-GQE-models"
TRAIN_DIR="results/train"

echo "=== Uploading model checkpoints to Hugging Face ==="
echo "Repo: ${HF_REPO}"
echo ""

# Create repo if it doesn't exist
huggingface-cli repo create "${HF_REPO}" --exist-ok 2>/dev/null || true

# Upload all .pt files
for f in "${TRAIN_DIR}"/*.pt; do
    if [ -f "$f" ]; then
        echo "Uploading: $(basename $f) ($(du -h "$f" | cut -f1))"
        huggingface-cli upload "${HF_REPO}" "$f" "$(basename $f)"
    fi
done

# Upload UCCSD dataset
if [ -f "${TRAIN_DIR}/uccsd_dataset/gqe_supervised_dataset.pt" ]; then
    echo "Uploading: uccsd_dataset/gqe_supervised_dataset.pt"
    huggingface-cli upload "${HF_REPO}" \
        "${TRAIN_DIR}/uccsd_dataset/gqe_supervised_dataset.pt" \
        "uccsd_dataset/gqe_supervised_dataset.pt"
fi

echo ""
echo "=== Upload complete ==="
echo "Models available at: https://huggingface.co/${HF_REPO}"
