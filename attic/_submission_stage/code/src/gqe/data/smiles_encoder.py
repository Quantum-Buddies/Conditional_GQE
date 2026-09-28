"""SMILES encoder for cross-molecule transfer learning.

This module implements a SMILES tokenization and embedding system that enables
the H-cGQE Transformer to generalize across different molecules by conditioning
on molecular structure encoded as SMILES strings.

The encoder:
1. Tokenizes SMILES strings using a chemistry-aware tokenizer
2. Embeds tokens into a latent vector via a small transformer or GRU
3. Produces a molecular embedding that can condition the H-cGQE circuit generator

Transfer learning protocol:
- Pretrain on source molecules (small, abundant: H2, LiH, BeH2, N2)
- Fine-tune on target molecules (larger: ethylene, benzene, iodobenzene)
- The SMILES embedding provides structural priors that transfer

Usage:
    from src.gqe.data.smiles_encoder import SmilesEncoder, SmilesTokenizer
    encoder = SmilesEncoder(vocab_size=100, embed_dim=128, hidden_dim=256)
    embedding = encoder.encode("N#N")  # N2 SMILES
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# SMILES vocabulary: atoms, bonds, and special tokens
SMILES_VOCAB = [
    "<PAD>", "<UNK>", "< SOS >", "<EOS>",
    # Atoms
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B",
    "H", "Si", "Se", "As", "Li", "Be", "Mg", "Ca", "Na", "Al",
    # Bonds
    "=", "#", "-", ":", "/", "\\",
    # Rings and branches
    "(", ")", "[", "]", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Charges
    "+", "-", "@", "@@",
    # Special
    ".", "c", "n", "o", "s", "p",
]

# Build vocab lookup
VOCAB_TO_IDX = {tok: i for i, tok in enumerate(SMILES_VOCAB)}
IDX_TO_VOCAB = {i: tok for i, tok in enumerate(SMILES_VOCAB)}
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


class SmilesTokenizer:
    """Chemistry-aware SMILES tokenizer.

    Handles multi-character atoms (Cl, Br, Si, Se, As) and
    single-character tokens (C, N, O, =, #, etc.).
    """

    MULTI_CHAR_TOKENS = ["Cl", "Br", "Si", "Se", "As", "@@", "Li", "Be", "Mg", "Ca", "Na", "Al"]
    SINGLE_CHAR_TOKENS = set(SMILES_VOCAB) - set(MULTI_CHAR_TOKENS) - {"<PAD>", "<UNK>", "< SOS >", "<EOS>"}

    def __init__(self, max_length: int = 128) -> None:
        self.max_length = max_length
        self.vocab_to_idx = VOCAB_TO_IDX
        self.idx_to_vocab = IDX_TO_VOCAB

    def tokenize(self, smiles: str) -> list[str]:
        """Tokenize a SMILES string into a list of tokens."""
        tokens = []
        i = 0
        smiles = smiles.strip()

        while i < len(smiles):
            matched = False

            # Try multi-character tokens first
            for mc in self.MULTI_CHAR_TOKENS:
                if smiles[i:i + len(mc)] == mc:
                    tokens.append(mc)
                    i += len(mc)
                    matched = True
                    break

            if not matched:
                ch = smiles[i]
                if ch in self.SINGLE_CHAR_TOKENS:
                    tokens.append(ch)
                else:
                    tokens.append("<UNK>")
                i += 1

        return tokens

    def encode(self, smiles: str) -> tuple[np.ndarray, int]:
        """Encode SMILES string to token indices.

        Returns:
            (token_indices, actual_length) padded to max_length.
        """
        tokens = ["< SOS >"] + self.tokenize(smiles) + ["<EOS>"]
        indices = [self.vocab_to_idx.get(t, UNK_IDX) for t in tokens]

        # Truncate if too long
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
            indices[-1] = EOS_IDX

        length = len(indices)

        # Pad to max_length
        while len(indices) < self.max_length:
            indices.append(PAD_IDX)

        return np.array(indices, dtype=np.int64), length

    def decode(self, indices: np.ndarray) -> str:
        """Decode token indices back to SMILES string."""
        tokens = []
        for idx in indices:
            idx = int(idx)
            if idx == PAD_IDX:
                continue
            if idx == EOS_IDX:
                break
            if idx == SOS_IDX:
                continue
            tokens.append(self.idx_to_vocab.get(idx, "<UNK>"))
        return "".join(tokens)


class SmilesEncoder(nn.Module):
    """SMILES molecular encoder using a small transformer.

    Encodes a SMILES string into a fixed-dimensional embedding vector
    that can be used to condition the H-cGQE circuit generator.

    Architecture:
    - Token embedding layer
    - Positional encoding
    - 2-4 transformer encoder layers
    - Mean pooling over non-pad tokens
    - Linear projection to output dimension
    """

    def __init__(
        self,
        vocab_size: int = len(SMILES_VOCAB),
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        output_dim: int = 256,
        max_length: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.max_length = max_length

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # Positional encoding (learned)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        self.tokenizer = SmilesTokenizer(max_length=max_length)

    def forward(self, token_indices: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass: token indices -> molecular embedding.

        Args:
            token_indices: (batch, max_length) token indices.
            lengths: (batch,) actual sequence lengths for masking.

        Returns:
            (batch, output_dim) molecular embeddings.
        """
        batch_size, seq_len = token_indices.shape

        # Create attention mask for padding
        if lengths is not None:
            mask = torch.arange(seq_len, device=token_indices.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = (token_indices != PAD_IDX)

        # Embeddings
        positions = torch.arange(seq_len, device=token_indices.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(token_indices) + self.position_embedding(positions)

        # Transformer
        # Convert mask to float attention mask (True = attend, False = mask)
        key_padding_mask = ~mask  # True = mask out (pad positions)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Mean pooling over non-pad tokens
        mask_f = mask.float().unsqueeze(-1)  # (batch, seq, 1)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        # Output projection
        embedding = self.output_proj(pooled)

        return embedding

    def encode(self, smiles: str | list[str], device: str = "cpu") -> torch.Tensor:
        """Encode SMILES string(s) to embedding vector(s).

        Args:
            smiles: Single SMILES string or list of strings.
            device: Device to run on.

        Returns:
            (output_dim,) or (batch, output_dim) embedding tensor.
        """
        was_single = isinstance(smiles, str)
        if was_single:
            smiles = [smiles]

        batch_indices = []
        batch_lengths = []
        for s in smiles:
            indices, length = self.tokenizer.encode(s)
            batch_indices.append(indices)
            batch_lengths.append(length)

        token_tensor = torch.from_numpy(np.stack(batch_indices)).to(device)
        length_tensor = torch.from_numpy(np.array(batch_lengths, dtype=np.int64)).to(device)

        self.eval()
        with torch.no_grad():
            embedding = self.forward(token_tensor, length_tensor)

        if was_single:
            return embedding[0]
        return embedding


# Known molecule SMILES strings in the dataset
MOLECULE_SMILES = {
    "h2": "[H][H]",
    "h2_0.5": "[H][H]",
    "h2_0.74": "[H][H]",
    "h2_1.0": "[H][H]",
    "h2_1.5": "[H][H]",
    "h2_2.0": "[H][H]",
    "lih": "[LiH]",
    "lih_1.2": "[LiH]",
    "lih_1.6": "[LiH]",
    "lih_2.0": "[LiH]",
    "lih_3.0": "[LiH]",
    "lih_1.6_full": "[LiH]",
    "lih_1.6_631g": "[LiH]",
    "beh2": "[BeH2]",
    "beh2_1.3": "[BeH2]",
    "beh2_2.0": "[BeH2]",
    "beh2_1.3_full": "[BeH2]",
    "n2": "N#N",
    "n2_1.1": "N#N",
    "n2_1.8": "N#N",
    "n2_2.5": "N#N",
    "n2_1.1_full": "N#N",
    "n2_1.1_631g_cas8": "N#N",
    "n2_ccpvdz": "N#N",
    "n2_ccpvdz_cas20": "N#N",
    "n2_ccpvdz_full": "N#N",
    "iodobenzene": "Ic1ccccc1",
    "iodobenzene_cas12": "Ic1ccccc1",
    "methyl_iodide": "CI",
    "methyl_iodide_cas12": "CI",
    "imeph": "ICc1ccccc1",
    "imeph_cas12": "ICc1ccccc1",
    "phenol": "Oc1ccccc1",
    "phenol_cas12": "Oc1ccccc1",
    "ethylene": "C=C",
    "formaldehyde": "C=O",
    "benzene_cas20": "c1ccccc1",
    "h2o_1.0_631g_cas8": "O",
    "ch3i": "CI",
}


def get_smiles_for_molecule(name: str) -> str:
    """Get SMILES string for a molecule name from the known mapping."""
    if name in MOLECULE_SMILES:
        return MOLECULE_SMILES[name]
    # Try fuzzy matching
    base = name.split("_")[0]
    if base in MOLECULE_SMILES:
        return MOLECULE_SMILES[base]
    raise KeyError(f"No SMILES known for molecule '{name}'. Add it to MOLECULE_SMILES.")


def build_transfer_learning_dataset(
    hamiltonian_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Build a transfer learning dataset mapping molecules to SMILES embeddings.

    Args:
        hamiltonian_path: Path to Hamiltonian JSON.
        out_path: Optional output path for the dataset.

    Returns:
        Dictionary with molecule names, SMILES, tokenized forms, and metadata.
    """
    with hamiltonian_path.open() as f:
        data = json.load(f)

    tokenizer = SmilesTokenizer()
    dataset = []

    for record in data.get("records", []):
        name = record.get("name", "")
        n_qubits = int(record.get("n_qubits", 0))
        n_terms = len(record.get("terms", []))

        try:
            smiles = get_smiles_for_molecule(name)
        except KeyError:
            continue

        tokens = tokenizer.tokenize(smiles)
        indices, length = tokenizer.encode(smiles)

        dataset.append({
            "molecule": name,
            "smiles": smiles,
            "n_qubits": n_qubits,
            "n_hamiltonian_terms": n_terms,
            "tokens": tokens,
            "token_indices": indices.tolist(),
            "token_length": length,
        })

    result = {
        "description": "Cross-molecule transfer learning dataset with SMILES encoding",
        "n_molecules": len(dataset),
        "vocab_size": len(SMILES_VOCAB),
        "max_length": tokenizer.max_length,
        "molecules": dataset,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"Transfer learning dataset saved to {out_path}")

    return result


if __name__ == "__main__":
    # Quick self-test
    print("=== SMILES Encoder Test ===")

    tokenizer = SmilesTokenizer()
    test_smiles = ["N#N", "[H][H]", "[LiH]", "C=C", "c1ccccc1", "CI"]
    for s in test_smiles:
        tokens = tokenizer.tokenize(s)
        indices, length = tokenizer.encode(s)
        decoded = tokenizer.decode(indices[:length])
        print(f"  {s:20s} -> tokens={tokens} -> decoded={decoded}")

    # Test encoder
    encoder = SmilesEncoder(embed_dim=64, output_dim=128)
    print(f"\nEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    for s in test_smiles:
        emb = encoder.encode(s)
        print(f"  {s:20s} -> embedding shape={emb.shape}, norm={emb.norm():.4f}")

    # Test batch encoding
    batch_emb = encoder.encode(test_smiles)
    print(f"\nBatch encoding: {len(test_smiles)} molecules -> shape={batch_emb.shape}")

    # Compute pairwise similarities (cosine)
    norms = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
    sim_matrix = norms @ norms.T
    print("\nCosine similarity matrix:")
    for i, s in enumerate(test_smiles):
        row = " ".join(f"{sim_matrix[i,j]:.2f}" for j in range(len(test_smiles)))
        print(f"  {s:20s} {row}")
