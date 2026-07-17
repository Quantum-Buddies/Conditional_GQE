"""Hierarchical Conditional Generative Quantum Eigensolver (H-cGQE) Transformer.

Architecture:
- HamiltonianEncoder: embeds a molecular Hamiltonian (list of Pauli terms + coeffs)
  into a fixed-size context vector via character-level Pauli embeddings and
  Transformer self-attention over terms.
- OperatorPoolDecoder: causal Transformer decoder that autoregressively predicts
  the next operator index from a unified vocabulary, cross-attending to the
  Hamiltonian encoding.

The model treats quantum circuit synthesis as a sequence-to-sequence translation
task: Hamiltonian -> operator sequence.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F


# Character vocabulary for Pauli strings
PAULI_CHAR_VOCAB = {"<PAD>": 0, "I": 1, "X": 2, "Y": 3, "Z": 4}
PAULI_CHAR_PAD_ID = 0
SPECIAL_TOKENS = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PauliStringEncoder(nn.Module):
    """Encode a Pauli string (e.g. 'IZIZ') into a fixed-size vector.

    Uses character-level embedding + 1D convolution + mean pooling.
    """

    def __init__(self, char_embed_dim: int = 32, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.char_embed = nn.Embedding(len(PAULI_CHAR_VOCAB), char_embed_dim, padding_idx=PAULI_CHAR_PAD_ID)
        self.conv = nn.Sequential(
            nn.Conv1d(char_embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, pauli_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # pauli_ids: (batch, max_pauli_len)
        x = self.char_embed(pauli_ids)  # (batch, max_pauli_len, char_embed_dim)
        x = x.permute(0, 2, 1)  # (batch, char_embed_dim, max_pauli_len)
        x = self.conv(x)  # (batch, hidden_dim, max_pauli_len)
        x = x.permute(0, 2, 1)  # (batch, max_pauli_len, hidden_dim)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            x = x.sum(dim=1) / lengths  # mean pooling
        else:
            x = x.mean(dim=1)
        return self.norm(x)


class HamiltonianEncoder(nn.Module):
    """Encode a full Hamiltonian as a sequence of (Pauli term + coefficient) pairs.

    Input:  pauli_ids   (batch, n_terms, max_pauli_len)
            coeffs      (batch, n_terms)
            term_mask   (batch, n_terms)   # 1 for real terms, 0 for padding
    Output: context     (batch, n_terms, d_model)  # per-term representations
            memory      (batch, n_terms, d_model)  # same, for cross-attention
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        char_embed_dim: int = 32,
        max_pauli_len: int = 24,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_pauli_len = max_pauli_len
        self.pauli_encoder = PauliStringEncoder(
            char_embed_dim=char_embed_dim, hidden_dim=d_model, dropout=dropout
        )
        self.coeff_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=512, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        pauli_ids: torch.Tensor,
        coeffs: torch.Tensor,
        term_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pauli_ids: (batch, n_terms, max_pauli_len)
        # coeffs:    (batch, n_terms)
        batch_size, n_terms, max_len = pauli_ids.shape

        # Flatten to encode all Pauli strings
        flat_ids = pauli_ids.reshape(batch_size * n_terms, max_len)
        flat_mask = (flat_ids != PAULI_CHAR_PAD_ID).float() if max_len > 0 else None
        pauli_embeds = self.pauli_encoder(flat_ids, flat_mask)  # (batch*n_terms, d_model)
        pauli_embeds = pauli_embeds.reshape(batch_size, n_terms, self.d_model)

        # Coefficient projection
        coeff_embeds = self.coeff_proj(coeffs.unsqueeze(-1))  # (batch, n_terms, d_model)

        # Combine: sum of Pauli embedding + coefficient embedding
        x = pauli_embeds + coeff_embeds  # (batch, n_terms, d_model)
        x = self.pos_enc(x)

        # Transformer over terms
        # term_mask for padding: True = padding, False = valid (nn.Transformer convention)
        src_key_padding_mask = None
        if term_mask is not None:
            src_key_padding_mask = ~term_mask.bool()

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return x, x  # (context, memory for cross-attn)


class OperatorPoolDecoder(nn.Module):
    """Causal Transformer decoder that autoregressively predicts operator tokens."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def _make_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        # (size, size) with True in upper triangle (including diagonal for causal)
        # nn.TransformerDecoder expects True = mask out
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(
        self,
        tgt_indices: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # tgt_indices: (batch, tgt_len)
        x = self.token_embed(tgt_indices) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        causal_mask = self._make_causal_mask(tgt_indices.size(1), tgt_indices.device)
        x = self.transformer(
            x,
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=~memory_mask.bool() if memory_mask is not None else None,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.head(x)  # (batch, tgt_len, vocab_size)

    def generate(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        bos_id: int = 1,
        eos_id: int = 2,
        max_len: int = 32,
        temperature: float = 1.0,
        z_only_token_mask: torch.Tensor | None = None,
        force_entanglement: bool = False,
        max_repeat: int = 4,
        sample: bool = False,
        length_mask: torch.Tensor | None = None,
        freq_penalty: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressive generation.

        Args:
            z_only_token_mask: Boolean tensor of shape (vocab_size,) where True
                marks tokens that correspond to Z-only operators (no X/Y).
            force_entanglement: If True, the model is forbidden from generating a
                sequence composed solely of Z-only operators. Once an entangler
                (X/Y) token is emitted, the constraint is relaxed.
            max_repeat: Stop generation if the same token repeats this many times.
            sample: If True, sample from the temperature-scaled logits; otherwise
                use greedy decoding.
            length_mask: Boolean tensor of shape (vocab_size,) where True marks
                tokens that are compatible with the target molecule's qubit count.
            freq_penalty: Subtract freq_penalty * count[token] from logits at each
                step to discourage repeated operators (prevents mode collapse).
        """
        batch_size = memory.size(0)
        device = memory.device
        tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        has_entangler = torch.zeros(batch_size, dtype=torch.bool, device=device)
        repeat_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        last_token = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = self.forward(tokens, memory, memory_mask)[:, -1, :]  # (batch, vocab_size)

            # Frequency penalty: penalize tokens already in the sequence
            if freq_penalty > 0.0:
                token_counts = torch.zeros(batch_size, logits.size(-1), device=device)
                for t in range(1, tokens.size(1)):
                    token_counts.scatter_(1, tokens[:, t:t+1], 1.0, reduce='add')
                logits = logits - freq_penalty * token_counts

            if temperature != 1.0:
                logits = logits / temperature

            if force_entanglement and z_only_token_mask is not None:
                # Only apply the constraint if we haven't generated an entangler yet.
                # The BOS token is ignored (token index 1, not in z_only mask by default).
                constrain = ~has_entangler
                if constrain.any():
                    logits[constrain] = logits[constrain].masked_fill(z_only_token_mask, float("-inf"))

            if length_mask is not None:
                logits[:, ~length_mask] = float("-inf")

            if sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            tokens = torch.cat([tokens, next_token], dim=1)
            next_token_flat = next_token.squeeze(-1)
            same = next_token_flat == last_token
            repeat_count = torch.where(same, repeat_count + 1, torch.zeros_like(repeat_count))
            last_token = next_token_flat
            finished |= next_token_flat == eos_id
            finished |= repeat_count >= max_repeat
            if force_entanglement and z_only_token_mask is not None:
                has_entangler |= ~z_only_token_mask[next_token_flat]
            if finished.all():
                break
        return tokens


class HcGQEModel(nn.Module):
    """Full H-cGQE: Hamiltonian -> operator sequence."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_pauli_len: int = 24,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = HamiltonianEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_pauli_len=max_pauli_len,
        )
        self.decoder = OperatorPoolDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        pauli_ids: torch.Tensor,
        coeffs: torch.Tensor,
        tgt_indices: torch.Tensor,
        term_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, memory = self.encoder(pauli_ids, coeffs, term_mask)
        memory_mask = term_mask
        logits = self.decoder(tgt_indices, memory, memory_mask, tgt_key_padding_mask)
        return logits

    @torch.no_grad()
    def generate(
        self,
        pauli_ids: torch.Tensor,
        coeffs: torch.Tensor,
        term_mask: torch.Tensor | None = None,
        bos_id: int = 1,
        eos_id: int = 2,
        max_len: int = 32,
        temperature: float = 1.0,
        vocab: dict[str, int] | None = None,
        force_entanglement: bool = False,
        max_repeat: int = 4,
        sample: bool = False,
        n_qubits: int | None = None,
        freq_penalty: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        _, memory = self.encoder(pauli_ids, coeffs, term_mask)

        z_only_token_mask = None
        length_mask = None
        if vocab is not None:
            if force_entanglement:
                z_only_token_mask = build_z_only_token_mask(vocab, device=memory.device)
            if n_qubits is not None:
                length_mask = build_length_token_mask(vocab, n_qubits, device=memory.device)

        return self.decoder.generate(
            memory,
            term_mask,
            bos_id,
            eos_id,
            max_len,
            temperature,
            z_only_token_mask=z_only_token_mask,
            force_entanglement=force_entanglement,
            max_repeat=max_repeat,
            sample=sample,
            length_mask=length_mask,
            freq_penalty=freq_penalty,
        )


def build_z_only_token_mask(
    vocab: dict[str, int], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Return a boolean mask of shape (vocab_size,) where True marks Z-only tokens.

    A Pauli word is Z-only if it contains only I and Z (no X or Y). Special
    tokens (<PAD>, <BOS>, <EOS>, <UNK>) are marked as Z-only so they are not
    used to satisfy the entanglement constraint; the constraint is only
    relaxed when an operator token with X/Y is generated.
    """
    vocab_size = max(vocab.values()) + 1
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for word, idx in vocab.items():
        if word in SPECIAL_TOKENS:
            mask[idx] = True
        else:
            # Z-only = no X or Y anywhere in the word
            mask[idx] = ("X" not in word) and ("Y" not in word)
    return mask


def build_length_token_mask(
    vocab: dict[str, int],
    n_qubits: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a boolean mask of shape (vocab_size,) where True marks tokens
    whose Pauli word length is compatible with the molecule's qubit count.

    Special tokens are always allowed. Operator tokens are allowed only if
    their word length is <= n_qubits (shorter words are padded during circuit
    construction).
    """
    vocab_size = max(vocab.values()) + 1
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for word, idx in vocab.items():
        if word in SPECIAL_TOKENS:
            mask[idx] = True
        else:
            mask[idx] = len(word) <= n_qubits
    return mask


def build_operator_vocab(pauli_words: Sequence[str]) -> dict[str, int]:
    """Build a unified vocabulary mapping Pauli words -> token IDs.

    Special tokens are reserved: <PAD>=0, <BOS>=1, <EOS>=2, <UNK>=3.
    Pauli words start from ID 4.
    """
    vocab: dict[str, int] = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    idx = 4
    for word in sorted(set(pauli_words)):
        if word not in vocab:
            vocab[word] = idx
            idx += 1
    return vocab


def tokenize_hamiltonian(
    terms: list[tuple[str, float]],
    vocab: dict[str, int],
    max_terms: int = 128,
    max_pauli_len: int = 24,
) -> dict[str, torch.Tensor]:
    """Convert a list of (pauli_word, coefficient) into model inputs.

    Returns a dict with keys: pauli_ids, coeffs, term_mask.
    """
    # Truncate/pad terms
    terms = terms[:max_terms]
    n = len(terms)
    pad_n = max_terms - n

    pauli_ids = torch.zeros(max_terms, max_pauli_len, dtype=torch.long)
    coeffs = torch.zeros(max_terms, dtype=torch.float)
    term_mask = torch.zeros(max_terms, dtype=torch.float)

    for i, (word, coeff) in enumerate(terms):
        term_mask[i] = 1.0
        coeffs[i] = coeff
        for j, ch in enumerate(word[:max_pauli_len]):
            pauli_ids[i, j] = PAULI_CHAR_VOCAB.get(ch, PAULI_CHAR_PAD_ID)

    return {
        "pauli_ids": pauli_ids,
        "coeffs": coeffs,
        "term_mask": term_mask,
    }


def tokenize_operator_sequence(
    operator_words: list[str],
    vocab: dict[str, int],
    max_len: int = 64,
) -> torch.Tensor:
    """Convert a list of operator Pauli words into a token ID tensor with BOS/EOS."""
    bos = vocab["<BOS>"]
    eos = vocab["<EOS>"]
    unk = vocab["<UNK>"]
    tokens = [bos]
    for w in operator_words[: max_len - 2]:
        tokens.append(vocab.get(w, unk))
    tokens.append(eos)
    # Pad
    if len(tokens) < max_len:
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len], dtype=torch.long)
