"""ScoreEncoder — Conformer encoder with 3-way input.

Input: phoneme IDs + MIDI pitch + duration (frames)
Output: phone-level hidden states [B, T_phone, 512]

3-way concat: phone_emb(512) + pitch_emb(512) + dur_emb(512) → proj(1536→512) → Conformer 4L
"""

import torch
import torch.nn as nn

from .modules import (
    ConformerBlock,
    SinusoidalPosEmb,
    PitchEmbedding,
)


class DurationEmbedding(nn.Module):
    """Log-duration → learned embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, durations: torch.Tensor) -> torch.Tensor:
        """durations: [B, T] int → [B, T, dim]"""
        # Log-scale duration (add 1 to avoid log(0))
        log_dur = torch.log1p(durations.float()).unsqueeze(-1)  # [B, T, 1]
        return self.net(log_dur)


class ScoreEncoder(nn.Module):
    """Conformer encoder with 3-way phoneme+pitch+duration input.

    Architecture:
        phone_emb(n_phones, 512) + pitch_emb(512) + dur_emb(512)
        → concat [B, T, 1536]
        → input_proj(1536, 512)
        → pos_enc
        → 4x ConformerBlock(512)
        → phone_hidden [B, T_phone, 512]

    ~35M params.
    """

    def __init__(
        self,
        n_phones: int = 80,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        kernel_size: int = 31,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 3-way input embeddings
        self.phone_emb = nn.Embedding(n_phones, hidden_dim, padding_idx=0)
        self.pitch_emb = PitchEmbedding(hidden_dim)
        self.dur_emb = DurationEmbedding(hidden_dim)

        # Projection: 3*hidden → hidden
        self.input_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.pos_enc = SinusoidalPosEmb(hidden_dim)

        # Conformer layers
        self.layers = nn.ModuleList([
            ConformerBlock(hidden_dim, n_heads, kernel_size, ff_expansion, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        phonemes: torch.Tensor,    # [B, T_phone] int
        pitch: torch.Tensor,       # [B, T_phone] int (MIDI)
        durations: torch.Tensor,   # [B, T_phone] int (frames)
        mask: torch.Tensor = None, # [B, T_phone] bool (True=pad)
    ) -> torch.Tensor:
        """Returns phone_hidden: [B, T_phone, hidden_dim]"""
        ph = self.phone_emb(phonemes)   # [B, T, D]
        pi = self.pitch_emb(pitch)      # [B, T, D]
        du = self.dur_emb(durations)    # [B, T, D]

        x = torch.cat([ph, pi, du], dim=-1)  # [B, T, 3D]
        x = self.input_proj(x)               # [B, T, D]
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x
