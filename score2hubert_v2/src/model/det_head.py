"""DetHubertHead — deterministic HuBERT feature regression head.

4 residual conv blocks with GroupNorm + GELU.
Receptive field = 4 * (7-1) + 1 = 25 frames = 500ms (syllable-level context).
Encoder provides global context via attention.

Input: [frame_hidden; f0_emb] → [B, T_frame, 1024]
Output: hubert_pred [B, T_frame, 768]

~8M params.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d + GroupNorm + GELU + Dropout, with residual connection."""

    def __init__(self, dim: int, kernel_size: int = 7, groups: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D, T] -> [B, D, T]"""
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class DetHubertHead(nn.Module):
    """Deterministic HuBERT regression head.

    Architecture:
        in_proj: Linear(2*hidden, hidden)   # [frame_hidden; f0_emb]
        N ConvBlock(hidden, k=7, gn=8)      # residual convolutions
        out_proj: Linear(hidden, 768)

    Run 8: phoneme identity skip — the last conv block's input gets an additive
    residual from the head's input (scaled by learnable `skip_scale`, init 0.5).
    Rationale: N=6 conv blocks of kernel=7 give receptive field 37 frames = 740ms.
    Inside sustained notes >1s, conv layers can "drift" away from the phoneme identity
    that's strongly present in frame_hidden (encoder output + pos features). The skip
    guarantees the last block still has direct access to the input representation.

    ~12M params with hidden=512, n_blocks=6.
    """

    def __init__(self, hidden_dim: int = 512, hubert_dim: int = 768,
                 n_blocks: int = 4, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList([
            ConvBlock(hidden_dim, kernel_size, groups=8, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(hidden_dim, hubert_dim)
        # Learnable skip from head input (post in_proj) to last block's input.
        # Init 0.5: noticeable but not dominant.
        self.skip_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, frame_hidden: torch.Tensor, f0_emb: torch.Tensor) -> torch.Tensor:
        """
        frame_hidden: [B, T_frame, hidden_dim]
        f0_emb:       [B, T_frame, hidden_dim]
        -> hubert_pred: [B, T_frame, 768]
        """
        x = torch.cat([frame_hidden, f0_emb], dim=-1)  # [B, T, 2*D]
        x = self.in_proj(x)                              # [B, T, D]
        skip = x  # keep for last-block identity skip

        # Conv blocks operate on [B, D, T]
        x = x.transpose(1, 2)
        skip_bdt = skip.transpose(1, 2)
        last_idx = len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            if i == last_idx:
                x = x + self.skip_scale * skip_bdt
            x = block(x)
        x = x.transpose(1, 2)                            # [B, T, D]

        return self.out_proj(x)                           # [B, T, 768]
