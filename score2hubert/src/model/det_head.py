"""
Deterministic HuBERT head.

Replaces the flow matching decoder with a direct regression model from
[frame_hidden | f0_emb] -> HuBERT features.

Why this exists (see project_score2hubert_decisions for full reasoning):
HuBERT features are ~80% deterministic given the score (cross-singer same-segment
cos sim ~0.80 on ACE-Opencpop). Flow matching's stochastic sampling actively
fights this — multiple noise trajectories collapse to the same target, wasting
capacity, and the model learns to nail the high-variance dimensions (timbre,
F0 envelope) while letting the small phoneme-discriminative subspace become noise.

A small frozen-encoder diagnostic head reached cos 0.51 in 100 steps from cold
start, and produced *better audio with identifiable lyrics* than Plan A's flow
decoder at cos 0.76 — confirming the regression target is well-defined and the
flow path was the bottleneck.

Architecture (purely local — encoder above already provides global context):
    in_proj   : Linear(2*d_cond -> d_hidden)
    N x       : Conv1d(k=7) + GroupNorm + GELU + Dropout, residual
    out_proj  : Linear(d_hidden -> d_hubert)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DetHubertHead(nn.Module):
    """Deterministic regression head: [frame_hidden; f0_emb] -> HuBERT features."""

    def __init__(
        self,
        d_cond: int = 512,
        d_hubert: int = 768,
        d_hidden: int = 768,
        n_blocks: int = 6,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_cond = d_cond
        self.d_hubert = d_hubert
        self.d_hidden = d_hidden

        self.in_proj = nn.Linear(2 * d_cond, d_hidden)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.Sequential(
                nn.Conv1d(d_hidden, d_hidden, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(8, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)

        self.out_proj = nn.Linear(d_hidden, d_hubert)

    def forward(
        self,
        frame_hidden: torch.Tensor,    # [B, T, d_cond]
        f0_emb: torch.Tensor,           # [B, T, d_cond]
        mask: Optional[torch.Tensor] = None,  # [B, T] bool, True=valid
    ) -> torch.Tensor:
        """Returns predicted HuBERT features [B, T, d_hubert]."""
        x = torch.cat([frame_hidden, f0_emb], dim=-1)   # [B, T, 2*d_cond]
        h = self.in_proj(x)                              # [B, T, d_hidden]

        h = h.transpose(1, 2)                            # [B, d_hidden, T] for Conv1d
        for block in self.blocks:
            h = h + block(h)
        h = h.transpose(1, 2)                            # [B, T, d_hidden]

        out = self.out_proj(h)                           # [B, T, d_hubert]
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out
