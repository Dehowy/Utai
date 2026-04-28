"""
Discrete HuBERT head — classifies each frame into a K-means codebook token.

Replaces continuous regression (DetHubertHead) with discrete classification.
Same conv architecture as DetHubertHead for weight reuse; only the output layer
changes from Linear(d_hidden, 768) to Linear(d_hidden, n_codes).

Why discrete: L1 regression converges to the conditional mean E[HuBERT|score],
which is "fuzzy" when the target distribution is multi-modal (Phases 3-4 proved
this). Cross-entropy classification picks ONE specific codebook entry per frame
— no averaging, no blurring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiscreteHubertHead(nn.Module):
    """[frame_hidden; f0_emb] -> token logits [B, T, n_codes]."""

    def __init__(
        self,
        d_cond: int = 512,
        n_codes: int = 1024,
        d_hidden: int = 512,
        n_blocks: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        codebook_path: str = "",
        d_hubert: int = 768,
    ):
        super().__init__()
        self.d_cond = d_cond
        self.n_codes = n_codes
        self.d_hidden = d_hidden
        self.d_hubert = d_hubert

        # Same conv architecture as DetHubertHead (for weight reuse)
        self.in_proj = nn.Linear(2 * d_cond, d_hidden)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(d_hidden, d_hidden, kernel_size, padding=kernel_size // 2),
                    nn.GroupNorm(8, d_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )

        # Classification head (replaces det_head's out_proj Linear(d_hidden, 768))
        self.logit_proj = nn.Linear(d_hidden, n_codes)

        # Codebook: K-means centroids, frozen. Used to convert tokens → features for RVC.
        self.codebook = nn.Embedding(n_codes, d_hubert)
        if codebook_path:
            self._load_codebook(codebook_path)
        self.codebook.weight.requires_grad = False  # frozen

    def _load_codebook(self, path: str):
        """Load K-means centroids into the codebook embedding."""
        centroids = torch.load(path, map_location="cpu", weights_only=True).float()
        assert centroids.shape == (self.n_codes, self.d_hubert), \
            f"Codebook shape mismatch: {centroids.shape} vs ({self.n_codes}, {self.d_hubert})"
        self.codebook.weight.data.copy_(centroids)

    def forward(
        self,
        frame_hidden: torch.Tensor,   # [B, T, d_cond]
        f0_emb: torch.Tensor,          # [B, T, d_cond]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns logits [B, T, n_codes]."""
        x = torch.cat([frame_hidden, f0_emb], dim=-1)  # [B, T, 2*d_cond]
        h = self.in_proj(x)                              # [B, T, d_hidden]

        h = h.transpose(1, 2)                            # [B, d_hidden, T]
        for block in self.blocks:
            h = h + block(h)
        h = h.transpose(1, 2)                            # [B, T, d_hidden]

        logits = self.logit_proj(h)                      # [B, T, n_codes]
        if mask is not None:
            logits = logits * mask.unsqueeze(-1).float()
        return logits

    def tokens_to_features(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token indices to continuous HuBERT features via codebook lookup.
        token_ids: [B, T] long → [B, T, 768]
        """
        return self.codebook(token_ids)

    def init_from_det_head(self, det_head_state: dict):
        """Copy conv block weights from a trained DetHubertHead (warm start).
        in_proj and conv blocks are architecturally identical; only out_proj differs.
        """
        own = self.state_dict()
        copied = 0
        for key, val in det_head_state.items():
            if key in own and key != "logit_proj.weight" and key != "logit_proj.bias":
                if own[key].shape == val.shape:
                    own[key].copy_(val)
                    copied += 1
        self.load_state_dict(own, strict=False)
        return copied
