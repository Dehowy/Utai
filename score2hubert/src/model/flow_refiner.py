"""
Flow Refiner: small OT-CFM module that refines det_head's coarse HuBERT prediction.

Architecture: Conv1d-based (same philosophy as DetHubertHead but with time-conditioned
velocity prediction for OT-CFM). Does NOT use DiT/AdaLN (avoids the architectural
shortcut that caused Plan A's failure — see project_score2hubert_decisions.md).

Training:
  Target: residual = hubert_gt - hubert_coarse (det_head output, DETACHED)
  OT-CFM: sample t~U(0,1), interpolate x_t = (1-t)*x_0 + t*residual,
  predict velocity v, loss = MSE(v_pred, v_target) where v_target = residual - x_0

Inference (Euler ODE):
  Start from noise x_0 ~ N(0, 1)
  Euler steps: x_{i+1} = x_i + v(x_i, t_i, cond) * dt
  hubert_refined = hubert_coarse + x_final
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal embedding for diffusion/flow time step t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] float in [0, 1] → [B, dim]"""
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        emb = t.unsqueeze(-1) * freq.unsqueeze(0)  # [B, half]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]


class FlowRefiner(nn.Module):
    """Small OT-CFM refiner on det_head residual."""

    def __init__(
        self,
        d_hubert: int = 768,
        d_cond: int = 1792,  # frame_hidden(512) + hubert_coarse(768) + f0_emb(512)
        d_hidden: int = 384,
        n_blocks: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        noise_scale: float = 1.0,  # Scale x_0 noise to match residual std (~0.18)
    ):
        super().__init__()
        self.d_hubert = d_hubert
        self.d_hidden = d_hidden
        self.noise_scale = noise_scale

        # Projections
        self.in_proj = nn.Linear(d_hubert, d_hidden)
        self.cond_proj = nn.Linear(d_cond, d_hidden)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
        )

        # Conv blocks with residual connections (same pattern as DetHubertHead)
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

        self.out_proj = nn.Linear(d_hidden, d_hubert)

    def _velocity(
        self,
        x_t: torch.Tensor,   # [B, T, d_hubert]
        t: torch.Tensor,      # [B]
        cond: torch.Tensor,   # [B, T, d_cond]
    ) -> torch.Tensor:
        """Predict velocity field v(x_t, t, cond) → [B, T, d_hubert]."""
        h = self.in_proj(x_t) + self.cond_proj(cond) + self.time_mlp(t).unsqueeze(1)

        h = h.transpose(1, 2)  # [B, d_hidden, T]
        for block in self.blocks:
            h = h + block(h)
        h = h.transpose(1, 2)  # [B, T, d_hidden]

        return self.out_proj(h)

    def forward(
        self,
        x_1: torch.Tensor,    # [B, T, d_hubert] target residual
        cond: torch.Tensor,    # [B, T, d_cond]
        mask: Optional[torch.Tensor] = None,  # [B, T] bool
    ) -> torch.Tensor:
        """OT-CFM training loss (scalar)."""
        B, T, D = x_1.shape

        t = torch.rand(B, device=x_1.device)
        x_0 = torch.randn_like(x_1) * self.noise_scale

        t_e = t.view(B, 1, 1)
        x_t = (1 - t_e) * x_0 + t_e * x_1
        v_target = x_1 - x_0

        v_pred = self._velocity(x_t, t, cond)

        if mask is not None:
            m = mask.unsqueeze(-1).float()  # [B, T, 1]
            loss = (F.mse_loss(v_pred, v_target, reduction="none") * m).sum() / m.sum() / D
        else:
            loss = F.mse_loss(v_pred, v_target)

        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,   # [B, T, d_cond]
        n_steps: int = 4,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Euler ODE: noise → predicted residual. [B, T, d_hubert]"""
        B, T, _ = cond.shape
        x = torch.randn(B, T, self.d_hubert, device=cond.device) * self.noise_scale * temperature
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=cond.device)
            v = self._velocity(x, t, cond)
            x = x + v * dt

        return x
