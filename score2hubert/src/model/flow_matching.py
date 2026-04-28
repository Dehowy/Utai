"""
Optimal Transport Conditional Flow Matching (OT-CFM) decoder.

Learns a vector field that transports Gaussian noise to the HuBERT feature
distribution, conditioned on the score encoder output.

Key design choices:
- OT-CFM with linear interpolation paths (Lipman et al., 2023)
- DiT-style Transformer for the vector field network
- AdaLN for time-step conditioning
- Cross-attention for encoder condition injection
- Classifier-free guidance support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .modules import DiTBlock, ContinuousSinusoidalEmb, AdaLN


class VectorFieldNetwork(nn.Module):
    """
    Transformer-based vector field estimator for flow matching.

    Takes noisy features x_t, time t, and condition c, and predicts
    the velocity field v(t, x_t, c) that points from noise to data.
    """

    def __init__(
        self,
        d_hubert: int = 768,
        d_cond: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_hubert = d_hubert
        d_time = d_cond  # time embedding dimension matches condition

        # Input projection: noisy HuBERT features → d_hubert
        self.input_proj = nn.Linear(d_hubert, d_hubert)

        # Time embedding
        self.time_embed = nn.Sequential(
            ContinuousSinusoidalEmb(d_time),
            nn.Linear(d_time, d_time * 4),
            nn.SiLU(),
            nn.Linear(d_time * 4, d_time),
        )

        # Condition projection (match d_hubert for cross-attention)
        self.cond_proj = nn.Linear(d_cond, d_hubert)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_hubert,
                n_heads=n_heads,
                d_ff=d_ff,
                d_cond=d_time,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.final_adaln = AdaLN(d_hubert, d_time)
        self.output_proj = nn.Linear(d_hubert, d_hubert)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,        # [B, T, d_hubert] noisy features at time t
        t: torch.Tensor,           # [B] time steps in [0, 1]
        condition: torch.Tensor,   # [B, T, d_cond] encoder output (frame-level)
        mask: Optional[torch.Tensor] = None,  # [B, T] bool mask
    ) -> torch.Tensor:
        """Returns predicted velocity field: [B, T, d_hubert]"""

        # Project inputs
        h = self.input_proj(x_t)  # [B, T, d_hubert]
        time_emb = self.time_embed(t)  # [B, d_time]
        cond = self.cond_proj(condition)  # [B, T, d_hubert]

        # DiT blocks
        for block in self.blocks:
            h = block(h, cond=cond, time_emb=time_emb, mask=mask)

        # Final projection
        h = self.final_adaln(h, time_emb)
        return self.output_proj(h)


class FlowMatchingDecoder(nn.Module):
    """
    OT-CFM decoder for generating HuBERT features from score conditions.

    Training:
    - Sample t ~ U[0,1], x_0 ~ N(0,I), x_1 = target HuBERT
    - x_t = (1 - t) * x_0 + t * x_1  (OT linear interpolation)
    - Loss = ||v_theta(x_t, t, cond) - (x_1 - x_0)||^2

    Inference:
    - Start from x_0 ~ N(0, sigma^2 * I)
    - Solve ODE: dx/dt = v_theta(x_t, t, cond) from t=0 to t=1
    - Use Euler or higher-order ODE solver
    """

    def __init__(
        self,
        d_hubert: int = 768,
        d_cond: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.d_hubert = d_hubert
        self.sigma_min = sigma_min
        # NOTE: CFG drop is now applied by the caller (Score2HuBERT.forward)
        # because we need to selectively drop only the phoneme part of the
        # condition while keeping F0. See score2hubert.py for the logic.

        self.vector_field = VectorFieldNetwork(
            d_hubert=d_hubert,
            d_cond=d_cond,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

    def compute_loss(
        self,
        x_1: torch.Tensor,        # [B, T, d_hubert] target HuBERT features
        condition: torch.Tensor,   # [B, T, d_cond] encoder output (CFG drop already applied by caller)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute OT-CFM training loss.

        CFG drop is NOT applied here — the caller (Score2HuBERT.forward) handles
        it because we want selective CFG (drop phoneme content but keep F0).

        Returns:
            loss   : scalar flow matching loss
            metrics: dict with logging metrics
            pred_v : [B, T, D] predicted velocity (for reconstruction loss in caller)
            t      : [B] sampled time steps
            x_t    : [B, T, D] interpolated point at time t (for reconstruction loss)
        """
        B, T, D = x_1.shape

        # Sample time uniformly
        t = torch.rand(B, device=x_1.device, dtype=x_1.dtype)

        # Sample noise
        x_0 = torch.randn_like(x_1)

        # OT linear interpolation path
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        # Target velocity: v* = x_1 - x_0 (straight line)
        target_v = x_1 - x_0

        # Predict velocity (caller already applied any CFG drop on condition)
        pred_v = self.vector_field(x_t, t, condition, mask=mask)

        # Masked MSE loss
        if mask is not None:
            mask_expand = mask.unsqueeze(-1).float()  # [B, T, 1]
            loss = ((pred_v - target_v) ** 2 * mask_expand).sum() / mask_expand.sum() / D
        else:
            loss = F.mse_loss(pred_v, target_v)

        metrics = {
            "flow_loss": loss.item(),
            "pred_v_norm": pred_v.norm(dim=-1).mean().item(),
            "target_v_norm": target_v.norm(dim=-1).mean().item(),
        }

        return loss, metrics, pred_v, t, x_t

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_steps: int = 10,
        solver: str = "euler",
        cfg_scale: float = 1.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate HuBERT features via ODE solving.

        Args:
            condition: [B, T, d_cond] encoder output
            mask: [B, T] padding mask
            n_steps: number of ODE solver steps
            solver: "euler" | "midpoint" | "rk4"
            cfg_scale: classifier-free guidance scale (1.0 = no guidance)
            temperature: noise temperature

        Returns:
            x_1: [B, T, d_hubert] generated HuBERT features
        """
        B, T, _ = condition.shape
        device = condition.device

        # Start from noise
        x = torch.randn(B, T, self.d_hubert, device=device) * temperature

        # Zero condition for CFG unconditional pass
        if cfg_scale != 1.0:
            uncond = torch.zeros_like(condition)

        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = step / n_steps
            t = torch.full((B,), t_val, device=device)

            if solver == "euler":
                v = self._get_velocity(x, t, condition, uncond if cfg_scale != 1.0 else None,
                                       cfg_scale, mask)
                x = x + v * dt

            elif solver == "midpoint":
                # Midpoint method (2nd order)
                v1 = self._get_velocity(x, t, condition,
                                        uncond if cfg_scale != 1.0 else None, cfg_scale, mask)
                t_mid = torch.full((B,), t_val + 0.5 * dt, device=device)
                x_mid = x + v1 * (0.5 * dt)
                v2 = self._get_velocity(x_mid, t_mid, condition,
                                        uncond if cfg_scale != 1.0 else None, cfg_scale, mask)
                x = x + v2 * dt

            elif solver == "rk4":
                # Runge-Kutta 4th order
                t1 = t
                t2 = torch.full((B,), t_val + 0.5 * dt, device=device)
                t3 = t2
                t4 = torch.full((B,), t_val + dt, device=device)

                k1 = self._get_velocity(x, t1, condition,
                                        uncond if cfg_scale != 1.0 else None, cfg_scale, mask)
                k2 = self._get_velocity(x + k1 * (0.5 * dt), t2, condition,
                                        uncond if cfg_scale != 1.0 else None, cfg_scale, mask)
                k3 = self._get_velocity(x + k2 * (0.5 * dt), t3, condition,
                                        uncond if cfg_scale != 1.0 else None, cfg_scale, mask)
                k4 = self._get_velocity(x + k3 * dt, t4, condition,
                                        uncond if cfg_scale != 1.0 else None, cfg_scale, mask)
                x = x + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

        return x

    def _get_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        uncond: Optional[torch.Tensor],
        cfg_scale: float,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Get velocity with optional classifier-free guidance."""
        if uncond is not None and cfg_scale != 1.0:
            # Batched: concat conditional and unconditional
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t, t], dim=0)
            c_double = torch.cat([condition, uncond], dim=0)
            m_double = torch.cat([mask, mask], dim=0) if mask is not None else None

            v_double = self.vector_field(x_double, t_double, c_double, mask=m_double)
            v_cond, v_uncond = v_double.chunk(2, dim=0)

            # CFG: v = v_uncond + w * (v_cond - v_uncond)
            return v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            return self.vector_field(x, t, condition, mask=mask)
