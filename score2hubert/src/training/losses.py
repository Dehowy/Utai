"""
Loss functions for Score2HuBERT training.

Multi-objective loss:
1. Flow matching loss (OT-CFM vector field regression)
2. F0 prediction loss (log-F0 L1 + voicing BCE)
3. Duration prediction loss (log-duration MSE, optional)
4. Variance prediction losses (energy, breathiness MSE)
5. Multi-scale HuBERT consistency loss (auxiliary)

The flow matching loss is the primary objective; others are regularizers
that ensure the conditioning signal is informative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class Score2HuBERTLoss(nn.Module):
    """
    Aggregated loss computation for Score2HuBERT model.

    This wraps the individual component losses and applies configured weights.
    Most of the actual loss computation happens inside the model's forward(),
    but this class handles any additional losses and the final aggregation.
    """

    def __init__(self, config: dict):
        super().__init__()
        loss_cfg = config["training"]["loss"]
        self.w_flow = loss_cfg["flow_matching"]
        self.w_f0 = loss_cfg["f0"]
        self.w_dur = loss_cfg.get("duration", 0.5)
        self.w_energy = loss_cfg.get("energy", 0.1)
        self.w_breath = loss_cfg.get("breathiness", 0.1)
        self.w_voicing = loss_cfg.get("voicing", 0.05)

        ms_cfg = loss_cfg.get("multiscale_hubert", {})
        self.ms_enabled = ms_cfg.get("enabled", False)
        self.ms_weight = ms_cfg.get("weight", 0.3)
        self.ms_scales = ms_cfg.get("scales", [1, 5, 10])

    def forward(self, model_output: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        Takes the output from Score2HuBERT.forward() and computes final weighted loss.

        The model already computes individual losses; this just provides a
        hook for any additional post-hoc losses and unified logging.
        """
        # Model already computes the total loss with weights
        total_loss = model_output["loss"]
        metrics = model_output.get("metrics", {})
        losses = model_output.get("losses", {})

        # Add weighted breakdown to metrics for logging
        metrics["total_loss"] = total_loss.item()
        for name, val in losses.items():
            metrics[f"loss/{name}"] = val

        return total_loss, metrics


class FeatureMatchingLoss(nn.Module):
    """
    Optional feature matching loss between generated and target HuBERT features.

    Used during the deterministic pre-training phase (Phase 1) where we train
    without flow matching to establish a strong encoder baseline.
    """

    def __init__(self, d_hubert: int = 768):
        super().__init__()
        self.d_hubert = d_hubert

    def forward(
        self,
        pred: torch.Tensor,     # [B, T, 768] predicted features
        target: torch.Tensor,   # [B, T, 768] target features
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """L1 + cosine similarity loss on HuBERT features."""
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            # L1
            l1 = (torch.abs(pred - target) * mask_f).sum() / mask_f.sum() / self.d_hubert
            # Cosine similarity
            cos = F.cosine_similarity(pred, target, dim=-1)  # [B, T]
            cos_loss = (1.0 - cos) * mask.float()
            cos_loss = cos_loss.sum() / mask.float().sum()
        else:
            l1 = F.l1_loss(pred, target)
            cos = F.cosine_similarity(pred, target, dim=-1)
            cos_loss = (1.0 - cos).mean()

        total = l1 + 0.5 * cos_loss

        metrics = {
            "feat_l1": l1.item(),
            "feat_cos_loss": cos_loss.item(),
            "feat_cos_sim": cos.mean().item(),
        }

        return total, metrics
