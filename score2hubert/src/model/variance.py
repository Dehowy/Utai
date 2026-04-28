"""
Variance Adaptor: predicts and injects frame-level variance features.

Predicted variances:
- Energy envelope (loudness contour)
- Breathiness (aspiration noise level)
- Voicing strength (vocal fold tension)

These are predicted from the encoder output and added back to the
hidden representation before flow matching decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .modules import Conv1DStack


class VariancePredictor(nn.Module):
    """Single variance dimension predictor (energy, breathiness, etc.)."""

    def __init__(self, d_cond: int, hidden_dim: int = 256, n_layers: int = 3,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.predictor = Conv1DStack(
            in_dim=d_cond,
            hidden_dim=hidden_dim,
            out_dim=1,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        # Embedding to inject predicted variance back into hidden state
        self.embedding = nn.Linear(1, d_cond)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, d_cond] input features
        target: [B, T] ground truth (if provided, use for teacher forcing)
        Returns: (predicted_value [B, T], embedding [B, T, d_cond])
        """
        pred = self.predictor(x, mask).squeeze(-1)  # [B, T]

        # Teacher forcing during training
        value = target if target is not None else pred
        emb = self.embedding(value.unsqueeze(-1))  # [B, T, d_cond]

        return pred, emb


class VarianceAdaptor(nn.Module):
    """
    Predicts multiple variance features and injects them into the hidden state.

    Acts as a bridge between the encoder (phoneme-level, expanded to frame-level)
    and the flow matching decoder.
    """

    def __init__(
        self,
        d_cond: int = 512,
        energy_cfg: Optional[dict] = None,
        breathiness_cfg: Optional[dict] = None,
        voicing_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.predictors = nn.ModuleDict()
        self.enabled = {}

        configs = {
            "energy": energy_cfg,
            "breathiness": breathiness_cfg,
            "voicing": voicing_cfg,
        }

        for name, cfg in configs.items():
            if cfg is not None and cfg.get("enabled", False):
                self.predictors[name] = VariancePredictor(
                    d_cond=d_cond,
                    hidden_dim=cfg.get("hidden_dim", 256),
                    n_layers=cfg.get("n_layers", 3),
                    kernel_size=cfg.get("kernel_size", 3),
                )
                self.enabled[name] = True

    def forward(
        self,
        x: torch.Tensor,   # [B, T_frame, d_cond]
        targets: Optional[Dict[str, torch.Tensor]] = None,  # {name: [B, T_frame]}
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            x: [B, T_frame, d_cond] hidden state with variance embeddings added
            predictions: dict of {name: [B, T_frame]} predicted variances
        """
        predictions = {}

        for name, predictor in self.predictors.items():
            target = targets.get(name) if targets is not None else None
            pred, emb = predictor(x, target=target, mask=mask)
            predictions[name] = pred
            x = x + emb

        return x, predictions

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MSE loss for each variance dimension."""
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        metrics = {}

        for name, pred in predictions.items():
            if name in targets:
                target = targets[name]
                if mask is not None:
                    loss = F.mse_loss(pred[mask], target[mask])
                else:
                    loss = F.mse_loss(pred, target)
                total_loss = total_loss + loss
                metrics[f"var_{name}_loss"] = loss.item()

        return total_loss, metrics
