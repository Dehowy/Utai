"""
F0 Predictor: predicts fundamental frequency contour from score conditions.

Dual-mode design:
- Singing mode (singing_ratio → 1): predicts F0 deviation from note pitch
  (log-scale offset in cents), plus vibrato parameters
- Rap mode (singing_ratio → 0): predicts absolute F0 contour from text/rhythm
  prosody (no reference pitch)
- Interpolation: output = ratio * singing_f0 + (1-ratio) * rap_f0

The F0 output matches RVC's expected format:
- Continuous Hz values at 100fps
- Range: 50-1100 Hz
- Quantized to 1-255 bins (mel-scale) for coarse F0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class F0Predictor(nn.Module):
    """
    Frame-level F0 predictor with singing/rap dual-mode support.

    Architecture:
    - Conv1D stack for local pitch pattern extraction
    - Bidirectional LSTM for long-range pitch dependencies
    - Separate singing/rap heads with smooth interpolation
    """

    def __init__(
        self,
        d_cond: int = 512,
        hidden_dim: int = 384,
        conv_layers: int = 5,
        conv_kernel: int = 5,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        f0_bins: int = 256,
    ):
        super().__init__()
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_bins = f0_bins

        # Precompute mel-scale boundaries for F0 quantization (matches RVC)
        f0_mel_min = 1127.0 * math.log(1.0 + f0_min / 700.0)
        f0_mel_max = 1127.0 * math.log(1.0 + f0_max / 700.0)
        self.register_buffer("f0_mel_min", torch.tensor(f0_mel_min))
        self.register_buffer("f0_mel_max", torch.tensor(f0_mel_max))

        # Note pitch embedding (frame-level, expanded from phoneme-level)
        self.note_pitch_proj = nn.Linear(1, hidden_dim)

        # Conv stack for local pattern extraction
        conv_blocks = []
        for i in range(conv_layers):
            d_in = d_cond + hidden_dim if i == 0 else hidden_dim
            conv_blocks.extend([
                nn.Conv1d(d_in, hidden_dim, conv_kernel, padding=conv_kernel // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
        self.conv_stack = nn.Sequential(*conv_blocks)

        # Bidirectional LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Singing head: predicts log-F0 deviation from note pitch (in cents/1200)
        self.singing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Rap head: predicts absolute log-F0
        self.rap_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Voicing head: predicts probability of voicing (for unvoiced frames)
        self.voicing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        condition: torch.Tensor,      # [B, T_frame, d_cond] encoder output (frame-level)
        note_pitch: torch.Tensor,      # [B, T_frame] MIDI pitch per frame (expanded)
        singing_ratio: torch.Tensor,   # [B] or [B, T_frame] float in [0, 1]
        mask: Optional[torch.Tensor] = None,  # [B, T_frame] bool
        voicing_target: Optional[torch.Tensor] = None,  # [B, T_frame] GT voicing (training)
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            f0_hz: [B, T_frame] predicted F0 in Hz
            voicing: [B, T_frame] voicing probability
            details: dict with singing/rap components for analysis
        """
        B, T, _ = condition.shape

        # Note pitch feature
        note_hz = self._midi_to_hz(note_pitch.float())  # [B, T]
        note_log = torch.log(note_hz.clamp(min=1.0) + 1e-6).unsqueeze(-1)  # [B, T, 1]
        note_feat = self.note_pitch_proj(note_log)  # [B, T, hidden]

        # Concatenate condition + note pitch feature
        x = torch.cat([condition, note_feat], dim=-1)  # [B, T, d_cond + hidden]

        # Conv stack
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.conv_stack(x)
        x = x.transpose(1, 2)  # [B, T, hidden]

        # LSTM
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            x, _ = self.lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=T)
        else:
            x, _ = self.lstm(x)

        # Singing mode: deviation from note pitch
        singing_dev = self.singing_head(x).squeeze(-1)  # [B, T] log-deviation
        singing_f0 = note_hz * torch.exp(singing_dev)  # [B, T] Hz

        # Rap mode: absolute F0
        rap_log_f0 = self.rap_head(x).squeeze(-1)  # [B, T] log-Hz (unnormalized)
        # Scale to valid F0 range
        rap_f0 = torch.sigmoid(rap_log_f0) * (self.f0_max - self.f0_min) + self.f0_min

        # Voicing (compute once, reuse for details)
        voicing_logits = self.voicing_head(x).squeeze(-1)  # [B, T]
        voicing = torch.sigmoid(voicing_logits)

        # Interpolate based on singing ratio
        if singing_ratio.dim() == 1:
            singing_ratio = singing_ratio.unsqueeze(1).expand(-1, T)
        f0_hz = singing_ratio * singing_f0 + (1 - singing_ratio) * rap_f0

        # Apply voicing mask (unvoiced → 0 Hz)
        # During training: use GT voicing for teacher forcing (differentiable path)
        # During inference: use predicted voicing with hard threshold
        if voicing_target is not None:
            voicing_mask = (voicing_target > 0).float()
        else:
            voicing_mask = (voicing > 0.5).float()
        f0_hz = f0_hz * voicing_mask

        # Clamp to valid range
        f0_hz = f0_hz.clamp(0, self.f0_max)

        details = {
            "singing_f0": singing_f0,
            "rap_f0": rap_f0,
            "singing_dev": singing_dev,
            "voicing_logits": voicing_logits,
        }

        return f0_hz, voicing, details

    def compute_loss(
        self,
        pred_f0: torch.Tensor,          # [B, T] predicted Hz
        target_f0: torch.Tensor,         # [B, T] target Hz
        pred_voicing: torch.Tensor,      # [B, T] predicted voicing prob
        mask: Optional[torch.Tensor] = None,
        details: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute F0 prediction loss.

        Uses log-F0 L1 loss for voiced frames + binary CE for voicing detection.
        """
        voiced_mask = target_f0 > self.f0_min  # [B, T]
        if mask is not None:
            voiced_mask = voiced_mask & mask

        # Log-F0 L1 loss (only on voiced frames)
        if voiced_mask.any():
            pred_log = torch.log(pred_f0[voiced_mask].clamp(min=1.0))
            target_log = torch.log(target_f0[voiced_mask].clamp(min=1.0))
            f0_loss = F.l1_loss(pred_log, target_log)
        else:
            f0_loss = torch.tensor(0.0, device=pred_f0.device)

        # Voicing BCE loss (use logits version for AMP safety)
        voicing_target = (target_f0 > self.f0_min).float()
        voicing_logits = details["voicing_logits"] if details else torch.logit(pred_voicing, eps=1e-6)
        if mask is not None:
            voicing_loss = F.binary_cross_entropy_with_logits(
                voicing_logits[mask], voicing_target[mask]
            )
        else:
            voicing_loss = F.binary_cross_entropy_with_logits(voicing_logits, voicing_target)

        total = f0_loss + 0.5 * voicing_loss

        metrics = {
            "f0_loss": f0_loss.item(),
            "voicing_loss": voicing_loss.item(),
            "f0_mae_cents": self._hz_to_cents_error(pred_f0, target_f0, voiced_mask),
        }

        return total, metrics

    def f0_to_coarse(self, f0: torch.Tensor) -> torch.Tensor:
        """Convert F0 (Hz) to RVC's coarse representation (1-255 mel-scale bins)."""
        f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
        f0_mel_norm = (f0_mel - self.f0_mel_min) / (self.f0_mel_max - self.f0_mel_min)
        f0_coarse = (f0_mel_norm * 254.0 + 1.0).long().clamp(1, 255)
        f0_coarse[f0 < self.f0_min] = 0
        return f0_coarse

    @staticmethod
    def _midi_to_hz(midi: torch.Tensor) -> torch.Tensor:
        """Convert MIDI note number to Hz. MIDI 69 = A4 = 440Hz."""
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

    @staticmethod
    def _hz_to_cents_error(pred: torch.Tensor, target: torch.Tensor,
                           mask: torch.Tensor) -> float:
        """Compute mean absolute error in cents for voiced frames."""
        if not mask.any():
            return 0.0
        pred_c = pred[mask].clamp(min=1.0)
        target_c = target[mask].clamp(min=1.0)
        cents = 1200.0 * torch.abs(torch.log2(pred_c / target_c))
        return cents.mean().item()
