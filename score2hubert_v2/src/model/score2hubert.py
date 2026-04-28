"""Score2HuBERT v2 — full model assembly.

Score input (phonemes, pitch, duration)
  → ScoreEncoder (Conformer 4L)
  → LengthRegulator (phone → frame)
  → F0Predictor (frame → f0_hz + voicing)
  → F0CondProj (f0 → embedding)
  → DetHubertHead (frame + f0_emb → HuBERT [768])

Total: ~49M params.
"""

import torch
import torch.nn as nn

from .encoder import ScoreEncoder
from .modules import LengthRegulator
from .f0_predictor import F0Predictor, F0CondProj
from .det_head import DetHubertHead


class Score2HuBERT(nn.Module):
    """Score2HuBERT v2: score → HuBERT features + F0.

    Args:
        n_phones: vocabulary size (default 80)
        hidden_dim: encoder/head hidden dimension (default 512)
        hubert_dim: HuBERT output dimension (default 768)
        n_enc_layers: number of Conformer layers (default 4)
        n_heads: attention heads (default 8)
        enc_kernel: Conformer conv kernel size (default 31)
        n_det_blocks: DetHubertHead conv blocks (default 4)
        det_kernel: DetHubertHead conv kernel size (default 7)
        dropout: dropout rate (default 0.1)
    """

    def __init__(
        self,
        n_phones: int = 80,
        hidden_dim: int = 512,
        hubert_dim: int = 768,
        n_enc_layers: int = 4,
        n_heads: int = 8,
        enc_kernel: int = 31,
        n_det_blocks: int = 4,
        det_kernel: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hubert_dim = hubert_dim

        self.encoder = ScoreEncoder(
            n_phones=n_phones,
            hidden_dim=hidden_dim,
            n_layers=n_enc_layers,
            n_heads=n_heads,
            kernel_size=enc_kernel,
            dropout=dropout,
        )
        self.length_reg = LengthRegulator(hidden_dim=hidden_dim)
        self.f0_predictor = F0Predictor(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.f0_cond = F0CondProj(hidden_dim=hidden_dim)
        self.det_head = DetHubertHead(
            hidden_dim=hidden_dim,
            hubert_dim=hubert_dim,
            n_blocks=n_det_blocks,
            kernel_size=det_kernel,
            dropout=dropout,
        )
        # Aux head: predict [pos_in_phone, pos_in_note] from frame_hidden.
        # Forces encoder to keep positional signal decodable → load-bear pos features.
        self.pos_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        phonemes: torch.Tensor,      # [B, T_phone] int
        pitch: torch.Tensor,         # [B, T_phone] int (MIDI)
        durations: torch.Tensor,     # [B, T_phone] int (frames)
        phone_mask: torch.Tensor = None,  # [B, T_phone] bool (True=pad)
        f0_target: torch.Tensor = None,   # [B, T_frame] float (Hz) — teacher-forced F0
    ) -> dict:
        """Forward pass. Returns dict with all outputs for loss computation.

        If f0_target is provided (training), uses it for HuBERT conditioning.
        Otherwise uses predicted F0 (inference).
        """
        # Fix: force pitch=0 at SP phoneme frames. Score tools often carry
        # the previous note's MIDI into SP frames — this leaks voiced signal
        # through F0Predictor + HuBERT conditioning, causing rest-frame artifacts.
        SP_ID = 3
        pitch = pitch.clone()
        pitch[phonemes == SP_ID] = 0

        # Encode phone-level
        phone_hidden = self.encoder(phonemes, pitch, durations, mask=phone_mask)

        # Expand to frame-level (intra-phoneme pos + phone dur + intra-note pos + note dur)
        n_frames = durations.sum(dim=1).max().item()
        frame_hidden = self.length_reg(
            phone_hidden, durations, pitch=pitch, max_frames=n_frames,
        )

        # Expand note pitch to frame-level (plain expansion, no pos features)
        note_pitch_frame = self.length_reg(
            pitch.unsqueeze(-1).float(), durations, max_frames=n_frames,
            add_pos_features=False,
        ).squeeze(-1).long()

        # Predict F0
        f0_pred, voicing_logit = self.f0_predictor(frame_hidden, note_pitch_frame)

        # F0 conditioning: use target in training, predicted in inference
        f0_for_cond = f0_target if f0_target is not None else f0_pred.detach()
        f0_emb = self.f0_cond(f0_for_cond)

        # Predict HuBERT
        hubert_pred = self.det_head(frame_hidden, f0_emb)

        # Aux pos prediction (Run 8): force encoder to keep pos signal decodable
        pos_pred = torch.sigmoid(self.pos_head(frame_hidden))  # [B, T, 2] in [0,1]

        return {
            "hubert_pred": hubert_pred,          # [B, T_frame, 768]
            "f0_pred": f0_pred,                  # [B, T_frame]
            "voicing_logit": voicing_logit,      # [B, T_frame]
            "note_pitch_frame": note_pitch_frame,  # [B, T_frame] long — for F0 loss outlier filter
            "phone_hidden": phone_hidden,        # [B, T_phone, 512]
            "frame_hidden": frame_hidden,        # [B, T_frame, 512]
            "pos_pred": pos_pred,                # [B, T_frame, 2] (phoneme, note)
        }

    def synthesize(
        self,
        phonemes: torch.Tensor,
        pitch: torch.Tensor,
        durations: torch.Tensor,
        f0_curve: torch.Tensor = None,
    ) -> dict:
        """Inference mode. Optionally takes user-drawn F0 curve.

        Args:
            phonemes: [1, T_phone] or [T_phone]
            pitch: [1, T_phone] or [T_phone]
            durations: [1, T_phone] or [T_phone]
            f0_curve: [1, T_frame] or [T_frame] optional user F0

        Returns dict with hubert_pred [1, T_frame, 768] and f0 [1, T_frame].
        """
        # Ensure batch dim
        if phonemes.dim() == 1:
            phonemes = phonemes.unsqueeze(0)
            pitch = pitch.unsqueeze(0)
            durations = durations.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(phonemes, pitch, durations, f0_target=f0_curve)

        return {
            "hubert": out["hubert_pred"],
            "f0": out["f0_pred"] if f0_curve is None else f0_curve,
        }

    def count_params(self) -> dict:
        """Count parameters by sub-module."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())
        return {
            "encoder": _count(self.encoder),
            "length_reg": _count(self.length_reg),
            "f0_predictor": _count(self.f0_predictor),
            "f0_cond": _count(self.f0_cond),
            "det_head": _count(self.det_head),
            "pos_head": _count(self.pos_head),
            "total": _count(self),
        }
