"""F0Predictor — predict F0 contour from frame-level hidden states.

Dual-mode singing/rap architecture, but singing_ratio hardcoded to 1.0 for v2.
User provides pitch_curve in production → F0Predictor is bypassed at inference.

Input: frame_hidden [B, T_frame, 512] + note_pitch [B, T_frame] (MIDI)
Output: f0_hz [B, T_frame], voicing [B, T_frame]

~6M params.
"""

import torch
import torch.nn as nn


class F0Predictor(nn.Module):
    """Predict F0 as deviation from note pitch + voicing probability.

    Architecture:
        5x Conv1d(512, 512, k=5) + LayerNorm + GELU
        2x BiLSTM(256) → 512
        singing_head: Linear(512, 1)   # log-F0 deviation from note
        voicing_head: Linear(512, 1)   # voicing probability (logit)

    F0 prediction strategy:
        f0 = note_f0 * 2^(deviation/1200)
        where deviation is in cents, note_f0 = 440 * 2^((midi-69)/12)

    ~6M params.
    """

    def __init__(self, hidden_dim: int = 512, n_conv_layers: int = 5,
                 conv_kernel: int = 5, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        # Conv stack
        convs = []
        for _ in range(n_conv_layers):
            convs.extend([
                nn.Conv1d(hidden_dim, hidden_dim, conv_kernel, padding=conv_kernel // 2),
                nn.LayerNorm(hidden_dim),  # will be applied after transpose
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.conv_stack = nn.ModuleList(convs)

        # BiLSTM
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, num_layers=lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0,
        )

        # Output heads
        self.singing_head = nn.Linear(hidden_dim, 1)   # deviation in cents
        self.voicing_head = nn.Linear(hidden_dim, 1)   # voicing logit

        # Note pitch projection (MIDI → Hz → log-space)
        self.note_scale = nn.Linear(1, hidden_dim)  # optional conditioning

    @staticmethod
    def midi_to_hz(midi: torch.Tensor) -> torch.Tensor:
        """Convert MIDI note to Hz. 0 MIDI → 0 Hz (rest)."""
        hz = 440.0 * (2.0 ** ((midi.float() - 69.0) / 12.0))
        hz = hz * (midi > 0).float()  # zero out rest notes
        return hz

    def forward(
        self,
        frame_hidden: torch.Tensor,  # [B, T_frame, D]
        note_pitch: torch.Tensor,    # [B, T_frame] int MIDI
    ) -> tuple:
        """Returns (f0_hz, voicing_logit).

        f0_hz: [B, T_frame] predicted F0 in Hz
        voicing_logit: [B, T_frame] pre-sigmoid voicing probability
        """
        x = frame_hidden

        # Conv stack (conv operates on [B, D, T])
        x = x.transpose(1, 2)
        for i in range(0, len(self.conv_stack), 4):
            conv, norm, act, drop = (self.conv_stack[i], self.conv_stack[i+1],
                                     self.conv_stack[i+2], self.conv_stack[i+3])
            x = conv(x)
            x = x.transpose(1, 2)  # [B, T, D] for LayerNorm
            x = norm(x)
            x = act(x)
            x = drop(x)
            x = x.transpose(1, 2)  # back to [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]

        # BiLSTM
        x, _ = self.lstm(x)  # [B, T, D]

        # Predict deviation (cents) and voicing
        deviation = self.singing_head(x).squeeze(-1)   # [B, T] cents
        voicing_logit = self.voicing_head(x).squeeze(-1)  # [B, T]

        # Convert deviation to F0
        note_f0 = self.midi_to_hz(note_pitch)           # [B, T]
        f0_hz = note_f0 * (2.0 ** (deviation / 1200.0))  # [B, T]
        f0_hz = f0_hz * (note_pitch > 0).float()         # zero at rest

        return f0_hz, voicing_logit


class F0CondProj(nn.Module):
    """Project F0 (Hz) into frame-level embedding for DetHubertHead conditioning.

    f0_hz → log-scale → sinusoidal features → linear → [B, T, hidden_dim]
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, f0_hz: torch.Tensor) -> torch.Tensor:
        """f0_hz: [B, T] → [B, T, hidden_dim]"""
        # Log-scale (add small eps to avoid log(0))
        log_f0 = torch.log1p(f0_hz).unsqueeze(-1)  # [B, T, 1]
        return self.proj(log_f0)
