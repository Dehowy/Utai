"""Reusable building blocks for Score2HuBERT v2.

Conformer block, LengthRegulator, sinusoidal embeddings, pitch embedding.
All shapes documented as: [B, T, D] where B=batch, T=seq_len, D=hidden_dim.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Sinusoidal Positional Encoding ──────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Fixed sinusoidal positional encoding (not learned)."""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D]"""
        return x + self.pe[:, :x.size(1)]


# ─── Pitch Embedding ────────────────────────────────────────────────────────

class PitchEmbedding(nn.Module):
    """Sinusoidal pitch embedding with octave + pitch-class decomposition.

    MIDI note 0-127 → sinusoidal encoding of semitone frequency + octave/pitch-class.
    More musically meaningful than a plain lookup table.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        # Learnable projection from 3 features (normalized_midi, octave, pitch_class)
        # to half the dimension, then sinusoidal expansion
        self.proj = nn.Linear(3, half)
        self.out = nn.Linear(half * 2, dim)  # concat sin+cos → dim

    def forward(self, midi: torch.Tensor) -> torch.Tensor:
        """midi: [B, T] int/long → [B, T, dim]"""
        midi_f = midi.float()
        # Decompose MIDI into musically meaningful features
        normalized = midi_f / 127.0                    # [0, 1]
        octave = (midi_f / 12.0).floor() / 10.0       # ~[0, 1]
        pitch_class = (midi_f % 12) / 12.0            # [0, 1)
        features = torch.stack([normalized, octave, pitch_class], dim=-1)  # [B, T, 3]

        h = self.proj(features)  # [B, T, half]
        # Sinusoidal expansion
        emb = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)  # [B, T, half*2]
        return self.out(emb)  # [B, T, dim]


# ─── Conformer Block ────────────────────────────────────────────────────────

class ConvModule(nn.Module):
    """Conformer convolution module: pointwise → GLU → depthwise → BN → Swish → pointwise."""

    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * 2)  # pointwise, expand for GLU
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.pw2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D]"""
        x = self.layer_norm(x)
        x = self.pw1(x)
        x = F.glu(x, dim=-1)                        # [B, T, D]
        x = x.transpose(1, 2)                        # [B, D, T]
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = x.transpose(1, 2)                        # [B, T, D]
        x = self.pw2(x)
        return self.dropout(x)


class FeedForward(nn.Module):
    """Conformer feed-forward module with expansion factor 4."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * expansion)
        self.act = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * expansion, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)


class ConformerBlock(nn.Module):
    """Single Conformer block: FF → MHSA → Conv → FF (Macaron-style).

    Uses pre-norm (LayerNorm before each sub-layer) and residual connections.
    Half-step residual on the two feed-forward modules (Conformer paper).
    """

    def __init__(self, dim: int, n_heads: int = 8, kernel_size: int = 31,
                 ff_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForward(dim, ff_expansion, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConvModule(dim, kernel_size, dropout)
        self.ff2 = FeedForward(dim, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """x: [B, T, D], mask: [B, T] bool (True=pad) -> [B, T, D]"""
        # FF1 (half-step)
        x = x + 0.5 * self.ff1(x)

        # MHSA
        x_norm = self.attn_norm(x)
        key_padding_mask = mask if mask is not None else None
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + self.attn_dropout(attn_out)

        # Conv
        x = x + self.conv(x)

        # FF2 (half-step)
        x = x + 0.5 * self.ff2(x)

        return self.final_norm(x)


# ─── Length Regulator ────────────────────────────────────────────────────────

class LengthRegulator(nn.Module):
    """Expand phone-level hidden states to frame-level using duration (in frames).

    Adds per-frame positional features (Run 8: two levels of pos encoding):
    - pos_in_phone ∈ [0, 1) — position within current phoneme
    - pos_in_note  ∈ [0, 1) — position within current musical note (= run of consecutive
      phonemes with same MIDI pitch). Captures attack/sustain/release envelope.
    - log(phone_duration) — global phone-length scalar
    - log(note_duration)  — global note-length scalar

    All projections Kaiming-init; together they give enough pos signal to break
    det_head's "long sustained = drift to adjacent cluster" failure mode.

    phone_hidden: [B, T_phone, D]
    durations:    [B, T_phone] (int, frames per phoneme)
    pitch:        [B, T_phone] optional (MIDI) — if provided, adds intra-note pos
    -> frame_hidden: [B, T_frame, D]

    Pass add_pos_features=False to get plain expansion (e.g. for pitch tensor of dim=1).
    """

    def __init__(self, hidden_dim: int = 512, pos_resolution: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_resolution = pos_resolution

        # Sinusoidal lookup table (reused for both intra-phoneme and intra-note pos)
        pe = torch.zeros(pos_resolution + 1, hidden_dim)
        positions = torch.arange(0, pos_resolution + 1, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("pos_pe", pe)

        # Intra-phoneme pos + phoneme-duration
        self.pos_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dur_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Intra-note pos + note-duration (Run 8). Separate from phoneme-level so model
        # learns two distinct positional signals (phoneme articulation vs note envelope).
        self.note_pos_proj = nn.Linear(hidden_dim, hidden_dim)
        self.note_dur_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable scaling on the ADDED positional features (Run 8). Init 3.0 so pos
        # starts with ~0.2 per-elem scale vs phone_hidden ~O(1) (Run 5 had ~0.06 at Kaiming
        # defaults, too small to load-bear). Gradient can shrink it if over-scaled.
        self.pos_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, phone_hidden: torch.Tensor, durations: torch.Tensor,
                pitch: torch.Tensor = None, max_frames: int = None,
                add_pos_features: bool = True) -> torch.Tensor:
        B, T_phone, D = phone_hidden.shape
        device = phone_hidden.device
        if max_frames is None:
            max_frames = int(durations.sum(dim=1).max().item())

        outputs = []
        for b in range(B):
            durs_b = durations[b].clamp(min=0).long()
            total_d = int(durs_b.sum().item())
            expanded = torch.repeat_interleave(phone_hidden[b], durs_b, dim=0)

            if add_pos_features and total_d > 0:
                # ── Intra-phoneme pos + phone dur ───────────────────────────────
                frame_to_phone = torch.repeat_interleave(
                    torch.arange(T_phone, device=device), durs_b
                )  # [T_exp]
                dur_per_frame = durs_b[frame_to_phone].float().clamp(min=1.0)
                cum_durs = torch.cumsum(durs_b, dim=0)
                phone_start = cum_durs - durs_b
                global_idx = torch.arange(total_d, device=device)
                local_offset = (global_idx - phone_start[frame_to_phone]).float()

                pos_norm = local_offset / dur_per_frame  # [0,1)
                pos_idx = (pos_norm * self.pos_resolution).long().clamp(max=self.pos_resolution)
                pos_feat = self.pos_proj(self.pos_pe[pos_idx])

                log_dur = torch.log1p(dur_per_frame).unsqueeze(-1)
                dur_feat = self.dur_proj(log_dur)

                # ── Intra-note pos + note dur (Run 8) ───────────────────────────
                note_pos_feat = 0
                note_dur_feat = 0
                if pitch is not None:
                    pitch_b = pitch[b, :T_phone]  # [T_phone] MIDI
                    # Note boundary = pitch change (SP with pitch=0 becomes own note)
                    if T_phone > 1:
                        change = (pitch_b[1:] != pitch_b[:-1]).long()
                        note_idx = torch.cat([
                            torch.zeros(1, device=device, dtype=torch.long),
                            change.cumsum(0),
                        ])  # [T_phone] — monotonic
                    else:
                        note_idx = torch.zeros(1, device=device, dtype=torch.long)

                    num_notes = int(note_idx.max().item()) + 1
                    note_durs = torch.zeros(num_notes, device=device, dtype=torch.long)
                    note_durs.scatter_add_(0, note_idx, durs_b)
                    note_cumsum = torch.cumsum(note_durs, 0)
                    note_starts = note_cumsum - note_durs

                    frame_note_idx = note_idx[frame_to_phone]
                    frame_note_start = note_starts[frame_note_idx].float()
                    frame_note_length = note_durs[frame_note_idx].float().clamp(min=1.0)

                    pos_in_note = (global_idx.float() - frame_note_start) / frame_note_length
                    note_pos_idx = (pos_in_note * self.pos_resolution).long().clamp(max=self.pos_resolution)
                    note_pos_feat = self.note_pos_proj(self.pos_pe[note_pos_idx])

                    log_note_dur = torch.log1p(frame_note_length).unsqueeze(-1)
                    note_dur_feat = self.note_dur_proj(log_note_dur)

                # All 4 pos features scaled by learnable pos_scale (init 3.0)
                expanded = expanded + self.pos_scale * (pos_feat + dur_feat + note_pos_feat + note_dur_feat)

            # Pad or truncate to max_frames
            if expanded.size(0) >= max_frames:
                outputs.append(expanded[:max_frames])
            else:
                pad = torch.zeros(max_frames - expanded.size(0), D,
                                  device=device, dtype=phone_hidden.dtype)
                outputs.append(torch.cat([expanded, pad], dim=0))

        return torch.stack(outputs, dim=0)
