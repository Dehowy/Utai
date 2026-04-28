"""
Score Encoder: converts musical score inputs to hidden representations.

Input: phoneme IDs, MIDI pitch, durations, singing ratio, optional pitch curve
Output: phone-level hidden states [B, T_phone, d_model]

Architecture: Embedding layers → Conformer stack → phone-level features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .modules import ConformerBlock, SinusoidalPosEmb, LengthRegulator


class PitchEmbedding(nn.Module):
    """
    Embed MIDI pitch using a combination of:
    1. Sinusoidal encoding (captures harmonic structure)
    2. Learned octave + pitch-class embedding (captures musical structure)
    """

    def __init__(self, d_model: int, max_pitch: int = 128):
        super().__init__()
        self.d_model = d_model
        # Sinusoidal part (half the dimension)
        self.sin_dim = d_model // 2
        # Learned part (other half)
        self.octave_embed = nn.Embedding(12, d_model // 4)  # 12 octaves (0-11)
        self.pitch_class_embed = nn.Embedding(12, d_model // 4)  # 12 semitones
        # Projection
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, pitch: torch.Tensor) -> torch.Tensor:
        """pitch: [B, T] MIDI note numbers (0-127), 0 for rap/unvoiced."""
        B, T = pitch.shape
        device = pitch.device

        # Sinusoidal encoding of pitch (continuous representation)
        half = self.sin_dim // 2
        div_term = torch.exp(
            torch.arange(0, half, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / half)
        )
        pitch_float = pitch.float().unsqueeze(-1)  # [B, T, 1]
        sin_emb = torch.cat([
            torch.sin(pitch_float * div_term),
            torch.cos(pitch_float * div_term),
        ], dim=-1)  # [B, T, sin_dim]

        # Learned octave + pitch class
        octave = (pitch // 12).clamp(0, 11)
        pitch_class = pitch % 12
        oct_emb = self.octave_embed(octave)  # [B, T, d/4]
        pc_emb = self.pitch_class_embed(pitch_class)  # [B, T, d/4]

        # Concatenate and project
        combined = torch.cat([sin_emb, oct_emb, pc_emb], dim=-1)  # [B, T, d_model]
        return self.proj(combined)


class ScoreEncoder(nn.Module):
    """
    Conformer-based encoder for musical score features.

    Takes phoneme sequence with note-level annotations and produces
    contextualized hidden states at phoneme granularity.
    """

    def __init__(
        self,
        phoneme_vocab_size: int = 512,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        max_pitch: int = 128,
        pitch_embed_dim: int = 256,
        use_relative_pos: bool = True,
        n_techniques: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_techniques = n_techniques

        # Input embeddings
        self.phoneme_embed = nn.Embedding(phoneme_vocab_size, d_model, padding_idx=0)
        self.pitch_embed = PitchEmbedding(d_model, max_pitch)

        # Duration embedding (log-scale continuous)
        self.duration_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Singing ratio embedding (0=rap, 1=singing, continuous)
        self.singing_ratio_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
        )

        # Language ID embedding
        self.lang_embed = nn.Embedding(8, d_model)  # room for future languages

        # Technique embedding (Phase 3): per-phoneme binary flags for 6 singing techniques
        # (mix_voice, falsetto, breathy, pharyngeal, vibrato, glissando) from GTSinger.
        # Initialized to zero so loading pre-Phase-3 checkpoints via --init-from
        # produces identical behavior at step 0 (zero contribution from this path).
        # Training will learn to use these flags as tech_proj moves away from zero.
        self.tech_proj = nn.Linear(n_techniques, d_model)
        nn.init.zeros_(self.tech_proj.weight)
        nn.init.zeros_(self.tech_proj.bias)

        # Combine all input features
        self.input_proj = nn.Sequential(
            nn.Linear(d_model * 4, d_model),  # phone + pitch + dur + singing_ratio
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding
        self.pos_enc = SinusoidalPosEmb(d_model)

        # Conformer stack
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                use_relative_pos=use_relative_pos,
            )
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        phonemes: torch.Tensor,       # [B, T_phone] int
        pitch: torch.Tensor,           # [B, T_phone] int (MIDI)
        duration: torch.Tensor,        # [B, T_phone] int (frames)
        singing_ratio: torch.Tensor,   # [B] or [B, T_phone] float in [0,1]
        lang_id: Optional[torch.Tensor] = None,  # [B] int
        phone_mask: Optional[torch.Tensor] = None,  # [B, T_phone] bool
        tech: Optional[torch.Tensor] = None,  # [B, T_phone, n_techniques] float 0/1
    ) -> torch.Tensor:
        """
        Returns phone-level hidden states: [B, T_phone, d_model]
        """
        B, T = phonemes.shape

        # Embed each input
        phone_emb = self.phoneme_embed(phonemes)  # [B, T, d]
        pitch_emb = self.pitch_embed(pitch)  # [B, T, d]

        # Log-scale duration embedding
        dur_float = torch.log1p(duration.float()).unsqueeze(-1)  # [B, T, 1]
        dur_emb = self.duration_proj(dur_float)  # [B, T, d]

        # Singing ratio
        if singing_ratio.dim() == 1:
            singing_ratio = singing_ratio.unsqueeze(1).expand(-1, T)
        sr_emb = self.singing_ratio_proj(singing_ratio.unsqueeze(-1))  # [B, T, d]

        # Combine
        combined = torch.cat([phone_emb, pitch_emb, dur_emb, sr_emb], dim=-1)
        x = self.input_proj(combined)  # [B, T, d_model]

        # Add language embedding if provided
        if lang_id is not None:
            x = x + self.lang_embed(lang_id).unsqueeze(1)

        # Add technique embedding if provided (Phase 3)
        # tech is [B, T_phone, n_techniques], each value in {0.0, 1.0} marking
        # whether a specific GTSinger technique is active on that phoneme.
        # tech_proj is zero-initialized, so passing all-zeros (or None) produces
        # identical output to the pre-Phase-3 encoder — backwards compatible.
        if tech is not None:
            x = x + self.tech_proj(tech)  # [B, T, d_model]

        # Add positional encoding
        x = x + self.pos_enc(x)

        # Conformer stack
        for block in self.conformer_blocks:
            x = block(x, mask=phone_mask)

        return self.output_norm(x)
