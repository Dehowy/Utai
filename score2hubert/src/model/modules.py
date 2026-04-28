"""
Shared neural network modules for Score2HuBERT.

Includes:
- Sinusoidal positional encoding (absolute + relative)
- Multi-head self-attention with relative positional bias
- Conformer block (FFN-Attention-Conv-FFN sandwich)
- Adaptive Layer Normalization (AdaLN, for flow matching time conditioning)
- Conv1D stack (for variance/duration predictors)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for continuous values (time steps, pitch, etc.)."""

    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] or [B, T, D] — adds positional encoding along T dimension."""
        if x.dim() == 2:
            return self.pe[:, : x.size(1), :]
        return x + self.pe[:, : x.size(1), :]


class ContinuousSinusoidalEmb(nn.Module):
    """Sinusoidal embedding for continuous scalar values (e.g., diffusion time t)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        emb = math.log(10000.0) / (self.half_dim - 1)
        self.register_buffer("emb_coeff", torch.exp(-emb * torch.arange(self.half_dim)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] scalar values in [0, 1] → [B, dim]."""
        emb = t.unsqueeze(-1) * self.emb_coeff.unsqueeze(0) * 1000.0
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding via learnable bias in attention."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe_k = nn.Embedding(2 * max_len + 1, d_model)

    def forward(self, length: int) -> torch.Tensor:
        """Returns relative position embeddings: [T, T, d_model]."""
        positions = torch.arange(length, device=self.pe_k.weight.device)
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # [T, T]
        relative_pos = relative_pos.clamp(-self.max_len, self.max_len) + self.max_len
        return self.pe_k(relative_pos)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional relative positional bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_relative_pos: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.use_relative_pos = use_relative_pos
        if use_relative_pos:
            self.rel_pos = RelativePositionalEncoding(self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, D] query input
        mask: [B, T] bool mask (True=keep, False=pad)
        kv: [B, S, D] optional cross-attention key/value (if None, self-attention)
        """
        B, T, _ = x.shape
        kv_input = kv if kv is not None else x
        S = kv_input.size(1)

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # [B, heads, T, S]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Relative positional bias (self-attention only)
        if self.use_relative_pos and kv is None:
            rel_emb = self.rel_pos(T)  # [T, T, d_k]
            # q: [B, heads, T, d_k], rel_emb: [T, T, d_k]
            rel_attn = torch.einsum("bhid,ijd->bhij", q, rel_emb)
            attn = attn + rel_attn * self.scale

        if mask is not None:
            if mask.dim() == 2:  # [B, S] padding mask
                mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
                attn = attn.masked_fill(~mask_expanded, float("-inf"))
            elif mask.dim() == 3:  # [B, T, S] custom mask
                attn = attn.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, heads, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class ConvModule(nn.Module):
    """Conformer convolution module: pointwise → GLU → depthwise → BN → Swish → pointwise."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pointwise1(x)  # [B, 2D, T]
        x = F.glu(x, dim=1)  # [B, D, T]
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)  # [B, T, D]


class FeedForward(nn.Module):
    """Position-wise feed-forward with SwiGLU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class ConformerBlock(nn.Module):
    """
    Conformer block: FFN(1/2) → MHSA → Conv → FFN(1/2)
    Each sub-layer has pre-norm and residual connection.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 conv_kernel_size: int = 31, dropout: float = 0.1,
                 use_relative_pos: bool = True):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_relative_pos)
        self.conv = ConvModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization — conditions normalization on external signal (e.g., time step)."""

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D], cond: [B, D_cond] or [B, T, D_cond]"""
        params = self.proj(cond)
        if params.dim() == 2:
            params = params.unsqueeze(1)
        scale, shift = params.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with AdaLN conditioning.
    Used in the flow matching decoder's vector field network.

    Structure: AdaLN → Self-Attention → AdaLN → Cross-Attention → AdaLN → FFN
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 d_cond: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention
        self.adaln_self = AdaLN(d_model, d_cond)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # Cross-attention (condition from encoder)
        self.adaln_cross = AdaLN(d_model, d_cond)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # Feed-forward
        self.adaln_ff = AdaLN(d_model, d_cond)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                time_emb: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, D] noisy features
        cond: [B, T, D_enc] encoder condition (frame-level)
        time_emb: [B, D_cond] time step embedding
        mask: [B, T] padding mask
        """
        # Self-attention
        h = self.adaln_self(x, time_emb)
        x = x + self.self_attn(h, mask=mask)

        # Cross-attention with encoder condition
        h = self.adaln_cross(x, time_emb)
        x = x + self.cross_attn(h, mask=mask, kv=cond)

        # Feed-forward
        h = self.adaln_ff(x, time_emb)
        x = x + self.ff(h)

        return x


class Conv1DStack(nn.Module):
    """Stack of Conv1D layers for duration/variance prediction."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Conv1d(d_in, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
        self.conv_stack = nn.ModuleList(layers)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, T, D] → [B, T, out_dim]"""
        h = x.transpose(1, 2)  # [B, D, T]
        for i in range(0, len(self.conv_stack), 4):
            conv = self.conv_stack[i]
            relu = self.conv_stack[i + 1]
            norm = self.conv_stack[i + 2]
            drop = self.conv_stack[i + 3]
            h = conv(h)
            h = relu(h)
            h = h.transpose(1, 2)  # [B, T, D] for LayerNorm
            h = norm(h)
            h = drop(h)
            h = h.transpose(1, 2)  # [B, D, T] for next Conv
        h = h.transpose(1, 2)  # [B, T, D]
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        return self.out_proj(h)


class LengthRegulator(nn.Module):
    """Expand phoneme-level features to frame-level using given durations."""

    def forward(self, x: torch.Tensor, durations: torch.Tensor,
                max_len: Optional[int] = None) -> torch.Tensor:
        """
        x: [B, T_phone, D]
        durations: [B, T_phone] integer frame counts per phoneme
        Returns: [B, T_frame, D]
        """
        outputs = []
        for i in range(x.size(0)):
            expanded = torch.repeat_interleave(x[i], durations[i], dim=0)
            outputs.append(expanded)

        if max_len is None:
            max_len = max(o.size(0) for o in outputs)

        padded = torch.zeros(x.size(0), max_len, x.size(2), device=x.device, dtype=x.dtype)
        for i, o in enumerate(outputs):
            length = min(o.size(0), max_len)
            padded[i, :length] = o[:length]

        return padded
