"""
Recursive Transformer Block.

A single transformer layer (attention + FFN) that gets applied multiple times
with the SAME weights. This is the "recursion" in Mixture of Recursions.

Standard transformer: 32 layers × 32 unique (Φ₁, Φ₂, ..., Φ₃₂)
MoR recursive block: 1 shared Φ applied 12 times: f(f(f(...f(h; Φ)...; Φ); Φ); Φ)

The residual connection at each step is critical:
    h_{r+1} = f(h_r; Φ) + h_r

Without it, repeatedly applying the same transform would collapse to a fixed point.
With it, each recursion REFINES the representation incrementally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with KV output for caching."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_compressor=None,
        quantize_kv: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: (batch, seq_len) bool — which KV positions are valid
            kv_compressor: optional TurboQuantCompressor. When provided together
                with quantize_kv=True, K and V are passed through
                compress->decompress before attention, so the quantization error
                of the KV cache actually enters the model output. This is how the
                inference-time quality cost of compression is measured.
            quantize_kv: enable the compress/decompress simulation above.

        Returns:
            (output, keys, values)
            output: (batch, seq_len, d_model)
            keys: (batch, n_heads, seq_len, head_dim)   — pre-quantization (for cache accounting)
            values: (batch, n_heads, seq_len, head_dim)
        """
        B, S, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        Q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)
        K = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2)

        # Simulate a compressed KV cache: quantize then dequantize the K/V that
        # attention reads. This injects the real reconstruction error into the
        # forward pass so perplexity reflects the cost of compression.
        # K_raw/V_raw are still returned for the cache's byte accounting.
        K_attn, V_attn = K, V
        if quantize_kv and kv_compressor is not None:
            K_attn = kv_compressor.decompress(kv_compressor.compress(K)).to(Q.dtype)
            V_attn = kv_compressor.decompress(kv_compressor.compress(V)).to(Q.dtype)

        K, V = K_attn, V_attn

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, S, S)

        # Causal mask (lower triangular)
        causal = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply KV validity mask (from recursion-aware cache)
        if attn_mask is not None:
            # attn_mask: (B, S) → (B, 1, 1, S) — mask invalid KV positions
            kv_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            scores = scores.masked_fill(~kv_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)  # (B, H, S, Dh)
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(output)

        return output, K, V


class FeedForward(nn.Module):
    """Standard FFN with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class RecursiveTransformerBlock(nn.Module):
    """One transformer block = Attention + FFN + LayerNorms.

    This block is designed to be applied recursively. The same instance
    gets called N times during a forward pass, with a residual connection
    between each application:

        h₁ = block(h₀) + h₀
        h₂ = block(h₁) + h₁
        ...
        hN = block(h_{N-1}) + h_{N-1}

    The residual is handled externally (in the model's recursion loop),
    not inside this module, so it can be gated by the router.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_compressor=None,
        quantize_kv: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: (batch, seq_len) bool — valid KV positions for attention
            kv_compressor / quantize_kv: forwarded to attention to optionally
                simulate a compressed KV cache (see MultiHeadAttention.forward).

        Returns:
            (output, keys, values)
            output: (batch, seq_len, d_model) — the transformed hidden state
            keys: (batch, n_heads, seq_len, head_dim) — for KV cache
            values: (batch, n_heads, seq_len, head_dim)
        """
        # Pre-norm attention (GPT-style)
        normed = self.norm1(x)
        attn_out, K, V = self.attention(
            normed, attn_mask, kv_compressor=kv_compressor, quantize_kv=quantize_kv
        )
        x_attn = x + self.dropout(attn_out)

        # Pre-norm FFN
        normed2 = self.norm2(x_attn)
        ffn_out = self.ffn(normed2)
        output = x_attn + self.dropout(ffn_out)

        return output, K, V

    def count_parameters(self) -> int:
        """Count trainable parameters in this block."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
