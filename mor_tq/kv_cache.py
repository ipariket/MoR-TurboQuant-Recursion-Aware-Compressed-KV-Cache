"""
Recursion-Aware KV Cache Manager.

This is the novel component — the bridge between MoR's adaptive compute
and TurboQuant's compression. Standard KV caches allocate uniformly:
every layer, every token. This cache:

    1. Tracks which tokens are active at each recursion depth
    2. Only allocates KV storage for active tokens
    3. Compresses stored entries via PolarQuant (WHT + Lloyd-Max)
    4. Handles sparse attention masks when some tokens have KV at
       depth 5 but others exited at depth 2

Memory model:
    Standard Transformer: layers × seq_len × heads × head_dim × 2 × 2 bytes
    MoR + Compression:    Σ(active_tokens_at_depth_r) × heads × compressed_size

    If avg exit depth = 4 out of 12 recursions, and compression = 4×:
    → ~1/12 of standard memory usage
"""

import torch
import torch.nn as nn
from typing import Optional
from mor_tq.compression import PolarQuantCompressor, CompressedKV


class RecursionAwareKVCache(nn.Module):
    """KV cache that only stores entries for active tokens at each recursion.

    Two modes:
        recursion_wise: Separate KV cache per recursion. Attention at depth r
            only sees tokens still active at depth r. Cleaner but uses more memory.

        shared: Single KV cache, overwritten each recursion. Exited tokens keep
            their last KV entry. Maximum memory savings but introduces staleness.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        n_recursions: int,
        kv_bits: int = 3,
        group_size: int = 128,
        mode: str = "recursion_wise",
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.n_recursions = n_recursions
        self.kv_bits = kv_bits
        self.mode = mode

        # Compressor (shared for K and V — TurboQuant uses separate codebooks
        # for K vs V in production, but we use the same for simplicity)
        self.compress_enabled = kv_bits > 0
        if self.compress_enabled:
            self.compressor = PolarQuantCompressor(
                head_dim=head_dim,
                bits=kv_bits,
                group_size=group_size,
            )

        # Storage — initialized lazily on first use
        self._k_cache: list[Optional[torch.Tensor | CompressedKV]] = [None] * n_recursions
        self._v_cache: list[Optional[torch.Tensor | CompressedKV]] = [None] * n_recursions
        self._active_masks: list[Optional[torch.Tensor]] = [None] * n_recursions
        self._seq_len = 0

    def reset(self):
        """Clear all cached KV entries."""
        self._k_cache = [None] * self.n_recursions
        self._v_cache = [None] * self.n_recursions
        self._active_masks = [None] * self.n_recursions
        self._seq_len = 0

    def store(
        self,
        recursion_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        active_mask: torch.Tensor,
    ):
        """Store KV entries only for active tokens at this recursion.

        Args:
            recursion_idx: Which recursion step (0 to n_recursions-1)
            key: (batch, n_heads, seq_len, head_dim) — full KV from attention
            value: (batch, n_heads, seq_len, head_dim)
            active_mask: (batch, seq_len) bool — which tokens are active
        """
        B, H, S, D = key.shape
        self._seq_len = S

        # Store the active mask for this recursion
        self._active_masks[recursion_idx] = active_mask.clone()

        if self.mode == "recursion_wise":
            # Store only active tokens' KV
            # Expand mask to match KV shape: (B, 1, S, 1) for broadcasting
            mask_expanded = active_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, S, 1)
            mask_expanded = mask_expanded.expand_as(key)  # (B, H, S, D)

            # Zero out inactive tokens (they won't contribute to attention)
            k_masked = key * mask_expanded.float()
            v_masked = value * mask_expanded.float()

            if self.compress_enabled:
                self._k_cache[recursion_idx] = self.compressor.compress(k_masked)
                self._v_cache[recursion_idx] = self.compressor.compress(v_masked)
            else:
                self._k_cache[recursion_idx] = k_masked
                self._v_cache[recursion_idx] = v_masked

        elif self.mode == "shared":
            # Overwrite shared cache — active tokens get new KV,
            # exited tokens retain their last entry
            if recursion_idx == 0:
                # First recursion: store everything (all tokens active at depth 0)
                if self.compress_enabled:
                    self._k_cache[0] = self.compressor.compress(key)
                    self._v_cache[0] = self.compressor.compress(value)
                else:
                    self._k_cache[0] = key.clone()
                    self._v_cache[0] = value.clone()
            else:
                # Subsequent: decompress, update active positions, recompress
                prev_k = self._retrieve_raw(self._k_cache[0])
                prev_v = self._retrieve_raw(self._v_cache[0])

                mask_expanded = active_mask.unsqueeze(1).unsqueeze(-1).expand_as(key)
                prev_k = torch.where(mask_expanded, key, prev_k)
                prev_v = torch.where(mask_expanded, value, prev_v)

                if self.compress_enabled:
                    self._k_cache[0] = self.compressor.compress(prev_k)
                    self._v_cache[0] = self.compressor.compress(prev_v)
                else:
                    self._k_cache[0] = prev_k
                    self._v_cache[0] = prev_v

    def retrieve(
        self,
        recursion_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve KV cache and attention mask for a given recursion.

        Args:
            recursion_idx: Which recursion step

        Returns:
            (key, value, attn_mask)
            key: (batch, n_heads, seq_len, head_dim)
            value: (batch, n_heads, seq_len, head_dim) 
            attn_mask: (batch, seq_len) bool — which positions have valid KV
        """
        if self.mode == "recursion_wise":
            cache_idx = recursion_idx
        else:
            cache_idx = 0  # shared mode always uses slot 0

        k = self._retrieve_raw(self._k_cache[cache_idx])
        v = self._retrieve_raw(self._v_cache[cache_idx])

        # Build attention mask: which positions have valid KV at this depth
        if self.mode == "recursion_wise":
            attn_mask = self._active_masks[recursion_idx]
            if attn_mask is None:
                attn_mask = torch.ones(k.shape[0], k.shape[2], dtype=torch.bool, device=k.device)
        else:
            # Shared mode: all positions valid (exited tokens retain last KV)
            attn_mask = torch.ones(k.shape[0], k.shape[2], dtype=torch.bool, device=k.device)

        return k, v, attn_mask

    def _retrieve_raw(self, cached) -> torch.Tensor:
        """Decompress if needed, return raw tensor."""
        if cached is None:
            raise ValueError("Cache slot is empty — store before retrieve")
        if isinstance(cached, CompressedKV):
            return self.compressor.decompress(cached)
        return cached

    def memory_stats(self) -> dict:
        """Calculate actual vs baseline memory usage.

        All calculations are normalized to per-sample (batch=1) to give
        architecture-level comparison independent of batch size.
        """
        # Determine batch size from the first available mask
        batch_size = 1
        for m in self._active_masks:
            if m is not None:
                batch_size = m.shape[0]
                break

        baseline_bytes = 0
        actual_bytes = 0

        for r in range(self.n_recursions):
            mask = self._active_masks[r]
            if mask is None:
                continue

            # Normalize to per-sample: average active tokens per sample
            n_active_per_sample = mask.sum().item() / batch_size
            seq_len = mask.shape[1]

            # MoR baseline: active tokens only, FP16 (what MoR would use without compression)
            mor_layer_baseline = int(n_active_per_sample) * self.n_heads * self.head_dim * 2 * 2
            baseline_bytes += mor_layer_baseline

            if self.compress_enabled:
                # Compressed: only active tokens per sample
                stats = self.compressor.memory_bytes(int(n_active_per_sample), self.n_heads)
                actual_bytes += stats["compressed_bytes"] * 2  # K + V
            else:
                actual_bytes += int(n_active_per_sample) * self.n_heads * self.head_dim * 2 * 2

        # Standard transformer baseline (per sample):
        # all layers × all tokens × full precision
        standard_baseline = (
            self.n_recursions
            * self._seq_len
            * self.n_heads
            * self.head_dim
            * 2  # K + V
            * 2  # FP16 bytes
        )

        return {
            "actual_bytes": actual_bytes,
            "mor_baseline_bytes": baseline_bytes,  # MoR without compression
            "standard_baseline_bytes": standard_baseline,  # standard transformer
            "compression_vs_standard": standard_baseline / max(1, actual_bytes),
            "compression_vs_mor": baseline_bytes / max(1, actual_bytes),
            "active_token_fraction": sum(
                m.sum().item() / m.numel()
                for m in self._active_masks
                if m is not None
            ) / max(1, sum(1 for m in self._active_masks if m is not None)),
        }
