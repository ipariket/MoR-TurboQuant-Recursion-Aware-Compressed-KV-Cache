"""
Full Mixture-of-Recursions Model with Compressed KV Cache.

Assembles:
    - Token embedding
    - Recursive transformer blocks (shared weights)
    - Adaptive router (decides early exit per token)
    - Recursion-aware KV cache (sparse allocation + compression)
    - Output head

The forward pass loop:
    for each recursion r in [0, N):
        1. Router scores all tokens → active_mask
        2. Apply transformer block to active tokens only
        3. Store KV only for active tokens (compressed)
        4. Residual update: h[active] = block(h[active]) + h[active]
        5. Tokens where active=False keep their frozen h
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from mor_tq.config import MoRConfig
from mor_tq.recursive_block import RecursiveTransformerBlock
from mor_tq.router import AdaptiveRouter
from mor_tq.kv_cache import RecursionAwareKVCache


@dataclass
class MoROutput:
    """Output from a MoR forward pass."""
    logits: torch.Tensor                  # (batch, seq_len, vocab_size)
    loss: Optional[torch.Tensor]          # language modeling loss if labels provided
    exit_depths: torch.Tensor             # (batch, seq_len) — recursion depth each token reached
    router_loss: torch.Tensor             # auxiliary load-balancing loss
    kv_stats: dict                        # memory statistics from the KV cache
    per_recursion_active: list[float]     # fraction of tokens active at each recursion


class MoRModel(nn.Module):
    """Mixture-of-Recursions language model with compressed KV cache.

    Architecture:
        [Embedding] → [Intro layers (unique)] → [Recursive core (shared Φ)] 
        → [Outro layers (unique)] → [LM Head]

    The recursive core applies the same weights N times, with the router
    deciding per-token early exit. KV cache only stores entries for tokens
    that are still active, compressed via PolarQuant.
    """

    def __init__(self, config: MoRConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Token + position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Build layer structure based on sharing strategy
        if config.sharing_strategy == "middle_cycle":
            # Unique intro layers
            self.intro_blocks = nn.ModuleList([
                RecursiveTransformerBlock(
                    config.d_model, config.n_heads, config.d_ff, config.dropout
                )
                for _ in range(config.n_unique_intro)
            ])
            # Shared recursive core (ONE block, applied multiple times)
            self.shared_block = RecursiveTransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            # Unique outro layers
            self.outro_blocks = nn.ModuleList([
                RecursiveTransformerBlock(
                    config.d_model, config.n_heads, config.d_ff, config.dropout
                )
                for _ in range(config.n_unique_outro)
            ])
        else:
            # Full sharing: one block for everything
            self.intro_blocks = nn.ModuleList()
            self.shared_block = RecursiveTransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            self.outro_blocks = nn.ModuleList()

        # Router
        self.router = AdaptiveRouter(
            d_model=config.d_model,
            strategy=config.routing_strategy,
            capacity_factor=config.capacity_factor,
            exit_threshold=config.exit_threshold,
        )

        # Recursion-aware KV cache with TurboQuant compression
        self.kv_cache = RecursionAwareKVCache(
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            n_recursions=config.n_recursions,
            kv_bits=config.kv_bits,
            group_size=config.kv_group_size,
            use_qjl=config.use_qjl,
        )

        # Output
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (embedding ↔ output head)
        self.lm_head.weight = self.token_emb.weight

        # Eval-time KV-compression simulation. When enabled, attention reads
        # K/V through compress->decompress so perplexity reflects the real
        # reconstruction error of the cache. Off by default (training is FP).
        self.quantize_kv_in_attn = False
        self._attn_compressor = (
            self.kv_cache.compressor if self.kv_cache.compress_enabled else None
        )

        self._init_weights()

    def set_kv_quant(self, enabled: bool):
        """Toggle the eval-time compressed-KV attention path.

        Returns the effective state (False if this model has no compressor).
        """
        self.quantize_kv_in_attn = bool(enabled) and self._attn_compressor is not None
        return self.quantize_kv_in_attn

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> MoROutput:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            labels: (batch, seq_len) target IDs for LM loss (optional)

        Returns:
            MoROutput with logits, loss, exit depth stats, and KV memory stats.
        """
        B, S = input_ids.shape
        device = input_ids.device
        config = self.config

        # Reset KV cache for this forward pass
        self.kv_cache.reset()

        # Handles for the optional compressed-KV attention path
        _kvc = self._attn_compressor
        _qkv = self.quantize_kv_in_attn

        # Embeddings
        positions = torch.arange(S, device=device).unsqueeze(0)
        h = self.token_emb(input_ids) + self.pos_emb(positions)
        h = self.emb_dropout(h)

        # Track per-token exit depth
        exit_depths = torch.zeros(B, S, dtype=torch.long, device=device)
        exited = torch.zeros(B, S, dtype=torch.bool, device=device)
        per_recursion_active = []
        total_router_loss = torch.tensor(0.0, device=device)

        recursion_idx = 0

        # === Phase 1: Unique intro layers (no routing — all tokens pass through) ===
        for block in self.intro_blocks:
            block_out, K, V = block(h, kv_compressor=_kvc, quantize_kv=_qkv)
            h = block_out  # residual is inside the block

            # Store KV for all tokens (no routing during intro)
            all_active = torch.ones(B, S, dtype=torch.bool, device=device)
            self.kv_cache.store(recursion_idx, K, V, all_active)
            exit_depths += 1

            per_recursion_active.append(1.0)
            recursion_idx += 1

        # === Phase 2: Shared recursive core (with routing) ===
        n_shared = config.n_shared_recursions
        for r in range(n_shared):
            # Router decides which tokens continue
            router_out = self.router(h, already_exited=exited)
            active_mask = router_out.active_mask

            frac_active = router_out.n_active / max(1, router_out.n_total)
            per_recursion_active.append(frac_active)

            # Load balance loss
            total_router_loss = total_router_loss + self.router.compute_load_balance_loss(
                router_out.router_scores
            )

            # Apply shared block to ALL tokens (simpler than sparse compute),
            # but only update active ones
            block_out, K, V = self.shared_block(h, kv_compressor=_kvc, quantize_kv=_qkv)

            # Residual update ONLY for active tokens
            # h[active] = block(h[active]) + h[active]  (residual)
            # h[exited] = h[exited]  (frozen)
            active_expanded = active_mask.unsqueeze(-1)  # (B, S, 1)
            h = torch.where(active_expanded, block_out, h)

            # Store KV only for active tokens (compressed)
            self.kv_cache.store(recursion_idx, K, V, active_mask)

            # Update exit tracking
            newly_exited = ~active_mask & ~exited
            exit_depths[newly_exited] = recursion_idx
            exited = exited | ~active_mask

            recursion_idx += 1

        # Tokens still active after all shared recursions get max depth
        exit_depths[~exited] = recursion_idx

        # === Phase 3: Unique outro layers (all tokens, no routing) ===
        for block in self.outro_blocks:
            block_out, K, V = block(h, kv_compressor=_kvc, quantize_kv=_qkv)
            h = block_out

            all_active = torch.ones(B, S, dtype=torch.bool, device=device)
            self.kv_cache.store(recursion_idx, K, V, all_active)

            per_recursion_active.append(1.0)
            recursion_idx += 1

        # Output
        h = self.final_norm(h)
        logits = self.lm_head(h)

        # Language modeling loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Average router loss
        avg_router_loss = total_router_loss / max(1, n_shared)

        return MoROutput(
            logits=logits,
            loss=loss,
            exit_depths=exit_depths,
            router_loss=avg_router_loss,
            kv_stats=self.kv_cache.memory_stats(),
            per_recursion_active=per_recursion_active,
        )

    def count_parameters(self) -> dict:
        """Compare parameter count vs equivalent standard transformer."""
        total = sum(p.numel() for p in self.parameters())

        # What a standard transformer with the same total depth would need
        equivalent_depth = self.config.n_recursions
        single_block_params = self.shared_block.count_parameters()
        standard_params = single_block_params * equivalent_depth

        # Our actual unique params (intro + 1 shared + outro + embeddings + head)
        embedding_params = sum(
            p.numel() for p in [*self.token_emb.parameters(), *self.pos_emb.parameters()]
        )
        intro_params = sum(b.count_parameters() for b in self.intro_blocks)
        outro_params = sum(b.count_parameters() for b in self.outro_blocks)
        shared_params = single_block_params  # counted once

        return {
            "total_params": total,
            "standard_equivalent_params": standard_params + embedding_params,
            "unique_block_params": intro_params + shared_params + outro_params,
            "shared_block_params": shared_params,
            "embedding_params": embedding_params,
            "parameter_savings": 1 - (total / (standard_params + embedding_params)),
        }
