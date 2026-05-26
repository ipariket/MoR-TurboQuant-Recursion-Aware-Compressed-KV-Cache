"""Tests for MoR-TurboQuant module."""

import torch
import pytest
from mor_tq import (
    MoRConfig,
    MoRModel,
    AdaptiveRouter,
    RecursionAwareKVCache,
    PolarQuantCompressor,
    RecursiveTransformerBlock,
)


# ============================================================
# Compression tests
# ============================================================

class TestPolarQuantCompressor:
    def setup_method(self):
        self.compressor = PolarQuantCompressor(head_dim=64, bits=3, group_size=64)

    def test_compress_decompress_roundtrip(self):
        """Compression → decompression should approximately reconstruct."""
        x = torch.randn(4, 8, 32, 64)  # (batch, heads, seq, head_dim)
        compressed = self.compressor.compress(x)
        reconstructed = self.compressor.decompress(compressed)

        assert reconstructed.shape == x.shape
        # Check relative reconstruction quality (cosine similarity per vector)
        x_flat = x.reshape(-1, 64)
        r_flat = reconstructed.reshape(-1, 64)
        cos_sim = torch.nn.functional.cosine_similarity(x_flat, r_flat, dim=-1).mean()
        assert cos_sim > 0.5, f"Cosine similarity too low: {cos_sim:.3f}"

    def test_3bit_packing_roundtrip(self):
        """3-bit pack/unpack should be lossless."""
        indices = torch.randint(0, 8, (1024,), dtype=torch.uint8)
        packed = self.compressor._pack_indices(indices.unsqueeze(0))
        unpacked = self.compressor._unpack_indices(packed, 1024)
        assert torch.equal(indices, unpacked)

    def test_4bit_packing_roundtrip(self):
        comp4 = PolarQuantCompressor(head_dim=64, bits=4, group_size=64)
        indices = torch.randint(0, 16, (1024,), dtype=torch.uint8)
        packed = comp4._pack_indices(indices.unsqueeze(0))
        unpacked = comp4._unpack_indices(packed, 1024)
        assert torch.equal(indices, unpacked)

    def test_compression_ratio(self):
        ratio = self.compressor.compression_ratio()
        # 3-bit with fp32 norms per 64-element group:
        # effective bits = 3 + 32/64 = 3.5 → ratio ≈ 16/3.5 ≈ 4.57
        assert ratio > 4.0
        assert ratio < 6.0

    def test_memory_bytes(self):
        stats = self.compressor.memory_bytes(n_tokens=1024, n_heads=8)
        assert stats["compressed_bytes"] < stats["fp16_bytes"]
        assert stats["ratio"] > 3.5


# ============================================================
# Router tests
# ============================================================

class TestAdaptiveRouter:
    def test_token_choice_basic(self):
        router = AdaptiveRouter(d_model=64, strategy="token", exit_threshold=0.5)
        h = torch.randn(2, 16, 64)
        out = router(h)

        assert out.active_mask.shape == (2, 16)
        assert out.active_mask.dtype == torch.bool
        assert 0 < out.n_active <= 32  # some should be active, some not

    def test_expert_choice_respects_capacity(self):
        router = AdaptiveRouter(d_model=64, strategy="expert", capacity_factor=0.5)
        h = torch.randn(1, 100, 64)
        out = router(h)

        # Should select approximately 50% of tokens
        expected = 50
        assert abs(out.n_active - expected) <= 2  # allow small rounding

    def test_already_exited_tokens_stay_exited(self):
        router = AdaptiveRouter(d_model=64, strategy="expert", capacity_factor=0.5)
        h = torch.randn(1, 20, 64)

        # Mark half as already exited
        already_exited = torch.zeros(1, 20, dtype=torch.bool)
        already_exited[0, :10] = True

        out = router(h, already_exited=already_exited)

        # None of the already-exited tokens should be active
        assert not out.active_mask[0, :10].any()

    def test_load_balance_loss(self):
        router = AdaptiveRouter(d_model=64, strategy="expert", capacity_factor=0.5)
        # Uniform scores near 0.5 → low loss
        scores_balanced = torch.full((1, 100), 0.5)
        loss_balanced = router.compute_load_balance_loss(scores_balanced)

        # All scores at 1.0 → high loss (all continue, far from 0.5 target)
        scores_imbalanced = torch.ones(1, 100)
        loss_imbalanced = router.compute_load_balance_loss(scores_imbalanced)

        assert loss_balanced < loss_imbalanced


# ============================================================
# KV Cache tests
# ============================================================

class TestRecursionAwareKVCache:
    def test_store_and_retrieve(self):
        cache = RecursionAwareKVCache(
            n_heads=4, head_dim=64, max_seq_len=32,
            n_recursions=4, kv_bits=0,  # no compression for basic test
        )
        K = torch.randn(1, 4, 32, 64)
        V = torch.randn(1, 4, 32, 64)
        mask = torch.ones(1, 32, dtype=torch.bool)

        cache.store(0, K, V, mask)
        k_out, v_out, attn_mask = cache.retrieve(0)

        assert k_out.shape == K.shape
        assert attn_mask.all()

    def test_sparse_storage_with_compression(self):
        cache = RecursionAwareKVCache(
            n_heads=4, head_dim=64, max_seq_len=32,
            n_recursions=4, kv_bits=3,
        )
        K = torch.randn(1, 4, 32, 64)
        V = torch.randn(1, 4, 32, 64)

        # Only half the tokens are active
        mask = torch.zeros(1, 32, dtype=torch.bool)
        mask[0, :16] = True

        cache.store(0, K, V, mask)
        k_out, v_out, attn_mask = cache.retrieve(0)

        # Inactive positions should be zeroed
        assert k_out[:, :, 16:, :].abs().sum() < 1e-3

    def test_memory_stats(self):
        cache = RecursionAwareKVCache(
            n_heads=8, head_dim=64, max_seq_len=128,
            n_recursions=8, kv_bits=3,
        )
        # Simulate decreasing active tokens across recursions
        for r in range(8):
            K = torch.randn(1, 8, 128, 64)
            V = torch.randn(1, 8, 128, 64)
            n_active = max(16, 128 - r * 16)
            mask = torch.zeros(1, 128, dtype=torch.bool)
            mask[0, :n_active] = True
            cache.store(r, K, V, mask)

        stats = cache.memory_stats()
        assert stats["compression_vs_standard"] > 1.0
        assert stats["active_token_fraction"] < 1.0


# ============================================================
# Full model tests
# ============================================================

class TestMoRModel:
    def setup_method(self):
        self.config = MoRConfig(
            d_model=128,
            n_heads=4,
            d_ff=256,
            n_recursions=6,
            capacity_factor=0.5,
            routing_strategy="expert",
            kv_bits=3,
            vocab_size=1000,
            max_seq_len=64,
            dropout=0.0,
            n_unique_intro=1,
            n_unique_outro=1,
        )
        self.model = MoRModel(self.config)

    def test_forward_shape(self):
        input_ids = torch.randint(0, 1000, (2, 32))
        output = self.model(input_ids)

        assert output.logits.shape == (2, 32, 1000)
        assert output.exit_depths.shape == (2, 32)

    def test_forward_with_labels(self):
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))
        output = self.model(input_ids, labels=labels)

        assert output.loss is not None
        assert output.loss.item() > 0

    def test_early_exit_happening(self):
        """Verify that tokens actually exit at different depths."""
        input_ids = torch.randint(0, 1000, (1, 64))
        output = self.model(input_ids)

        depths = output.exit_depths[0]
        # Should have variation in exit depths (not all same)
        assert depths.unique().numel() > 1, (
            f"All tokens exited at same depth: {depths.unique()}"
        )

    def test_kv_memory_savings(self):
        """KV cache should use less memory than standard transformer baseline."""
        input_ids = torch.randint(0, 1000, (1, 64))
        output = self.model(input_ids)

        stats = output.kv_stats
        assert stats["compression_vs_standard"] > 1.5, (
            f"Expected significant savings, got ratio {stats['compression_vs_standard']:.2f}"
        )

    def test_parameter_savings(self):
        """MoR should use fewer unique parameters than equivalent standard transformer."""
        param_stats = self.model.count_parameters()
        assert param_stats["parameter_savings"] > 0.3, (
            f"Expected >30% parameter savings, got {param_stats['parameter_savings']:.1%}"
        )

    def test_per_recursion_active_decreases(self):
        """Active token fraction should generally decrease across recursions."""
        input_ids = torch.randint(0, 1000, (1, 64))  # must be <= max_seq_len
        output = self.model(input_ids)

        active = output.per_recursion_active
        # Intro layer is always 1.0, then shared recursions should decrease
        # (not strictly monotonic, but trend should be downward)
        shared_active = active[1:-1]  # skip intro and outro
        if len(shared_active) >= 2:
            assert shared_active[-1] <= shared_active[0] + 0.1


# ============================================================
# Integration test
# ============================================================

class TestEndToEnd:
    def test_gradient_flow(self):
        """Verify gradients flow through the entire model including router."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=4, vocab_size=100,
            max_seq_len=32, kv_bits=0,
            dropout=0.0,
        )
        model = MoRModel(config)

        input_ids = torch.randint(0, 100, (1, 16))
        labels = torch.randint(0, 100, (1, 16))

        output = model(input_ids, labels=labels)
        total_loss = output.loss + 0.01 * output.router_loss
        total_loss.backward()

        # Router should have gradients
        assert model.router.gate.weight.grad is not None
        assert model.router.gate.weight.grad.abs().sum() > 0

        # Shared block should have gradients
        for p in model.shared_block.parameters():
            if p.requires_grad:
                assert p.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
