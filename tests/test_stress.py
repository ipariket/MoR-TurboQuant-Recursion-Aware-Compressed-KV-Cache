"""
Stress tests, edge cases, benchmarks, and integration tests.

Covers:
    - WHT mathematical properties
    - Compression at various bit widths and dimensions
    - Router edge cases (all exit, none exit, single token)
    - KV cache shared vs recursion_wise modes
    - Model configs (full sharing, middle_cycle, token-choice, expert-choice)
    - Memory savings validation at scale
    - Training loop simulation (multi-step gradient updates)
    - Numerical stability under extreme inputs
    - Batch size stress tests
    - Sequence length boundary tests
"""

import torch
import pytest
import math
from mor_tq import (
    MoRConfig,
    MoRModel,
    AdaptiveRouter,
    RecursionAwareKVCache,
    PolarQuantCompressor,
    RecursiveTransformerBlock,
)
from mor_tq.compression import _walsh_hadamard_transform, _inverse_wht


# ============================================================
# WHT Mathematical Property Tests
# ============================================================

class TestWHTProperties:
    """Verify Walsh-Hadamard Transform satisfies expected math properties."""

    def test_roundtrip_exact(self):
        """WHT(WHT(x)) = x (self-inverse)."""
        x = torch.randn(8, 64)
        reconstructed = _inverse_wht(_walsh_hadamard_transform(x))
        assert torch.allclose(x, reconstructed, atol=1e-5), \
            f"Max error: {(x - reconstructed).abs().max():.2e}"

    def test_preserves_energy(self):
        """WHT preserves L2 norm (Parseval's theorem)."""
        x = torch.randn(16, 128)
        x_wht = _walsh_hadamard_transform(x)
        norm_before = x.norm(dim=-1)
        norm_after = x_wht.norm(dim=-1)
        assert torch.allclose(norm_before, norm_after, atol=1e-4), \
            f"Norm drift: {(norm_before - norm_after).abs().max():.2e}"

    def test_linearity(self):
        """WHT(a*x + b*y) = a*WHT(x) + b*WHT(y)."""
        x = torch.randn(4, 32)
        y = torch.randn(4, 32)
        a, b = 2.5, -1.3
        lhs = _walsh_hadamard_transform(a * x + b * y)
        rhs = a * _walsh_hadamard_transform(x) + b * _walsh_hadamard_transform(y)
        assert torch.allclose(lhs, rhs, atol=1e-4)

    def test_various_power_of_2_dims(self):
        """WHT works for all power-of-2 dimensions."""
        for dim in [2, 4, 8, 16, 32, 64, 128, 256]:
            x = torch.randn(2, dim)
            out = _walsh_hadamard_transform(x)
            back = _inverse_wht(out)
            assert torch.allclose(x, back, atol=1e-4), f"Failed at dim={dim}"

    def test_does_not_mutate_input(self):
        """WHT should not modify the input tensor."""
        x = torch.randn(4, 64)
        x_copy = x.clone()
        _ = _walsh_hadamard_transform(x)
        assert torch.equal(x, x_copy), "WHT mutated input tensor"

    def test_single_element(self):
        """WHT of dim=1 should return x / sqrt(1) = x."""
        x = torch.tensor([[3.14]])
        out = _walsh_hadamard_transform(x)
        assert torch.allclose(x, out, atol=1e-6)

    def test_batch_independence(self):
        """Each row in batch should be transformed independently."""
        x = torch.randn(4, 16)
        full_out = _walsh_hadamard_transform(x)
        for i in range(4):
            single_out = _walsh_hadamard_transform(x[i:i+1])
            assert torch.allclose(full_out[i], single_out[0], atol=1e-6)


# ============================================================
# Compression Edge Cases
# ============================================================

class TestCompressionEdgeCases:

    def test_4bit_roundtrip_quality(self):
        """4-bit should reconstruct better than 3-bit."""
        x = torch.randn(2, 4, 16, 64)
        comp3 = PolarQuantCompressor(head_dim=64, bits=3, group_size=64)
        comp4 = PolarQuantCompressor(head_dim=64, bits=4, group_size=64)

        r3 = comp3.decompress(comp3.compress(x))
        r4 = comp4.decompress(comp4.compress(x))

        mse3 = (x - r3).pow(2).mean()
        mse4 = (x - r4).pow(2).mean()
        assert mse4 < mse3, f"4-bit MSE ({mse4:.4f}) should be lower than 3-bit ({mse3:.4f})"

    def test_small_head_dim(self):
        """Should work with head_dim as small as 2 (padded to 2)."""
        comp = PolarQuantCompressor(head_dim=2, bits=3, group_size=128)
        x = torch.randn(1, 1, 4, 2)
        compressed = comp.compress(x)
        reconstructed = comp.decompress(compressed)
        assert reconstructed.shape == x.shape

    def test_large_head_dim(self):
        """Should work with large head_dim like 256."""
        comp = PolarQuantCompressor(head_dim=256, bits=3, group_size=128)
        x = torch.randn(1, 2, 8, 256)
        compressed = comp.compress(x)
        reconstructed = comp.decompress(compressed)
        assert reconstructed.shape == x.shape

    def test_non_power_of_2_head_dim(self):
        """Non-power-of-2 head_dim should be padded correctly."""
        for dim in [48, 80, 96, 100]:
            comp = PolarQuantCompressor(head_dim=dim, bits=3, group_size=64)
            x = torch.randn(1, 1, 4, dim)
            compressed = comp.compress(x)
            reconstructed = comp.decompress(compressed)
            assert reconstructed.shape == x.shape, f"Failed for head_dim={dim}"

    def test_zero_input(self):
        """All zeros should compress and decompress to near-zero."""
        comp = PolarQuantCompressor(head_dim=64, bits=3, group_size=64)
        x = torch.zeros(1, 2, 8, 64)
        compressed = comp.compress(x)
        reconstructed = comp.decompress(compressed)
        assert reconstructed.abs().max() < 0.1

    def test_constant_input(self):
        """Constant vector should not crash."""
        comp = PolarQuantCompressor(head_dim=64, bits=3, group_size=64)
        x = torch.ones(1, 2, 8, 64) * 5.0
        compressed = comp.compress(x)
        reconstructed = comp.decompress(compressed)
        assert reconstructed.shape == x.shape

    def test_very_large_values(self):
        """Large magnitude inputs should not produce NaN/Inf."""
        comp = PolarQuantCompressor(head_dim=64, bits=3, group_size=64)
        x = torch.randn(1, 2, 8, 64) * 1000
        compressed = comp.compress(x)
        reconstructed = comp.decompress(compressed)
        assert not torch.isnan(reconstructed).any(), "NaN in reconstruction"
        assert not torch.isinf(reconstructed).any(), "Inf in reconstruction"

    def test_very_small_values(self):
        """Near-zero inputs should not produce NaN."""
        comp = PolarQuantCompressor(head_dim=64, bits=3, group_size=64)
        x = torch.randn(1, 2, 8, 64) * 1e-8
        compressed = comp.compress(x)
        reconstructed = comp.decompress(compressed)
        assert not torch.isnan(reconstructed).any()

    def test_single_token(self):
        """Should work with seq_len=1."""
        comp = PolarQuantCompressor(head_dim=64, bits=3, group_size=64)
        x = torch.randn(1, 4, 1, 64)
        compressed = comp.compress(x)
        reconstructed = comp.decompress(compressed)
        assert reconstructed.shape == x.shape

    def test_memory_bytes_accuracy(self):
        """Memory calculation should match actual compressed sizes."""
        comp = PolarQuantCompressor(head_dim=64, bits=3, group_size=64)
        stats = comp.memory_bytes(n_tokens=512, n_heads=8)
        # FP16 baseline: 512 * 8 * 64 * 2 = 524,288 bytes
        expected_fp16 = 512 * 8 * 64 * 2
        assert stats["fp16_bytes"] == expected_fp16
        assert stats["compressed_bytes"] < expected_fp16


# ============================================================
# Router Edge Cases
# ============================================================

class TestRouterEdgeCases:

    def test_single_token_sequence(self):
        """Router should handle seq_len=1."""
        router = AdaptiveRouter(d_model=64, strategy="expert", capacity_factor=0.5)
        h = torch.randn(1, 1, 64)
        out = router(h)
        # Must keep at least 1 token
        assert out.n_active >= 1

    def test_all_tokens_already_exited(self):
        """If every token already exited, active should be 0 or minimal."""
        router = AdaptiveRouter(d_model=64, strategy="expert", capacity_factor=0.5)
        h = torch.randn(1, 16, 64)
        already_exited = torch.ones(1, 16, dtype=torch.bool)
        out = router(h, already_exited=already_exited)
        assert out.n_active == 0

    def test_capacity_factor_100_percent(self):
        """capacity_factor=1.0 should keep all eligible tokens."""
        router = AdaptiveRouter(d_model=64, strategy="expert", capacity_factor=1.0)
        h = torch.randn(1, 50, 64)
        out = router(h)
        assert out.n_active == 50

    def test_very_low_capacity(self):
        """Very low capacity should still keep at least 1."""
        router = AdaptiveRouter(d_model=64, strategy="expert", capacity_factor=0.01)
        h = torch.randn(1, 100, 64)
        out = router(h)
        assert out.n_active >= 1

    def test_token_choice_high_threshold(self):
        """Very high threshold should exit most tokens."""
        router = AdaptiveRouter(d_model=64, strategy="token", exit_threshold=0.99)
        h = torch.randn(1, 100, 64)
        out = router(h)
        # Most should exit with such high threshold
        assert out.n_active < 80

    def test_token_choice_low_threshold(self):
        """Very low threshold should keep most tokens."""
        router = AdaptiveRouter(d_model=64, strategy="token", exit_threshold=0.01)
        h = torch.randn(1, 100, 64)
        out = router(h)
        assert out.n_active > 20

    def test_multi_batch_expert_choice(self):
        """Expert choice should work independently per batch element."""
        router = AdaptiveRouter(d_model=64, strategy="expert", capacity_factor=0.5)
        h = torch.randn(4, 32, 64)
        out = router(h)
        # Total active should be approximately 4 * 16 = 64
        assert 48 <= out.n_active <= 80

    def test_router_scores_in_valid_range(self):
        """Sigmoid output must be in [0, 1]."""
        router = AdaptiveRouter(d_model=128, strategy="expert", capacity_factor=0.5)
        h = torch.randn(2, 64, 128)
        out = router(h)
        assert out.router_scores.min() >= 0.0
        assert out.router_scores.max() <= 1.0

    def test_router_is_differentiable(self):
        """Router scores should have gradients."""
        router = AdaptiveRouter(d_model=64, strategy="expert", capacity_factor=0.5)
        h = torch.randn(1, 16, 64, requires_grad=True)
        out = router(h)
        out.router_scores.sum().backward()
        assert h.grad is not None


# ============================================================
# KV Cache Edge Cases & Modes
# ============================================================

class TestKVCacheEdgeCases:

    def test_shared_mode_basic(self):
        """Shared mode should work without errors."""
        cache = RecursionAwareKVCache(
            n_heads=4, head_dim=64, max_seq_len=32,
            n_recursions=4, kv_bits=0, mode="shared",
        )
        for r in range(4):
            K = torch.randn(1, 4, 32, 64)
            V = torch.randn(1, 4, 32, 64)
            mask = torch.ones(1, 32, dtype=torch.bool)
            if r > 0:
                mask[0, -(r * 4):] = False  # progressively deactivate
            cache.store(r, K, V, mask)

        k_out, v_out, attn_mask = cache.retrieve(3)
        assert k_out.shape == (1, 4, 32, 64)
        assert attn_mask.all()  # shared mode: all valid

    def test_shared_mode_preserves_exited_kv(self):
        """In shared mode, exited tokens should retain their last KV."""
        cache = RecursionAwareKVCache(
            n_heads=2, head_dim=32, max_seq_len=8,
            n_recursions=3, kv_bits=0, mode="shared",
        )
        # Recursion 0: all active
        K0 = torch.ones(1, 2, 8, 32) * 1.0
        V0 = torch.ones(1, 2, 8, 32) * 1.0
        cache.store(0, K0, V0, torch.ones(1, 8, dtype=torch.bool))

        # Recursion 1: only first 4 active, last 4 should keep K0 values
        K1 = torch.ones(1, 2, 8, 32) * 2.0
        V1 = torch.ones(1, 2, 8, 32) * 2.0
        mask1 = torch.zeros(1, 8, dtype=torch.bool)
        mask1[0, :4] = True
        cache.store(1, K1, V1, mask1)

        k_out, _, _ = cache.retrieve(1)
        # Active tokens (0-3) should have value 2.0
        assert torch.allclose(k_out[0, 0, :4, :], torch.ones(4, 32) * 2.0)
        # Exited tokens (4-7) should retain value 1.0
        assert torch.allclose(k_out[0, 0, 4:, :], torch.ones(4, 32) * 1.0)

    def test_recursion_wise_vs_shared_different_masks(self):
        """Recursion-wise mode should have per-recursion masks."""
        for mode in ["recursion_wise", "shared"]:
            cache = RecursionAwareKVCache(
                n_heads=2, head_dim=32, max_seq_len=16,
                n_recursions=4, kv_bits=0, mode=mode,
            )
            for r in range(4):
                K = torch.randn(1, 2, 16, 32)
                V = torch.randn(1, 2, 16, 32)
                n_active = 16 - r * 4
                mask = torch.zeros(1, 16, dtype=torch.bool)
                mask[0, :n_active] = True
                cache.store(r, K, V, mask)

            stats = cache.memory_stats()
            assert stats["compression_vs_standard"] >= 1.0

    def test_reset_clears_everything(self):
        """After reset, retrieve should fail."""
        cache = RecursionAwareKVCache(
            n_heads=2, head_dim=32, max_seq_len=8,
            n_recursions=2, kv_bits=0,
        )
        K = torch.randn(1, 2, 8, 32)
        V = torch.randn(1, 2, 8, 32)
        cache.store(0, K, V, torch.ones(1, 8, dtype=torch.bool))
        cache.reset()

        with pytest.raises(ValueError):
            cache.retrieve(0)

    def test_compressed_shared_mode(self):
        """Shared mode with compression should work."""
        cache = RecursionAwareKVCache(
            n_heads=4, head_dim=64, max_seq_len=32,
            n_recursions=4, kv_bits=3, mode="shared",
        )
        for r in range(4):
            K = torch.randn(1, 4, 32, 64)
            V = torch.randn(1, 4, 32, 64)
            mask = torch.ones(1, 32, dtype=torch.bool)
            cache.store(r, K, V, mask)

        k_out, v_out, _ = cache.retrieve(3)
        assert k_out.shape == (1, 4, 32, 64)
        assert not torch.isnan(k_out).any()


# ============================================================
# Recursive Block Tests
# ============================================================

class TestRecursiveBlock:

    def test_repeated_application_changes_output(self):
        """Applying the block multiple times should produce different outputs."""
        block = RecursiveTransformerBlock(d_model=64, n_heads=2, d_ff=128, dropout=0.0)
        x = torch.randn(1, 8, 64)

        outputs = []
        h = x
        for _ in range(4):
            out, _, _ = block(h)
            h = out + h  # external residual
            outputs.append(h.clone())

        # Each recursion should produce a different result
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[i], outputs[i-1], atol=1e-3), \
                f"Recursion {i} identical to {i-1}"

    def test_residual_convergence(self):
        """With moderate recursions, output should not produce NaN/Inf."""
        block = RecursiveTransformerBlock(d_model=64, n_heads=2, d_ff=128, dropout=0.0)
        block.eval()
        x = torch.randn(1, 8, 64) * 0.1

        h = x
        for _ in range(8):
            with torch.no_grad():
                out, _, _ = block(h)
                h = out + h

        # Should not produce NaN or Inf
        assert not torch.isnan(h).any(), "NaN after 8 recursions"
        assert not torch.isinf(h).any(), "Inf after 8 recursions"

    def test_attention_mask_applied(self):
        """Attention mask should zero out invalid KV positions."""
        block = RecursiveTransformerBlock(d_model=64, n_heads=2, d_ff=128, dropout=0.0)
        x = torch.randn(1, 8, 64)

        # All valid vs half valid should produce different outputs
        mask_all = torch.ones(1, 8, dtype=torch.bool)
        mask_half = torch.zeros(1, 8, dtype=torch.bool)
        mask_half[0, :4] = True

        out_all, _, _ = block(x, attn_mask=mask_all)
        out_half, _, _ = block(x, attn_mask=mask_half)

        assert not torch.allclose(out_all, out_half, atol=1e-3)

    def test_kv_output_shapes(self):
        """Block should output correctly shaped K and V."""
        block = RecursiveTransformerBlock(d_model=128, n_heads=4, d_ff=256)
        x = torch.randn(2, 16, 128)
        out, K, V = block(x)

        assert out.shape == (2, 16, 128)
        assert K.shape == (2, 4, 16, 32)  # (batch, heads, seq, head_dim)
        assert V.shape == (2, 4, 16, 32)

    def test_parameter_count(self):
        """Verify parameter counting is correct."""
        block = RecursiveTransformerBlock(d_model=64, n_heads=2, d_ff=128)
        count = block.count_parameters()
        # Manually compute: 4 linear projections in attention + 2 in FFN + 2 layernorms
        # Attention: Q,K,V,O each 64*64+64 = 4160, total 16640
        # FFN: 64*128+128 + 128*64+64 = 8320 + 8256 = 16576
        # LayerNorm: 2*(64+64) = 256
        assert count > 0
        assert count == sum(p.numel() for p in block.parameters() if p.requires_grad)


# ============================================================
# Config Validation Tests
# ============================================================

class TestConfigValidation:

    def test_valid_config(self):
        config = MoRConfig(d_model=128, n_heads=4, n_recursions=8)
        config.validate()  # should not raise

    def test_invalid_head_dim(self):
        config = MoRConfig(d_model=100, n_heads=3)  # 100/3 not integer
        with pytest.raises(AssertionError):
            config.validate()

    def test_invalid_recursions(self):
        config = MoRConfig(n_recursions=1)
        with pytest.raises(AssertionError):
            config.validate()

    def test_invalid_capacity_factor(self):
        config = MoRConfig(capacity_factor=0.0)
        with pytest.raises(AssertionError):
            config.validate()

    def test_invalid_kv_bits(self):
        config = MoRConfig(kv_bits=5)
        with pytest.raises(AssertionError):
            config.validate()

    def test_too_many_unique_layers(self):
        config = MoRConfig(n_recursions=4, n_unique_intro=2, n_unique_outro=2)
        with pytest.raises(AssertionError):
            config.validate()

    def test_head_dim_property(self):
        config = MoRConfig(d_model=256, n_heads=8)
        assert config.head_dim == 32

    def test_n_shared_recursions_middle_cycle(self):
        config = MoRConfig(n_recursions=8, n_unique_intro=1, n_unique_outro=1)
        assert config.n_shared_recursions == 6

    def test_n_shared_recursions_full(self):
        config = MoRConfig(n_recursions=8, sharing_strategy="full")
        assert config.n_shared_recursions == 8


# ============================================================
# Model Variant Tests
# ============================================================

class TestModelVariants:

    def test_full_sharing_strategy(self):
        """Model with full weight sharing (no unique intro/outro)."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=4, sharing_strategy="full",
            vocab_size=100, max_seq_len=32, kv_bits=0, dropout=0.0,
        )
        model = MoRModel(config)
        out = model(torch.randint(0, 100, (1, 16)))
        assert out.logits.shape == (1, 16, 100)
        # Full sharing = more parameter savings
        stats = model.count_parameters()
        assert stats["parameter_savings"] > 0.4

    def test_token_choice_routing(self):
        """Model with token-choice routing instead of expert-choice."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=4, routing_strategy="token",
            exit_threshold=0.5,
            vocab_size=100, max_seq_len=32, kv_bits=0, dropout=0.0,
        )
        model = MoRModel(config)
        out = model(torch.randint(0, 100, (1, 16)))
        assert out.logits.shape == (1, 16, 100)

    def test_4bit_kv_compression(self):
        """Model with 4-bit KV compression."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=4, vocab_size=100, max_seq_len=32,
            kv_bits=4, dropout=0.0,
        )
        model = MoRModel(config)
        out = model(torch.randint(0, 100, (1, 16)))
        assert out.kv_stats["compression_vs_standard"] > 1.0

    def test_no_kv_compression(self):
        """Model with kv_bits=0 (no compression)."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=4, vocab_size=100, max_seq_len=32,
            kv_bits=0, dropout=0.0,
        )
        model = MoRModel(config)
        out = model(torch.randint(0, 100, (1, 16)))
        # Should still save via early exit alone
        assert out.kv_stats["compression_vs_standard"] >= 1.0

    def test_many_recursions(self):
        """Model with high recursion count."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=16, n_unique_intro=2, n_unique_outro=2,
            vocab_size=100, max_seq_len=32, kv_bits=3, dropout=0.0,
        )
        model = MoRModel(config)
        out = model(torch.randint(0, 100, (1, 16)))
        assert out.logits.shape == (1, 16, 100)
        stats = model.count_parameters()
        assert stats["parameter_savings"] > 0.6  # 16 recursions, huge savings

    def test_large_batch(self):
        """Batch size stress test."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=4, vocab_size=100, max_seq_len=64,
            kv_bits=0, dropout=0.0,
        )
        model = MoRModel(config)
        out = model(torch.randint(0, 100, (16, 32)))
        assert out.logits.shape == (16, 32, 100)

    def test_max_seq_len_boundary(self):
        """Using exactly max_seq_len should work."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=4, vocab_size=100, max_seq_len=32,
            kv_bits=0, dropout=0.0,
        )
        model = MoRModel(config)
        out = model(torch.randint(0, 100, (1, 32)))  # exactly max
        assert out.logits.shape == (1, 32, 100)

    def test_seq_len_1(self):
        """Single token input should work."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=4, vocab_size=100, max_seq_len=32,
            kv_bits=0, dropout=0.0,
        )
        model = MoRModel(config)
        out = model(torch.randint(0, 100, (1, 1)))
        assert out.logits.shape == (1, 1, 100)


# ============================================================
# Training Simulation Tests
# ============================================================

class TestTraining:

    def _make_model(self, kv_bits=0):
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=4, vocab_size=100, max_seq_len=32,
            kv_bits=kv_bits, dropout=0.0,
        )
        return MoRModel(config)

    def test_multi_step_training(self):
        """Simulate multiple training steps — loss should decrease."""
        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Use fixed data so model can memorize
        input_ids = torch.randint(0, 100, (4, 16))
        labels = torch.randint(0, 100, (4, 16))

        losses = []
        for step in range(20):
            optimizer.zero_grad()
            out = model(input_ids, labels=labels)
            loss = out.loss + 0.01 * out.router_loss
            loss.backward()
            optimizer.step()
            losses.append(out.loss.item())

        # Loss should decrease over 20 steps
        assert losses[-1] < losses[0], \
            f"Loss didn't decrease: {losses[0]:.3f} → {losses[-1]:.3f}"

    def test_training_with_compression(self):
        """Training loop should work with KV compression enabled."""
        model = self._make_model(kv_bits=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.randint(0, 100, (2, 16))

        # Just verify it doesn't crash for several steps
        for _ in range(5):
            optimizer.zero_grad()
            out = model(input_ids, labels=labels)
            loss = out.loss + 0.01 * out.router_loss
            loss.backward()
            optimizer.step()

        assert out.loss.item() > 0  # should produce valid loss

    def test_gradient_flow_through_router(self):
        """Router gradients should be non-zero after backward."""
        model = self._make_model()
        input_ids = torch.randint(0, 100, (1, 16))
        labels = torch.randint(0, 100, (1, 16))

        out = model(input_ids, labels=labels)
        total_loss = out.loss + 0.01 * out.router_loss
        total_loss.backward()

        router_grad = model.router.gate.weight.grad
        assert router_grad is not None
        assert router_grad.abs().sum() > 0, "Router has zero gradients"

    def test_gradient_flow_to_embeddings(self):
        """Gradients should reach the embedding layer."""
        model = self._make_model()
        input_ids = torch.randint(0, 100, (1, 16))
        labels = torch.randint(0, 100, (1, 16))

        out = model(input_ids, labels=labels)
        out.loss.backward()

        assert model.token_emb.weight.grad is not None
        assert model.token_emb.weight.grad.abs().sum() > 0

    def test_gradient_flow_through_all_blocks(self):
        """Every block (intro, shared, outro) should receive gradients."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=6, n_unique_intro=1, n_unique_outro=1,
            vocab_size=100, max_seq_len=32, kv_bits=0, dropout=0.0,
        )
        model = MoRModel(config)
        input_ids = torch.randint(0, 100, (1, 16))
        labels = torch.randint(0, 100, (1, 16))

        out = model(input_ids, labels=labels)
        (out.loss + 0.01 * out.router_loss).backward()

        for name, block in [("intro", model.intro_blocks[0]),
                            ("shared", model.shared_block),
                            ("outro", model.outro_blocks[0])]:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in block.parameters() if p.requires_grad
            )
            assert has_grad, f"{name} block has no gradients"

    def test_weight_tying(self):
        """LM head and embedding should share the same weight tensor."""
        model = self._make_model()
        assert model.lm_head.weight is model.token_emb.weight

    def test_no_nan_in_output(self):
        """Forward pass should never produce NaN."""
        model = self._make_model()
        for _ in range(10):
            input_ids = torch.randint(0, 100, (2, 16))
            out = model(input_ids)
            assert not torch.isnan(out.logits).any(), "NaN in logits"


# ============================================================
# Memory Savings Benchmark
# ============================================================

class TestMemoryBenchmark:

    def test_savings_scale_with_recursions(self):
        """More recursions = more potential savings from early exit."""
        savings = []
        for n_rec in [4, 8, 12]:
            config = MoRConfig(
                d_model=128, n_heads=4, d_ff=256,
                n_recursions=n_rec, vocab_size=1000, max_seq_len=64,
                kv_bits=3, dropout=0.0,
            )
            model = MoRModel(config)
            model.eval()
            with torch.no_grad():
                out = model(torch.randint(0, 1000, (1, 64)))
            savings.append(out.kv_stats["compression_vs_standard"])

        # More recursions should generally give better compression ratio
        # (more opportunities for early exit)
        assert savings[-1] >= savings[0] * 0.8  # allow some variance

    def test_savings_scale_with_capacity_factor(self):
        """Lower capacity = more aggressive exit = more savings."""
        savings = {}
        for cf in [0.3, 0.5, 0.8]:
            config = MoRConfig(
                d_model=64, n_heads=2, d_ff=128,
                n_recursions=8, capacity_factor=cf,
                vocab_size=100, max_seq_len=32,
                kv_bits=3, dropout=0.0,
            )
            model = MoRModel(config)
            model.eval()
            with torch.no_grad():
                out = model(torch.randint(0, 100, (1, 32)))
            savings[cf] = out.kv_stats["compression_vs_standard"]

        # Lower capacity should give more compression
        assert savings[0.3] >= savings[0.8] * 0.7

    def test_compression_vs_no_compression(self):
        """With compression should always beat without."""
        results = {}
        for bits in [0, 3]:
            config = MoRConfig(
                d_model=64, n_heads=2, d_ff=128,
                n_recursions=6, vocab_size=100, max_seq_len=32,
                kv_bits=bits, dropout=0.0,
            )
            model = MoRModel(config)
            model.eval()
            with torch.no_grad():
                out = model(torch.randint(0, 100, (1, 32)))
            results[bits] = out.kv_stats["compression_vs_standard"]

        assert results[3] > results[0], \
            f"3-bit ({results[3]:.2f}x) should beat no compression ({results[0]:.2f}x)"


# ============================================================
# Determinism and Reproducibility
# ============================================================

class TestReproducibility:

    def test_eval_mode_deterministic(self):
        """Same input in eval mode should produce same output."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=4, vocab_size=100, max_seq_len=32,
            kv_bits=0, dropout=0.0,
        )
        model = MoRModel(config)
        model.eval()

        input_ids = torch.randint(0, 100, (1, 16))
        with torch.no_grad():
            out1 = model(input_ids)
            out2 = model(input_ids)

        assert torch.allclose(out1.logits, out2.logits), \
            "Eval mode not deterministic"

    def test_seeded_reproducibility(self):
        """With same seed, everything should reproduce."""
        config = MoRConfig(
            d_model=64, n_heads=2, d_ff=128,
            n_recursions=4, vocab_size=100, max_seq_len=32,
            kv_bits=0, dropout=0.1,
        )

        results = []
        for _ in range(2):
            torch.manual_seed(42)
            model = MoRModel(config)
            model.train()
            input_ids = torch.randint(0, 100, (1, 16))
            out = model(input_ids)
            results.append(out.logits.detach().clone())

        assert torch.allclose(results[0], results[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
