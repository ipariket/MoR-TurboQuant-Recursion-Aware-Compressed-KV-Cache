# MoR-TurboQuant: Architecture & Roadmap

> This document covers internal architecture, component design, known limitations, and the path to a research-credible system. The [README](../README.md) covers API usage and results.

---

## Table of Contents

1. [What Problem This Solves](#1-what-problem-this-solves)
2. [Architecture Overview](#2-architecture-overview)
3. [Component Deep Dive](#3-component-deep-dive)
4. [Data Flow (Forward Pass)](#4-data-flow-forward-pass)
5. [Memory Model](#5-memory-model)
6. [Current Limitations](#6-current-limitations)
7. [Roadmap](#7-roadmap)

---

## 1. What Problem This Solves

### KV Cache Memory is the Real Bottleneck

When a transformer generates text, it stores Key and Value vectors for every token at every layer so it doesn't recompute them. This is called the **KV cache**.

For a standard 32-layer model with 32 heads, 128 head_dim, 4096-token context:

```
32 layers × 4096 tokens × 32 heads × 128 dims × 2 (K+V) × 2 bytes
= ~2 GB per sequence
```

At batch size 8, that's 16 GB — the entire GPU — just for the cache. This is what limits how many users you can serve simultaneously.

### Two Existing Solutions, Neither Complete

| Approach | What it does | What it misses |
|----------|-------------|----------------|
| **TurboQuant / HIGGS** | Quantize KV entries to 3–4 bit | Still allocates for every token at every layer |
| **Mixture of Recursions (MoR)** | Tokens exit early, skip later layers | No compression on the KV entries that do exist |

**This project combines both**: tokens that exit early skip KV allocation entirely, and the entries that survive get quantized. The savings multiply rather than add.

---

## 2. Architecture Overview

```
Input IDs  (batch, seq_len)
      │
      ▼
┌─────────────────────────────────┐
│  Token Embedding                │
│  + Positional Embedding         │
│  + Dropout                      │
└────────────────┬────────────────┘
                 │  h = (batch, seq_len, d_model)
                 ▼
┌─────────────────────────────────┐
│  Intro Block(s)                 │  unique weights
│  n_unique_intro = 1 (default)   │  all tokens, no routing
│  Attention + FFN + LayerNorm    │
└────────────────┬────────────────┘
                 │
                 ▼
╔═════════════════════════════════════════════════════╗
║            SHARED RECURSIVE CORE                    ║
║                                                     ║
║  for r in range(n_shared_recursions):               ║
║                                                     ║
║    ┌────────────────────────────┐                   ║
║    │  AdaptiveRouter            │                   ║
║    │  g_t = σ(θᵀ · h_t)        │  one linear layer ║
║    │  expert-choice: top 50%   │  shared across all ║
║    └────────────┬───────────────┘  recursions       ║
║                 │                                   ║
║           active_mask (bool)                        ║
║                 │                                   ║
║    ┌────────────▼───────────────┐                   ║
║    │  Shared Block  Φ           │  ← SAME weights   ║
║    │  Attention + FFN           │    every loop     ║
║    │  (runs on all tokens)      │                   ║
║    └────────────┬───────────────┘                   ║
║                 │                                   ║
║    h = where(active, block_out, h)                  ║
║    (exited tokens: hidden state frozen)             ║
║                 │                                   ║
║    KV cache: store only active tokens               ║
║    → compress: WHT → normalize → Lloyd-Max          ║
║                                                     ║
╚═════════════════════════════════════════════════════╝
                 │
                 ▼
┌─────────────────────────────────┐
│  Outro Block(s)                 │  unique weights
│  n_unique_outro = 1 (default)   │  all tokens, no routing
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  LayerNorm                      │
│  LM Head                        │  weight-tied to token embedding
└────────────────┬────────────────┘
                 │
                 ▼
  Logits  (batch, seq_len, vocab_size)
```

### Default Configuration

With `MoRConfig` defaults (`d_model=512, n_heads=8, n_recursions=8, capacity_factor=0.5`):

```
1 intro block       — unique weights, all tokens pass
6 shared recursions — same Φ applied 6 times, with routing
1 outro block       — unique weights, all tokens pass
────────────────────────────────────────────────────
8 total depth, but only 1 shared weight set for the core
```

---

## 3. Component Deep Dive

### 3.1 RecursiveTransformerBlock (`recursive_block.py`)

One transformer block (Attention + FFN) applied N times with the **same weights**. Each application refines the representation:

```
h₁ = Φ(h₀) + h₀
h₂ = Φ(h₁) + h₁   ← identical Φ, different input
h₃ = Φ(h₂) + h₂
```

The residual connection is critical — without it, repeated application of the same linear map would collapse to a fixed point. With it, each recursion makes an incremental update.

The block is standard pre-norm GPT style:
```
normed    = LayerNorm(x)
attn_out  = MultiHeadAttention(normed)
x         = x + Dropout(attn_out)
normed2   = LayerNorm(x)
ffn_out   = FFN(normed2)            # Linear → GELU → Linear
output    = x + Dropout(ffn_out)
```

### 3.2 AdaptiveRouter (`router.py`)

Decides which tokens need more processing at each recursion. The core gate is a single learned weight vector θ:

```
g_t = σ(θᵀ · h_t)     # scalar in [0, 1] per token
```

High score = token representation is still unsettled, needs more recursions.
Low score = token has converged, can exit early.

**Two routing strategies:**

| Strategy | How | Tradeoff |
|----------|-----|----------|
| `token` | Each token independently: exit if g_t < threshold | Simple, but unbalanced batch sizes |
| `expert` | System picks top `capacity_factor` fraction per step | Balanced GPU utilization, better in practice |

Expert-choice is the default and recommended. It guarantees exactly 50% of eligible tokens continue at each recursion, keeping compute predictable.

**Load balance loss** (from Switch Transformer):
```
L_balance = (mean_score - capacity_factor)²
```
Penalizes both collapse cases: all tokens always continue, or all tokens always exit.

### 3.3 RecursionAwareKVCache (`kv_cache.py`)

The novel component. Standard caches allocate uniformly. This cache:

1. Tracks which tokens are active at each recursion depth
2. Stores KV **only for active tokens** at depth r
3. Compresses stored entries via PolarQuant
4. Handles the resulting sparse attention pattern

**Two cache modes:**

| Mode | Storage | When to use |
|------|---------|-------------|
| `recursion_wise` | Separate cache slot per depth | Research / analysis — exact tracking |
| `shared` | One slot, updated in-place | Maximum memory savings |

### 3.4 PolarQuantCompressor (`compression.py`)

Implements TurboQuant+-style compression. Pipeline per vector:

```
raw vector (FP16, head_dim)
     │
     ▼  pad to next power-of-2
     │
     ▼  Walsh-Hadamard Transform (WHT)
     │  O(d log d), decorrelates components
     │  quantization error spreads evenly across dims
     │
     ▼  split into groups of group_size=128
     │
     ▼  extract per-group L2 norm → store as FP32
     │
     ▼  normalize each group to unit sphere
     │
     ▼  Lloyd-Max scalar quantization
     │  3-bit → 8 centroids, 4-bit → 16 centroids
     │  centroids pre-fitted for N(0,1) distribution
     │  (WHT output is approximately Gaussian)
     │
     ▼  bit-pack indices
        3-bit: 8 indices into 3 bytes
        4-bit: 2 nibbles per byte

compressed: (packed_indices, per_group_norms)
```

Decompression reverses: unpack → centroid lookup → rescale by norm → inverse WHT.

WHT is self-inverse (H·H = d·I), so the same function is used both directions.

---

## 4. Data Flow (Forward Pass)

```python
# 1. Embed
h = token_emb(input_ids) + pos_emb(positions)   # (B, S, D)

# 2. Intro blocks — all tokens, no routing
for block in intro_blocks:
    h, K, V = block(h)
    kv_cache.store(r, K, V, all_active_mask)

# 3. Shared recursive core
exited = zeros(B, S, bool)

for r in range(n_shared_recursions):
    router_out = router(h, already_exited=exited)
    active_mask = router_out.active_mask           # (B, S) bool

    block_out, K, V = shared_block(h)              # ← runs on ALL tokens (current limitation)

    h = where(active_mask, block_out, h)           # only update active tokens
    kv_cache.store(r, K, V, active_mask)           # only store active KV

    exited = exited | ~active_mask

# 4. Outro blocks — all tokens
for block in outro_blocks:
    h, K, V = block(h)
    kv_cache.store(r, K, V, all_active_mask)

# 5. Output
logits = lm_head(layer_norm(h))

# 6. Loss
lm_loss     = cross_entropy(logits, labels)
router_loss = router.compute_load_balance_loss(scores)
total_loss  = lm_loss + 0.01 * router_loss
```

---

## 5. Memory Model

### Standard Transformer (Baseline)
```
layers × seq_len × n_heads × head_dim × 2 (K+V) × 2 bytes (FP16)
```

### MoR + TurboQuant (This System)
```
Σ_r [ active_tokens_at_r × n_heads × compressed_bytes_per_vector ]

where compressed_bytes = (bits/8) × head_dim  +  (32/8) × n_groups  (norm overhead)
```

### Example (Medium Config: d=256, 8 recursions, 128 tokens)

| System | KV Memory | How |
|--------|-----------|-----|
| Standard Transformer | 1,048,576 bytes | All 8 layers, all 128 tokens, FP16 |
| MoR only (no compression) | ~393,216 bytes | Early exit, ~37% active avg, FP16 |
| MoR + 3-bit PolarQuant | **97,792 bytes** | Early exit + compressed survivors |
| **Reduction** | **10.7×** | ~2.7× from exit × ~4× from compression |

---

## 6. Current Limitations

### 6.1 Compute Savings Are Not Real Yet (Critical)

The shared block runs on all tokens every recursion:

```python
# model.py:193 — runs full batch even for inactive tokens
block_out, K, V = self.shared_block(h)
h = torch.where(active_expanded, block_out, h)   # only keeps active results
```

This saves **KV memory** but not **FLOPs**. Real compute savings require passing only active token indices into the block — which needs either:
- Index-based slicing with re-padding (doable in pure PyTorch, ~70% of theoretical savings)
- Custom Triton sparse attention kernels (full theoretical savings)

### 6.2 No Real Training Benchmark

The existing training demonstration uses repeated fixed batches (memorization). There is no evaluation on a standard dataset (WikiText-103, OpenWebText, The Pile).

### 6.3 4-bit Codebook Asymmetry

The Lloyd-Max centroids for 4-bit quantization in `compression.py` are not symmetric around zero. The 3-bit codebook is correctly symmetric. This causes degraded quality at 4-bit vs 3-bit, which is the opposite of expected behavior.

### 6.4 No Autoregressive Generation Support

`kv_cache.reset()` is called on every forward pass. The cache doesn't support incremental decoding (appending one token at a time), which is required for actual text generation. Training is fine; inference for generation is not implemented.

### 6.5 No Baseline Comparison

There are no side-by-side numbers against a same-size standard transformer on the same dataset, which is the minimum required to make any claim about quality vs memory tradeoff.

---

## 7. Roadmap

### Phase 1 — Correctness (1–2 weeks, CPU / MacBook)

| Task | Why |
|------|-----|
| Fix 4-bit Lloyd-Max codebook symmetry | Currently produces worse results than 3-bit |
| Add unit test: 4-bit quality > 3-bit quality | Prevents regression |
| Train on TinyStories with real dataloader | Validate loss decreases on real text |
| Log `per_recursion_active` during training | Confirm router learns non-trivial exit depths |
| Enable compression after N warmup steps | Prevent quantization from blocking early learning |

**Target:** Router settles to 30–60% active per recursion, loss curves down on real data.

---

### Phase 2 — Research Credible (2–4 weeks, single A100)

| Task | Why |
|------|-----|
| Train Tier 2 config (d=256, 5M params) on WikiText-103 | Standard benchmark |
| Evaluate perplexity vs standard transformer same size | The core claim |
| Plot: KV memory vs perplexity (sweep capacity_factor, kv_bits) | The key figure |
| Add autoregressive generation with incremental KV cache | Required for real inference demo |
| Add LR schedule (cosine warmup) and gradient clipping | Standard training hygiene |

**Target:** Within 5–10% perplexity of baseline while using 5–10× less KV memory.

**Estimated compute cost:** Single A100 for 1–2 days, ~$30–80 on Lambda/RunPod.

---

### Phase 3 — Compute Savings Real (4–8 weeks)

| Task | Why |
|------|-----|
| Sparse forward pass: only compute block on active indices | Make compute claim real, not theoretical |
| GPU wall-clock benchmarks: tokens/sec, memory usage | Reviewers will require this |
| Compare against SnapKV, H2O, GriffinKV on same task | Position in literature |
| Separate K vs V codebooks | TurboQuant uses this; improves quality |

**Target:** Demonstrated wall-clock speedup, not just theoretical FLOP reduction.

---

### Phase 4 — Submission Ready (8–16 weeks)

| Task | Why |
|------|-----|
| Scale to GPT-2 size (124M params) | Reviewers expect >100M param experiments |
| Ablation: MoR-only vs compression-only vs combined | Justify the combination |
| Multi-dataset evaluation | Generalization |
| Triton sparse attention kernel | Production-quality compute savings |

**Venue target:** MLSys, NeurIPS Systems Track, or ICLR (efficiency track).

---

## Quick Reference: Config Knobs

| Parameter | Controls | Tradeoff |
|-----------|----------|----------|
| `n_recursions` | Total depth | More = better quality, more memory |
| `capacity_factor` | Fraction of tokens that continue per recursion | Lower = faster, less accurate |
| `kv_bits` | Quantization precision (0=off, 3, 4) | Lower = less memory, lower quality |
| `routing_strategy` | `expert` or `token` | Expert = balanced GPU, token = flexible |
| `sharing_strategy` | `full` or `middle_cycle` | middle_cycle = intro/outro layers separate |
| `n_unique_intro/outro` | Non-shared layers at start/end | Improves quality at parameter cost |

---

*Last updated: 2026-05-29*
