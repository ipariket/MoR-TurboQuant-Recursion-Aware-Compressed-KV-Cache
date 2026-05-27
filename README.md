# MoR-TurboQuant: Recursion-Aware Compressed KV Cache

A PyTorch module bridging **Mixture of Recursions (MoR)** adaptive compute with **TurboQuant+** style KV cache compression. This is the missing middle layer — tokens that exit early skip KV allocation entirely, and the KV entries that *do* exist get compressed via PolarQuant + Walsh-Hadamard Transform.

## The Gap This Fills

| System | KV entries produced | KV entry size | Total KV memory |
|--------|-------------------|---------------|-----------------|
| Standard Transformer | All layers × all tokens | FP16 (full) | **Baseline (100%)** |
| Standard + TurboQuant | All layers × all tokens | 3-4 bit | ~25% of baseline |
| MoR (no compression) | Only active recursions | FP16 (full) | ~40% of baseline |
| **MoR + TurboQuant (this project)** | Only active recursions | 3-4 bit | **~10% of baseline** |

## Architecture

```
Token Embeddings
      │
      ▼
┌─────────────────────────┐
│   Recursive Block (Φ)   │◄── Same weights, applied N times
│   Attention + FFN        │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   Adaptive Router        │── g_t = σ(θᵀ · h_t)
│   token-choice or        │
│   expert-choice          │
└─────┬─────────┬─────────┘
      │         │
   continue    exit
      │         │
      ▼         ▼
  next      freeze h_t,
  recursion   skip KV
      │
      ▼
┌─────────────────────────┐
│  Recursion-Aware KV Cache│── Only stores KV for active tokens
│  + PolarQuant compression │── WHT + Lloyd-Max on stored entries
└─────────────────────────┘
```

## Components

- `mor_tq/recursive_block.py` — Weight-shared transformer block with residual recursion
- `mor_tq/router.py` — Adaptive depth router (token-choice + expert-choice)
- `mor_tq/kv_cache.py` — Recursion-aware KV cache manager with sparse allocation
- `mor_tq/compression.py` — PolarQuant-style compression (WHT + Lloyd-Max codebook)
- `mor_tq/model.py` — Full MoR model assembling all components
- `mor_tq/config.py` — Configuration dataclass with three efficiency knobs

## Results

### KV Memory Compression

Benchmarks run on CPU with expert-choice routing (capacity_factor=0.5) and 3-bit PolarQuant compression.

| Model Config | Params | Param Savings | KV Compression | Active Tokens | Mean Exit Depth |
|---|---|---|---|---|---|
| Small (d=64, 4 recursions) | 172K | 16% | **5.8×** | 68.8% | 1.8 / 4 |
| Medium (d=256, 8 recursions) | 5.1M | 44% | **10.7×** | 37.3% | 2.0 / 8 |
| Large (d=512, 12 recursions) | 32.4M | 41% | **11.0×** | 41.7% | 3.0 / 12 |

### Ablation: MoR Early Exit vs Compression

Using the Medium config (d=256, 8 recursions, seq_len=128):

| Configuration | KV Memory Reduction | How |
|---|---|---|
| Standard Transformer | 1.0× (baseline) | All layers, FP16 |
| MoR only (kv_bits=0) | 2.7× | Early exit skips KV allocation |
| MoR + 3-bit PolarQuant | **10.7×** | Early exit + compressed survivors |

The savings multiply: ~2.7× from early exit × ~4× from 3-bit compression ≈ 10.7× total.

### Training Convergence

Training a small model (d=128, 6 recursions, 3-bit KV) with Adam (lr=1e-3) on fixed data:

| Step | Loss | Mean Exit Depth |
|---|---|---|
| 1 | 6.939 | 1.9 |
| 10 | 4.631 | 1.9 |
| 25 | 2.572 | 1.9 |
| 50 | 0.468 | 1.9 |

Loss decreases steadily despite non-differentiable quantization in the KV cache. Gradients flow through the router, all transformer blocks, and embeddings.

### Actual KV Memory Usage

Medium config, 128 tokens:

```
Standard transformer (8 layers, FP16):  1,048,576 bytes
MoR + 3-bit compression:                  97,792 bytes  (10.7× reduction)
```

## Test Suite

84 tests across two files covering every component:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

| Category | Tests | What It Validates |
|---|---|---|
| WHT Math Properties | 7 | Roundtrip, energy preservation, linearity, input immutability |
| Compression Edge Cases | 10 | 3-bit vs 4-bit quality, extreme dimensions, NaN/Inf safety, zero/constant inputs |
| Router Edge Cases | 9 | Single token, all exited, capacity bounds, differentiability |
| KV Cache Modes | 5 | Shared vs recursion-wise, exited token preservation, reset, compressed shared |
| Recursive Block | 5 | Repeated application, attention masking, KV shapes, stability |
| Config Validation | 9 | Invalid head dims, recursion counts, capacity factors, bit widths |
| Model Variants | 8 | Full sharing, token-choice, 4-bit, no compression, seq_len=1, batch=16 |
| Training Simulation | 7 | Multi-step loss decrease, gradient flow through router/embeddings/all blocks |
| Memory Benchmarks | 3 | Savings vs recursion count, capacity factor, compression on/off |
| Reproducibility | 2 | Deterministic eval mode, seeded reproducibility |

## Quick Start

```python
from mor_tq import MoRModel, MoRConfig

config = MoRConfig(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_recursions=8,
    capacity_factor=0.5,      # expert-choice: keep top 50% per recursion
    routing_strategy="expert", # "expert" or "token"
    kv_bits=3,                 # TurboQuant-style compression
    vocab_size=32000,
)

model = MoRModel(config)

# Forward pass — tokens auto-route through adaptive recursions
output = model(input_ids)

# output.logits          — (batch, seq_len, vocab_size)
# output.exit_depths     — per-token recursion depth
# output.kv_stats        — KV memory usage vs baselines
# output.router_loss     — auxiliary load-balancing loss
# output.loss            — language modeling loss (if labels provided)
```

## Install

```bash
pip install -e .
```

## References

- [Mixture of Recursions (MoR)](https://github.com/raymin0223/mixture_of_recursions) — Google DeepMind
- [TurboQuant+](https://pypi.org/project/turboquant-plus-vllm/) — Varjosoft
- [HIGGS](https://arxiv.org/abs/2411.17525) — Weight quantization algorithm
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) — KV cache quantization (ICLR 2026)
