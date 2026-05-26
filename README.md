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
- `mor_tq/benchmarks.py` — Memory + FLOP comparison utilities

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
logits, stats = model(input_ids)

# stats contains:
#   - exit_depths: per-token recursion depth
#   - kv_memory_bytes: actual KV memory used
#   - kv_memory_baseline: what standard transformer would use
#   - compression_ratio: combined savings
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
