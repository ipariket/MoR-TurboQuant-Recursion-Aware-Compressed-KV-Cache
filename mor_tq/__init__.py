"""
MoR-TurboQuant: Recursion-Aware Compressed KV Cache

Bridges Mixture of Recursions (MoR) adaptive compute with 
TurboQuant-style KV cache compression. Tokens that exit early 
skip KV allocation; surviving entries get compressed via 
PolarQuant (WHT + Lloyd-Max codebook).
"""

from mor_tq.config import MoRConfig
from mor_tq.model import MoRModel
from mor_tq.router import AdaptiveRouter
from mor_tq.kv_cache import RecursionAwareKVCache
from mor_tq.compression import PolarQuantCompressor
from mor_tq.recursive_block import RecursiveTransformerBlock

__version__ = "0.1.0"

__all__ = [
    "MoRConfig",
    "MoRModel",
    "AdaptiveRouter",
    "RecursionAwareKVCache",
    "PolarQuantCompressor",
    "RecursiveTransformerBlock",
]
