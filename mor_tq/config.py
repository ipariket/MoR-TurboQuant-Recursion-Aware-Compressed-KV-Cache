"""Model configuration for MoR-TurboQuant."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MoRConfig:
    """Configuration for a Mixture-of-Recursions model with compressed KV cache.

    The three efficiency knobs:
        1. n_recursions + sharing_strategy → parameter savings (weight reuse)
        2. capacity_factor + routing_strategy → compute savings (early exit)
        3. kv_bits → memory savings (compressed KV cache)
    """

    # ----- Model dimensions -----
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dropout: float = 0.1

    # ----- Recursion -----
    n_recursions: int = 8
    sharing_strategy: Literal["full", "middle_cycle"] = "middle_cycle"
    # middle_cycle: unique intro layers, shared recursive core, unique outro layers
    n_unique_intro: int = 1  # unique layers before shared block
    n_unique_outro: int = 1  # unique layers after shared block

    # ----- Router -----
    routing_strategy: Literal["token", "expert"] = "expert"
    capacity_factor: float = 0.5
    # expert-choice: at each recursion, process top capacity_factor fraction of tokens
    # token-choice: each token exits when g_t < exit_threshold
    exit_threshold: float = 0.5  # only used for token-choice routing

    # ----- KV Cache Compression -----
    kv_bits: int = 3  # 0 = no compression, 3 = 2-bit PQ + 1-bit QJL, 4 = 3-bit PQ + 1-bit QJL
    kv_group_size: int = 128  # compression group size
    use_qjl: bool = True  # enable QJL residual correction (full TurboQuant)

    # ----- Derived -----
    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def n_shared_recursions(self) -> int:
        """Number of recursions using the shared weight block."""
        if self.sharing_strategy == "full":
            return self.n_recursions
        return max(1, self.n_recursions - self.n_unique_intro - self.n_unique_outro)

    def validate(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.n_recursions >= 2, "Need at least 2 recursions"
        assert 0.0 < self.capacity_factor <= 1.0, "capacity_factor must be in (0, 1]"
        assert self.kv_bits in (0, 3, 4), "kv_bits must be 0 (none), 3 (2-bit PQ + 1-bit QJL), or 4 (3-bit PQ + 1-bit QJL)"
        if self.sharing_strategy == "middle_cycle":
            total_unique = self.n_unique_intro + self.n_unique_outro
            assert total_unique < self.n_recursions, (
                f"Unique layers ({total_unique}) must be less than total recursions ({self.n_recursions})"
            )
        return self
