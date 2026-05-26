"""
Adaptive Router for Mixture of Recursions.

Decides which tokens continue to the next recursion and which exit early.
Two strategies:

    Token-choice: Each token independently exits when g_t < threshold.
        + Simple, intuitive
        - Unbalanced compute (some recursions process 90%, others 10%)
        - GPU utilization suffers

    Expert-choice: System picks fixed budget — top capacity_factor% of tokens proceed.
        + Perfectly balanced compute per recursion
        + GPU-friendly (predictable batch sizes)
        - Slightly less flexible per-token
        
The paper found expert-choice gives better validation loss across scales.
"""

import torch
import torch.nn as nn
from typing import Literal, NamedTuple


class RouterOutput(NamedTuple):
    """Output of the adaptive router at one recursion step."""
    active_mask: torch.Tensor       # (batch, seq_len) bool — which tokens continue
    router_scores: torch.Tensor     # (batch, seq_len) float — raw sigmoid scores
    n_active: int                   # number of active tokens this step
    n_total: int                    # total tokens


class AdaptiveRouter(nn.Module):
    """Learned router that scores each token's need for further processing.

    The core operation is trivially simple:
        g_t = σ(θᵀ · h_t)
    
    θ is a single learned vector (d_model dims). The dot product with the 
    token's hidden state measures "how unsettled is this representation?"
    Sigmoid squashes to [0, 1]. High = needs more work. Low = converged.
    """

    def __init__(
        self,
        d_model: int,
        strategy: Literal["token", "expert"] = "expert",
        capacity_factor: float = 0.5,
        exit_threshold: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.strategy = strategy
        self.capacity_factor = capacity_factor
        self.exit_threshold = exit_threshold

        # The router is just a single linear projection to scalar
        # θ is (d_model,) — one vector for the entire model
        self.gate = nn.Linear(d_model, 1, bias=False)

        # Initialize small so early training doesn't route aggressively
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_exited: torch.Tensor | None = None,
    ) -> RouterOutput:
        """Score tokens and decide who continues.

        Args:
            hidden_states: (batch, seq_len, d_model)
            already_exited: (batch, seq_len) bool — tokens that already exited
                in a previous recursion. These are never reactivated.

        Returns:
            RouterOutput with active_mask and scores.
        """
        B, S, D = hidden_states.shape

        # Compute router scores: g_t = σ(θᵀ · h_t)
        scores = torch.sigmoid(self.gate(hidden_states).squeeze(-1))  # (B, S)

        if self.strategy == "token":
            active_mask = self._token_choice(scores, already_exited)
        elif self.strategy == "expert":
            active_mask = self._expert_choice(scores, already_exited)
        else:
            raise ValueError(f"Unknown routing strategy: {self.strategy}")

        # Tokens that already exited stay exited
        if already_exited is not None:
            active_mask = active_mask & ~already_exited

        n_active = active_mask.sum().item()
        n_total = B * S

        return RouterOutput(
            active_mask=active_mask,
            router_scores=scores,
            n_active=n_active,
            n_total=n_total,
        )

    def _token_choice(
        self,
        scores: torch.Tensor,
        already_exited: torch.Tensor | None,
    ) -> torch.Tensor:
        """Each token independently decides: score >= threshold → continue.

        Simple but leads to unbalanced compute across recursion steps.
        """
        mask = scores >= self.exit_threshold

        if already_exited is not None:
            # Don't let already-exited tokens come back
            mask = mask & ~already_exited

        return mask

    def _expert_choice(
        self,
        scores: torch.Tensor,
        already_exited: torch.Tensor | None,
    ) -> torch.Tensor:
        """System picks the top-k most-needy tokens to continue.

        Guarantees exactly capacity_factor * n_eligible tokens per recursion,
        giving perfectly balanced GPU utilization.
        """
        B, S = scores.shape

        # Mask out already-exited tokens by setting their scores to -inf
        effective_scores = scores.clone()
        if already_exited is not None:
            effective_scores[already_exited] = -float("inf")

        # Count eligible tokens per batch
        if already_exited is not None:
            n_eligible = (~already_exited).sum(dim=-1)  # (B,)
        else:
            n_eligible = torch.full((B,), S, device=scores.device)

        # Select top-k per batch element
        k_per_batch = (n_eligible.float() * self.capacity_factor).ceil().long()
        k_per_batch = k_per_batch.clamp(min=1)  # always keep at least 1

        # Create mask via top-k selection
        active_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b in range(B):
            k = min(k_per_batch[b].item(), S)
            if k > 0:
                _, top_indices = effective_scores[b].topk(k)
                active_mask[b, top_indices] = True

        return active_mask

    def compute_load_balance_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss encouraging balanced routing across tokens.

        Prevents the router from always selecting the same tokens or
        collapsing to trivial all-continue / all-exit solutions.

        From the Switch Transformer paper (Fedus et al., 2022):
            L_balance = n * Σ_i (f_i · p_i)
        where f_i is fraction routed to expert i, p_i is mean probability.
        
        Adapted for binary continue/exit routing.
        """
        # Mean probability of continuing
        mean_prob = scores.mean()
        # Variance of probabilities (want this to be moderate, not 0 or max)
        var_prob = scores.var()

        # Penalize both extremes: all-continue (mean≈1) and all-exit (mean≈0)
        # Sweet spot is around capacity_factor
        target = self.capacity_factor
        balance_loss = (mean_prob - target) ** 2

        return balance_loss
