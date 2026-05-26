"""
Example: Run MoR-TurboQuant and see the savings.

Shows:
    1. Model creation with config
    2. Forward pass with adaptive routing
    3. Per-token exit depth analysis
    4. KV memory savings vs standard transformer
    5. Parameter savings from weight sharing
"""

import torch
from mor_tq import MoRConfig, MoRModel


def main():
    print("=" * 70)
    print("MoR-TurboQuant: Recursion-Aware Compressed KV Cache Demo")
    print("=" * 70)

    # Create model
    config = MoRConfig(
        d_model=256,
        n_heads=8,
        d_ff=1024,
        n_recursions=8,
        capacity_factor=0.5,
        routing_strategy="expert",
        kv_bits=3,
        vocab_size=10000,
        max_seq_len=512,
        dropout=0.0,
        sharing_strategy="middle_cycle",
        n_unique_intro=1,
        n_unique_outro=1,
    )
    config.validate()

    model = MoRModel(config)
    model.eval()

    # --- Parameter analysis ---
    param_stats = model.count_parameters()
    print(f"\n--- Parameter Analysis ---")
    print(f"Total parameters:               {param_stats['total_params']:>12,}")
    print(f"Standard equivalent (8 layers):  {param_stats['standard_equivalent_params']:>12,}")
    print(f"Shared block params:             {param_stats['shared_block_params']:>12,}")
    print(f"Parameter savings:               {param_stats['parameter_savings']:>11.1%}")

    # --- Forward pass ---
    print(f"\n--- Forward Pass (batch=2, seq_len=128) ---")
    input_ids = torch.randint(0, 10000, (2, 128))

    with torch.no_grad():
        output = model(input_ids)

    # --- Exit depth analysis ---
    depths = output.exit_depths[0]  # first batch element
    print(f"\nExit depth distribution (batch element 0):")
    for d in sorted(depths.unique().tolist()):
        count = (depths == d).sum().item()
        bar = "█" * (count // 2)
        print(f"  Depth {d:2d}: {count:3d} tokens {bar}")

    print(f"\n  Mean exit depth: {depths.float().mean():.1f} / {config.n_recursions}")
    print(f"  Min: {depths.min().item()}, Max: {depths.max().item()}")

    # --- Per-recursion activity ---
    print(f"\nTokens active at each recursion:")
    for i, frac in enumerate(output.per_recursion_active):
        bar = "▓" * int(frac * 40)
        phase = "intro" if i == 0 else ("outro" if i == len(output.per_recursion_active) - 1 else "shared")
        print(f"  R{i}: {frac:5.1%} active  {bar}  ({phase})")

    # --- KV Memory analysis ---
    stats = output.kv_stats
    print(f"\n--- KV Cache Memory ---")
    print(f"Standard transformer (all layers, FP16): {stats['standard_baseline_bytes']:>12,} bytes")
    print(f"MoR only (early exit, no compression):   {stats['mor_baseline_bytes']:>12,} bytes")
    print(f"MoR + TurboQuant (early exit + 3-bit):   {stats['actual_bytes']:>12,} bytes")
    print(f"")
    print(f"Compression vs standard transformer:     {stats['compression_vs_standard']:>10.1f}×")
    print(f"Compression vs MoR-only:                 {stats['compression_vs_mor']:>10.1f}×")
    print(f"Average active token fraction:            {stats['active_token_fraction']:>10.1%}")

    # --- Combined savings summary ---
    print(f"\n{'=' * 70}")
    print(f"COMBINED SAVINGS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Parameter reduction:  {param_stats['parameter_savings']:.0%} fewer unique weights")
    print(f"  KV memory reduction:  {stats['compression_vs_standard']:.1f}× less KV cache")
    print(f"  Compute reduction:    ~{(1 - stats['active_token_fraction']):.0%} FLOPs saved (early exits)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
