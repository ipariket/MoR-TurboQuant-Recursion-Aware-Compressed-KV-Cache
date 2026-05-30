"""
Inference KV Memory Benchmark

Measures KV cache memory two ways:
  1. Analytical: exact byte count from architecture math
  2. Hardware: actual CUDA memory delta (when GPU available)

Run after training: python benchmark_kv.py

Produces tables showing real measurements for:
  - Standard transformer (all layers, FP16)
  - MoR only (early exit, FP16)
  - MoR + 3-bit (early exit + compression)
"""

import torch
import json
import gc
from mor_tq import MoRConfig, MoRModel


def measure_cuda_memory(model, input_ids):
    """Measure actual CUDA memory consumed by a forward pass.

    Returns the memory delta (peak - before) which isolates
    the KV cache + activation cost from model weight cost.
    """
    if not torch.cuda.is_available():
        return None

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    with torch.no_grad():
        output = model(input_ids)

    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_delta = mem_peak - mem_before

    return {
        "mem_before_bytes": mem_before,
        "mem_peak_bytes": mem_peak,
        "mem_delta_bytes": mem_delta,
        "kv_stats": output.kv_stats,
        "exit_depths": output.exit_depths,
    }


def benchmark_config(name, config, seq_lengths, device="cpu"):
    """Run inference at various seq lengths and measure KV memory."""
    model = MoRModel(config).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    param_stats = model.count_parameters()

    results = {
        "name": name,
        "total_params": total_params,
        "param_savings": round(param_stats["parameter_savings"] * 100, 1),
        "seq_length_results": [],
    }

    for seq_len in seq_lengths:
        if seq_len > config.max_seq_len:
            continue

        input_ids = torch.randint(0, config.vocab_size, (1, seq_len)).to(device)

        # Analytical measurement
        with torch.no_grad():
            output = model(input_ids)

        stats = output.kv_stats
        entry = {
            "seq_len": seq_len,
            "standard_bytes": stats["standard_baseline_bytes"],
            "mor_bytes": stats["mor_baseline_bytes"],
            "actual_bytes": stats["actual_bytes"],
            "compression_vs_standard": round(stats["compression_vs_standard"], 2),
            "compression_vs_mor": round(stats["compression_vs_mor"], 2),
            "active_fraction": round(stats["active_token_fraction"] * 100, 1),
        }

        # Hardware measurement (CUDA only)
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            hw = measure_cuda_memory(model, input_ids)
            if hw:
                entry["cuda_delta_bytes"] = hw["mem_delta_bytes"]

        results["seq_length_results"].append(entry)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def main():
    print("=" * 80)
    print("KV Cache Memory Benchmark: Inference-Time Measurement")
    print("=" * 80)

    device = "cpu"  # CPU is fine for memory measurement
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print(f"Device: {device}\n")

    MAX_SEQ = 2048
    SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048]

    COMMON = dict(
        d_model=512,
        n_heads=8,
        d_ff=2048,
        vocab_size=1000,
        max_seq_len=MAX_SEQ,
        dropout=0.0,
    )

    configs = {
        "Standard 8-layer": MoRConfig(
            **COMMON, n_recursions=8, sharing_strategy="full",
            capacity_factor=1.0, routing_strategy="expert", kv_bits=0,
        ),
        "MoR only (no compress)": MoRConfig(
            **COMMON, n_recursions=8, sharing_strategy="middle_cycle",
            n_unique_intro=1, n_unique_outro=1,
            capacity_factor=0.5, routing_strategy="expert", kv_bits=0,
        ),
        "MoR + 3-bit (Ours)": MoRConfig(
            **COMMON, n_recursions=8, sharing_strategy="middle_cycle",
            n_unique_intro=1, n_unique_outro=1,
            capacity_factor=0.5, routing_strategy="expert", kv_bits=3,
        ),
    }

    all_results = {}
    for name, cfg in configs.items():
        print(f"Benchmarking: {name}...")
        all_results[name] = benchmark_config(name, cfg, SEQ_LENGTHS, device)

    # Print comparison table
    print(f"\n{'=' * 100}")
    print("KV Cache Memory Usage (bytes per sample, batch=1)")
    print(f"{'=' * 100}")

    header = f"{'Seq Len':>8}"
    for name in configs:
        short = name.split("(")[0].strip()[:20]
        header += f" | {short:>16}"
    header += " | {'Compression':>12}"
    print(f"{'Seq Len':>8} | {'Standard':>16} | {'MoR only':>16} | {'MoR+3bit (Ours)':>16} | {'vs Standard':>12} | {'vs MoR':>12}")
    print("-" * 100)

    names = list(configs.keys())
    for i, seq_len in enumerate(SEQ_LENGTHS):
        row = f"{seq_len:>8}"

        bytes_list = []
        for name in names:
            res = all_results[name]["seq_length_results"]
            if i < len(res):
                b = res[i]["actual_bytes"]
                bytes_list.append(b)
                row += f" | {b:>16,}"
            else:
                bytes_list.append(0)
                row += f" | {'N/A':>16}"

        if len(bytes_list) == 3 and bytes_list[2] > 0:
            vs_std = bytes_list[0] / max(1, bytes_list[2])
            vs_mor = bytes_list[1] / max(1, bytes_list[2])
            row += f" | {vs_std:>11.2f}x | {vs_mor:>11.2f}x"

        print(row)

    print(f"{'=' * 100}")

    # Summary
    ours_results = all_results[names[2]]["seq_length_results"]
    if ours_results:
        avg_compression = sum(r["compression_vs_standard"] for r in ours_results) / len(ours_results)
        avg_active = sum(r["active_fraction"] for r in ours_results) / len(ours_results)
        print(f"\nAverage compression vs standard: {avg_compression:.2f}x")
        print(f"Average active token fraction: {avg_active:.1f}%")
        print(f"\nBreakdown:")
        print(f"  Early exit contribution: {sum(r['compression_vs_standard']/r['compression_vs_mor'] for r in ours_results)/len(ours_results):.2f}x")
        print(f"  Compression contribution: {sum(r['compression_vs_mor'] for r in ours_results)/len(ours_results):.2f}x")
        print(f"  Combined (multiplicative): {avg_compression:.2f}x")

    # Scaling analysis
    print(f"\n{'=' * 80}")
    print("Scaling: KV Memory at 2048 tokens (what matters for real inference)")
    print(f"{'=' * 80}")
    for name in names:
        res = all_results[name]["seq_length_results"]
        if res:
            last = res[-1]
            mb = last["actual_bytes"] / (1024 * 1024)
            print(f"  {name:<30} {last['actual_bytes']:>12,} bytes ({mb:.2f} MB)")

    # Save
    with open("kv_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to kv_benchmark_results.json")

    # Hardware memory measurement (CUDA only)
    if device == "cuda":
        print(f"\n{'=' * 80}")
        print("HARDWARE VALIDATION: Actual CUDA Memory Delta (torch.cuda)")
        print(f"{'=' * 80}")
        print(f"{'Seq Len':>8} | {'Standard Delta':>16} | {'MoR+3bit Delta':>16} | {'HW Reduction':>14}")
        print("-" * 70)

        std_results = all_results[names[0]]["seq_length_results"]
        ours_results = all_results[names[2]]["seq_length_results"]

        for i in range(len(std_results)):
            seq_len = std_results[i]["seq_len"]
            std_delta = std_results[i].get("cuda_delta_bytes", 0)
            ours_delta = ours_results[i].get("cuda_delta_bytes", 0)
            if std_delta > 0 and ours_delta > 0:
                hw_ratio = std_delta / ours_delta
                print(f"{seq_len:>8} | {std_delta:>14,} B | {ours_delta:>14,} B | {hw_ratio:>13.2f}x")

        print(f"\nNote: CUDA deltas include activations + KV cache. Analytical")
        print(f"measurement isolates KV cache only and is more precise.")


if __name__ == "__main__":
    main()
