"""
KV Cache Memory Benchmark — 4-way comparison.

Measures KV cache memory for:
  1. Standard transformer (baseline)
  2. Standard + TurboQuant (compression only)
  3. MoR only (early exit only)
  4. MoR + TurboQuant (our full system)

Run: python benchmark_kv.py
"""

import torch
import json
import gc
from mor_tq import MoRConfig, MoRModel


def benchmark_config(name, config, seq_lengths, device="cpu"):
    model = MoRModel(config).to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())

    results = {"name": name, "total_params": total_params, "seq_length_results": []}

    for seq_len in seq_lengths:
        if seq_len > config.max_seq_len:
            continue
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len)).to(device)

        with torch.no_grad():
            output = model(input_ids)

        stats = output.kv_stats
        entry = {
            "seq_len": seq_len,
            "standard_bytes": stats["standard_baseline_bytes"],
            "mor_bytes": stats["mor_baseline_bytes"],
            "actual_bytes": stats["actual_bytes"],
            "compression_vs_standard": round(stats["compression_vs_standard"], 2),
            "active_fraction": round(stats["active_token_fraction"] * 100, 1),
        }

        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            with torch.no_grad():
                output = model(input_ids)
            torch.cuda.synchronize()
            entry["cuda_delta_bytes"] = torch.cuda.max_memory_allocated() - mem_before

        results["seq_length_results"].append(entry)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def main():
    print("=" * 100)
    print("KV Cache Memory Benchmark: 4-Way Comparison")
    print("=" * 100)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print(f"Device: {device}\n")

    MAX_SEQ = 2048
    SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048]

    COMMON = dict(d_model=512, n_heads=8, d_ff=2048, vocab_size=1000, max_seq_len=MAX_SEQ, dropout=0.0)

    configs = {
        "Standard": MoRConfig(
            **COMMON, n_recursions=8, sharing_strategy="full",
            capacity_factor=1.0, routing_strategy="expert", kv_bits=0,
        ),
        "TurboQuant only": MoRConfig(
            **COMMON, n_recursions=8, sharing_strategy="full",
            capacity_factor=1.0, routing_strategy="expert",
            kv_bits=3, use_qjl=True,
        ),
        "MoR only": MoRConfig(
            **COMMON, n_recursions=8, sharing_strategy="middle_cycle",
            n_unique_intro=1, n_unique_outro=1,
            capacity_factor=0.5, routing_strategy="expert", kv_bits=0,
        ),
        "MoR+TurboQuant": MoRConfig(
            **COMMON, n_recursions=8, sharing_strategy="middle_cycle",
            n_unique_intro=1, n_unique_outro=1,
            capacity_factor=0.5, routing_strategy="expert",
            kv_bits=3, use_qjl=True,
        ),
    }

    all_results = {}
    for name, cfg in configs.items():
        print(f"Benchmarking: {name}...")
        all_results[name] = benchmark_config(name, cfg, SEQ_LENGTHS, device)

    names = list(configs.keys())

    print(f"\n{'=' * 110}")
    print("KV Cache Memory (bytes per sample, batch=1)")
    print(f"{'=' * 110}")
    print(f"{'Seq':>6} | {'Standard':>14} | {'TurboQuant':>14} | {'MoR only':>14} | {'MoR+TQ (Ours)':>14} | {'TQ ratio':>10} | {'Ours ratio':>10}")
    print("-" * 110)

    for i, seq_len in enumerate(SEQ_LENGTHS):
        bytes_list = []
        for name in names:
            res = all_results[name]["seq_length_results"]
            b = res[i]["actual_bytes"] if i < len(res) else 0
            bytes_list.append(b)

        tq_ratio = bytes_list[0] / max(1, bytes_list[1]) if bytes_list[1] > 0 else 0
        ours_ratio = bytes_list[0] / max(1, bytes_list[3]) if bytes_list[3] > 0 else 0

        print(f"{seq_len:>6} | {bytes_list[0]:>14,} | {bytes_list[1]:>14,} | {bytes_list[2]:>14,} | {bytes_list[3]:>14,} | {tq_ratio:>9.2f}x | {ours_ratio:>9.2f}x")

    print(f"{'=' * 110}")

    # Summary
    print(f"\nCompression Breakdown at seq_len=2048:")
    for name in names:
        res = all_results[name]["seq_length_results"][-1]
        mb = res["actual_bytes"] / (1024 * 1024)
        print(f"  {name:<20} {res['actual_bytes']:>12,} bytes ({mb:.2f} MB) = {res['compression_vs_standard']:.2f}x vs standard")

    # Multiplicative analysis
    tq_ratio = all_results["Standard"]["seq_length_results"][-1]["actual_bytes"] / \
               max(1, all_results["TurboQuant only"]["seq_length_results"][-1]["actual_bytes"])
    mor_ratio = all_results["Standard"]["seq_length_results"][-1]["actual_bytes"] / \
                max(1, all_results["MoR only"]["seq_length_results"][-1]["actual_bytes"])
    combined = all_results["Standard"]["seq_length_results"][-1]["actual_bytes"] / \
               max(1, all_results["MoR+TurboQuant"]["seq_length_results"][-1]["actual_bytes"])

    print(f"\n  TurboQuant contribution: {tq_ratio:.2f}x")
    print(f"  MoR contribution:       {mor_ratio:.2f}x")
    print(f"  Combined:               {combined:.2f}x (multiplicative: {tq_ratio:.2f} × {mor_ratio:.2f} = {tq_ratio*mor_ratio:.2f}x)")

    with open("kv_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to kv_benchmark_results.json")


if __name__ == "__main__":
    main()
