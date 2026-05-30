"""
MoR-TurboQuant Training on WikiText-103
Run: python train.py

Optimized for Apple M4 Mac Mini (16GB)
Uses MPS (Metal Performance Shaders) for GPU acceleration
"""

import torch
import math
import time
import json
import os
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Config
# ============================================================
SEQ_LEN = 256
BATCH_SIZE = 16          # smaller batch for 16GB RAM
N_EPOCHS = 2             # 2 epochs balances quality vs time
LR = 3e-4
EVAL_EVERY = 500
LOG_EVERY = 50
GRAD_CLIP = 1.0
ROUTER_LOSS_WEIGHT = 0.01

# ============================================================
# Device setup
# ============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M4 GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU (will be slow)")

# ============================================================
# Load dataset
# ============================================================
print("\nStep 1: Loading WikiText-103...")
from datasets import load_dataset
import tiktoken

raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
enc = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = enc.n_vocab
print(f"Vocab size: {VOCAB_SIZE}")


def tokenize_split(split_name):
    texts = raw_dataset[split_name]["text"]
    all_tokens = []
    for text in texts:
        if text.strip():
            all_tokens.extend(enc.encode(text))
    return torch.tensor(all_tokens, dtype=torch.long)


print("Tokenizing train...")
train_tokens = tokenize_split("train")
print("Tokenizing validation...")
val_tokens = tokenize_split("validation")
print("Tokenizing test...")
test_tokens = tokenize_split("test")
print(f"Train: {len(train_tokens):,} | Val: {len(val_tokens):,} | Test: {len(test_tokens):,} tokens")


# ============================================================
# Dataset
# ============================================================
class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_chunks = len(tokens) // seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start: start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


train_loader = DataLoader(
    TokenDataset(train_tokens, SEQ_LEN),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False,
)
val_loader = DataLoader(
    TokenDataset(val_tokens, SEQ_LEN),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
)
test_loader = DataLoader(
    TokenDataset(test_tokens, SEQ_LEN),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
)
print(f"Seq length: {SEQ_LEN} | Batch size: {BATCH_SIZE} | Train batches: {len(train_loader)}")


# ============================================================
# Model configs
# ============================================================
print("\nStep 2: Setting up models...")
from mor_tq import MoRConfig, MoRModel

COMMON = dict(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    vocab_size=VOCAB_SIZE,
    max_seq_len=SEQ_LEN,
    dropout=0.1,
)

configs = {
    "Standard 8-layer": MoRConfig(
        **COMMON,
        n_recursions=8,
        sharing_strategy="full",
        capacity_factor=1.0,
        routing_strategy="expert",
        kv_bits=0,
    ),
    "MoR + 3-bit KV (Ours)": MoRConfig(
        **COMMON,
        n_recursions=8,
        sharing_strategy="middle_cycle",
        n_unique_intro=1,
        n_unique_outro=1,
        capacity_factor=0.5,
        routing_strategy="expert",
        kv_bits=3,
    ),
}

# Print param counts
print(f"\n{'Model':<30} {'Params':>12}")
print("-" * 45)
for name, cfg in configs.items():
    m = MoRModel(cfg)
    total = sum(p.numel() for p in m.parameters())
    print(f"{name:<30} {total:>12,}")
    del m


# ============================================================
# Training functions
# ============================================================
@torch.no_grad()
def evaluate(model, loader, max_batches=None):
    model.eval()
    total_loss = 0
    total_tokens = 0
    kv_stats_sum = defaultdict(float)
    n_batches = 0

    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        output = model(x, labels=y)

        total_loss += output.loss.item() * x.shape[0] * x.shape[1]
        total_tokens += x.shape[0] * x.shape[1]

        for k, v in output.kv_stats.items():
            if isinstance(v, (int, float)):
                kv_stats_sum[k] += v
        n_batches += 1

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
    avg_kv_stats = {k: v / max(1, n_batches) for k, v in kv_stats_sum.items()}

    return perplexity, avg_loss, avg_kv_stats


def train_model(name, config):
    print(f"\n{'=' * 60}")
    print(f"Training: {name}")
    print(f"{'=' * 60}")

    model = MoRModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1,
    )
    total_steps = N_EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=LR / 10)

    global_step = 0
    best_val_ppl = float("inf")
    val_ppls = []
    start_time = time.time()
    save_path = f"{name.replace(' ', '_').replace('(', '').replace(')', '')}_best.pt"

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_tokens = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            output = model(x, labels=y)
            loss = output.loss + ROUTER_LOSS_WEIGHT * output.router_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += output.loss.item() * x.shape[0] * x.shape[1]
            epoch_tokens += x.shape[0] * x.shape[1]
            global_step += 1

            if global_step % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / epoch_tokens
                ppl = math.exp(min(avg_loss, 20))
                depth = output.exit_depths.float().mean().item()
                active = output.kv_stats.get("active_token_fraction", 1.0)
                print(f"  Step {global_step:5d} | Loss {avg_loss:.3f} | PPL {ppl:.1f} | "
                      f"Depth {depth:.1f} | Active {active:.0%} | {elapsed:.0f}s")

            if global_step % EVAL_EVERY == 0:
                val_ppl, val_loss, val_kv = evaluate(model, val_loader, max_batches=50)
                val_ppls.append((global_step, val_ppl))
                kv_comp = val_kv.get("compression_vs_standard", 1.0)
                print(f"  >>> Val PPL: {val_ppl:.2f} | KV compression: {kv_comp:.1f}x")
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    torch.save(model.state_dict(), save_path)
                    print(f"  >>> Saved best checkpoint")
                model.train()

        # End of epoch
        val_ppl, _, _ = evaluate(model, val_loader)
        train_ppl = math.exp(min(epoch_loss / epoch_tokens, 20))
        print(f"\n  Epoch {epoch + 1}/{N_EPOCHS} | Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

    # Final test eval
    print(f"\nLoading best checkpoint...")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    test_ppl, test_loss, test_kv = evaluate(model, test_loader)
    param_stats = model.count_parameters()
    total_time = time.time() - start_time

    results = {
        "name": name,
        "test_ppl": round(test_ppl, 2),
        "best_val_ppl": round(best_val_ppl, 2),
        "total_params": total_params,
        "param_savings": round(param_stats["parameter_savings"] * 100, 1),
        "kv_compression": round(test_kv.get("compression_vs_standard", 1.0), 1),
        "active_token_fraction": round(test_kv.get("active_token_fraction", 1.0) * 100, 1),
        "training_time_min": round(total_time / 60, 1),
    }

    print(f"\n  FINAL: Test PPL={test_ppl:.2f} | KV={results['kv_compression']}x | "
          f"Params={total_params:,} | Time={total_time / 60:.1f}min")

    del model, optimizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


# ============================================================
# Train all variants
# ============================================================
print(f"\nStep 3: Training ({N_EPOCHS} epochs each)...")
print(f"Estimated time: ~4-6 hours on M4 Mac Mini\n")

all_results = []
for name, cfg in configs.items():
    result = train_model(name, cfg)
    all_results.append(result)

# ============================================================
# Print results table
# ============================================================
print(f"\n\n{'=' * 80}")
print("RESULTS: WikiText-103 Language Modeling")
print(f"{'=' * 80}")
print(f"{'Model':<30} {'Test PPL':>10} {'KV Compress':>12} {'Params':>12} {'Savings':>10}")
print("-" * 80)
for r in all_results:
    kv = f"{r['kv_compression']}x"
    save = f"{r['param_savings']}%" if r['param_savings'] > 1 else "—"
    print(f"{r['name']:<30} {r['test_ppl']:>10} {kv:>12} {r['total_params']:>12,} {save:>10}")
print("=" * 80)

if len(all_results) == 2:
    baseline = all_results[0]['test_ppl']
    ours = all_results[1]['test_ppl']
    kv = all_results[1]['kv_compression']
    diff = ours - baseline
    print(f"\nKey finding: MoR + 3-bit KV achieves {kv}x KV memory reduction")
    print(f"with {diff:+.2f} perplexity difference vs standard transformer.")

# Save results
with open("training_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to training_results.json")
print("Done!")
