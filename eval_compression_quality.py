"""
KV compression reconstruction fidelity.

Measures how well TurboQuant reconstructs the K/V vectors a model actually
produces, across bit-widths and with/without QJL. This is the quantization
quality table for the paper — it explains *why* compressed perplexity moves
the way it does in train.py.

Optionally loads a trained checkpoint:
    python eval_compression_quality.py [checkpoint.pt]
With no argument it uses a random-init model (distributions are still
representative for a fidelity sanity check, but a trained checkpoint is better).

Run: python eval_compression_quality.py
"""

import sys
import torch
import torch.nn.functional as F
from mor_tq import MoRConfig, MoRModel
from mor_tq.compression import TurboQuantCompressor


def capture_kv(model, input_ids):
    """Run a forward pass and capture K, V from the shared block's attention."""
    grabbed = {}

    def hook(_module, _inp, out):
        # attention returns (output, K, V)
        grabbed["K"] = out[1].detach().float()
        grabbed["V"] = out[2].detach().float()

    handle = model.shared_block.attention.register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return grabbed["K"], grabbed["V"]


def fidelity(compressor, x):
    """Cosine similarity and relative L2 error of compress->decompress."""
    recon = compressor.decompress(compressor.compress(x))
    x_flat = x.reshape(-1, x.shape[-1])
    r_flat = recon.reshape(-1, x.shape[-1])
    cos = F.cosine_similarity(x_flat, r_flat, dim=-1).mean().item()
    rel_l2 = ((x_flat - r_flat).norm(dim=-1) / (x_flat.norm(dim=-1) + 1e-8)).mean().item()
    return cos, rel_l2


def main():
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )

    config = MoRConfig(
        d_model=512, n_heads=8, d_ff=2048, n_recursions=8,
        sharing_strategy="middle_cycle", n_unique_intro=1, n_unique_outro=1,
        capacity_factor=0.5, routing_strategy="expert",
        kv_bits=3, use_qjl=True, vocab_size=50257, max_seq_len=256, dropout=0.0,
    )
    model = MoRModel(config).to(device).eval()

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt:
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("No checkpoint given — using random-init model (representative, not trained).")

    input_ids = torch.randint(0, config.vocab_size, (4, 256)).to(device)
    K, V = capture_kv(model, input_ids)
    print(f"Captured K{tuple(K.shape)} V{tuple(V.shape)} on {device}\n")

    settings = [
        ("3-bit, QJL on  (2-bit PQ + 1-bit QJL)", 3, True),
        ("3-bit, QJL off (3-bit PQ)",             3, False),
        ("4-bit, QJL on  (3-bit PQ + 1-bit QJL)", 4, True),
        ("4-bit, QJL off (4-bit PQ)",             4, False),
    ]

    head_dim = config.head_dim
    gsize = config.kv_group_size

    print(f"{'Setting':<42} {'K cos':>7} {'K relL2':>8} {'V cos':>7} {'V relL2':>8} {'ratio':>7}")
    print("-" * 86)
    rows = []
    for label, bits, use_qjl in settings:
        comp = TurboQuantCompressor(head_dim=head_dim, bits=bits,
                                    group_size=gsize, use_qjl=use_qjl).to(device)
        kc, kl = fidelity(comp, K)
        vc, vl = fidelity(comp, V)
        ratio = comp.compression_ratio()
        rows.append((label, kc, kl, vc, vl, ratio))
        print(f"{label:<42} {kc:>7.3f} {kl:>8.3f} {vc:>7.3f} {vl:>8.3f} {ratio:>6.2f}x")
    print("-" * 86)
    print("cos = cosine similarity to original (1.0 = perfect). relL2 = mean relative L2 error.")
    print("ratio = analytical compression vs FP16 per entry.")

    print("\nNote: if 3-bit/QJL-on shows lower cosine than 3-bit/QJL-off, the QJL")
    print("residual stage is hurting at this bit budget — that is a real finding,")
    print("not a bug. It tells you which configuration to report as the headline.")


if __name__ == "__main__":
    main()
