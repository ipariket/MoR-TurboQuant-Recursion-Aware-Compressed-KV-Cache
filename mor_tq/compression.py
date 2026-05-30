"""
TurboQuant KV cache compression (ICLR 2026, Google Research).

Implements the full two-stage algorithm:
    Stage 1 — PolarQuant:
        1. Walsh-Hadamard Transform (WHT) — decorrelates the vector
        2. Per-group normalization — extract norm, normalize to unit sphere
        3. Lloyd-Max quantization — (b-1)-bit MSE-optimal codebook
        4. Pack indices into compact bit representation

    Stage 2 — QJL Residual Correction:
        5. Compute residual (original - quantized) in WHT domain
        6. Project residual through random Gaussian matrix
        7. Store only the signs (1 bit per projection)
        8. On decompression, reconstruct unbiased correction

    Total: (b-1) bits PolarQuant + 1 bit QJL = b bits total

References:
    - TurboQuant: arXiv 2504.19874 (Zandieh et al., Google Research)
    - PolarQuant: AISTATS 2026
    - QJL: AAAI 2025
"""

import torch
import torch.nn as nn
import math
from typing import NamedTuple


class CompressedKV(NamedTuple):
    """Compressed key or value vectors."""
    indices: torch.Tensor      # packed quantization indices, uint8
    norms: torch.Tensor        # per-group norms, float32
    qjl_signs: torch.Tensor    # QJL sign bits, packed uint8 (None if QJL disabled)
    residual_norms: torch.Tensor  # per-vector residual norms for QJL (None if disabled)
    shape: tuple               # original shape for reconstruction
    bits: int                  # total bits (e.g. 3 = 2-bit PolarQuant + 1-bit QJL)
    qjl_enabled: bool          # whether QJL correction is active


def _generate_lloyd_max_codebook(bits: int) -> torch.Tensor:
    """Generate MSE-optimal Lloyd-Max centroids for Gaussian-distributed data.

    After WHT + normalization, the components are approximately N(0, 1/d).
    Lloyd-Max gives the MSE-optimal quantizer for this distribution.

    Args:
        bits: number of bits for PolarQuant stage (NOT total bits)
    """
    if bits == 2:
        # 4 centroids for N(0,1) — used when total=3 (2-bit PQ + 1-bit QJL)
        centroids = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104])
    elif bits == 3:
        # 8 centroids for N(0,1) — used when total=4 (3-bit PQ + 1-bit QJL)
        centroids = torch.tensor([
            -1.7479, -1.0500, -0.5006, -0.1257,
             0.1257,  0.5006,  1.0500,  1.7479
        ])
    elif bits == 4:
        # 16 centroids for N(0,1) — standalone 4-bit without QJL
        centroids = torch.tensor([
            -2.4008, -1.8435, -1.4371, -1.0993,
            -0.7995, -0.5224, -0.2582,  0.0000,
             0.2582,  0.5224,  0.7995,  1.0993,
             1.4371,  1.8435,  2.0483,  2.4008
        ])
    else:
        raise ValueError(f"Unsupported PolarQuant bits: {bits}")

    assert len(centroids) == 2 ** bits
    return centroids


def _walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform.

    O(d log d) via iterative butterfly operations.
    Decorrelates the input vector so quantization error distributes evenly.
    """
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"WHT requires power-of-2 dimension, got {d}"

    result = x.clone()
    orig_shape = result.shape
    flat = result.reshape(-1, d)

    h = 1
    while h < d:
        for i in range(0, d, 2 * h):
            a = flat[:, i:i + h].clone()
            b = flat[:, i + h:i + 2 * h].clone()
            flat[:, i:i + h] = a + b
            flat[:, i + h:i + 2 * h] = a - b
        h *= 2

    result = flat.reshape(orig_shape) / math.sqrt(d)
    return result


def _inverse_wht(x: torch.Tensor) -> torch.Tensor:
    """Inverse WHT — same as forward (involution up to scaling)."""
    return _walsh_hadamard_transform(x)


class TurboQuantCompressor(nn.Module):
    """Full TurboQuant compressor for KV cache vectors.

    Two-stage pipeline:
        Stage 1 (PolarQuant): WHT → group norm → Lloyd-Max at (b-1) bits
        Stage 2 (QJL): residual → random projection → sign bits (1 bit)

    Total: b bits per coordinate.

    Args:
        head_dim: dimension per attention head
        bits: total bits per coordinate (3 = 2-bit PQ + 1-bit QJL)
        group_size: group size for per-group normalization
        use_qjl: whether to enable QJL correction (True = full TurboQuant)
        qjl_dim: number of QJL projection dimensions (default = head_dim)
    """

    def __init__(
        self,
        head_dim: int,
        bits: int = 3,
        group_size: int = 128,
        use_qjl: bool = True,
        qjl_dim: int = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.total_bits = bits
        self.use_qjl = use_qjl

        # PolarQuant uses (bits-1) if QJL enabled, else all bits
        self.pq_bits = bits - 1 if use_qjl else bits
        self.n_centroids = 2 ** self.pq_bits

        # Pad head_dim to next power of 2 for WHT
        self.padded_dim = 1 << (head_dim - 1).bit_length()
        self.group_size = min(group_size, self.padded_dim)

        # QJL projection dimension
        self.qjl_dim = qjl_dim or self.padded_dim

        # Lloyd-Max codebook for PolarQuant stage
        codebook = _generate_lloyd_max_codebook(self.pq_bits)
        self.register_buffer("codebook", codebook)

        # Decision boundaries
        boundaries = (codebook[:-1] + codebook[1:]) / 2
        self.register_buffer("boundaries", boundaries)

        # Fixed random projection matrix for QJL (not learned)
        if use_qjl:
            # Gaussian random matrix, normalized
            proj = torch.randn(self.padded_dim, self.qjl_dim) / math.sqrt(self.qjl_dim)
            self.register_buffer("qjl_projection", proj)

    def compress(self, x: torch.Tensor) -> CompressedKV:
        """Compress key or value vectors using full TurboQuant.

        Args:
            x: (batch, n_heads, seq_len, head_dim)

        Returns:
            CompressedKV with packed PolarQuant indices + QJL signs.
        """
        original_shape = x.shape
        device = x.device

        # Flatten to (N, head_dim)
        x_flat = x.reshape(-1, self.head_dim).float()
        N = x_flat.shape[0]

        # Pad to power-of-2
        if self.head_dim != self.padded_dim:
            x_padded = torch.zeros(N, self.padded_dim, device=device)
            x_padded[:, :self.head_dim] = x_flat
            x_flat = x_padded

        # === Stage 1: PolarQuant ===
        # WHT rotation
        x_wht = _walsh_hadamard_transform(x_flat)

        # Group and extract norms
        n_groups = self.padded_dim // self.group_size
        x_grouped = x_wht.view(N, n_groups, self.group_size)
        norms = x_grouped.norm(dim=-1, keepdim=True)  # (N, n_groups, 1)
        norms_flat = norms.squeeze(-1)

        # Normalize
        x_normed = x_grouped / (norms + 1e-8)

        # Lloyd-Max quantization
        x_for_quant = x_normed.reshape(-1)
        indices = torch.searchsorted(self.boundaries, x_for_quant)
        indices = indices.clamp(0, self.n_centroids - 1)
        indices = indices.view(N, n_groups * self.group_size)

        # Reconstruct quantized version (for residual computation)
        quantized_normed = self.codebook[indices.long()].view(N, n_groups, self.group_size)
        quantized_wht = (quantized_normed * norms).view(N, self.padded_dim)

        # Pack PolarQuant indices
        packed_indices = self._pack_indices(indices)

        # === Stage 2: QJL Residual Correction ===
        qjl_signs = None
        residual_norms = None

        if self.use_qjl:
            # Residual in WHT domain
            residual = x_wht - quantized_wht  # (N, padded_dim)

            # Store residual norm for reconstruction scaling
            residual_norms = residual.norm(dim=-1)  # (N,)

            # Project through random Gaussian matrix
            projected = residual @ self.qjl_projection  # (N, qjl_dim)

            # Keep only signs (1 bit per projection)
            signs = (projected >= 0).to(torch.uint8)  # (N, qjl_dim)

            # Pack sign bits into uint8
            qjl_signs = self._pack_bits(signs)

        return CompressedKV(
            indices=packed_indices,
            norms=norms_flat,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            shape=original_shape,
            bits=self.total_bits,
            qjl_enabled=self.use_qjl,
        )

    def decompress(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress back to full-precision vectors."""
        device = compressed.norms.device
        N = compressed.norms.shape[0]
        n_groups = self.padded_dim // self.group_size

        # === Stage 1: PolarQuant reconstruction ===
        total_scalars = N * n_groups * self.group_size
        indices = self._unpack_indices(compressed.indices, total_scalars)
        indices = indices.view(N, n_groups, self.group_size)

        x_normed = self.codebook[indices.long()]
        norms = compressed.norms.unsqueeze(-1)
        x_scaled = x_normed * norms
        x_wht = x_scaled.view(N, self.padded_dim)

        # === Stage 2: QJL correction ===
        if compressed.qjl_enabled and compressed.qjl_signs is not None:
            # Unpack sign bits
            signs = self._unpack_bits(compressed.qjl_signs, N * self.qjl_dim)
            signs = signs.view(N, self.qjl_dim).float()

            # Convert 0/1 to -1/+1
            signs = 2.0 * signs - 1.0

            # Reconstruct residual estimate: R_hat = ||r|| * P^T * sign(P * r) / qjl_dim
            residual_norms = compressed.residual_norms.unsqueeze(-1)  # (N, 1)
            correction = signs @ self.qjl_projection.t()  # (N, padded_dim)

            # Scale by residual norm / sqrt(qjl_dim)
            correction = correction * residual_norms / math.sqrt(self.qjl_dim)

            x_wht = x_wht + correction

        # Inverse WHT
        x_restored = _inverse_wht(x_wht)

        # Remove padding
        if self.head_dim != self.padded_dim:
            x_restored = x_restored[:, :self.head_dim]

        return x_restored.view(compressed.shape)

    def _pack_bits(self, bits_tensor: torch.Tensor) -> torch.Tensor:
        """Pack binary tensor into uint8 (8 bits per byte)."""
        flat = bits_tensor.reshape(-1).to(torch.uint8)
        # Pad to multiple of 8
        pad_len = (8 - flat.shape[0] % 8) % 8
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8, device=flat.device)])

        flat = flat.view(-1, 8)
        packed = torch.zeros(flat.shape[0], dtype=torch.uint8, device=flat.device)
        for i in range(8):
            packed |= (flat[:, i] << (7 - i))
        return packed

    def _unpack_bits(self, packed: torch.Tensor, total_bits: int) -> torch.Tensor:
        """Unpack uint8 back to individual bits."""
        result = torch.zeros(packed.shape[0] * 8, dtype=torch.uint8, device=packed.device)
        for i in range(8):
            result[i::8] = (packed >> (7 - i)) & 1
        return result[:total_bits]

    def _pack_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Pack quantization indices into uint8."""
        indices = indices.to(torch.uint8)

        if self.pq_bits == 4:
            assert indices.shape[-1] % 2 == 0
            even = indices[..., 0::2]
            odd = indices[..., 1::2]
            packed = (even << 4) | odd
            return packed

        elif self.pq_bits == 3:
            flat = indices.reshape(-1)
            pad_len = (8 - flat.shape[0] % 8) % 8
            if pad_len > 0:
                flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8, device=flat.device)])
            flat = flat.view(-1, 8)
            byte0 = (flat[:, 0] << 5) | (flat[:, 1] << 2) | (flat[:, 2] >> 1)
            byte1 = ((flat[:, 2] & 1) << 7) | (flat[:, 3] << 4) | (flat[:, 4] << 1) | (flat[:, 5] >> 2)
            byte2 = ((flat[:, 5] & 3) << 6) | (flat[:, 6] << 3) | flat[:, 7]
            packed = torch.stack([byte0, byte1, byte2], dim=-1).reshape(-1)
            return packed

        elif self.pq_bits == 2:
            # 4 indices per byte
            flat = indices.reshape(-1)
            pad_len = (4 - flat.shape[0] % 4) % 4
            if pad_len > 0:
                flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8, device=flat.device)])
            flat = flat.view(-1, 4)
            packed = (flat[:, 0] << 6) | (flat[:, 1] << 4) | (flat[:, 2] << 2) | flat[:, 3]
            return packed

        else:
            raise ValueError(f"Unsupported PolarQuant bits: {self.pq_bits}")

    def _unpack_indices(self, packed: torch.Tensor, total_elements: int) -> torch.Tensor:
        """Unpack uint8 back to individual indices."""
        if self.pq_bits == 4:
            high = (packed >> 4) & 0x0F
            low = packed & 0x0F
            unpacked = torch.stack([high, low], dim=-1).reshape(-1)
            return unpacked[:total_elements]

        elif self.pq_bits == 3:
            packed = packed.view(-1, 3)
            b0, b1, b2 = packed[:, 0], packed[:, 1], packed[:, 2]
            i0 = (b0 >> 5) & 0x07
            i1 = (b0 >> 2) & 0x07
            i2 = ((b0 & 0x03) << 1) | ((b1 >> 7) & 0x01)
            i3 = (b1 >> 4) & 0x07
            i4 = (b1 >> 1) & 0x07
            i5 = ((b1 & 0x01) << 2) | ((b2 >> 6) & 0x03)
            i6 = (b2 >> 3) & 0x07
            i7 = b2 & 0x07
            unpacked = torch.stack([i0, i1, i2, i3, i4, i5, i6, i7], dim=-1).reshape(-1)
            return unpacked[:total_elements]

        elif self.pq_bits == 2:
            i0 = (packed >> 6) & 0x03
            i1 = (packed >> 4) & 0x03
            i2 = (packed >> 2) & 0x03
            i3 = packed & 0x03
            unpacked = torch.stack([i0, i1, i2, i3], dim=-1).reshape(-1)
            return unpacked[:total_elements]

        else:
            raise ValueError(f"Unsupported PolarQuant bits: {self.pq_bits}")

    def compression_ratio(self) -> float:
        """Theoretical compression ratio vs FP16."""
        fp16_bits = 16
        # PolarQuant: pq_bits per element + amortized norm storage
        norm_bits = 32 / self.group_size  # fp32 norm amortized
        pq_cost = self.pq_bits + norm_bits

        # QJL: 1 bit per projection dim + amortized residual norm
        qjl_cost = 0
        if self.use_qjl:
            qjl_cost = (self.qjl_dim / self.padded_dim)  # 1 bit per qjl_dim, amortized over padded_dim
            qjl_cost += 32 / self.padded_dim  # residual norm (fp32) amortized

        total_cost = pq_cost + qjl_cost
        return fp16_bits / total_cost

    def memory_bytes(self, n_tokens: int, n_heads: int) -> dict:
        """Calculate actual memory usage for a given cache size."""
        n_groups = self.padded_dim // self.group_size
        total_scalars = n_tokens * n_heads * self.padded_dim
        N = n_tokens * n_heads  # total vectors

        # PolarQuant indices
        if self.pq_bits == 4:
            index_bytes = total_scalars // 2
        elif self.pq_bits == 3:
            index_bytes = (total_scalars * 3) // 8
        elif self.pq_bits == 2:
            index_bytes = total_scalars // 4
        else:
            index_bytes = 0

        # Per-group norms (fp32)
        norm_bytes = N * n_groups * 4

        # QJL sign bits
        qjl_bytes = 0
        residual_norm_bytes = 0
        if self.use_qjl:
            qjl_bytes = (N * self.qjl_dim + 7) // 8  # packed bits
            residual_norm_bytes = N * 4  # fp32 residual norms

        # FP16 baseline
        fp16_bytes = n_tokens * n_heads * self.head_dim * 2

        compressed_total = index_bytes + norm_bytes + qjl_bytes + residual_norm_bytes

        return {
            "compressed_bytes": compressed_total,
            "fp16_bytes": fp16_bytes,
            "ratio": fp16_bytes / max(1, compressed_total),
            "breakdown": {
                "pq_indices": index_bytes,
                "group_norms": norm_bytes,
                "qjl_signs": qjl_bytes,
                "residual_norms": residual_norm_bytes,
            }
        }


# Backward compatibility alias
PolarQuantCompressor = TurboQuantCompressor
