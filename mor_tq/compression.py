"""
PolarQuant-style KV cache compression.

Implements the core math from TurboQuant+:
    1. Walsh-Hadamard Transform (WHT) — decorrelates the vector
    2. Per-group normalization — extract norm, normalize to unit sphere  
    3. Lloyd-Max quantization — MSE-optimal codebook for the normalized values
    4. Pack indices into compact bit representation

This is the scalar case of HIGGS (Malinovskii et al., NAACL 2025).
"""

import torch
import torch.nn as nn
import math
from typing import NamedTuple


class CompressedKV(NamedTuple):
    """Compressed key or value vectors."""
    indices: torch.Tensor     # packed quantization indices, uint8
    norms: torch.Tensor       # per-group norms, float32
    shape: tuple              # original shape for reconstruction
    bits: int                 # 3 or 4


def _generate_lloyd_max_codebook(bits: int) -> torch.Tensor:
    """Generate MSE-optimal Lloyd-Max centroids for Gaussian-distributed data.

    After WHT + normalization, the components are approximately N(0, 1/d).
    Lloyd-Max gives the MSE-optimal quantizer for this distribution.

    For 3-bit (8 centroids) and 4-bit (16 centroids), these are well-known
    tabulated values for the standard normal distribution.
    """
    if bits == 3:
        # 8 centroids for N(0,1) — classical Lloyd-Max solution
        centroids = torch.tensor([
            -1.7479, -1.0500, -0.5006, -0.1257,
             0.1257,  0.5006,  1.0500,  1.7479
        ])
    elif bits == 4:
        # 16 centroids for N(0,1)
        centroids = torch.tensor([
            -2.1519, -1.6104, -1.2044, -0.8605,
            -0.5540, -0.2698, -0.0942,  0.0942,
             0.2698,  0.5540,  0.8605,  1.2044,
             1.6104,  2.1519,  2.4010,  2.7500
        ])
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    assert len(centroids) == 2 ** bits, f"Expected {2**bits} centroids, got {len(centroids)}"
    return centroids


def _walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform.

    O(d log d) via iterative butterfly operations.
    Decorrelates the input vector so quantization error distributes evenly.

    Args:
        x: (..., d) where d must be a power of 2
    
    Returns:
        WHT of x (new tensor, input not modified)
    """
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"WHT requires power-of-2 dimension, got {d}"

    # Work on a copy to avoid mutating input
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
    """Inverse WHT — same as forward WHT (it's an involution up to scaling).
    
    WHT is self-inverse: H * H = d * I, so H^{-1} = H / d.
    Since forward already divides by sqrt(d), applying it again gives the inverse.
    """
    return _walsh_hadamard_transform(x)


class PolarQuantCompressor(nn.Module):
    """PolarQuant-style compressor for KV cache vectors.

    Pipeline per vector:
        1. WHT rotation (decorrelate components)
        2. Group into chunks of group_size
        3. Extract and store per-group L2 norm
        4. Normalize each group to unit norm
        5. Quantize each scalar to nearest Lloyd-Max centroid
        6. Pack indices into uint8

    Decompression reverses: unpack → centroid lookup → rescale by norm → inverse WHT.
    """

    def __init__(self, head_dim: int, bits: int = 3, group_size: int = 128):
        super().__init__()
        self.head_dim = head_dim
        self.bits = bits
        self.n_centroids = 2 ** bits

        # Pad head_dim to next power of 2 for WHT
        self.padded_dim = 1 << (head_dim - 1).bit_length()

        # Group size can't exceed padded_dim
        self.group_size = min(group_size, self.padded_dim)

        # Register codebook as buffer (not a parameter — no grad)
        codebook = _generate_lloyd_max_codebook(bits)
        self.register_buffer("codebook", codebook)

        # Precompute decision boundaries (midpoints between centroids)
        boundaries = (codebook[:-1] + codebook[1:]) / 2
        self.register_buffer("boundaries", boundaries)

    def compress(self, x: torch.Tensor) -> CompressedKV:
        """Compress key or value vectors.

        Args:
            x: (batch, n_heads, seq_len, head_dim) or (n_tokens, head_dim)

        Returns:
            CompressedKV with packed indices and per-group norms.
        """
        original_shape = x.shape
        device = x.device

        # Flatten to (N, head_dim) for processing
        x_flat = x.reshape(-1, self.head_dim).float()
        N = x_flat.shape[0]

        # Step 1: Pad to power-of-2 if needed
        if self.head_dim != self.padded_dim:
            x_padded = torch.zeros(N, self.padded_dim, device=device)
            x_padded[:, :self.head_dim] = x_flat
            x_flat = x_padded

        # Step 2: Walsh-Hadamard Transform
        x_wht = _walsh_hadamard_transform(x_flat)

        # Step 3: Group and extract norms
        n_groups = self.padded_dim // self.group_size
        x_grouped = x_wht.view(N, n_groups, self.group_size)

        # Per-group L2 norm
        norms = x_grouped.norm(dim=-1, keepdim=True)  # (N, n_groups, 1)
        norms_flat = norms.squeeze(-1)  # (N, n_groups)

        # Step 4: Normalize to unit norm per group
        x_normed = x_grouped / (norms + 1e-8)  # (N, n_groups, group_size)

        # Step 5: Scalar quantization via Lloyd-Max
        # searchsorted finds which bin each value falls into
        x_for_quant = x_normed.reshape(-1)  # flatten all scalars
        indices = torch.searchsorted(self.boundaries, x_for_quant)
        indices = indices.clamp(0, self.n_centroids - 1)
        indices = indices.view(N, n_groups * self.group_size)

        # Step 6: Pack into uint8
        packed = self._pack_indices(indices)

        return CompressedKV(
            indices=packed,
            norms=norms_flat,
            shape=original_shape,
            bits=self.bits,
        )

    def decompress(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress back to full-precision vectors.

        Args:
            compressed: CompressedKV from compress()

        Returns:
            Tensor with original shape, approximately reconstructed.
        """
        device = compressed.norms.device
        N = compressed.norms.shape[0]
        n_groups = self.padded_dim // self.group_size

        # Unpack indices
        total_scalars = N * n_groups * self.group_size
        indices = self._unpack_indices(compressed.indices, total_scalars)
        indices = indices.view(N, n_groups, self.group_size)

        # Centroid lookup
        x_normed = self.codebook[indices.long()]  # (N, n_groups, group_size)

        # Rescale by stored norms
        norms = compressed.norms.unsqueeze(-1)  # (N, n_groups, 1)
        x_scaled = x_normed * norms

        # Reshape back to (N, padded_dim)
        x_wht = x_scaled.view(N, self.padded_dim)

        # Inverse WHT
        x_restored = _inverse_wht(x_wht)

        # Remove padding
        if self.head_dim != self.padded_dim:
            x_restored = x_restored[:, :self.head_dim]

        # Reshape to original
        return x_restored.view(compressed.shape)

    def _pack_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Pack quantization indices into uint8.

        For 4-bit: 2 indices per byte (simple nibble packing)
        For 3-bit: 8 indices per 3 bytes (tighter packing)
        """
        indices = indices.to(torch.uint8)

        if self.bits == 4:
            # Two 4-bit indices per byte
            assert indices.shape[-1] % 2 == 0
            even = indices[..., 0::2]
            odd = indices[..., 1::2]
            packed = (even << 4) | odd
            return packed

        elif self.bits == 3:
            # Pack 8 indices (24 bits) into 3 bytes
            flat = indices.reshape(-1)
            # Pad to multiple of 8
            pad_len = (8 - flat.shape[0] % 8) % 8
            if pad_len > 0:
                flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8, device=flat.device)])

            flat = flat.view(-1, 8)  # (groups_of_8, 8)

            # Pack 8 × 3-bit values into 3 bytes
            byte0 = (flat[:, 0] << 5) | (flat[:, 1] << 2) | (flat[:, 2] >> 1)
            byte1 = ((flat[:, 2] & 1) << 7) | (flat[:, 3] << 4) | (flat[:, 4] << 1) | (flat[:, 5] >> 2)
            byte2 = ((flat[:, 5] & 3) << 6) | (flat[:, 6] << 3) | flat[:, 7]

            packed = torch.stack([byte0, byte1, byte2], dim=-1).reshape(-1)
            return packed

        else:
            raise ValueError(f"Unsupported bits: {self.bits}")

    def _unpack_indices(self, packed: torch.Tensor, total_elements: int) -> torch.Tensor:
        """Unpack uint8 back to individual indices."""
        if self.bits == 4:
            high = (packed >> 4) & 0x0F
            low = packed & 0x0F
            unpacked = torch.stack([high, low], dim=-1).reshape(-1)
            return unpacked[:total_elements]

        elif self.bits == 3:
            packed = packed.view(-1, 3)  # groups of 3 bytes
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

        else:
            raise ValueError(f"Unsupported bits: {self.bits}")

    def compression_ratio(self) -> float:
        """Theoretical compression ratio vs FP16."""
        fp16_bits_per_element = 16
        # Compressed: bits per element + amortized norm storage
        norm_bits = 32  # fp32 per group
        compressed_bits = self.bits + (norm_bits / self.group_size)
        return fp16_bits_per_element / compressed_bits

    def memory_bytes(self, n_tokens: int, n_heads: int) -> dict:
        """Calculate actual memory usage for a given cache size."""
        n_groups = self.padded_dim // self.group_size
        total_scalars = n_tokens * n_heads * self.padded_dim

        if self.bits == 4:
            index_bytes = total_scalars // 2
        elif self.bits == 3:
            index_bytes = (total_scalars * 3) // 8
        else:
            index_bytes = 0

        norm_bytes = n_tokens * n_heads * n_groups * 4  # fp32 norms

        fp16_bytes = n_tokens * n_heads * self.head_dim * 2

        return {
            "compressed_bytes": index_bytes + norm_bytes,
            "fp16_bytes": fp16_bytes,
            "ratio": fp16_bytes / max(1, index_bytes + norm_bytes),
        }
