"""MLX DAC 44.1 kHz decoder for Zonos-2.

Mirrors the decode path in Zonos2_TTS-ComfyUI/native.py (shear_up) and
Zonos2_TTS-ComfyUI/runtime.py (Zonos2DAC.decode), then implements the
DacModel decoder in pure MLX.

Decode path
-----------
delayed_codes (frames, codebooks) int64 numpy
  -> shear_up()           -- un-delay the multi-codebook pattern
  -> eos trim             -- keep only valid frames
  -> clamp 0..1023
  -> (1, codebooks, frames)
  -> RVQ.from_codes()     -- per-codebook embedding lookup + out_proj sum
  -> DacDecoder           -- conv + 4x DecoderBlock (Snake+ConvTranspose+ResUnits) + Snake + conv + tanh
  -> (1, samples)  44.1 kHz

Weight layout differences (torch -> MLX)
-----------------------------------------
torch Conv1d        weight (out, in, kernel)       -> MLX (out, kernel, in)  : transpose (0,2,1)
torch ConvTranspose1d weight (in, out, kernel)     -> MLX (out, kernel, in)  : transpose (1,2,0)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

mx.set_memory_limit(int(45 * (1 << 30)))


# ---------------------------------------------------------------------------
# shear_up  (numpy-only, CPU)
# ---------------------------------------------------------------------------

def shear_up(x: np.ndarray, pad_id: int) -> np.ndarray:
    """Un-delay the multi-codebook token pattern.

    For codebook j, column j has been delayed by j rows relative to codebook 0.
    shear_up shifts each column j upward by j positions so all codebooks are
    time-aligned again, filling the vacated tail with *pad_id*.

    Args:
        x:      (frames, codebooks) int64 array of delayed codes.
        pad_id: Fill value for out-of-range positions.

    Returns:
        (frames, codebooks) int64 array, same shape as *x*.
    """
    x = np.asarray(x)
    frames, codebooks = x.shape[-2], x.shape[-1]
    out = np.full_like(x, pad_id)
    for j in range(codebooks):
        valid = frames - j
        if valid > 0:
            out[..., :valid, j] = x[..., j:, j]
    return out


# ---------------------------------------------------------------------------
# MLX building blocks
# ---------------------------------------------------------------------------

class Snake1d(nn.Module):
    """Snake activation: x + (1/(alpha+eps)) * sin^2(alpha * x).

    alpha shape: (1, channels, 1) in torch (channels-first).
    In our MLX code we work channels-last  (B, T, C) so alpha is stored as
    (channels,) and broadcast accordingly.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)  alpha: (C,) -> broadcasts fine
        a = self.alpha + 1e-9
        return x + (1.0 / a) * mx.sin(a * x) ** 2


class ResidualUnit(nn.Module):
    """Snake -> Conv1d(k=7, dilation) -> Snake -> Conv1d(k=1) + residual."""

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, padding=pad, dilation=dilation)
        self.snake2 = Snake1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        h = self.conv1(self.snake1(x))
        h = self.conv2(self.snake2(h))
        # Trim residual if length changed (shouldn't happen with same padding)
        t_h = h.shape[1]
        t_x = x.shape[1]
        if t_x != t_h:
            trim = (t_x - t_h) // 2
            x = x[:, trim : trim + t_h, :]
        return x + h


class DecoderBlock(nn.Module):
    """Snake -> ConvTranspose1d(stride) -> 3x ResidualUnit."""

    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        kernel_size = 2 * stride
        padding = math.ceil(stride / 2)
        self.snake1 = Snake1d(input_dim)
        # MLX ConvTranspose1d: weight stored (out, kernel, in); padding trims output
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.res_unit1 = ResidualUnit(output_dim, dilation=1)
        self.res_unit2 = ResidualUnit(output_dim, dilation=3)
        self.res_unit3 = ResidualUnit(output_dim, dilation=9)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.snake1(x)
        x = self.conv_t1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        return x


class VectorQuantize(nn.Module):
    """Single VQ layer (decode path only): codebook embed + out_proj."""

    def __init__(self, codebook_size: int, codebook_dim: int, hidden_size: int):
        super().__init__()
        # codebook: (codebook_size, codebook_dim) — standard embedding
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        # out_proj is a kernel=1 Conv1d: (hidden_size, codebook_dim, 1) in MLX layout
        self.out_proj = nn.Conv1d(codebook_dim, hidden_size, kernel_size=1)

    def from_codes(self, codes: mx.array) -> mx.array:
        """Embed codes and project to hidden_size.

        Args:
            codes: (B, T) int32 indices.

        Returns:
            (B, T, hidden_size) float32.
        """
        # embedding lookup: (B, T) -> (B, T, codebook_dim)
        z = self.codebook(codes)
        # out_proj (kernel=1 conv1d): (B, T, codebook_dim) -> (B, T, hidden_size)
        return self.out_proj(z)


class RVQ(nn.Module):
    """Residual Vector Quantizer — decode path only."""

    def __init__(self, n_codebooks: int, codebook_size: int, codebook_dim: int, hidden_size: int):
        super().__init__()
        self.quantizers = [
            VectorQuantize(codebook_size, codebook_dim, hidden_size)
            for _ in range(n_codebooks)
        ]

    def from_codes(self, audio_codes: mx.array) -> mx.array:
        """Sum per-codebook projected embeddings.

        Args:
            audio_codes: (B, n_codebooks, T) int32.

        Returns:
            (B, T, hidden_size) float32 — summed quantized representation.
        """
        n_codebooks = audio_codes.shape[1]
        out = None
        for i in range(n_codebooks):
            z_i = self.quantizers[i].from_codes(audio_codes[:, i, :])  # (B, T, H)
            out = z_i if out is None else out + z_i
        return out  # type: ignore[return-value]


class DacDecoder(nn.Module):
    """Full DAC decoder: conv1 -> 4x DecoderBlock -> Snake -> conv2 -> tanh."""

    def __init__(
        self,
        hidden_size: int,
        decoder_hidden_size: int,
        upsampling_ratios: list[int],
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, decoder_hidden_size, kernel_size=7, padding=3)

        blocks = []
        dim = decoder_hidden_size
        for stride in upsampling_ratios:
            out_dim = dim // 2
            blocks.append(DecoderBlock(dim, out_dim, stride))
            dim = out_dim
        self.block = blocks

        self.snake1 = Snake1d(dim)
        self.conv2 = nn.Conv1d(dim, 1, kernel_size=7, padding=3)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, hidden_size)
        x = self.conv1(x)
        for blk in self.block:
            x = blk(x)
        x = self.snake1(x)
        x = self.conv2(x)
        return mx.tanh(x)


# ---------------------------------------------------------------------------
# Top-level module
# ---------------------------------------------------------------------------

class Dac44k(nn.Module):
    """MLX DAC 44.1 kHz decode-only model."""

    def __init__(
        self,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        hidden_size: int = 1024,
        decoder_hidden_size: int = 1536,
        upsampling_ratios: list[int] | None = None,
    ):
        super().__init__()
        if upsampling_ratios is None:
            upsampling_ratios = [8, 8, 4, 2]
        self.n_codebooks = n_codebooks
        self.quantizer = RVQ(n_codebooks, codebook_size, codebook_dim, hidden_size)
        self.decoder = DacDecoder(hidden_size, decoder_hidden_size, upsampling_ratios)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, dac_dir: Union[str, Path]) -> "Dac44k":
        """Load a HuggingFace DacModel safetensors checkpoint.

        Handles the torch->MLX weight transpositions:
          - Conv1d:           (out, in, k) -> (out, k, in)
          - ConvTranspose1d:  (in, out, k) -> (out, k, in)
        """
        import json

        dac_dir = Path(dac_dir)
        with open(dac_dir / "config.json") as f:
            cfg = json.load(f)

        model = cls(
            n_codebooks=cfg["n_codebooks"],
            codebook_size=cfg["codebook_size"],
            codebook_dim=cfg["codebook_dim"],
            hidden_size=cfg["hidden_size"],
            decoder_hidden_size=cfg["decoder_hidden_size"],
            upsampling_ratios=cfg["upsampling_ratios"],
        )

        raw = mx.load(str(dac_dir / "model.safetensors"))

        # Build remapped weight dict.  The HF key prefix matches our module path
        # exactly except for the Conv weight shapes that need transposing.
        remapped: dict[str, mx.array] = {}
        for k, v in raw.items():
            # Only keep decoder and quantizer weights
            if not (k.startswith("decoder.") or k.startswith("quantizer.")):
                continue

            if k.endswith(".weight"):
                if "conv_t1" in k:
                    # torch ConvTranspose1d: (in, out, kernel) -> MLX (out, kernel, in)
                    v = v.transpose(1, 2, 0)
                elif "conv" in k or "in_proj" in k or "out_proj" in k:
                    # torch Conv1d: (out, in, kernel) -> MLX (out, kernel, in)
                    v = v.transpose(0, 2, 1)
                # codebook.weight: (codebook_size, codebook_dim) — no transpose needed
            # alpha parameters: (1, channels, 1) in torch channels-first.
            # Our Snake1d stores (channels,) so squeeze.
            if k.endswith(".alpha"):
                v = v.squeeze()  # (channels,)

            remapped[k] = v

        model.load_weights(list(remapped.items()), strict=False)
        mx.eval(model.parameters())
        return model

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(
        self,
        delayed_codes: np.ndarray,
        pad_id: int,
        eos_frame: int | None,
    ) -> np.ndarray:
        """Decode delayed codes to a 44.1 kHz waveform.

        Args:
            delayed_codes: (frames, n_codebooks) int64 numpy array.
            pad_id:        Padding token id (used by shear_up).
            eos_frame:     Index of the EOS frame (exclusive upper bound).
                           If None, trims the n_codebooks-1 tail rows.

        Returns:
            (1, samples) float32 numpy array at 44100 Hz.
        """
        # 1. Un-delay
        codes = shear_up(delayed_codes, pad_id)  # (frames, codebooks)

        # 2. Trim
        if eos_frame is not None and eos_frame >= 0:
            codes = codes[:int(eos_frame)]
        else:
            keep = max(0, codes.shape[0] - (self.n_codebooks - 1))
            codes = codes[:keep]

        # 3. Clamp
        codes = np.clip(codes, 0, 1023)

        # 4. (frames, codebooks) -> (1, codebooks, frames) for RVQ
        codes_mx = mx.array(codes.T[np.newaxis], dtype=mx.int32)  # (1, 9, T)

        # 5. RVQ decode: (1, 9, T) -> (1, T, hidden_size)
        z = self.quantizer.from_codes(codes_mx)

        # 6. Decoder: (1, T, H) -> (1, T, 1)
        out = self.decoder(z)  # (1, T, 1)

        # 7. Reshape to (1, samples) and return as numpy
        # out shape: (1, T, 1) -> (1, samples)
        mx.eval(out)
        out_np = np.array(out).squeeze(-1)  # (1, T*upsample)
        return out_np.astype(np.float32)
