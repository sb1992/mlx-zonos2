"""MLX ECAPA-TDNN speaker encoder + LDA projection + enrollment cache.

Pure-MLX port of the torch ECAPA-TDNN in
``weights/zonos2-bf16/speaker_encoder/modeling_ecapa_tdnn.py``. Produces a
2048-d x-vector from a log-mel spectrogram; the LDA projection (which lives in
the 8B DiT checkpoint) maps that 2048 -> 1024 for injection into the transformer
trunk as the speaker condition.

Layout differences (torch -> MLX)
---------------------------------
torch is channels-first ``(B, C, T)``; MLX ``nn.Conv1d`` is channels-last
``(B, T, C)``. So torch "time" ops (``dim=2``) become MLX ``axis=1`` and torch
"channel" concats (``dim=1``) become MLX ``axis=-1``. The mel input is already
``(B, T, mel)`` channels-last so we do NOT transpose it.

torch Conv1d weight ``(out, in, k)`` -> MLX ``(out, k, in)`` via
``.transpose(0, 2, 1)`` (same idiom as ``dac.py``).

Padding
-------
The torch convs use ``padding="same", padding_mode="reflect"``. MLX ``mx.pad``
has no reflect mode, so we reflect-pad manually then run a valid (padding=0)
conv. All effective paddings here are symmetric: ``pad = dilation*(k-1)//2``.

Precision
---------
The oracle ran ECAPA in float32 (transformers upcast the bf16-stored weights to
float32). We cast all loaded weights to float32 and run the encoder in float32 —
the model is tiny so this is cheap and maximizes parity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

mx.set_memory_limit(int(45 * (1 << 30)))


# ---------------------------------------------------------------------------
# Reflect padding (channels-last, along the time axis)
# ---------------------------------------------------------------------------

def reflect_pad_time(x: mx.array, pad: int) -> mx.array:
    """Reflect-pad ``x`` (B, T, C) by ``pad`` on each side along time (axis=1).

    Matches torch ``F.pad(..., mode="reflect")``: mirrors WITHOUT repeating the
    edge sample. E.g. ``[1,2,3,4,5]`` with pad=2 -> ``[3,2,1,2,3,4,5,4,3]``.
    """
    if pad == 0:
        return x
    if pad >= x.shape[1]:
        raise ValueError(f"reflect pad {pad} >= time length {x.shape[1]} (torch reflect would error)")
    left = x[:, 1:pad + 1, :][:, ::-1, :]      # mirror of indices [1 .. pad] -> [pad .. 1]
    right = x[:, -pad - 1:-1, :][:, ::-1, :]    # mirror of indices [T-pad-1 .. T-2] -> [T-2 .. T-pad-1]
    return mx.concatenate([left, x, right], axis=1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TimeDelayNetBlock(nn.Module):
    """reflect-pad -> valid Conv1d -> ReLU (a TDNN layer)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        h = self.conv(reflect_pad_time(x, self.pad))
        return nn.relu(h)


class Res2NetBlock(nn.Module):
    """Multi-scale Res2Net block built from TDNN sub-blocks (scale-1 of them)."""

    def __init__(self, in_channels: int, out_channels: int, scale: int = 8,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.scale = scale
        self.blocks = [
            TimeDelayNetBlock(in_channel, hidden_channel, kernel_size=kernel_size, dilation=dilation)
            for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C). Split channels into `scale` chunks of equal size.
        chunks = mx.split(x, self.scale, axis=-1)
        outputs = []
        output_part = None
        for i, part in enumerate(chunks):
            if i == 0:
                output_part = part
            elif i == 1:
                output_part = self.blocks[i - 1](part)
            else:
                output_part = self.blocks[i - 1](part + output_part)
            outputs.append(output_part)
        return mx.concatenate(outputs, axis=-1)


class SqueezeExcitationBlock(nn.Module):
    """Channel-wise squeeze-and-excitation (mean over time, two 1x1 convs)."""

    def __init__(self, in_channels: int, se_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(se_channels, out_channels, kernel_size=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C). Mean over time (axis=1) -> (B, 1, C).
        s = mx.mean(x, axis=1, keepdims=True)
        s = nn.relu(self.conv1(s))
        s = mx.sigmoid(self.conv2(s))
        return x * s  # broadcast over time


class SqueezeExcitationRes2NetBlock(nn.Module):
    """TDNN -> Res2Net -> TDNN -> SE, with residual."""

    def __init__(self, in_channels: int, out_channels: int, res2net_scale: int = 8,
                 se_channels: int = 128, kernel_size: int = 1, dilation: int = 1):
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        h = self.tdnn1(x)
        h = self.res2net_block(h)
        h = self.tdnn2(h)
        h = self.se_block(h)
        return h + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling -> concatenated weighted mean and std.

    Single full-length utterance => the torch length-mask is all-ones, so the
    masked computation simplifies to uniform-weight stats + a softmax over time
    with no -inf masking. (Numerically identical to the torch path with
    ``lengths = ones``.)
    """

    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, kernel_size=1, dilation=1)
        self.conv = nn.Conv1d(attention_channels, channels, kernel_size=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        T = x.shape[1]

        # Uniform-weight statistics (mask / total == 1/T everywhere).
        m = 1.0 / T
        mean = mx.sum(m * x, axis=1)                                   # (B, C)
        std = mx.sqrt(mx.maximum(mx.sum(m * (x - mean[:, None, :]) ** 2, axis=1), self.eps))

        mean_bc = mx.broadcast_to(mean[:, None, :], x.shape)           # (B, T, C)
        std_bc = mx.broadcast_to(std[:, None, :], x.shape)             # (B, T, C)
        attn_in = mx.concatenate([x, mean_bc, std_bc], axis=-1)        # (B, T, 3C)

        a = self.conv(mx.tanh(self.tdnn(attn_in)))                     # (B, T, C)
        a = mx.softmax(a, axis=1)                                      # softmax over time

        wmean = mx.sum(a * x, axis=1)                                  # (B, C)
        wstd = mx.sqrt(mx.maximum(mx.sum(a * (x - wmean[:, None, :]) ** 2, axis=1), self.eps))
        pooled = mx.concatenate([wmean, wstd], axis=-1)               # (B, 2C)
        return pooled[:, None, :]                                      # (B, 1, 2C)


# ---------------------------------------------------------------------------
# Top-level ECAPA-TDNN
# ---------------------------------------------------------------------------

class EcapaTDNN(nn.Module):
    """ECAPA-TDNN x-vector encoder (log-mel -> 2048-d embedding)."""

    def __init__(
        self,
        mel_dim: int = 128,
        enc_dim: int = 2048,
        enc_channels: list[int] | None = None,
        enc_kernel_sizes: list[int] | None = None,
        enc_dilations: list[int] | None = None,
        enc_attention_channels: int = 128,
        enc_res2net_scale: int = 8,
        enc_se_channels: int = 128,
    ):
        super().__init__()
        if enc_channels is None:
            enc_channels = [512, 512, 512, 512, 1536]
        if enc_kernel_sizes is None:
            enc_kernel_sizes = [5, 3, 3, 3, 1]
        if enc_dilations is None:
            enc_dilations = [1, 2, 3, 4, 1]

        blocks: list[nn.Module] = []
        # Initial TDNN layer.
        blocks.append(
            TimeDelayNetBlock(mel_dim, enc_channels[0], enc_kernel_sizes[0], enc_dilations[0])
        )
        # SE-Res2Net layers.
        for i in range(1, len(enc_channels) - 1):
            blocks.append(
                SqueezeExcitationRes2NetBlock(
                    enc_channels[i - 1],
                    enc_channels[i],
                    res2net_scale=enc_res2net_scale,
                    se_channels=enc_se_channels,
                    kernel_size=enc_kernel_sizes[i],
                    dilation=enc_dilations[i],
                )
            )
        self.blocks = blocks

        # Multi-layer feature aggregation.
        self.mfa = TimeDelayNetBlock(
            enc_channels[-1], enc_channels[-1], enc_kernel_sizes[-1], enc_dilations[-1]
        )
        # Attentive statistical pooling.
        self.asp = AttentiveStatisticsPooling(
            enc_channels[-1], attention_channels=enc_attention_channels
        )
        # Final linear transformation (1x1 conv).
        self.fc = nn.Conv1d(enc_channels[-1] * 2, enc_dim, kernel_size=1, padding=0)

    def __call__(self, mel: mx.array) -> mx.array:
        # mel: (B, T, mel_dim) — already channels-last, no transpose.
        h = mel.astype(mx.float32)
        collected = []
        for block in self.blocks:
            h = block(h)
            collected.append(h)

        # MFA aggregates the SE-Res2Net outputs only (blocks[1:]), NOT block[0].
        mfa_in = mx.concatenate(collected[1:], axis=-1)   # (B, T, 1536)
        h = self.mfa(mfa_in)

        h = self.asp(h)                                   # (B, 1, 3072)
        h = self.fc(h)                                    # (B, 1, 2048)
        return h[:, 0, :]                                 # (B, 2048)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, speaker_encoder_dir: Union[str, Path]) -> "EcapaTDNN":
        """Load the HF ECAPA-TDNN safetensors checkpoint into the MLX module.

        Transposes every conv weight ``(out, in, k) -> (out, k, in)`` and casts
        all tensors to float32. ``strict=True`` proves every one of the 76
        tensors maps to a parameter slot.
        """
        speaker_encoder_dir = Path(speaker_encoder_dir)
        with open(speaker_encoder_dir / "config.json") as f:
            cfg = json.load(f)

        model = cls(
            mel_dim=cfg["mel_dim"],
            enc_dim=cfg["enc_dim"],
            enc_channels=cfg["enc_channels"],
            enc_kernel_sizes=cfg["enc_kernel_sizes"],
            enc_dilations=cfg["enc_dilations"],
            enc_attention_channels=cfg["enc_attention_channels"],
            enc_res2net_scale=cfg["enc_res2net_scale"],
            enc_se_channels=cfg["enc_se_channels"],
        )

        raw = mx.load(str(speaker_encoder_dir / "model.safetensors"))
        remapped: dict[str, mx.array] = {}
        for k, v in raw.items():
            if k.endswith(".weight"):
                # All weights here are conv weights (out, in, k) -> (out, k, in).
                v = v.transpose(0, 2, 1)
            remapped[k] = v.astype(mx.float32)

        model.load_weights(list(remapped.items()), strict=True)
        mx.eval(model.parameters())
        return model


# ---------------------------------------------------------------------------
# LDA projection (lives in the 8B DiT checkpoint)
# ---------------------------------------------------------------------------

class SpeakerLDA:
    """Applies the DiT's ``speaker_lda_projection`` (nn.Linear 2048 -> 1024)."""

    def __init__(self, weight: mx.array, bias: mx.array):
        # weight: (1024, 2048) torch nn.Linear convention; bias: (1024,)
        self.weight = weight
        self.bias = bias

    def __call__(self, emb: mx.array) -> mx.array:
        # emb: (B, 2048) -> (B, 1024) via nn.Linear form emb @ W.T + b.
        return emb.astype(mx.float32) @ self.weight.T + self.bias

    @classmethod
    def from_dit(cls, dit_safetensors_path: Union[str, Path]) -> "SpeakerLDA":
        """Load ONLY the two LDA tensors from the DiT safetensors.

        ``mx.load`` memory-maps, so the untouched 8B of DiT weights are never
        materialized — we eval only the two LDA tensors.

        Accepts either key naming: the upstream bf16 checkpoint uses
        ``speaker_lda_projection.*``; our converted/quantized checkpoints rename
        it to ``speaker_lda.*`` (kept bf16 in every tier). Both resolve here so
        enrolment works from any tier's trunk.
        """
        raw = mx.load(str(dit_safetensors_path))
        if "speaker_lda_projection.weight" in raw:
            wkey, bkey = "speaker_lda_projection.weight", "speaker_lda_projection.bias"
        elif "speaker_lda.weight" in raw:
            wkey, bkey = "speaker_lda.weight", "speaker_lda.bias"
        else:
            raise KeyError(
                f"no speaker LDA projection (speaker_lda_projection.* / speaker_lda.*) "
                f"in {dit_safetensors_path}"
            )
        weight = raw[wkey].astype(mx.float32)
        bias = raw[bkey].astype(mx.float32)
        mx.eval(weight, bias)
        return cls(weight, bias)


# ---------------------------------------------------------------------------
# Enrollment cache
# ---------------------------------------------------------------------------

class SpeakerProfile:
    """Container for the cached 1024-d LDA speaker vector (the inject vector)."""

    def __init__(self, lda: np.ndarray, compat: str):
        self.lda = np.asarray(lda, dtype=np.float32).reshape(-1)
        self.compat = str(compat)

    @staticmethod
    def save(path: Union[str, Path], lda_vec_np: np.ndarray, model_compat_hash: str) -> None:
        """Persist the LDA vector + a model-compat string to EXACTLY ``path``.

        ``np.savez`` appends ``.npz`` when handed a filename; writing through an
        open file handle suppresses that, so ``save("voice.zonos")`` produces
        ``voice.zonos`` (not ``voice.zonos.npz``). It is an npz container
        regardless of the extension.
        """
        lda = np.asarray(lda_vec_np, dtype=np.float32).reshape(-1)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            np.savez(fh, lda=lda, compat=np.asarray(str(model_compat_hash)))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SpeakerProfile":
        """Load a saved profile. Falls back to ``<path>.npz`` for older saves."""
        path = str(path)
        if not Path(path).exists() and not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path, allow_pickle=False)
        lda = np.asarray(data["lda"], dtype=np.float32).reshape(-1)
        compat = str(data["compat"])
        return cls(lda, compat)
