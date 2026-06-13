"""Safetensors header utilities for Zonos2.

Pure stdlib — no mlx/torch imports. Reads only the JSON header so tests
stay fast and CPU-only.

MLX naming scheme (upstream → MLX module path)
================================================

Top-level:
  multi_embedder.embedders.{j}.weight   → embed.embedders.{j}.weight
  multi_output.weight                   → lm_head.weight
  out_norm.weight                       → out_norm.weight
  speaker_lda_projection.weight/bias    → speaker_lda.weight / speaker_lda.bias
  speaker_projection.weight/bias        → speaker_proj.weight / speaker_proj.bias

Per-layer (both dense and MoE share the attention keys):
  layers.{i}.attention.wq.weight        → layers.{i}.attn.wq.weight
  layers.{i}.attention.wkv.weight       → layers.{i}.attn.wkv.weight   (kept fused; T2 splits at runtime)
  layers.{i}.attention.wo.weight        → layers.{i}.attn.wo.weight
  layers.{i}.attention.temp             → layers.{i}.attn.temp
  layers.{i}.attention.gater.weight     → layers.{i}.attn.gater.weight
  layers.{i}.attention_norm.weight      → layers.{i}.attn_norm.weight
  layers.{i}.ffn_norm.weight            → layers.{i}.ffn_norm.weight

Dense FFN (layers 0, 1, 2, 27):
  layers.{i}.feed_forward.w_in.weight   → layers.{i}.ffn.w_in.weight
  layers.{i}.feed_forward.w_out.weight  → layers.{i}.ffn.w_out.weight

MoE FFN (layers 3-26):
  layers.{i}.feed_forward.experts.w13            → layers.{i}.moe.experts.w13          (kept fused)
  layers.{i}.feed_forward.experts.w2             → layers.{i}.moe.experts.w2
  layers.{i}.feed_forward.router.balancing_biases          → layers.{i}.moe.router.balancing_biases
  layers.{i}.feed_forward.router.down_proj.weight          → layers.{i}.moe.router.down_proj.weight
  layers.{i}.feed_forward.router.down_proj.bias            → layers.{i}.moe.router.down_proj.bias
  layers.{i}.feed_forward.router.rmsnorm_eda.weight        → layers.{i}.moe.router.rmsnorm_eda.weight
  layers.{i}.feed_forward.router.router_mlp.0.weight       → layers.{i}.moe.router.mlp.0.weight
  layers.{i}.feed_forward.router.router_mlp.0.bias         → layers.{i}.moe.router.mlp.0.bias
  layers.{i}.feed_forward.router.router_mlp.2.weight       → layers.{i}.moe.router.mlp.2.weight
  layers.{i}.feed_forward.router.router_mlp.2.bias         → layers.{i}.moe.router.mlp.2.bias
  layers.{i}.feed_forward.router.router_mlp.4.weight       → layers.{i}.moe.router.mlp.4.weight
  layers.{i}.feed_forward.router.router_states_scale       → layers.{i}.moe.router.states_scale
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path


def load_safetensors_header(path: str | Path) -> dict:
    """Read and return the JSON metadata header from a safetensors file.

    Returns the raw dict including the ``__metadata__`` key if present.
    Does NOT read any tensor data.
    """
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n <= 0 or n > 100 * 1024 * 1024:
            raise ValueError(f"safetensors header length {n} is implausible — corrupt file?")
        return json.loads(f.read(n))


def scan_layers(hdr: dict) -> tuple[list[int], list[int]]:
    """Classify each transformer layer as MoE or dense.

    A layer is **MoE** if it contains a ``feed_forward.experts.w13`` tensor.
    A layer is **dense** if it contains a ``feed_forward.w_in`` tensor.

    Returns ``(sorted_moe_indices, sorted_dense_indices)``.
    """
    moe: set[int] = set()
    dense: set[int] = set()

    moe_pat = re.compile(r"^layers\.(\d+)\.feed_forward\.experts\.w13$")
    dense_pat = re.compile(r"^layers\.(\d+)\.feed_forward\.w_in\.")

    for key in hdr:
        if key == "__metadata__":
            continue
        m = moe_pat.match(key)
        if m:
            moe.add(int(m.group(1)))
            continue
        m = dense_pat.match(key)
        if m:
            dense.add(int(m.group(1)))

    overlap = set(moe) & set(dense)
    if overlap:
        raise ValueError(f"scan_layers: layers in both MoE and dense: {sorted(overlap)}")

    return sorted(moe), sorted(dense)


# ---------------------------------------------------------------------------
# Key remap: upstream safetensors path → MLX module attribute path
# ---------------------------------------------------------------------------

# Fixed mappings for top-level tensors
_TOP_LEVEL_MAP: dict[str, str] = {
    "multi_output.weight": "lm_head.weight",
    "out_norm.weight": "out_norm.weight",
    "speaker_lda_projection.weight": "speaker_lda.weight",
    "speaker_lda_projection.bias": "speaker_lda.bias",
    "speaker_projection.weight": "speaker_proj.weight",
    "speaker_projection.bias": "speaker_proj.bias",
}

# Attention key fragments: upstream suffix → MLX suffix
_ATTN_MAP: dict[str, str] = {
    "attention.wq.weight": "attn.wq.weight",
    "attention.wkv.weight": "attn.wkv.weight",
    "attention.wo.weight": "attn.wo.weight",
    "attention.temp": "attn.temp",
    "attention.gater.weight": "attn.gater.weight",
    "attention_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
}

# Dense FFN key fragments
_DENSE_FFN_MAP: dict[str, str] = {
    "feed_forward.w_in.weight": "ffn.w_in.weight",
    "feed_forward.w_out.weight": "ffn.w_out.weight",
}

# MoE FFN key fragments
_MOE_FFN_MAP: dict[str, str] = {
    "feed_forward.experts.w13": "moe.experts.w13",
    "feed_forward.experts.w2": "moe.experts.w2",
    "feed_forward.router.balancing_biases": "moe.router.balancing_biases",
    "feed_forward.router.down_proj.weight": "moe.router.down_proj.weight",
    "feed_forward.router.down_proj.bias": "moe.router.down_proj.bias",
    "feed_forward.router.rmsnorm_eda.weight": "moe.router.rmsnorm_eda.weight",
    "feed_forward.router.router_mlp.0.weight": "moe.router.mlp.0.weight",
    "feed_forward.router.router_mlp.0.bias": "moe.router.mlp.0.bias",
    "feed_forward.router.router_mlp.2.weight": "moe.router.mlp.2.weight",
    "feed_forward.router.router_mlp.2.bias": "moe.router.mlp.2.bias",
    "feed_forward.router.router_mlp.4.weight": "moe.router.mlp.4.weight",
    "feed_forward.router.router_states_scale": "moe.router.states_scale",
}

_LAYER_KEY_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
_EMBEDDER_RE = re.compile(r"^multi_embedder\.embedders\.(\d+)\.weight$")


def remap_keys(src_keys: list[str]) -> dict[str, str]:
    """Map every upstream safetensors key to an MLX module attribute path.

    The mapping is:
    - total (every key in *src_keys* appears exactly once as a dict key)
    - collision-free (every value is unique)

    Returns ``{upstream_key: mlx_path}``.
    """
    out: dict[str, str] = {}

    for key in src_keys:
        # --- embedders ---------------------------------------------------
        m = _EMBEDDER_RE.match(key)
        if m:
            out[key] = f"embed.embedders.{m.group(1)}.weight"
            continue

        # --- fixed top-level keys ----------------------------------------
        if key in _TOP_LEVEL_MAP:
            out[key] = _TOP_LEVEL_MAP[key]
            continue

        # --- per-layer keys ----------------------------------------------
        m = _LAYER_KEY_RE.match(key)
        if m:
            idx, suffix = m.group(1), m.group(2)
            prefix = f"layers.{idx}."

            # attention / norm keys
            if suffix in _ATTN_MAP:
                out[key] = prefix + _ATTN_MAP[suffix]
                continue

            # dense FFN
            if suffix in _DENSE_FFN_MAP:
                out[key] = prefix + _DENSE_FFN_MAP[suffix]
                continue

            # MoE FFN
            if suffix in _MOE_FFN_MAP:
                out[key] = prefix + _MOE_FFN_MAP[suffix]
                continue

        # If we fall through here the key is unmapped — raise immediately
        # so we notice during testing rather than silently dropping tensors.
        raise ValueError(f"remap_keys: no mapping defined for upstream key {key!r}")

    return out
