from pathlib import Path

import pytest

from zonos2_mlx.weights import load_safetensors_header, scan_layers, remap_keys

_ROOT = Path(__file__).resolve().parent.parent

SAFE = str(_ROOT / "weights/zonos2-bf16/zonos2-bf16.safetensors")

# Reads the bf16 safetensors header (weights are gitignored) — skip on a fresh clone.
pytestmark = pytest.mark.skipif(
    not Path(SAFE).exists(),
    reason="needs weights/zonos2-bf16/zonos2-bf16.safetensors",
)


def test_header_loads():
    hdr = load_safetensors_header(SAFE)
    keys = [k for k in hdr if k != "__metadata__"]
    assert len(keys) == 507


def test_layer_typing():
    hdr = load_safetensors_header(SAFE)
    moe, dense = scan_layers(hdr)          # -> (sorted list, sorted list) of layer indices
    assert len(moe) == 24 and len(dense) == 4 and len(moe) + len(dense) == 28
    assert dense == [0, 1, 2, 27]          # per oracle notes


def test_remap_total():
    hdr = load_safetensors_header(SAFE)
    src = [k for k in hdr if k != "__metadata__"]
    mapping = remap_keys(src)              # upstream key -> mlx module path (str)
    assert len(mapping) == len(src)        # every tensor mapped, no orphans
    assert all(isinstance(v, str) and v for v in mapping.values())
    # no collisions
    assert len(set(mapping.values())) == len(mapping)
    # assert target (MLX) paths — locks the current naming scheme
    assert mapping["layers.0.attention.wq.weight"] == "layers.0.attn.wq.weight"
    assert mapping["layers.0.attention.wkv.weight"] == "layers.0.attn.wkv.weight"
    assert mapping["layers.0.attention.temp"] == "layers.0.attn.temp"
    assert mapping["layers.0.attention.gater.weight"] == "layers.0.attn.gater.weight"
    assert mapping["layers.3.feed_forward.experts.w13"] == "layers.3.moe.experts.w13"
    assert mapping["layers.3.feed_forward.router.down_proj.weight"] == "layers.3.moe.router.down_proj.weight"
    assert mapping["layers.0.feed_forward.w_in.weight"] == "layers.0.ffn.w_in.weight"
    assert mapping["multi_output.weight"] == "lm_head.weight"
    assert mapping["multi_embedder.embedders.4.weight"] == "embed.embedders.4.weight"
    assert mapping["speaker_lda_projection.weight"] == "speaker_lda.weight"
    assert mapping["speaker_projection.weight"] == "speaker_proj.weight"
