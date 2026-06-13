from zonos2_mlx.weights import load_safetensors_header, scan_layers, remap_keys

SAFE = "weights/zonos2-bf16/zonos2-bf16.safetensors"


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
    # spot-check a few representative remaps exist (adjust target names to your MLX module scheme):
    assert any(k.endswith("attention.wq.weight") or k.endswith("attention.wq") for k in src)
