from zonos2_mlx.config import Zonos2Config


def test_config_core():
    c = Zonos2Config.load("outputs/fixtures/config.json")
    assert (c.n_layers, c.dim, c.head_dim, c.n_heads, c.n_kv_heads) == (28, 2048, 128, 16, 4)
    assert c.n_codebooks == 9 and c.codebook_size == 1024
    assert c.moe_n_experts == 16 and float(c.rope_theta) == 10000.0
    assert c.eoa_id == 1024 and c.audio_pad_id == 1025
