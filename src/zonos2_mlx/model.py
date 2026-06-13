"""MLX Zonos2 8B MoE transformer trunk.

Mirrors ``Zonos2_TTS-ComfyUI/native.py`` op-for-op (see docs/research/00).
Loads the bf16 safetensors checkpoint via ``weights.remap_keys`` and runs a
prefill forward returning per-layer hidden states + multi-codebook logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

mx.set_memory_limit(int(45 * (1 << 30)))

from .config import Zonos2Config  # noqa: E402
from .layers import (  # noqa: E402
    Attention,
    DenseFeedForward,
    MoEFeedForward,
    rms_norm_weightless,
    rope_tables,
    softcap,
)
from .layers import RMSNorm as RMSNorm  # noqa: E402
from .weights import load_safetensors_header, remap_keys  # noqa: E402


@dataclass
class TrunkOutput:
    layers: dict[int, mx.array]  # captured per-layer hidden states (T, dim)
    last: mx.array               # final-layer hidden states (T, dim)


def _is_moe_layer(cfg: Zonos2Config, layer_id: int) -> bool:
    """native.py Zonos2Config.is_moe_layer (145-150)."""
    if cfg.moe_n_experts <= 1:
        return False
    if layer_id < cfg.moe_start_from_layer:
        return False
    return (cfg.n_layers - layer_id) > cfg.moe_end_from_layer


def _top_k_for_layer(cfg: Zonos2Config, layer_id: int) -> int:
    """native.py top_k_for_layer (152-153). special_topk_layers may carry
    str keys (loaded from JSON), so probe both."""
    special = cfg.special_topk_layers or {}
    if layer_id in special:
        return int(special[layer_id])
    if str(layer_id) in special:
        return int(special[str(layer_id)])
    return int(cfg.moe_router_topk)


class MultiEmbedding(nn.Module):
    """Sum of per-codebook + text embedders (native.py MultiEmbedding 187-204).

    Input codes are (B, T, 10): cols 0..8 = the 9 audio codebooks, col 9 =
    the text stream. All 10 embedder lookups are summed.
    """

    def __init__(self, n_codebooks: int, codebook_size: int, text_vocab: int, dim: int):
        super().__init__()
        self.embedders = [
            nn.Embedding(codebook_size + 2, dim) for _ in range(n_codebooks)
        ] + [nn.Embedding(text_vocab + 1, dim)]

    def __call__(self, codes: mx.array) -> mx.array:
        result = self.embedders[0](codes[..., 0])
        for i in range(1, codes.shape[-1]):
            result = result + self.embedders[i](codes[..., i])
        return result


class TransformerBlock(nn.Module):
    """Pre-norm attention + FFN/MoE residual block (native.py 570-604)."""

    def __init__(self, cfg: Zonos2Config, layer_id: int, is_moe: bool, top_k: int):
        super().__init__()
        self.is_moe = is_moe
        self.attn = Attention(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        if is_moe:
            use_eda = layer_id != cfg.moe_start_from_layer
            self.moe = MoEFeedForward(
                cfg.dim, cfg.intermediate_size, cfg.moe_router_dim,
                cfg.moe_n_experts, top_k, use_eda,
            )
        else:
            self.ffn = DenseFeedForward(cfg.dim, cfg.intermediate_size)

    def __call__(self, x, cos, sin, router_states):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        normed = self.ffn_norm(x)
        if self.is_moe:
            ff, router_states = self.moe(normed, router_states)
        else:
            ff = self.ffn(normed)
            router_states = None
        return x + ff, router_states


class Zonos2Model(nn.Module):
    """The 8B MoE trunk + multi-codebook embed/head + speaker injection."""

    def __init__(self, cfg: Zonos2Config):
        super().__init__()
        self.cfg = cfg
        self.embed = MultiEmbedding(cfg.n_codebooks, cfg.codebook_size, cfg.text_vocab, cfg.dim)
        self.speaker_proj = nn.Linear(cfg.speaker_lda_dim, cfg.dim, bias=True)
        # speaker_lda_projection (2048->1024); present in the checkpoint but
        # not exercised here since fixtures pass the already-projected lda vec.
        self.speaker_lda = nn.Linear(cfg.speaker_embedding_dim, cfg.speaker_lda_dim, bias=True)

        self.layers = []
        for i in range(cfg.n_layers):
            is_moe = _is_moe_layer(cfg, i)
            top_k = _top_k_for_layer(cfg, i)
            self.layers.append(TransformerBlock(cfg, i, is_moe, top_k))

        self.out_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.n_codebooks * (cfg.codebook_size + 2), bias=False)

        cos, sin = rope_tables(cfg.head_dim, cfg.max_seqlen, cfg.rope_theta)
        self._rope_cos = cos
        self._rope_sin = sin

    # ------------------------------------------------------------------
    def forward(
        self,
        ids: mx.array,
        speaker_lda: mx.array | None = None,
        speaker_position: int | None = None,
        capture_layers: list[int] | None = None,
    ) -> TrunkOutput:
        cfg = self.cfg
        capture = set(capture_layers or [])
        x = self.embed(ids)  # (B, T, dim)

        # Speaker injection: overwrite the hidden at speaker_position with
        # speaker_proj(lda) (native.py 692-697). lda is the already-projected
        # 1024-d vector (the fixture), so only speaker_proj is applied here.
        if speaker_lda is not None and speaker_position is not None:
            if 0 <= speaker_position < x.shape[1]:
                projected = self.speaker_proj(speaker_lda.astype(self.speaker_proj.weight.dtype))
                projected = projected.astype(x.dtype).reshape(x.shape[0], cfg.dim)
                # x[:, speaker_position] = projected
                rows = []
                for b in range(x.shape[0]):
                    row = mx.concatenate(
                        [x[b, :speaker_position], projected[b][None], x[b, speaker_position + 1 :]],
                        axis=0,
                    )
                    rows.append(row[None])
                x = mx.concatenate(rows, axis=0)

        # Weightless RMSNorm over the whole sequence (native.py 698-703).
        x = rms_norm_weightless(x, cfg.norm_eps)

        seqlen = x.shape[1]
        cos = self._rope_cos[:seqlen].reshape(1, seqlen, 1, cfg.head_dim // 2).astype(x.dtype)
        sin = self._rope_sin[:seqlen].reshape(1, seqlen, 1, cfg.head_dim // 2).astype(x.dtype)

        captured: dict[int, mx.array] = {}
        router_states = None
        for i, layer in enumerate(self.layers):
            x, router_states = layer(x, cos, sin, router_states)
            if i in capture:
                captured[i] = x[0]  # (T, dim)
        return TrunkOutput(layers=captured, last=x)

    def head(self, last: mx.array) -> mx.array:
        """Multi-output head on the LAST position (native.py 713-722)."""
        cfg = self.cfg
        hidden = self.out_norm(last[:, -1])  # (B, dim)
        logits = self.lm_head(hidden).reshape(
            hidden.shape[0], cfg.n_codebooks, cfg.codebook_size + 2
        )
        logits = softcap(logits, cfg.loss_softcap)
        return logits[0]  # (n_codebooks, codebook_size+2)

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, weights_dir: str) -> "Zonos2Model":
        wdir = Path(weights_dir)
        # Prefer the model's own config.json; fall back to the oracle fixtures
        # config (same resolved values) so a checkout without a weights-dir
        # config still gets special_topk_layers={26:2} etc. The defaults in
        # Zonos2Config are a last resort.
        cfg_candidates = [
            wdir / "config.json",
            Path(__file__).resolve().parents[2] / "outputs/fixtures/config.json",
        ]
        cfg = None
        for cfg_path in cfg_candidates:
            if cfg_path.exists():
                cfg = Zonos2Config.load(cfg_path)
                break
        if cfg is None:
            cfg = Zonos2Config()
        # special_topk_layers is intrinsic to this 28-layer zonos2 model; if a
        # bare-defaults config slipped through, restore the known value so the
        # top-2 layer (26) is not silently demoted to top-1.
        if not cfg.special_topk_layers and cfg.n_layers == 28 and cfg.moe_n_experts == 16:
            cfg.special_topk_layers = {"26": 2}

        st_files = sorted(wdir.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"no .safetensors under {wdir}")
        st_path = st_files[0]

        model = cls(cfg)
        hdr = load_safetensors_header(st_path)
        src_keys = [k for k in hdr if k != "__metadata__"]
        mapping = remap_keys(src_keys)

        raw = mx.load(str(st_path))
        params = _assemble_params(raw, mapping, cfg)
        model.update(params)
        mx.eval(model.parameters())
        return model


# ----------------------------------------------------------------------
def _assemble_params(raw: dict, mapping: dict[str, str], cfg: Zonos2Config) -> dict:
    """Build the nested parameter tree matching the module structure from the
    flat remapped {mlx_path: tensor} dict.

    Handles the structural differences between the checkpoint key layout and
    our module attribute names:
      - wkv [2,512,2048]            -> flatten to [1024, 2048]
      - ffn.w_in [2,3072,2048]      -> flatten to [6144, 2048]
      - router mlp.{0,2,4}          -> mlp_0 / mlp_2 / mlp_4
      - router states_scale         -> states_scale (bare param)
      - temp                        -> bare param
      - embedders/layers lists
    """
    # Resolve {mlx_path: tensor}
    flat: dict[str, mx.array] = {}
    for src, dst in mapping.items():
        flat[dst] = raw[src]

    tree: dict = {}

    # --- top-level -----------------------------------------------------
    # embedders
    embedders = []
    j = 0
    while f"embed.embedders.{j}.weight" in flat:
        embedders.append({"weight": flat[f"embed.embedders.{j}.weight"]})
        j += 1
    tree["embed"] = {"embedders": embedders}

    tree["lm_head"] = {"weight": flat["lm_head.weight"]}
    tree["out_norm"] = {"weight": flat["out_norm.weight"]}
    tree["speaker_lda"] = {
        "weight": flat["speaker_lda.weight"],
        "bias": flat["speaker_lda.bias"],
    }
    tree["speaker_proj"] = {
        "weight": flat["speaker_proj.weight"],
        "bias": flat["speaker_proj.bias"],
    }

    # --- layers --------------------------------------------------------
    layers = []
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        block: dict = {}
        # attention
        wkv = flat[p + "attn.wkv.weight"]  # [2,512,2048]
        wkv = wkv.reshape(wkv.shape[0] * wkv.shape[1], wkv.shape[2])  # [1024,2048]
        block["attn"] = {
            "wq": {"weight": flat[p + "attn.wq.weight"]},
            "wkv": {"weight": wkv},
            "wo": {"weight": flat[p + "attn.wo.weight"]},
            "gater": {"weight": flat[p + "attn.gater.weight"]},
            "temp": flat[p + "attn.temp"],
        }
        block["attn_norm"] = {"weight": flat[p + "attn_norm.weight"]}
        block["ffn_norm"] = {"weight": flat[p + "ffn_norm.weight"]}

        if _is_moe_layer(cfg, i):
            router: dict = {
                "down_proj": {
                    "weight": flat[p + "moe.router.down_proj.weight"],
                    "bias": flat[p + "moe.router.down_proj.bias"],
                },
                "mlp_0": {
                    "weight": flat[p + "moe.router.mlp.0.weight"],
                    "bias": flat[p + "moe.router.mlp.0.bias"],
                },
                "mlp_2": {
                    "weight": flat[p + "moe.router.mlp.2.weight"],
                    "bias": flat[p + "moe.router.mlp.2.bias"],
                },
                "mlp_4": {"weight": flat[p + "moe.router.mlp.4.weight"]},
                "rmsnorm_eda": {"weight": flat[p + "moe.router.rmsnorm_eda.weight"]},
                "balancing_biases": flat[p + "moe.router.balancing_biases"],
            }
            scale_key = p + "moe.router.states_scale"
            if scale_key in flat:
                router["states_scale"] = flat[scale_key]
            block["moe"] = {
                "router": router,
                "experts": {
                    "w13": flat[p + "moe.experts.w13"],
                    "w2": flat[p + "moe.experts.w2"],
                },
            }
        else:
            w_in = flat[p + "ffn.w_in.weight"]  # [2,3072,2048]
            w_in = w_in.reshape(w_in.shape[0] * w_in.shape[1], w_in.shape[2])  # [6144,2048]
            block["ffn"] = {
                "w_in": {"weight": w_in},
                "w_out": {"weight": flat[p + "ffn.w_out.weight"]},
            }
        layers.append(block)
    tree["layers"] = layers
    return tree
