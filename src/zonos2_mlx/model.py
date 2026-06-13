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
    KVCache,
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
    last: mx.array               # final-layer hidden states (T, dim) — batch stripped


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

    def __call__(self, x, cos, sin, router_states, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
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
        x = self._inject_speaker(x, speaker_lda, speaker_position)

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
        return TrunkOutput(layers=captured, last=x[0])  # (T, dim)

    def head(self, last: mx.array) -> mx.array:
        """Multi-output head on the LAST position (native.py 713-722).

        ``last`` is (T, dim) — batch-stripped final hidden states.
        Returns (n_codebooks, codebook_size+2).
        """
        cfg = self.cfg
        hidden = self.out_norm(last[-1])     # (dim,) — last token position
        hidden = hidden[None]                # (1, dim) for lm_head
        logits = self.lm_head(hidden).reshape(
            1, cfg.n_codebooks, cfg.codebook_size + 2
        )
        logits = softcap(logits, cfg.loss_softcap)
        return logits[0]  # (n_codebooks, codebook_size+2)

    # ------------------------------------------------------------------
    # KV-cached incremental decode path (native.py 683-722 + LayerKVCache).
    # ------------------------------------------------------------------
    def _inject_speaker(self, x, speaker_lda, speaker_position):
        """Overwrite x[:, speaker_position] with speaker_proj(lda) when both
        args are given and the position is in range (native.py 692-697)."""
        cfg = self.cfg
        if speaker_lda is None or speaker_position is None:
            return x
        if not (0 <= speaker_position < x.shape[1]):
            return x
        projected = self.speaker_proj(speaker_lda.astype(self.speaker_proj.weight.dtype))
        projected = projected.astype(x.dtype).reshape(x.shape[0], cfg.dim)
        rows = []
        for b in range(x.shape[0]):
            row = mx.concatenate(
                [x[b, :speaker_position], projected[b][None], x[b, speaker_position + 1 :]],
                axis=0,
            )
            rows.append(row[None])
        return mx.concatenate(rows, axis=0)

    def make_kv_caches(self, max_len: int) -> list[KVCache]:
        """One KVCache per layer (native.py create_kv_cache 652-669).

        ``max_len`` is accepted for parity / future bounds-checking; the MLX
        cache grows by concat so no pre-allocation is required.
        """
        del max_len  # cache grows lazily; nothing to preallocate.
        return [KVCache() for _ in range(self.cfg.n_layers)]

    def forward_cached(
        self,
        ids: mx.array,
        caches: list[KVCache],
        speaker_lda: mx.array | None = None,
        speaker_position: int | None = None,
    ) -> mx.array:
        """KV-cached forward returning the LAST-position logits (1, 9, 1026).

        The RoPE offset is ``caches[0].length`` BEFORE this call, exactly
        mirroring the oracle ``start = cache.length`` / ``cos = rope[start:end]``
        (native.py 347-372). Speaker injection happens only when both speaker
        args are passed (prefill). Each layer threads its own per-layer cache.
        """
        cfg = self.cfg
        x = self.embed(ids)  # (B, T, dim)
        x = self._inject_speaker(x, speaker_lda, speaker_position)

        # Weightless RMSNorm over the sequence (native.py 698-703).
        x = rms_norm_weightless(x, cfg.norm_eps)

        start = caches[0].length
        seqlen = x.shape[1]
        end = start + seqlen
        cos = self._rope_cos[start:end].reshape(1, seqlen, 1, cfg.head_dim // 2).astype(x.dtype)
        sin = self._rope_sin[start:end].reshape(1, seqlen, 1, cfg.head_dim // 2).astype(x.dtype)

        router_states = None
        for layer, cache in zip(self.layers, caches):
            x, router_states = layer(x, cos, sin, router_states, cache)

        # LAST-position multi-codebook logits (native.py 713-722).
        return self.head(x[0])[None]  # (1, n_codebooks, codebook_size+2)

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, weights_dir: str, quant=None) -> "Zonos2Model":
        """Load the trunk from ``weights_dir``.

        bf16 (default): the original op-for-op path. When a ``quant_config.json``
        sidecar is present in ``weights_dir`` (or ``quant`` is passed), the
        int8/int4 path is taken instead: the selected attention/ffn/lm_head
        Linears are rebuilt as ``QuantizedLinear`` and the MoE experts use the
        quantized gather-matmul (weights stay packed in memory). The forward /
        generate API is identical at any tier.

        ``quant`` may be a ``QuantizationConfig``, a path to a sidecar JSON, or
        ``None`` to auto-detect ``weights_dir/quant_config.json``.
        """
        wdir = Path(weights_dir)
        cfg_path = wdir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{cfg_path} is required (carries special_topk_layers etc.)"
                " — copy it alongside the weights"
            )
        cfg = Zonos2Config.load(cfg_path)

        st_files = sorted(wdir.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"no .safetensors under {wdir}")
        st_path = st_files[0]

        qcfg = cls._resolve_quant(wdir, quant)
        if qcfg is not None:
            return cls._from_pretrained_quantized(cfg, st_path, qcfg)

        model = cls(cfg)
        hdr = load_safetensors_header(st_path)
        src_keys = [k for k in hdr if k != "__metadata__"]
        mapping = remap_keys(src_keys)

        raw = mx.load(str(st_path))
        params = _assemble_params(raw, mapping, cfg)
        model.update(params)
        mx.eval(model.parameters())
        return model

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_quant(wdir: Path, quant):
        """Resolve a QuantizationConfig from an arg or an auto-detected sidecar."""
        from .quantize import QuantizationConfig

        if quant is None:
            sidecar = wdir / "quant_config.json"
            if sidecar.exists():
                import json

                return QuantizationConfig.from_dict(json.loads(sidecar.read_text()))
            return None
        if isinstance(quant, QuantizationConfig):
            return quant
        # A path to a sidecar JSON.
        import json

        return QuantizationConfig.from_dict(json.loads(Path(quant).read_text()))

    @classmethod
    def _from_pretrained_quantized(cls, cfg, st_path: Path, qcfg) -> "Zonos2Model":
        """Build a quantized trunk and load the packed int8/int4 weights.

        The quantized safetensors keys are ALREADY in MLX module-path layout
        (quantize.export_quantized dumped ``tree_flatten(model.parameters())``),
        so no remap_keys is needed. The Linears are converted to QuantizedLinear
        via the SAME predicate used at export, then the packed params are loaded;
        the experts are switched to the quantized gather-matmul path.
        """
        import mlx.nn as nn

        from .quantize import quant_linear_predicate

        model = cls(cfg)
        nn.quantize(
            model, group_size=qcfg.group_size, bits=qcfg.bits,
            class_predicate=quant_linear_predicate,
        )

        raw = mx.load(str(st_path))

        # --- install the quantized experts per MoE layer -------------------
        for base in qcfg.quantized_expert_bases:
            # base like "layers.3.moe.experts"
            parts = base.split(".")
            layer_id = int(parts[1])
            experts = model.layers[layer_id].moe.experts
            packed = {
                f"{name}_w_{kind}": raw[f"{base}.{name}_w_{kind}"]
                for name in ("gate", "up", "down")
                for kind in ("q", "scales", "biases")
            }
            experts.set_quantized(
                packed, bits=qcfg.expert_bits, group_size=qcfg.group_size
            )

        # --- load every non-expert tensor into the (now-quantized) tree ----
        expert_keys = {
            f"{base}.{name}_w_{kind}"
            for base in qcfg.quantized_expert_bases
            for name in ("gate", "up", "down")
            for kind in ("q", "scales", "biases")
        }
        params = _assemble_quantized_params(
            {k: v for k, v in raw.items() if k not in expert_keys}, cfg, qcfg,
        )
        model.update(params)
        mx.eval(model.parameters())
        return model


# ----------------------------------------------------------------------
def _assemble_quantized_params(raw: dict, cfg: Zonos2Config, qcfg) -> dict:
    """Unflatten the quantized safetensors (already in MLX module-path layout)
    into the nested param tree ``model.update`` expects.

    The exporter dumped ``tree_flatten(model.parameters())`` keys directly, so
    every key is already a dotted MLX path (``layers.3.attn.wq.weight``,
    ``layers.3.attn.wq.scales``/``.biases`` for the QuantizedLinears, plus the
    bf16-kept embeddings/norms/router/speaker). The fused wkv / ffn.w_in were
    already flattened at export, so no reshape is needed here. List indices
    (``layers.N``, ``embed.embedders.N``) and bare params (``attn.temp``,
    ``balancing_biases``) are handled by ``tree_unflatten``.
    """
    del cfg, qcfg  # the layout is fully self-describing via the flat keys.
    from mlx.utils import tree_unflatten

    return tree_unflatten(list(raw.items()))


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
