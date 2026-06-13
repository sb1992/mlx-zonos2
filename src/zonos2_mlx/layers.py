"""MLX layer primitives for the Zonos2 8B MoE transformer trunk.

Mirrors the plain-PyTorch reference (``Zonos2_TTS-ComfyUI/native.py``)
op-for-op. All line references below are into that ``native.py``.

bf16 weights, bf16 compute. The parity target is cosine >= 0.999 against
the torch oracle (dumped on MPS / bf16), not bit-exactness.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def softcap(logits: mx.array, cap: float) -> mx.array:
    """``cap * tanh(logits / cap)`` (native.py 719-721)."""
    if cap <= 0:
        return logits
    return cap * mx.tanh(logits / cap)


class RMSNorm(nn.Module):
    """Weighted RMSNorm (native.py RMSNorm / F.rms_norm).

    ``F.rms_norm(x, weight, eps)`` computes
    ``x / sqrt(mean(x^2) + eps) * weight``. The reduction is done in the
    input dtype; we match by normalising in the working dtype.
    """

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = mx.ones((size,))
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        var = mx.mean(x.astype(mx.float32) * x.astype(mx.float32), axis=-1, keepdims=True)
        normed = (x.astype(mx.float32) * mx.rsqrt(var + self.eps)).astype(x.dtype)
        return normed * self.weight


def rms_norm_weightless(x: mx.array, eps: float) -> mx.array:
    """RMSNorm with no learned weight (native.py 698-703 sequence norm;
    also the q/k QK-norm at lines 365/367 with eps=1e-6)."""
    var = mx.mean(x.astype(mx.float32) * x.astype(mx.float32), axis=-1, keepdims=True)
    return (x.astype(mx.float32) * mx.rsqrt(var + eps)).astype(x.dtype)


def apply_interleaved_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Interleaved (non-neox) RoPE (native.py _apply_interleaved_rope 223-235).

    ``x`` is (B, T, H, head_dim). cos/sin are (1, T, 1, head_dim//2).
    Pairs adjacent dims (even, odd), rotates, and re-interleaves.
    """
    pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    even = pair[..., 0]
    odd = pair[..., 1]
    rot_even = even * cos - odd * sin
    rot_odd = odd * cos + even * sin
    rotated = mx.stack((rot_even, rot_odd), axis=-1)
    return rotated.reshape(x.shape)


def rope_tables(head_dim: int, max_seqlen: int, theta: float):
    """Precompute cos/sin tables (native.py 259-269)."""
    inv_freq = 1.0 / (
        theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    )
    positions = mx.arange(max_seqlen, dtype=mx.float32)
    freqs = mx.outer(positions, inv_freq)
    return mx.cos(freqs), mx.sin(freqs)


class KVCache:
    """Per-layer incremental K/V cache (native.py LayerKVCache 207-220).

    Holds k,v as ``(B, n_kv_heads, T, head_dim)`` grown by concat along the time
    axis (axis=2). ``length`` tracks how many positions are populated and drives
    the RoPE cos/sin offset slice (``cos = rope[start:end]``).

    The cache stores the PRE-GQA-repeat kv (n_kv_heads heads), exactly like the
    oracle's ``cache.key`` which is shaped on ``num_kv_heads``.
    """

    def __init__(self):
        self.k: mx.array | None = None
        self.v: mx.array | None = None
        self.length = 0

    def update(self, new_k: mx.array, new_v: mx.array) -> tuple[mx.array, mx.array]:
        """Append ``new_k``/``new_v`` (B, n_kv_heads, qlen, head_dim) and return
        the full ``(B, n_kv_heads, length, head_dim)`` k/v."""
        if self.k is None:
            self.k = new_k
            self.v = new_v
        else:
            self.k = mx.concatenate([self.k, new_k], axis=2)
            self.v = mx.concatenate([self.v, new_v], axis=2)
        self.length = self.k.shape[2]
        return self.k, self.v


class Attention(nn.Module):
    """GQA attention with QK-RMSNorm, learned per-head query temperature,
    and a learned sigmoid output gate (native.py Attention 238-389).
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.repeat = n_heads // n_kv_heads
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        # wkv stays fused: weight [2, 512, 2048] -> flattened [1024, 2048].
        self.wkv = nn.Linear(dim, 2 * n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.gater = nn.Linear(dim, n_heads, bias=False)
        # temp is a bare parameter [1, 16, 1]; per-head query temperature.
        self.temp = mx.zeros((1, n_heads, 1))

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        cache: "KVCache | None" = None,
    ) -> mx.array:
        """Attention.

        ``cos``/``sin`` are the RoPE tables ALREADY sliced for this call's
        positions (``rope[start:end]``) — the model passes the offset slice.

        - ``cache is None``: the T2 full-causal path (unchanged math).
        - ``cache is not None``: compute q/k/v for the new tokens, append k/v to
          the cache, attend q against the FULL cached k/v. Causal mask only when
          qlen>1 (multi-token prefill); single-step decode uses no mask
          (native.py 381).
        """
        batch, seqlen, _ = x.shape

        # Output gate from the pre-projection normalized input (native.py 355).
        gate = mx.sigmoid(self.gater(x))  # (B, T, n_heads)

        q = self.wq(x).reshape(batch, seqlen, self.n_heads, self.head_dim)
        kv = self.wkv(x).reshape(batch, seqlen, 2, self.n_kv_heads, self.head_dim)
        k = kv[:, :, 0]
        v = kv[:, :, 1]

        # QK-RMSNorm (eps=1e-6, no weight), then per-head query temperature
        # which REPLACES the usual 1/sqrt(head_dim) scaling (native.py 365-367).
        q = rms_norm_weightless(q, 1e-6)
        # temp: (1, n_heads, 1) -> (1, 1, n_heads, 1) to broadcast over (B,T,H,d).
        q = q * mx.abs(self.temp).reshape(1, 1, self.n_heads, 1).astype(q.dtype)
        k = rms_norm_weightless(k, 1e-6)

        # RoPE applied after norm+temp (native.py 373-374).
        q = apply_interleaved_rope(q, cos, sin)
        k = apply_interleaved_rope(k, cos, sin)

        # To (B, H, T, d) for SDPA.
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Cached path: append the new (pre-GQA) k/v, attend against the full
        # accumulated k/v (native.py 376-381). A multi-token call is only valid
        # into an EMPTY cache (prefill at start==0); chunked prefill into a
        # non-empty cache would make the `seqlen > 1` causal mask below wrong
        # (it assumes square Q==K). Guard it explicitly.
        if cache is not None:
            if seqlen > 1 and cache.length != 0:
                raise NotImplementedError(
                    "multi-token attention into a non-empty KV cache "
                    "(chunked prefill) is unsupported"
                )
            k, v = cache.update(k, v)

        # GQA: repeat kv heads to n_heads.
        if self.repeat > 1:
            k = mx.repeat(k, self.repeat, axis=1)
            v = mx.repeat(v, self.repeat, axis=1)

        # The learned per-head `temp` multiplies q BEFORE the attention scale;
        # native.py calls torch SDPA with no explicit scale, so the default
        # 1/sqrt(head_dim) still applies on top of temp (lines 310-326, 366).
        # Causal mask only when this call adds >1 query token at the sequence
        # start (prefill); single-step decode against the cache uses no mask.
        scale = 1.0 / math.sqrt(self.head_dim)
        mask = "causal" if seqlen > 1 else None
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask
        )  # (B, H, T, d)

        # Back to (B, T, H, d), apply per-head gate (native.py 388).
        attended = attended.transpose(0, 2, 1, 3)
        attended = attended * gate[..., None]
        attended = attended.reshape(batch, seqlen, self.n_heads * self.head_dim)
        return self.wo(attended)


class DenseFeedForward(nn.Module):
    """SiLU-gated dense FFN (native.py DenseFeedForward 392-404).

    ``w_in`` is fused [2, 3072, 2048] -> flattened [6144, 2048]. The first
    3072 outputs are ``up``, the second 3072 are ``gate``:
    ``out = w_out(up * silu(gate))``.
    """

    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.intermediate = intermediate
        self.w_in = nn.Linear(dim, 2 * intermediate, bias=False)
        self.w_out = nn.Linear(intermediate, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        projected = self.w_in(x)
        up = projected[..., : self.intermediate]
        gate = projected[..., self.intermediate :]
        return self.w_out(up * nn.silu(gate))


class Router(nn.Module):
    """MoE router (native.py Router 494-542).

    down_proj (+bias) -> optional EDA carry-in add -> snapshot pre-norm state
    -> rmsnorm_eda -> MLP(Linear-GELU-Linear-GELU-Linear) -> softmax over
    16 experts (in fp32) -> add balancing_biases -> top-k by the biased score
    -> gather the *unbiased* softmax probs as the routing weights.
    """

    def __init__(self, dim: int, router_dim: int, n_experts: int, top_k: int, use_eda: bool):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_eda = use_eda
        self.down_proj = nn.Linear(dim, router_dim, bias=True)
        # router_mlp.{0,2,4}; GELU sits at indices 1,3 (no params).
        self.mlp_0 = nn.Linear(router_dim, router_dim, bias=True)
        self.mlp_2 = nn.Linear(router_dim, router_dim, bias=True)
        self.mlp_4 = nn.Linear(router_dim, n_experts, bias=False)
        self.rmsnorm_eda = RMSNorm(router_dim, 1e-5)
        if use_eda:
            self.states_scale = mx.zeros((router_dim,))
        self.balancing_biases = mx.zeros((n_experts,))

    def __call__(self, hidden: mx.array, router_states: mx.array | None):
        projected = self.down_proj(hidden)
        if self.use_eda and router_states is not None:
            projected = projected + router_states * self.states_scale
        next_states = projected
        mlp_out = self.mlp_4(nn.gelu(self.mlp_2(nn.gelu(self.mlp_0(self.rmsnorm_eda(projected))))))
        expert_prob = mx.softmax(mlp_out.astype(mx.float32), axis=-1)
        routing_scores = expert_prob + self.balancing_biases.astype(mx.float32)
        # top-k indices by biased score; route weights are the unbiased probs.
        idx = mx.argpartition(-routing_scores, self.top_k - 1, axis=-1)[..., : self.top_k]
        # Order within the top-k does not matter for a weighted sum; the torch
        # ref sorts experts by id when summing, which is also order-invariant.
        route_prob = mx.take_along_axis(expert_prob, idx, axis=-1)
        return route_prob, idx, next_states


class SonicExperts(nn.Module):
    """Fused-weight experts (native.py SonicExperts 407-491).

    ``w13[e]`` is [6144, 2048]; gate = x @ w13[e][0::2].T, up = x @ w13[e][1::2].T
    (INTERLEAVED stride-2, not split-half). ``w2[e]`` is [2048, 3072].
    Each selected expert's output is weighted by its routing prob and summed.

    Two weight layouts are supported, selected at load time:
      * bf16 (default): the fused ``w13``/``w2`` arrays + the per-token gather
        matmul above. This is the unchanged T2/T6 path.
      * quantized: per-expert int8/int4 ``gate_w``/``up_w``/``down_w`` (packed
        uint32 + scales + biases). The forward routes each token to its expert
        via ``mx.gather_qmm`` (the quantized gather-matmul), keeping the weights
        quantized in memory (no dequant-to-bf16). Enabled by ``set_quantized``.
    """

    def __init__(self, n_experts: int, dim: int, intermediate: int):
        super().__init__()
        self.n_experts = n_experts
        self.dim = dim
        self.intermediate = intermediate
        # bf16 fused layout: w13 [E, 2*intermediate, dim]; w2 [E, dim, intermediate]
        self.w13 = mx.zeros((n_experts, 2 * intermediate, dim))
        self.w2 = mx.zeros((n_experts, dim, intermediate))
        # Quantized layout (populated by set_quantized; None => bf16 path).
        # Per-projection bits: gate/up share gate_up_bits, down has down_bits.
        self._quant_bits: dict[str, int] | None = None
        self._quant_group_size: int = 64

    def set_quantized(
        self,
        packed: dict,
        gate_up_bits: int,
        down_bits: int,
        group_size: int,
    ) -> None:
        """Switch this expert block to the quantized gather-matmul path.

        ``packed`` carries ``{gate,up,down}_w_{q,scales,biases}`` stacked over
        the expert axis (built by quantize.export_quantized). The experts use
        **per-projection** bits: gate_w/up_w at ``gate_up_bits`` (int4) and down_w
        at ``down_bits`` (int8) — see docs/research/02. The bf16 ``w13``/``w2``
        placeholders are dropped to free memory; the quantized tensors are
        registered as bare params so ``model.update`` / ``mx.eval`` see them.
        """
        self._quant_bits = {
            "gate": int(gate_up_bits),
            "up": int(gate_up_bits),
            "down": int(down_bits),
        }
        self._quant_group_size = int(group_size)
        # Drop the bf16 placeholders so they don't sit in memory.
        self.w13 = None
        self.w2 = None
        for name in ("gate", "up", "down"):
            setattr(self, f"{name}_w_q", packed[f"{name}_w_q"])
            setattr(self, f"{name}_w_scales", packed[f"{name}_w_scales"])
            setattr(self, f"{name}_w_biases", packed[f"{name}_w_biases"])

    def _qproj(self, x: mx.array, name: str, ids: mx.array) -> mx.array:
        """Per-token quantized projection ``x[t] @ W[ids[t]].T`` via gather_qmm.

        ``x`` is (tokens, in); ``ids`` is (tokens,) expert indices. Returns
        (tokens, out). ``mx.gather_qmm`` picks ``W[ids[t]]`` per row. The bit width
        is per-projection (gate/up vs down), so it MUST match the bits each
        weight was quantized at in the exporter.
        """
        out = mx.gather_qmm(
            x[:, None, :],
            getattr(self, f"{name}_w_q"),
            getattr(self, f"{name}_w_scales"),
            getattr(self, f"{name}_w_biases"),
            rhs_indices=ids,
            transpose=True,
            group_size=self._quant_group_size,
            bits=self._quant_bits[name],
        )
        return out.reshape(x.shape[0], -1)

    def __call__(self, x: mx.array, route_prob: mx.array, expert_ids: mx.array) -> mx.array:
        # x: (tokens, dim); expert_ids/route_prob: (tokens, top_k)
        tokens = x.shape[0]
        top_k = expert_ids.shape[-1]
        output = mx.zeros_like(x)

        if self._quant_bits is not None:
            # Quantized gather-matmul path: weights stay int8/int4 in memory.
            for slot in range(top_k):
                ids = expert_ids[:, slot].astype(mx.uint32)  # (tokens,)
                weights = route_prob[:, slot]                # (tokens,)
                gate = self._qproj(x, "gate", ids)           # (tokens, intermediate)
                up = self._qproj(x, "up", ids)               # (tokens, intermediate)
                h = nn.silu(gate) * up                       # (tokens, intermediate)
                expert_out = self._qproj(h, "down", ids)     # (tokens, dim)
                output = output + expert_out * weights[:, None].astype(expert_out.dtype)
            return output.reshape(tokens, self.dim)

        # bf16 fused path (unchanged).
        # Interleaved gate/up split, precomputed views per call.
        gate_w = self.w13[:, 0::2, :]  # (E, intermediate, dim)
        up_w = self.w13[:, 1::2, :]    # (E, intermediate, dim)
        for slot in range(top_k):
            ids = expert_ids[:, slot]          # (tokens,)
            weights = route_prob[:, slot]      # (tokens,)
            # Gather per-token expert weight matrices.
            g = gate_w[ids]                    # (tokens, intermediate, dim)
            u = up_w[ids]                      # (tokens, intermediate, dim)
            d = self.w2[ids]                   # (tokens, dim, intermediate)
            xt = x[:, None, :]                 # (tokens, 1, dim)
            gate = mx.sum(xt * g, axis=-1)     # (tokens, intermediate)
            up = mx.sum(xt * u, axis=-1)       # (tokens, intermediate)
            h = (nn.silu(gate) * up)[:, None, :]  # (tokens, 1, intermediate)
            expert_out = mx.sum(h * d, axis=-1)   # (tokens, dim)
            output = output + expert_out * weights[:, None].astype(expert_out.dtype)
        return output.reshape(tokens, self.dim)


class MoEFeedForward(nn.Module):
    """Router + experts (native.py MoEFeedForward 545-567)."""

    def __init__(self, dim: int, intermediate: int, router_dim: int,
                 n_experts: int, top_k: int, use_eda: bool):
        super().__init__()
        self.router = Router(dim, router_dim, n_experts, top_k, use_eda)
        self.experts = SonicExperts(n_experts, dim, intermediate)

    def __call__(self, x: mx.array, router_states: mx.array | None):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        prev = router_states.reshape(-1, router_states.shape[-1]) if router_states is not None else None
        route_prob, expert_ids, next_states = self.router(flat, prev)
        out = self.experts(flat, route_prob, expert_ids)
        return out.reshape(shape), next_states.reshape(*shape[:-1], next_states.shape[-1])
