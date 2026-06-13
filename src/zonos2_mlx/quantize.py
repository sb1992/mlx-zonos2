"""Offline quantizer for the Zonos2 8B MoE trunk (int8 / int4).

Why this is more than ``nn.quantize``
-------------------------------------
~90% of the 8B params live in the **MoE experts** (24 MoE layers x 16 experts x
``w13``[6144,2048] + ``w2``[2048,3072]). Those are stored as bare ``mx.array``
params (``layers.N.moe.experts.w13``/``w2``) used via custom fused matmuls, NOT
``nn.Linear``. So ``nn.quantize`` alone touches only the small attention/FFN
Linears and yields almost no memory win. To actually cut the footprint we must
quantize the experts AND keep them quantized in memory (``mx.gather_qmm`` at
run time — see ``SonicExperts`` in layers.py).

What this exporter does (dots LLM-only playbook, extended to the experts)
-------------------------------------------------------------------------
QUANTIZE (bits=8 or 4, group_size=64):
  * per-layer attention ``wq``/``wkv``/``wo``/``gater`` (nn.Linear)
  * dense-FFN ``w_in``/``w_out`` (nn.Linear)
  * ``lm_head`` (nn.Linear)
  * the MoE experts ``w13``/``w2`` (the bulk) -- pre-split w13 into gate/up.

KEEP BF16:
  * embeddings (``embed.embedders.*``), all RMSNorm weights
  * the router (tiny: down_proj, mlp_0/2/4, rmsnorm_eda, balancing_biases,
    states_scale) -- quantizing it would hurt routing for ~no memory benefit
  * ``speaker_lda`` / ``speaker_proj`` / ``out_norm``

Expert split scheme (``split_gate_up_interleaved``)
---------------------------------------------------
``w13``[E,6144,2048] is interleaved gate/up (``w13[:,0::2,:]``=gate,
``w13[:,1::2,:]``=up). We pre-split at export to ``gate_w``[E,3072,2048] and
``up_w``[E,3072,2048], and rename ``w2``->``down_w``[E,2048,3072]. Each expert's
2D matrix is quantized independently and the per-expert ``(w_q, scales, biases)``
are stacked back to ``[E, ...]`` so run time can ``mx.gather_qmm`` with a
per-token expert index.

Output: ``zonos2_int{8,4}.safetensors`` + a ``quant_config.json`` sidecar.

Dev-only build tool. Pure MLX (``mlx`` only); no torch. LOCAL artifacts only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

mx.set_memory_limit(int(45 * (1 << 30)))

from .model import Zonos2Model  # noqa: E402

# Module paths (under the model tree) whose nn.Linear we DO quantize. The check
# is a substring match against the QuantizedLinear path nn.quantize hands the
# predicate, so these fragments uniquely select attention/ffn/lm_head and never
# the router or speaker Linears.
_QUANT_LINEAR_FRAGMENTS = (
    ".attn.wq",
    ".attn.wkv",
    ".attn.wo",
    ".attn.gater",
    ".ffn.w_in",
    ".ffn.w_out",
    "lm_head",
)

# Module paths we must NEVER quantize even if a fragment above were to match.
_NEVER_QUANT_FRAGMENTS = (
    ".moe.router.",
    "speaker_lda",
    "speaker_proj",
)


@dataclass
class QuantizationConfig:
    """Describes a quantization tier + which keys were quantized.

    ``bits`` is the nn.Linear (attention/ffn/lm_head) precision; ``expert_bits``
    the MoE-expert precision (may differ for a mixed tier; 0 => experts kept
    bf16). The loader uses ``expert_bits`` for the expert ``gather_qmm``.
    """

    bits: int
    group_size: int = 64
    expert_bits: int = 0
    # Linear module paths (nn.quantize) that were converted to QuantizedLinear.
    quantized_linear_paths: list[str] = field(default_factory=list)
    # Expert tensor base keys (e.g. "layers.3.moe.experts") whose
    # gate_w/up_w/down_w were quantized (empty when expert_bits == 0).
    quantized_expert_bases: list[str] = field(default_factory=list)
    expert_split_scheme: str = "split_gate_up_interleaved"

    def to_dict(self) -> dict:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "expert_bits": self.expert_bits,
            "quantized_linear_paths": sorted(self.quantized_linear_paths),
            "quantized_expert_bases": sorted(self.quantized_expert_bases),
            "expert_split_scheme": self.expert_split_scheme,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuantizationConfig":
        return cls(
            bits=int(d["bits"]),
            group_size=int(d.get("group_size", 64)),
            # Back-compat: older sidecars had no expert_bits => experts shared bits.
            expert_bits=int(d.get("expert_bits", d["bits"])),
            quantized_linear_paths=list(d.get("quantized_linear_paths", [])),
            quantized_expert_bases=list(d.get("quantized_expert_bases", [])),
            expert_split_scheme=d.get("expert_split_scheme", "split_gate_up_interleaved"),
        )


def quant_linear_predicate(path: str, module: nn.Module) -> bool:
    """``nn.quantize`` class_predicate: select ONLY the attention/ffn/lm_head
    nn.Linear layers (exclude the router + speaker Linears).

    ``path`` is the dotted module path; ``module`` the candidate. Only Linears
    with a quantizable last dim (divisible by group_size) are eligible — every
    selected layer's input dim (2048/3072) qualifies.
    """
    if not isinstance(module, nn.Linear):
        return False
    if any(frag in path for frag in _NEVER_QUANT_FRAGMENTS):
        return False
    return any(frag in path for frag in _QUANT_LINEAR_FRAGMENTS)


def split_experts_interleaved(w13: mx.array, w2: mx.array):
    """Split the fused interleaved ``w13`` into gate/up and rename ``w2``.

    ``w13``[E, 2*intermediate, dim] -> gate_w[E, intermediate, dim] (rows 0::2),
    up_w[E, intermediate, dim] (rows 1::2). ``w2``[E, dim, intermediate] ->
    down_w (unchanged). All three are the "output-major" (out, in) layout that
    ``mx.quantize`` / ``mx.gather_qmm(transpose=True)`` expect.
    """
    gate_w = w13[:, 0::2, :]
    up_w = w13[:, 1::2, :]
    down_w = w2
    return gate_w, up_w, down_w


def quantize_experts_stacked(w: mx.array, bits: int, group_size: int):
    """Quantize a stacked expert tensor ``w``[E, out, in] per-expert.

    Quantizes each expert's 2D ``[out, in]`` matrix independently and stacks the
    results, yielding ``(w_q[E, out, in_packed], scales[E, out, groups],
    biases[E, out, groups])`` — the batch layout ``mx.gather_qmm`` consumes.
    """
    n_experts = w.shape[0]
    wqs, scales, biases = [], [], []
    for e in range(n_experts):
        wq, sc, bi = mx.quantize(w[e].astype(mx.bfloat16), group_size=group_size, bits=bits)
        wqs.append(wq)
        scales.append(sc)
        biases.append(bi)
    return mx.stack(wqs), mx.stack(scales), mx.stack(biases)


def export_quantized(
    weights_dir: str,
    out_path: str,
    bits: int,
    group_size: int = 64,
    expert_bits: int | None = None,
) -> dict:
    """Quantize the bf16 trunk to ``bits`` and write the safetensors + sidecar.

    Loads the bf16 trunk (``weights_dir``), quantizes the selected nn.Linear via
    ``nn.quantize`` and the experts via per-expert ``mx.quantize``, writes
    ``out_path`` (``zonos2_int{bits}.safetensors``) + a ``quant_config.json``
    sidecar beside it. Returns a summary dict.

    ``expert_bits`` defaults to ``bits`` (uniform tier). Pass a different value
    for a MIXED tier — e.g. ``bits=4, expert_bits=8`` (int4 attention/ffn,
    int8 experts) or ``expert_bits=0`` to keep the experts bf16 (no expert
    quantization at all). This is the fallback ladder for the top-1-MoE
    quantization-sensitivity documented in the eval.
    """
    if bits not in (4, 8):
        raise ValueError(f"--bits must be 4 or 8 (got {bits})")
    if expert_bits is None:
        expert_bits = bits
    if expert_bits not in (0, 4, 8):
        raise ValueError(f"--expert-bits must be 0 (bf16), 4, or 8 (got {expert_bits})")

    wdir = Path(weights_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the full bf16 model (this is the 15 GB load — GPU, serial).
    # from_pretrained validates config.json exists alongside the weights.
    model = Zonos2Model.from_pretrained(str(wdir))

    # --- 1. Quantize the nn.Linear set (attention/ffn/lm_head) in place. ----
    quantized_linear_paths: list[str] = []

    def _record_predicate(path: str, module: nn.Module):
        keep = quant_linear_predicate(path, module)
        if keep:
            quantized_linear_paths.append(path)
        return keep

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=_record_predicate)

    # --- 2. Pull the (now-mixed) param tree out as a flat dict. -------------
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.parameters()))

    # --- 3. Quantize the experts: split w13, quantize gate/up/down per expert,
    #        drop the bf16 w13/w2, insert the *_q/*_scales/*_biases. ---------
    quantized_expert_bases: list[str] = []
    out_tensors: dict[str, mx.array] = {}

    expert_w13_keys = sorted(k for k in flat if k.endswith(".moe.experts.w13"))
    expert_bases = [k[: -len(".w13")] for k in expert_w13_keys]
    expert_param_keys: set[str] = set()

    if expert_bits != 0:
        # Quantize the experts at expert_bits (the bulk of the memory win).
        for base in expert_bases:
            expert_param_keys.add(base + ".w13")
            expert_param_keys.add(base + ".w2")
        for base in expert_bases:
            w13 = flat[base + ".w13"]
            w2 = flat[base + ".w2"]
            gate_w, up_w, down_w = split_experts_interleaved(w13, w2)
            for name, w in (("gate", gate_w), ("up", up_w), ("down", down_w)):
                wq, sc, bi = quantize_experts_stacked(
                    w, bits=expert_bits, group_size=group_size
                )
                out_tensors[f"{base}.{name}_w_q"] = wq
                out_tensors[f"{base}.{name}_w_scales"] = sc
                out_tensors[f"{base}.{name}_w_biases"] = bi
            quantized_expert_bases.append(base)
    # else: experts stay bf16 — their w13/w2 fall through the copy below.

    # --- 4. Copy every remaining tensor through (quantized Linears already in
    #        packed form; everything kept bf16 stays bf16). ------------------
    for k, v in flat.items():
        if k in expert_param_keys:
            continue  # replaced by the per-expert split above
        # Keep packed uint32 weights as-is; cast float params to bf16.
        out_tensors[k] = v if v.dtype == mx.uint32 else v.astype(mx.bfloat16)

    # Realise everything before saving (avoids a lazy 15 GB graph at write).
    mx.eval(out_tensors)

    qcfg = QuantizationConfig(
        bits=bits,
        group_size=group_size,
        expert_bits=expert_bits,
        quantized_linear_paths=quantized_linear_paths,
        quantized_expert_bases=quantized_expert_bases,
    )

    mx.save_safetensors(str(out_path), out_tensors)
    sidecar = out_path.parent / "quant_config.json"
    sidecar.write_text(json.dumps(qcfg.to_dict(), indent=2))

    return {
        "out": str(out_path),
        "sidecar": str(sidecar),
        "bits": bits,
        "expert_bits": expert_bits,
        "group_size": group_size,
        "n_quantized_linear": len(quantized_linear_paths),
        "n_quantized_expert_layers": len(quantized_expert_bases),
        "n_tensors": len(out_tensors),
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description="Quantize the Zonos2 bf16 trunk to int8/int4.")
    ap.add_argument("--weights-dir", default="weights/zonos2-bf16", help="bf16 trunk dir")
    ap.add_argument("--out", required=True, help="output safetensors path")
    ap.add_argument("--bits", type=int, choices=(4, 8), required=True)
    ap.add_argument(
        "--expert-bits", type=int, choices=(0, 4, 8), default=None,
        help="MoE-expert precision (default=--bits; 0 keeps experts bf16). "
        "Use for a mixed tier, e.g. --bits 4 --expert-bits 8.",
    )
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()
    summary = export_quantized(
        args.weights_dir, args.out, bits=args.bits,
        group_size=args.group_size, expert_bits=args.expert_bits,
    )
    print(f"[quantize] wrote {summary}", flush=True)


if __name__ == "__main__":
    _main()
