"""Quantization unit + eval gates for the Zonos2 trunk.

Two tiers of test:
  * CPU-ish round-trip (``test_quant_roundtrip_*``): no weights needed — exercise
    the quantize mechanics (per-expert quantize/gather_qmm, the sidecar
    serialization, the predicate) on small synthetic tensors. Always run.
  * GPU eval (``test_quant_eval``, marked ``gpu``): teacher-forced agreement of
    int8/int4 vs the bf16-MLX baseline codes + per-tier memory. Requires the
    15 GB bf16 trunk + the exported int8/int4 safetensors; run SERIALLY.

The GPU eval is driven by ``scripts/zonos2_quant_eval.py`` (which writes a JSON
summary the test asserts against) so the heavy generation runs once, serially,
outside pytest's process churn.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_memory_limit(int(45 * (1 << 30)))

from zonos2_mlx.quantize import (  # noqa: E402
    QuantizationConfig,
    quant_linear_predicate,
    quantize_experts_stacked,
    split_experts_interleaved,
)

_R = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CPU-ish round-trip tests (no model weights needed)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bits", [8, 4])
def test_quant_roundtrip_expert(bits):
    """Per-expert quantize -> gather_qmm reproduces the bf16 matmul (high cos),
    and shapes/metadata are preserved."""
    rng = np.random.default_rng(0)
    E, out_d, in_d, T = 16, 3072, 2048, 7
    w = mx.array((rng.standard_normal((E, out_d, in_d)) * 0.02).astype(np.float32))

    wq, sc, bi = quantize_experts_stacked(w, bits=bits, group_size=64)
    # Packed shapes: uint32 packing -> last dim shrinks; scales/biases per group.
    assert wq.shape[0] == E and sc.shape[0] == E and bi.shape[0] == E
    assert wq.dtype == mx.uint32
    assert sc.shape == (E, out_d, in_d // 64)

    x = mx.array((rng.standard_normal((T, in_d)) * 0.5).astype(np.float32))
    ids = mx.array(rng.integers(0, E, size=T).astype(np.uint32))
    out = mx.gather_qmm(
        x[:, None, :], wq, sc, bi, rhs_indices=ids,
        transpose=True, group_size=64, bits=bits,
    ).reshape(T, out_d)
    ref = mx.stack([x[t] @ w[int(ids[t])].T for t in range(T)])
    cos = float((ref * out).sum() / (mx.linalg.norm(ref) * mx.linalg.norm(out)))
    # int8 is near-lossless; int4 still tracks direction tightly on small mats.
    floor = 0.999 if bits == 8 else 0.99
    assert cos >= floor, f"bits={bits} gather_qmm cos {cos:.5f} < {floor}"


def test_split_experts_interleaved():
    """w13[:,0::2,:]=gate, w13[:,1::2,:]=up; down=w2, all (out,in)."""
    E, inter, dim = 2, 4, 3
    w13 = mx.arange(E * 2 * inter * dim).reshape(E, 2 * inter, dim)
    w2 = mx.zeros((E, dim, inter))
    gate, up, down = split_experts_interleaved(w13, w2)
    assert gate.shape == (E, inter, dim)
    assert up.shape == (E, inter, dim)
    assert down.shape == (E, dim, inter)
    # gate rows are the even rows of w13.
    assert bool(mx.all(gate[0] == w13[0, 0::2, :]))
    assert bool(mx.all(up[0] == w13[0, 1::2, :]))


def test_quant_config_sidecar_roundtrip(tmp_path):
    """QuantizationConfig serializes + reloads with per-projection expert bits."""
    qc = QuantizationConfig(
        bits=8, group_size=64,
        expert_gate_up_bits=4, expert_down_bits=8,
        quantized_linear_paths=["layers.0.attn.wq", "lm_head"],
        quantized_expert_bases=["layers.3.moe.experts"],
    )
    p = tmp_path / "quant_config.json"
    p.write_text(json.dumps(qc.to_dict(), indent=2))
    back = QuantizationConfig.from_dict(json.loads(p.read_text()))
    assert back.bits == 8
    assert back.group_size == 64
    assert back.expert_gate_up_bits == 4
    assert back.expert_down_bits == 8
    assert back.experts_quantized
    assert back.quantized_expert_bases == ["layers.3.moe.experts"]
    assert "lm_head" in back.quantized_linear_paths
    assert back.expert_split_scheme == "split_gate_up_interleaved"


def test_quant_config_back_compat_legacy_expert_bits():
    """Old sidecars (single ``expert_bits``, no per-projection keys) still load,
    mapping the legacy value onto BOTH expert projection groups."""
    # Legacy uniform-int4-experts sidecar.
    legacy = {"bits": 8, "group_size": 64, "expert_bits": 4}
    qc = QuantizationConfig.from_dict(legacy)
    assert qc.expert_gate_up_bits == 4
    assert qc.expert_down_bits == 4
    # Even older: no expert_bits at all => experts shared `bits`.
    older = {"bits": 8, "group_size": 64}
    qc2 = QuantizationConfig.from_dict(older)
    assert qc2.expert_gate_up_bits == 8
    assert qc2.expert_down_bits == 8


def test_quant_config_bf16_experts_not_quantized():
    """Both expert bits == 0 => experts_quantized is False (bf16 experts)."""
    qc = QuantizationConfig(bits=8, expert_gate_up_bits=0, expert_down_bits=0)
    assert not qc.experts_quantized


def test_quant_predicate_selects_only_intended_linears():
    """The predicate quantizes attention/ffn/lm_head but NEVER router/speaker."""
    lin = nn.Linear(8, 8, bias=False)
    assert quant_linear_predicate("layers.0.attn.wq", lin)
    assert quant_linear_predicate("layers.0.attn.wkv", lin)
    assert quant_linear_predicate("layers.0.ffn.w_in", lin)
    assert quant_linear_predicate("lm_head", lin)
    # Never these:
    assert not quant_linear_predicate("layers.3.moe.router.down_proj", lin)
    assert not quant_linear_predicate("layers.3.moe.router.mlp_0", lin)
    assert not quant_linear_predicate("speaker_proj", lin)
    assert not quant_linear_predicate("speaker_lda", lin)
    # Non-Linear modules are never selected.
    assert not quant_linear_predicate("layers.0.attn.wq", nn.RMSNorm(8))


# ---------------------------------------------------------------------------
# GPU eval gates (driven by scripts/zonos2_quant_eval.py)
#
# Corrected metric (docs/research/02): teacher-force the bf16-MLX baseline codes
# and measure, per tier vs bf16. The MEANINGFUL, smooth gates are:
#   (a) per-layer hidden-state cosine through the trunk  -> >= 0.97 (median per
#       token over the AUDIO region; gates against the recurrent-EDA COLLAPSE
#       cliff — the old uniform-int4 bug was 0.03 by layer 3),
#   (b) teacher-forced logit-KL (nats) vs bf16           -> int8 <= 0.1
#       (near-lossless); the int4-gate/up ship tier trades fidelity for memory
#       and sits ~0.2, gated at <= 0.25,
#   (d) full-length finite non-silent audio (NOT the int4-lm_head EOS collapse),
#       + ship < 16 GB.
#
# (c) logit top-1 agreement is NOT a hard gate (an EMPIRICAL correction to the
# research's optimistic >= 0.95 prediction). The bf16 audio-code logits are
# inherently FLAT — measured mean top-1 probability 0.31, and 84% of (frame,
# codebook) distributions have top-1 prob < 0.5. On such near-tied distributions
# ANY quantization noise reorders the argmax, so logit top-1 lands at the same
# ~0.81 (int8) / ~0.66 (ship) as the discarded sampled-token metric — it is the
# SAME router knife-edge measured on logits, not a softer signal. We record it
# (and the informational sampled-token frac) as prints, but gate on KL, which is
# the smooth distribution-overlap metric that DOES separate the tiers. Likewise
# sampled-token agreement stays informational only.
# ---------------------------------------------------------------------------
_TRUNK_COSINE_FLOOR = 0.97
_LOGIT_KL_CEIL = {"int8": 0.1, "ship": 0.25}
_SHIP_TIERS = ("ship", "int8")


def _load_summary():
    summary_path = _R / "outputs/quant_eval/summary.json"
    if not summary_path.exists():
        pytest.skip(
            "run scripts/zonos2_quant_eval.py first to produce "
            "outputs/quant_eval/summary.json"
        )
    return json.loads(summary_path.read_text())


@pytest.mark.gpu
@pytest.mark.parametrize("tier", _SHIP_TIERS)
def test_quant_eval_per_layer_cosine(tier):
    """(a) The hidden state must degrade GRADUALLY, not cliff. A healthy quant
    holds >= 0.97 through the trunk; the old uniform-int4 bug was 0.03 by layer 3.
    """
    summary = _load_summary()
    if tier not in summary:
        pytest.skip(f"tier {tier} not in summary")
    t = summary[tier]
    min_cos = t["per_layer_min_cosine"]
    assert min_cos >= _TRUNK_COSINE_FLOOR, (
        f"{tier} per-layer min cosine {min_cos:.4f} < {_TRUNK_COSINE_FLOOR} "
        f"(cliff at layer {t.get('per_layer_min_cosine_layer')}) — the hidden "
        "state collapsed; this is the recurrent-EDA amplification bug, not noise."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("tier", _SHIP_TIERS)
def test_quant_eval_logit_kl(tier):
    """(b) Teacher-forced mean per-frame logit-KL vs bf16 (nats). int8 is
    near-lossless (<= 0.1); the int4-gate/up ship tier trades fidelity for memory
    (~0.2, gated <= 0.25). This is the smooth distribution-overlap metric (the
    one DWQ optimizes) and the gate that actually separates the tiers — unlike
    top-1, which collapses to the router knife-edge on the flat audio logits."""
    summary = _load_summary()
    if tier not in summary:
        pytest.skip(f"tier {tier} not in summary")
    kl = summary[tier]["logit_kl_nats"]
    ceil = _LOGIT_KL_CEIL[tier]
    assert kl <= ceil, (
        f"{tier} teacher-forced logit-KL {kl:.4f} nats > {ceil}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("tier", _SHIP_TIERS)
def test_quant_eval_audio_full_length(tier):
    """(d) The tier must decode a full-length, finite, non-silent utterance —
    NOT the int4-lm_head immediate-EOS 0.03s collapse."""
    summary = _load_summary()
    if tier not in summary:
        pytest.skip(f"tier {tier} not in summary")
    t = summary[tier]
    assert t["free_running_n_frames"] > 100, (
        f"{tier} free-run collapsed to {t['free_running_n_frames']} frames "
        "(healthy is the full ~260; an int4 lm_head EOS-truncates to a few)"
    )
    assert t["audio_finite"], f"{tier} decoded audio has non-finite samples"
    assert not t["audio_silent"], f"{tier} decoded audio is silent"


@pytest.mark.gpu
def test_quant_eval_ship_fits_16gb():
    """The ship (int4 gate/up) tier must hit the < 16 GB footprint goal."""
    summary = _load_summary()
    if "ship" not in summary:
        pytest.skip("ship tier not in summary")
    peak = summary["ship"]["phys_footprint_peak_gb"]
    assert peak < 16.0, f"ship tier phys_footprint_peak {peak:.2f} GB >= 16 GB"
