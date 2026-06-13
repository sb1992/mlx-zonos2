"""Per-tier quantization eval for the Zonos2 trunk — CORRECTED METRIC (GPU, SERIAL).

docs/research/02 diagnosed the prior T7 anomalies: sampled-token agreement on a
top-1 router is a discrete knife-edge (~0.85 noise floor) that measures router
chaos, NOT quant quality. The real signals are smooth:

  (a) per-layer hidden-state cosine through the trunk (the bisection diagnostic;
      a healthy quant degrades GRADUALLY — the old uniform-int4 bug cliffed to
      ~0.03 by layer 3 because the int4 lm_head/linears corrupted layer 0 and the
      recurrent EDA router-state carry amplified it across 24 layers),
  (b) teacher-forced logit-KL vs bf16 (nats) — the smooth distribution-overlap
      metric (what DWQ optimizes); the gate that separates the tiers,
  (c) logit top-1 agreement (argmax per codebook) — INFORMATIONAL only: the bf16
      audio logits are flat (mean top-1 prob ~0.31, 84% < 0.5), so any quant
      reorders the argmax and top-1 lands at the same ~0.66-0.81 as the discarded
      sampled-token metric — the same router knife-edge on logits, not a gate,
  (d) full-length finite non-silent audio (an int4 lm_head EOS-truncates to a
      few frames).

All metrics teacher-force the bf16-MLX baseline codes
(``outputs/fixtures/baseline_bf16_codes.npy``) so the trajectory is identical
across tiers and the only variable is the quant. The bf16 run additionally caches
its per-layer captures + per-frame logits to disk so each tier (run in its own
process, one trunk in memory at a time) can diff against them.

    python scripts/zonos2_quant_eval.py --tier bf16
    python scripts/zonos2_quant_eval.py --tier int8 --weights weights/zonos2-int8
    python scripts/zonos2_quant_eval.py --tier int4 --weights weights/zonos2-int4
    python scripts/zonos2_quant_eval.py --tier int8lin-int4exp \
        --weights weights/_diag-int8lin-int4exp   # bisection-confirm tier (diag)
    python scripts/zonos2_quant_eval.py --tier summarize

Pure MLX (+ numpy host-side). No torch.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_memory_limit(int(45 * (1 << 30)))

from zonos2_mlx.generate import SamplingOptions, sample_codes  # noqa: E402
from zonos2_mlx.model import Zonos2Model  # noqa: E402
from zonos2_mlx.pipeline import synthesize  # noqa: E402
from zonos2_mlx.textnorm import build_prompt  # noqa: E402

_R = Path(__file__).resolve().parent.parent
_F = _R / "outputs/fixtures"
_OUT = _R / "outputs/quant_eval"

_TEXT = "The quick brown fox jumps over the lazy dog."
_BASELINE_CODES = _F / "baseline_bf16_codes.npy"
_BASELINE_LAYERS = _F / "baseline_bf16_layers.npz"   # per-layer captures (bf16)
_BASELINE_LOGITS = _F / "baseline_bf16_logits.npy"   # per-frame logits (bf16)
# The DAC codec is tier-independent (it only decodes audio codes), so every tier
# reuses the bf16 dir's dac_44khz rather than duplicating it per tier.
_DAC_DIR = _R / "weights/zonos2-bf16/dac_44khz"


# ---------------------------------------------------------------------------
# Memory measurement: proc_pid_rusage -> ri_phys_footprint (current footprint).
# Poll after mx.eval + mx.synchronize; max across phases = the per-tier peak.
# ---------------------------------------------------------------------------
class _RUsageInfoV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
    ]


_LIBC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_RUSAGE_INFO_V2 = 2


def phys_footprint_gb() -> float:
    """Current process physical footprint in GB (mx.synchronize first)."""
    mx.synchronize()
    ri = _RUsageInfoV2()
    rc = _LIBC.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V2, ctypes.byref(ri))
    if rc != 0:
        return float("nan")
    return ri.ri_phys_footprint / (1 << 30)


# ---------------------------------------------------------------------------
def _greedy_opts() -> SamplingOptions:
    return SamplingOptions(
        temperature=0.0,
        repetition_window=50,
        repetition_penalty=1.2,
        repetition_codebooks=8,
    )


def _teacher_forced_sequence(cfg, ref_codes: np.ndarray):
    """Build the full teacher-forced token sequence (1, T, n_cb+1) and the
    speaker position for a single (non-cached) ``forward`` capture.

    The prompt is the conditioning + text prefix; then one row per baseline audio
    frame: ``[codes[t, 0..8], text_vocab]``. The hidden at sequence position
    ``prompt_len-1+t`` is what predicts frame ``t`` (same as the cached loop)."""
    prompt, speaker_position = build_prompt(cfg, _TEXT, has_speaker=True)
    prompt = np.asarray(prompt)  # (1, P, 10)
    text_vocab = cfg.text_vocab
    n_cb = cfg.n_codebooks
    audio_rows = np.concatenate(
        [ref_codes.astype(np.int32),
         np.full((ref_codes.shape[0], 1), text_vocab, dtype=np.int32)],
        axis=1,
    )[None]  # (1, n_frames, n_cb+1)
    seq = np.concatenate([prompt, audio_rows], axis=1)  # (1, P+n_frames, 10)
    speaker_position = int(speaker_position) if speaker_position is not None else 0
    return mx.array(seq), speaker_position, prompt.shape[1], n_cb


def _capture_per_layer(model, ref_codes: np.ndarray) -> tuple[dict[int, np.ndarray], int]:
    """Run the non-cached full-causal forward over the teacher-forced sequence,
    capturing every layer's hidden state (T, dim). Returns ({layer: np.float32},
    prompt_len) so the cosine can be measured over the AUDIO-frame region only."""
    cfg = model.cfg
    lda = mx.array(np.load(_F / "lda.npy").astype(np.float32))
    seq, speaker_position, prompt_len, _n_cb = _teacher_forced_sequence(cfg, ref_codes)
    out = model.forward(
        seq, speaker_lda=lda, speaker_position=speaker_position,
        capture_layers=list(range(cfg.n_layers)),
    )
    captured = {i: np.array(h.astype(mx.float32)) for i, h in out.layers.items()}
    mx.eval(out.last)
    return captured, prompt_len


def _per_layer_cosine(tier_layers: dict, ref_layers: dict, prompt_len: int) -> dict:
    """MEDIAN per-token cosine per captured layer over the AUDIO-frame region
    (positions >= prompt_len), tier vs bf16 reference.

    Two corrections over a naive whole-tensor cosine (which masked the real
    signal in the first pass):
      * slice to the audio region — the prompt tokens are token-identical across
        tiers and barely move, so including them inflates cosine toward 1.0 and
        hides the audio-frame divergence that actually drives the decode;
      * median-per-token (not whole-tensor) — a handful of router knife-edge
        tokens can sit at low cosine without affecting the bulk, exactly as
        tests/test_trunk_parity gates the bf16 trunk. Median tracks the typical
        token; a genuine COLLAPSE (the old recurrent-EDA cliff) tanks every token
        and so the median too."""
    per_layer = {}
    for i in sorted(ref_layers):
        a = ref_layers[i][prompt_len:].astype(np.float64)   # (n_frames, dim)
        b = tier_layers[i][prompt_len:].astype(np.float64)
        num = (a * b).sum(-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
        per_layer[i] = float(np.median(num / den))
    min_layer = min(per_layer, key=per_layer.get)
    return {
        "per_layer_cosine": {str(k): v for k, v in per_layer.items()},
        "per_layer_min_cosine": per_layer[min_layer],
        "per_layer_min_cosine_layer": int(min_layer),
    }


def _teacher_forced_logits(model, ref_codes: np.ndarray) -> np.ndarray:
    """Per-frame last-position logits under teacher forcing (cached loop). Feeds
    the baseline codes as history and collects ``(n_frames, n_cb, vocab)`` raw
    (post-softcap) logits — the same logits the AR loop samples from."""
    cfg = model.cfg
    lda = mx.array(np.load(_F / "lda.npy").astype(np.float32))
    n_frames = ref_codes.shape[0]
    text_vocab = cfg.text_vocab
    n_cb = cfg.n_codebooks

    prompt, speaker_position = build_prompt(cfg, _TEXT, has_speaker=True)
    prompt = mx.array(prompt)
    speaker_position = int(speaker_position) if speaker_position is not None else 0

    caches = model.make_kv_caches(prompt.shape[1] + n_frames)
    logits = model.forward_cached(
        prompt, caches, speaker_lda=lda, speaker_position=speaker_position
    )
    mx.eval(logits, *[c.k for c in caches], *[c.v for c in caches])

    out = [np.array(logits[0].astype(mx.float32))]  # frame 0 (n_cb, vocab)
    for t in range(1, n_frames):
        prev_row = ref_codes[t - 1].astype(np.int32)
        next_row = mx.array(
            np.concatenate([prev_row, [text_vocab]]).reshape(1, 1, n_cb + 1)
        )
        logits = model.forward_cached(next_row, caches)
        mx.eval(logits, *[c.k for c in caches], *[c.v for c in caches])
        out.append(np.array(logits[0].astype(mx.float32)))
    return np.stack(out)  # (n_frames, n_cb, vocab)


def _logit_kl_and_top1(tier_logits: np.ndarray, ref_logits: np.ndarray) -> dict:
    """Mean per-frame KL(bf16 || tier) in nats + logit top-1 agreement.

    KL uses the bf16 distribution as the reference P (what the bf16 model would
    sample) and the tier as Q. top-1 is argmax(tier) == argmax(bf16) per
    (frame, codebook)."""
    p = ref_logits.astype(np.float64)
    q = tier_logits.astype(np.float64)
    # softmax over vocab (last axis), per (frame, codebook).
    p_log = p - p.max(-1, keepdims=True)
    p_log = p_log - np.log(np.exp(p_log).sum(-1, keepdims=True))
    q_log = q - q.max(-1, keepdims=True)
    q_log = q_log - np.log(np.exp(q_log).sum(-1, keepdims=True))
    p_prob = np.exp(p_log)
    kl = (p_prob * (p_log - q_log)).sum(-1)  # (frames, n_cb)
    top1 = (q.argmax(-1) == p.argmax(-1))    # (frames, n_cb)
    # Sampled-token (informational only): greedy argmax of raw tier logits == ref
    # codes per codebook (no rep-penalty; this is the chaos metric we DROPPED).
    return {
        "logit_kl_nats": float(kl.mean()),
        "logit_kl_max_nats": float(kl.max()),
        "logit_top1_agreement": float(top1.mean()),
    }


def _sampled_token_frac(model, ref_codes: np.ndarray) -> dict:
    """INFORMATIONAL ONLY (not a gate): teacher-forced sampled-token agreement —
    the discrete knife-edge metric docs/research/02 says to stop gating on. Kept
    as a print so the bisection narrative is auditable."""
    cfg = model.cfg
    lda = mx.array(np.load(_F / "lda.npy").astype(np.float32))
    n_frames = ref_codes.shape[0]
    text_vocab = cfg.text_vocab
    n_cb = cfg.n_codebooks
    opts = _greedy_opts()

    prompt, speaker_position = build_prompt(cfg, _TEXT, has_speaker=True)
    prompt = mx.array(prompt)
    speaker_position = int(speaker_position) if speaker_position is not None else 0

    caches = model.make_kv_caches(prompt.shape[1] + n_frames)
    logits = model.forward_cached(
        prompt, caches, speaker_lda=lda, speaker_position=speaker_position
    )
    mx.eval(logits, *[c.k for c in caches], *[c.v for c in caches])
    pred0 = sample_codes(logits, [], opts)
    frame0_exact = bool(np.array_equal(pred0, ref_codes[0]))

    agree = total = 0
    for t in range(1, n_frames):
        prev_row = ref_codes[t - 1].astype(np.int32)
        next_row = mx.array(
            np.concatenate([prev_row, [text_vocab]]).reshape(1, 1, n_cb + 1)
        )
        logits = model.forward_cached(next_row, caches)
        mx.eval(logits, *[c.k for c in caches], *[c.v for c in caches])
        history = [ref_codes[i] for i in range(t)]
        pred = sample_codes(logits, history, opts)
        agree += int((pred == ref_codes[t]).sum())
        total += n_cb
    return {
        "sampled_token_frac_informational": agree / total if total else 0.0,
        "frame0_exact": frame0_exact,
    }


def run_tier(tier: str, weights_dir: str) -> dict:
    _OUT.mkdir(parents=True, exist_ok=True)
    mem_after_import = phys_footprint_gb()

    model = Zonos2Model.from_pretrained(weights_dir)
    mx.eval(model.parameters())
    mem_loaded = phys_footprint_gb()

    # --- baseline establishment (bf16 only) -------------------------------
    if tier == "bf16":
        lda = np.load(_F / "lda.npy").astype(np.float32)
        # (1) baseline codes (greedy free-run) — the teacher-forcing target.
        res = synthesize(
            _TEXT, speaker_lda=lda, model=model, greedy=True, seed=0,
            max_new_tokens=1024, return_codes=True,
        )
        baseline = res.codes.astype(np.int64)
        np.save(_BASELINE_CODES, baseline)
        print(f"[{tier}] saved baseline codes {baseline.shape} -> {_BASELINE_CODES}")

        # (2) cache bf16 per-layer captures + per-frame logits for the tiers.
        ref_layers, _prompt_len = _capture_per_layer(model, baseline)
        np.savez(_BASELINE_LAYERS, **{str(k): v for k, v in ref_layers.items()})
        ref_logits = _teacher_forced_logits(model, baseline)
        np.save(_BASELINE_LOGITS, ref_logits)
        print(f"[{tier}] cached {len(ref_layers)} layer captures + logits "
              f"{ref_logits.shape}")
        mem_generate = phys_footprint_gb()

        # Decode a wav for the ear.
        from zonos2_mlx.dac import Dac44k
        dac = Dac44k.from_pretrained(str(_DAC_DIR))
        wav = dac.decode(baseline, pad_id=model.cfg.audio_pad_id, eos_frame=res.eos_frame)
        wav_np = np.asarray(wav).reshape(-1)
        import soundfile as sf
        wav_path = _OUT / f"quant_{tier}.wav"
        sf.write(str(wav_path), wav_np, 44100)
        summary = {
            "tier": tier,
            "per_layer_min_cosine": 1.0,
            "per_layer_min_cosine_layer": -1,
            "logit_kl_nats": 0.0,
            "logit_top1_agreement": 1.0,
            "free_running_n_frames": int(baseline.shape[0]),
            "audio_finite": bool(np.isfinite(wav_np).all()),
            "audio_silent": bool(float(np.abs(wav_np).max()) < 1e-4),
            "audio_abs_max": float(np.abs(wav_np).max()),
            "frame0_exact": True,
            "sampled_token_frac_informational": 1.0,
            "mem_loaded_gb": mem_loaded,
            "mem_generate_gb": mem_generate,
            "phys_footprint_peak_gb": max(mem_after_import, mem_loaded, mem_generate),
            "wav": str(wav_path),
        }
    else:
        ref_codes = np.load(_BASELINE_CODES).astype(np.int64)
        ref_layers = {int(k): v for k, v in np.load(_BASELINE_LAYERS).items()}
        ref_logits = np.load(_BASELINE_LOGITS)

        # (a) per-layer hidden-state cosine (audio region, median per token).
        tier_layers, prompt_len = _capture_per_layer(model, ref_codes)
        cos = _per_layer_cosine(tier_layers, ref_layers, prompt_len)
        # (b)/(c) teacher-forced logit-KL + top-1.
        tier_logits = _teacher_forced_logits(model, ref_codes)
        kl = _logit_kl_and_top1(tier_logits, ref_logits)
        # informational sampled-token frac.
        st = _sampled_token_frac(model, ref_codes)
        mem_tf = phys_footprint_gb()

        # (d) full-length free-run audio + decode for the ear.
        lda = np.load(_F / "lda.npy").astype(np.float32)
        wav_path = _OUT / f"quant_{tier}.wav"
        res = synthesize(
            _TEXT, speaker_lda=lda, model=model, greedy=True, seed=0,
            max_new_tokens=1024, dac_dir=str(_DAC_DIR), out_wav=str(wav_path),
        )
        mem_generate = phys_footprint_gb()
        wav_np = np.asarray(res.wav).reshape(-1) if res.wav is not None else np.zeros(1)

        summary = {
            "tier": tier,
            **cos,
            **kl,
            **st,
            "free_running_n_frames": int(res.codes.shape[0]),
            "audio_finite": bool(np.isfinite(wav_np).all()),
            "audio_silent": bool(float(np.abs(wav_np).max()) < 1e-4),
            "audio_abs_max": float(np.abs(wav_np).max()),
            "mem_loaded_gb": mem_loaded,
            "mem_teacher_forced_gb": mem_tf,
            "mem_generate_gb": mem_generate,
            "phys_footprint_peak_gb": max(
                mem_after_import, mem_loaded, mem_tf, mem_generate
            ),
            "wav": str(wav_path),
        }
        # Bisection print: the per-layer cosine ladder (the diagnostic tool).
        print(f"[{tier}] per-layer cosine through the trunk:")
        for k in sorted(cos["per_layer_cosine"], key=int):
            print(f"    L{int(k):2d}  {cos['per_layer_cosine'][k]:.4f}")
        print(f"[{tier}] min cosine {cos['per_layer_min_cosine']:.4f} at layer "
              f"{cos['per_layer_min_cosine_layer']}; logit-KL "
              f"{kl['logit_kl_nats']:.4f} nats; logit top-1 "
              f"{kl['logit_top1_agreement']:.4f}; sampled-token (info) "
              f"{st['sampled_token_frac_informational']:.4f}; free-run "
              f"{summary['free_running_n_frames']} frames")

    out_path = _OUT / f"{tier}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[{tier}] {json.dumps({k: v for k, v in summary.items() if k != 'per_layer_cosine'}, indent=2)}")
    return summary


def summarize() -> dict:
    merged = {}
    for tier in ("bf16", "int8", "int4", "int8lin-int4exp", "uniform-int4"):
        p = _OUT / f"{tier}.json"
        if p.exists():
            merged[tier] = json.loads(p.read_text())
    (_OUT / "summary.json").write_text(json.dumps(merged, indent=2))
    print(json.dumps(
        {t: {k: v for k, v in d.items() if k != "per_layer_cosine"}
         for t, d in merged.items()}, indent=2,
    ))
    return merged


def _main() -> None:
    ap = argparse.ArgumentParser(description="Per-tier Zonos2 quant eval (serial GPU).")
    ap.add_argument(
        "--tier", required=True,
        choices=("bf16", "int8", "int4", "int8lin-int4exp", "uniform-int4", "summarize"),
    )
    ap.add_argument("--weights", default=None, help="weights dir for this tier")
    args = ap.parse_args()

    if args.tier == "summarize":
        summarize()
        return

    weights = args.weights or {
        "bf16": str(_R / "weights/zonos2-bf16"),
        "int8": str(_R / "weights/zonos2-int8"),
        "int4": str(_R / "weights/zonos2-int4"),
        "int8lin-int4exp": str(_R / "weights/_diag-int8lin-int4exp"),
        "uniform-int4": str(_R / "weights/_diag-uniform-int4"),
    }[args.tier]
    run_tier(args.tier, weights)


if __name__ == "__main__":
    _main()
