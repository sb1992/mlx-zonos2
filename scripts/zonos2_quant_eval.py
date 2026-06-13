"""Per-tier quantization eval for the Zonos2 trunk (GPU, SERIAL).

Establishes the bf16-MLX baseline ONCE, then evaluates int8 / int4 against THAT
baseline (NOT the cross-framework MPS oracle — same-framework isolates quant
drift from the bf16 knife-edge documented in T6 / docs/research/00).

Per tier it reports:
  * teacher-forced per-codebook agreement vs the bf16-MLX baseline codes
    (feed the baseline codes as history, compare the tier's greedy next-token),
  * free-running per-codebook agreement (informational),
  * phys_footprint_peak GB (loaded + generate), via proc_pid_rusage,
  * a decoded wav for the user's ear A/B.

Run one tier per process so only one trunk lives in memory at a time:

    python scripts/zonos2_quant_eval.py --tier bf16
    python scripts/zonos2_quant_eval.py --tier int8 --weights weights/zonos2-int8
    python scripts/zonos2_quant_eval.py --tier int4 --weights weights/zonos2-int4

Then ``--tier summarize`` merges the per-tier JSONs into outputs/quant_eval/
summary.json (what tests/test_quant.py asserts against).

Pure MLX (+ numpy host-side sampling). No torch.
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
# The DAC codec is tier-independent (it only decodes audio codes), so every
# tier reuses the bf16 dir's dac_44khz rather than duplicating it per tier.
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


def _teacher_forced_agreement(model, ref_codes: np.ndarray) -> dict:
    """Per-codebook agreement: feed the baseline codes as history, compare the
    model's greedy next-token at each frame to the baseline.

    Mirrors tests/test_e2e_parity.py: prefill -> frame-0 argmax exact check ->
    teacher-forced loop over t = 1..n-1.
    """
    cfg = model.cfg
    lda = mx.array(np.load(_F / "lda.npy").astype(np.float32))
    n_frames = ref_codes.shape[0]
    text_vocab = cfg.text_vocab
    n_cb = cfg.n_codebooks

    prompt, speaker_position = build_prompt(cfg, _TEXT, has_speaker=True)
    prompt = mx.array(prompt)
    speaker_position = int(speaker_position) if speaker_position is not None else 0

    opts = _greedy_opts()
    caches = model.make_kv_caches(prompt.shape[1] + n_frames)
    logits = model.forward_cached(
        prompt, caches, speaker_lda=lda, speaker_position=speaker_position
    )
    mx.eval(logits, *[c.k for c in caches], *[c.v for c in caches])

    pred0 = sample_codes(logits, [], opts)
    frame0_exact = bool(np.array_equal(pred0, ref_codes[0]))

    agree = 0
    total = 0
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

    agreement = agree / total if total else 0.0
    return {
        "teacher_forced_agreement": agreement,
        "frame0_exact": frame0_exact,
        "n_frames": int(n_frames),
        "agree": int(agree),
        "total": int(total),
    }


def _free_running_agreement(codes: np.ndarray, ref_codes: np.ndarray) -> float:
    """Per-codebook agreement of two free-run trajectories over the common
    prefix (informational; greedy compounding makes this drift)."""
    n = min(codes.shape[0], ref_codes.shape[0])
    if n == 0:
        return 0.0
    a = codes[:n]
    b = ref_codes[:n]
    return float((a == b).mean())


def run_tier(tier: str, weights_dir: str) -> dict:
    _OUT.mkdir(parents=True, exist_ok=True)
    mem_after_import = phys_footprint_gb()

    # from_pretrained auto-detects a quant_config.json sidecar in weights_dir;
    # bf16 dirs have none, so the path is transparent per tier.
    model = Zonos2Model.from_pretrained(weights_dir)
    mx.eval(model.parameters())
    mem_loaded = phys_footprint_gb()

    # --- baseline establishment (bf16 only) -------------------------------
    if tier == "bf16":
        lda = np.load(_F / "lda.npy").astype(np.float32)
        res = synthesize(
            _TEXT, speaker_lda=lda, model=model, greedy=True, seed=0,
            max_new_tokens=1024, return_codes=True,
        )
        baseline = res.codes.astype(np.int64)
        np.save(_BASELINE_CODES, baseline)
        print(f"[{tier}] saved baseline codes {baseline.shape} -> {_BASELINE_CODES}")
        mem_generate = phys_footprint_gb()
        # Decode a wav for the ear.
        from zonos2_mlx.dac import Dac44k
        dac = Dac44k.from_pretrained(str(_DAC_DIR))
        wav = dac.decode(baseline, pad_id=model.cfg.audio_pad_id, eos_frame=res.eos_frame)
        import soundfile as sf
        wav_path = _OUT / f"quant_{tier}.wav"
        sf.write(str(wav_path), np.asarray(wav).reshape(-1), 44100)
        summary = {
            "tier": tier,
            "teacher_forced_agreement": 1.0,
            "frame0_exact": True,
            "free_running_agreement": 1.0,
            "n_frames": int(baseline.shape[0]),
            "mem_loaded_gb": mem_loaded,
            "mem_generate_gb": mem_generate,
            "phys_footprint_peak_gb": max(mem_after_import, mem_loaded, mem_generate),
            "wav": str(wav_path),
        }
    else:
        ref_codes = np.load(_BASELINE_CODES).astype(np.int64)
        tf = _teacher_forced_agreement(model, ref_codes)
        mem_tf = phys_footprint_gb()

        # Free-run this tier + decode for the ear.
        lda = np.load(_F / "lda.npy").astype(np.float32)
        wav_path = _OUT / f"quant_{tier}.wav"
        res = synthesize(
            _TEXT, speaker_lda=lda, model=model, greedy=True, seed=0,
            max_new_tokens=1024, dac_dir=str(_DAC_DIR),
            out_wav=str(wav_path),
        )
        mem_generate = phys_footprint_gb()
        free_agree = _free_running_agreement(res.codes.astype(np.int64), ref_codes)

        summary = {
            "tier": tier,
            **tf,
            "free_running_agreement": free_agree,
            "free_running_n_frames": int(res.codes.shape[0]),
            "mem_loaded_gb": mem_loaded,
            "mem_teacher_forced_gb": mem_tf,
            "mem_generate_gb": mem_generate,
            "phys_footprint_peak_gb": max(
                mem_after_import, mem_loaded, mem_tf, mem_generate
            ),
            "wav": str(wav_path),
        }

    out_path = _OUT / f"{tier}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[{tier}] {json.dumps(summary, indent=2)}")
    return summary


def summarize() -> dict:
    merged = {}
    for tier in ("bf16", "int8", "int4"):
        p = _OUT / f"{tier}.json"
        if p.exists():
            merged[tier] = json.loads(p.read_text())
    (_OUT / "summary.json").write_text(json.dumps(merged, indent=2))
    print(json.dumps(merged, indent=2))
    return merged


def _main() -> None:
    ap = argparse.ArgumentParser(description="Per-tier Zonos2 quant eval (serial GPU).")
    ap.add_argument(
        "--tier", required=True, choices=("bf16", "int8", "int4", "summarize")
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
    }[args.tier]
    run_tier(args.tier, weights)


if __name__ == "__main__":
    _main()
