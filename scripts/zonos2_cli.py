"""Drop-in CLI for the pure-MLX ZONOS2 8B-MoE TTS engine (text + voice -> wav).

Mirrors the dots/miso flag style. The engine logic lives in
``zonos2_mlx.pipeline.synthesize``; this wrapper just resolves the speaker
source (``--ref`` enrols, ``--profile`` loads a cached vector), picks a quant
tier, and writes the wav.

Examples
--------
    # enrol a reference voice on the fly + synthesize at int8
    uv run python scripts/zonos2_cli.py \
        --text "The quick brown fox jumps over the lazy dog." \
        --ref outputs/fixtures/ref.wav \
        --out outputs/cli/fox_int8.wav \
        --quant int8

    # reuse a pre-enrolled .zonos profile at the int4 tier
    uv run python scripts/zonos2_cli.py \
        --text "Hello there." \
        --profile outputs/voices/myvoice.zonos \
        --out outputs/cli/hello_int4.wav \
        --quant int4

Speaker source is exactly one of ``--ref`` (raw audio; enrols via the
torchaudio mel -> MLX ECAPA -> LDA path, identical to scripts/zonos2_enroll.py)
or ``--profile`` (a cached SpeakerProfile .zonos/.npz carrying the 1024-d LDA
vector). torchaudio is used ONLY for the ref-enrol mel here — the engine stays
pure-MLX.

Quant tiers map to weight dirs:
    bf16 -> weights/zonos2-bf16        (~44 GB peak; 64 GB Macs)
    int8 -> weights/zonos2-int8        (~13 GB peak)
    int4 -> weights/zonos2-int4        (~10.6 GB peak; 16 GB Macs)

The quantized weight dirs ship only the trunk safetensors; the tier-independent
DAC codec + speaker encoder always come from weights/zonos2-bf16.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_memory_limit(int(45 * (1 << 30)))

# Make `zonos2_mlx` importable when run as a plain script from the repo root.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from zonos2_mlx.pipeline import SAMPLE_RATE, synthesize, synthesize_long  # noqa: E402
from zonos2_mlx.speaker import SpeakerProfile  # noqa: E402

# Quant tier -> trunk weights dir.
_TIER_DIRS = {
    "bf16": "weights/zonos2-bf16",
    "int8": "weights/zonos2-int8",
    "int4": "weights/zonos2-int4",
}
# Tier-independent assets always live in the bf16 dir.
_BF16_DIR = "weights/zonos2-bf16"


def _enrol_ref(ref_path: str, speaker_dir: str, dit_path: str) -> np.ndarray:
    """Enrol a raw reference clip into a 1024-d LDA vector.

    Reuses scripts/zonos2_enroll.py's exact mel -> ECAPA -> LDA path (torchaudio
    mel, then the pure-MLX EcapaTDNN + SpeakerLDA).
    """
    from zonos2_mlx.speaker import EcapaTDNN, SpeakerLDA

    from zonos2_enroll import _prepare_mel  # the sanctioned scripts/ torch mel

    mel_np, duration = _prepare_mel(ref_path)
    mel = mx.array(mel_np)
    enc = EcapaTDNN.from_pretrained(speaker_dir)
    lda = SpeakerLDA.from_dit(dit_path)
    emb = enc(mel)            # (1, 2048)
    vec = lda(emb)            # (1, 1024)
    mx.eval(vec)
    print(f"  enrolled {ref_path} ({duration:.2f}s) -> LDA {tuple(vec.shape)}")
    return np.array(vec.astype(mx.float32)).reshape(1, -1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pure-MLX ZONOS2 TTS — text + voice -> wav.",
    )
    ap.add_argument("--text", required=True, help="Text to synthesize.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ref", help="Reference audio clip to enrol the speaker.")
    src.add_argument("--profile", help="Cached .zonos/.npz SpeakerProfile.")
    ap.add_argument("--out", required=True, help="Output wav path.")
    ap.add_argument(
        "--quant",
        choices=tuple(_TIER_DIRS),
        default="int8",
        help="Quant tier (bf16/int8/int4). Default: int8. Ignored if --model-dir is set.",
    )
    ap.add_argument(
        "--model-dir",
        default=None,
        help="A self-contained MLX weights folder — e.g. one tier of an `hf "
        "download shraey/zonos2-mlx` (trunk safetensors + dac_44khz/ + "
        "speaker_encoder/). Overrides --quant; point it at your download.",
    )
    ap.add_argument(
        "--speaking-rate",
        type=int,
        default=-1,
        help="Speaking-rate bucket 0..7, or -1 for unset (default).",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--accurate-mode",
        dest="accurate_mode",
        action="store_true",
        help="Accurate mode (default).",
    )
    mode.add_argument(
        "--expressive",
        dest="accurate_mode",
        action="store_false",
        help="Expressive mode (disables the accurate-mode token).",
    )
    ap.set_defaults(accurate_mode=True)
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed (only affects --sample; greedy is deterministic).")
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="AR decode cap (frames). Default 1024.",
    )
    ap.add_argument("--long", dest="long_form", action="store_true",
                    help="Long-form: split on sentence boundaries, pack into chunks, concatenate.")
    ap.add_argument("--max-seconds", type=float, default=40.0,
                    help="Long-form per-chunk target in estimated audio seconds (default 40).")
    ap.add_argument("--gap-ms", type=int, default=80,
                    help="Long-form silence between chunks, ms (default 80).")
    ap.add_argument("--chars-per-sec", type=float, default=None,
                    help="Long-form: override the script-aware duration estimate (chars/sec).")
    ap.add_argument("--language", default=None,
                    help="Optional language code (e.g. ZH/HI) to pick chunking rates.")
    ap.add_argument(
        "--greedy",
        dest="greedy",
        action="store_true",
        default=True,
        help="Greedy (temperature 0) decode — the default, parity path.",
    )
    ap.add_argument(
        "--sample",
        dest="greedy",
        action="store_false",
        help="Stochastic sampling (the oracle SamplingOptions defaults).",
    )
    args = ap.parse_args()

    if args.model_dir:
        # Self-contained tier folder (an HF download): trunk + dac_44khz +
        # speaker_encoder all live here. The LDA tensors are in the trunk
        # (from_dit accepts the quantized key too).
        adir = Path(args.model_dir).expanduser()
        weights_dir = str(adir)
        assets_dir = adir
    else:
        # Dev layout: weights/zonos2-<tier> trunk; shared DAC + speaker encoder
        # live in the bf16 dir (the quant dirs ship only the trunk).
        weights_dir = str(_REPO / _TIER_DIRS[args.quant])
        assets_dir = _REPO / _BF16_DIR
    dac_dir = str(assets_dir / "dac_44khz")
    speaker_dir = str(assets_dir / "speaker_encoder")
    _trunk = next(iter(sorted(Path(weights_dir).glob("*.safetensors"))), None)
    if _trunk is None:
        ap.error(f"no trunk *.safetensors found in {weights_dir!r} — check --model-dir / --quant.")
    dit_path = str(_trunk)

    # --- resolve the speaker LDA vector ---
    if args.ref is not None:
        speaker_lda = _enrol_ref(args.ref, speaker_dir, dit_path)
    else:
        prof = SpeakerProfile.load(args.profile)
        speaker_lda = prof.lda.reshape(1, -1)
        print(f"  loaded profile {args.profile} (compat={prof.compat})")

    tier_label = f"model-dir={args.model_dir}" if args.model_dir else f"quant={args.quant}"
    print(
        f"[zonos2] {tier_label} weights={weights_dir} "
        f"accurate_mode={args.accurate_mode} greedy={args.greedy}"
    )

    t0 = time.perf_counter()
    if args.long_form:
        result = synthesize_long(
            args.text,
            speaker_lda=speaker_lda,
            weights_dir=weights_dir,
            dac_dir=dac_dir,
            speaker_dir=speaker_dir,
            greedy=args.greedy,
            seed=args.seed,
            max_seconds=args.max_seconds,
            gap_ms=args.gap_ms,
            chars_per_sec=args.chars_per_sec,
            language=args.language,
            speaking_rate_bucket=args.speaking_rate,
            accurate_mode=args.accurate_mode,
            out_wav=args.out,
            progress=True,
        )
    else:
        result = synthesize(
            args.text,
            speaker_lda=speaker_lda,
            weights_dir=weights_dir,
            dac_dir=dac_dir,
            speaker_dir=speaker_dir,
            greedy=args.greedy,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            speaking_rate_bucket=args.speaking_rate,
            accurate_mode=args.accurate_mode,
            out_wav=args.out,
        )
    elapsed = time.perf_counter() - t0

    wav = np.asarray(result.wav).reshape(-1)
    n_samples = wav.shape[0]
    audio_secs = n_samples / SAMPLE_RATE
    abs_max = float(np.abs(wav).max()) if n_samples else 0.0
    finite = bool(np.isfinite(wav).all())
    per_audio_sec = elapsed / audio_secs if audio_secs > 0 else float("nan")

    print(f"[zonos2] wrote {args.out}")
    print(
        f"  frames={result.codes.shape[0]}  samples={n_samples}  "
        f"audio={audio_secs:.2f}s  eos_frame={result.eos_frame}"
    )
    print(f"  finite={finite}  abs_max={abs_max:.4f}  silent={abs_max < 1e-4}")
    print(
        f"  wall={elapsed:.1f}s  ->  {per_audio_sec:.2f} s/audio-sec "
        f"({args.model_dir or args.quant}, M-series, serial)"
    )

    if not finite or abs_max < 1e-4:
        print("ERROR: non-finite or silent audio.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
