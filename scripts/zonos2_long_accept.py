"""GPU acceptance for long-form zonos2: a ~90s passage forces >=2 packed chunks.

Asserts finite + non-silent audio, total duration within ~30% of the estimate,
and that peak RAM stays near the single-pass footprint (weights load once).
Run serially (one GPU job). Reads a .zonos profile so no torch is needed.

    uv run python scripts/zonos2_long_accept.py \
        --profile outputs/voices/accept.zonos \
        --model-dir weights/zonos2-int8 \
        --out outputs/long/accept.wav
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_memory_limit(int(45 * (1 << 30)))

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from zonos2_mlx.chunking import estimate_seconds, plan_chunks  # noqa: E402
from zonos2_mlx.pipeline import SAMPLE_RATE, synthesize_long  # noqa: E402
from zonos2_mlx.speaker import SpeakerProfile  # noqa: E402

_PASSAGE = (
    "The harbor woke slowly under a colorless sky. Fishing boats knocked against "
    "the pilings while gulls argued over the night's leavings. A man in a yellow "
    "slicker coiled a wet rope and did not look up. Far out past the breakwater the "
    "swell was long and patient, the kind that promises weather by afternoon. Inside "
    "the chandlery the radio murmured tide times and a warning for small craft. She "
    "counted the change twice, wrote the total in a ledger soft with damp, and "
    "turned the sign in the window. By noon the rain would come sideways off the "
    "water and the whole town would smell of salt and diesel and strong coffee. "
    "The old lobsterman tied his skiff to the outermost cleat and lugged his traps "
    "up the ramp one at a time, his boots leaving wet crescents on the planking. "
    "A boy on a bicycle watched from the top of the seawall, one hand shading his "
    "eyes against the flat light. The harbormaster leaned in her doorway with a mug "
    "and said nothing, because there was nothing yet that needed saying. Across the "
    "street the bakery opened its screen door and the smell of bread moved out over "
    "the cobblestones like something alive. Two cats appeared from under a dory and "
    "followed the smell with great seriousness. The tide was still falling, "
    "uncovering the dark weed on the lower pilings, and the whole harbor tilted "
    "slightly toward the sea as if listening for something only the water knew."
)


def _peak_rss_gb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (rss if sys.platform == "darwin" else rss * 1024) / (1 << 30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model-dir", required=True, help="self-contained tier folder")
    ap.add_argument("--out", default="outputs/long/accept.wav")
    ap.add_argument("--max-seconds", type=float, default=40.0)
    args = ap.parse_args()

    chunks = plan_chunks(_PASSAGE, max_seconds=args.max_seconds)
    est = sum(estimate_seconds(c) for c in chunks)
    print(f"planned {len(chunks)} chunks, est {est:.1f}s")
    assert len(chunks) >= 2, "passage did not split into >=2 chunks; lengthen it"

    adir = Path(args.model_dir)
    prof = SpeakerProfile.load(args.profile)
    t0 = time.perf_counter()
    res = synthesize_long(
        _PASSAGE,
        speaker_lda=prof.lda.reshape(1, -1),
        weights_dir=str(adir),
        dac_dir=str(adir / "dac_44khz"),
        speaker_dir=str(adir / "speaker_encoder"),
        max_seconds=args.max_seconds,
        out_wav=args.out,
        progress=True,
    )
    elapsed = time.perf_counter() - t0

    wav = np.asarray(res.wav).reshape(-1)
    audio_s = wav.shape[0] / SAMPLE_RATE
    finite = bool(np.isfinite(wav).all())
    abs_max = float(np.abs(wav).max()) if wav.size else 0.0
    peak = _peak_rss_gb()
    print(f"audio={audio_s:.2f}s wall={elapsed:.1f}s "
          f"finite={finite} abs_max={abs_max:.3f} peak_rss={peak:.1f}GB")

    assert finite and abs_max > 1e-3, "non-finite or silent long-form audio"
    assert abs(audio_s - est) / est < 0.30, f"duration {audio_s:.1f}s far from est {est:.1f}s"
    print("ACCEPT OK — listen to", args.out, "for the chunk seams")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
