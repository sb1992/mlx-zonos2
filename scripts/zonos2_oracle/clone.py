"""Oracle voice-clone utility — clone an arbitrary reference with natural sampling.

Reuses the plain-torch ZONOS2 bundle from dump_fixtures (MPS bf16, CPU fallback) and
the fork's high-level `generate_zonos2_audio`. Does NOT touch the golden fixtures.

Usage:
  uv run --extra oracle python scripts/zonos2_oracle/clone.py \
      --ref /path/to/voice.mp3 --text "..." --out outputs/clone/out.wav
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import zonos2_ref.native as native       # noqa: E402
import zonos2_ref.runtime as runtime      # noqa: E402
from dump_fixtures import _build_bundle, _pick_device  # noqa: E402

SR_REF = 24000  # speaker encoder target; resampling handled inside, but feed clean mono


def _load_ref(path: str, work: Path) -> dict:
    """Load reference audio as a {waveform, sample_rate} dict (ffmpeg -> wav -> tensor)."""
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "ref_in.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR_REF), str(wav)],
        check=True,
    )
    audio, sr = sf.read(str(wav), dtype="float32")
    return {"waveform": torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0), "sample_rate": sr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference audio (any format ffmpeg reads)")
    ap.add_argument("--text", default="Hey, this is a quick test of my own voice, "
                    "cloned and running entirely on device. Pretty wild, right?")
    ap.add_argument("--out", default=str(_HERE.parents[1] / "outputs/clone/newvoice_clone.wav"))
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device, dtype = _pick_device()
    print(f"Device: {device}  dtype: {dtype}", flush=True)
    t0 = time.time()
    try:
        bundle = _build_bundle(device, dtype)
    except Exception as exc:  # noqa: BLE001
        if device.type == "mps":
            print(f"MPS bundle failed ({exc}); falling back to CPU fp32.", flush=True)
            device, dtype = torch.device("cpu"), torch.float32
            bundle = _build_bundle(device, dtype)
        else:
            raise
    print(f"Bundle loaded in {time.time()-t0:.1f}s.", flush=True)

    ref = _load_ref(args.ref, out.parent)
    dur = ref["waveform"].shape[-1] / ref["sample_rate"]
    print(f"Reference: {args.ref}  {dur:.1f}s @ {ref['sample_rate']}Hz", flush=True)

    options = native.SamplingOptions(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed
    )
    quality_buckets = [None] * len(bundle.config.quality_features)

    t1 = time.time()
    result = runtime.generate_zonos2_audio(
        bundle,
        text=args.text,
        options=options,
        speaking_rate_bucket=-1,
        quality_buckets=quality_buckets,
        reference_audio=ref,
        clean_speaker_background=False,
        accurate_mode=True,
    )
    wav = result["waveform"]
    if hasattr(wav, "detach"):
        wav = wav.detach().float().cpu().numpy()
    wav = np.asarray(wav).squeeze()
    sf.write(str(out), wav, int(result["sample_rate"]))
    print(f"Generated {wav.shape[-1]/result['sample_rate']:.2f}s in {time.time()-t1:.1f}s "
          f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
