"""Enroll a reference voice into a Zonos2 speaker profile (``.zonos``).

Computes the log-mel front-end EXACTLY as
``Zonos2_TTS-ComfyUI/runtime.py::Zonos2SpeakerEncoder`` does (torchaudio is the
sanctioned mel path here — the engine stays torch-free), then runs the pure-MLX
``EcapaTDNN`` + ``SpeakerLDA`` and caches the 1024-d LDA vector.

Usage:
    uv run python scripts/zonos2_enroll.py \
        --ref outputs/fixtures/ref.wav \
        --out outputs/voices/myvoice.zonos

NOTE: a pure-MLX mel front-end (MLX STFT/mel) is an explicitly-deferred
follow-up — out of scope for T4. torchaudio here matches the oracle exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_memory_limit(int(45 * (1 << 30)))

# Make `zonos2_mlx` importable when run as a plain script from the repo root.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from zonos2_mlx.speaker import EcapaTDNN, SpeakerLDA, SpeakerProfile  # noqa: E402

# Constants mirrored from runtime.py.
SPEAKER_SAMPLE_RATE = 24_000
MAX_REFERENCE_SECONDS = 60.0
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
# Compat tag identifying which engine produced this vector.
COMPAT = "zonos2-bf16/ecapa-tdnn+lda/v1"


def _prepare_mel(ref_path: str) -> np.ndarray:
    """Load audio and compute the log-mel exactly as runtime.py does.

    Returns a numpy (1, T, 128) float32 mel.
    """
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    import torchaudio

    # Load exactly as the oracle does: soundfile -> (channels, samples).
    wav, sample_rate = sf.read(ref_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(wav.T).contiguous()  # (channels, samples)

    # --- _prepare_audio ---
    audio = waveform
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 2:
        audio = audio.mean(dim=0, keepdim=True)
    elif audio.ndim == 1:
        audio = audio.unsqueeze(0)

    source_sample_rate = int(sample_rate)
    duration_seconds = audio.shape[-1] / source_sample_rate
    if duration_seconds > MAX_REFERENCE_SECONDS:
        maximum_samples = round(MAX_REFERENCE_SECONDS * source_sample_rate)
        audio = audio[..., :maximum_samples]

    audio = audio.to(dtype=torch.float32)
    if source_sample_rate != SPEAKER_SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(source_sample_rate, SPEAKER_SAMPLE_RATE)(audio)
    if audio.shape[-1] < WIN_LENGTH:
        audio = F.pad(audio, (0, WIN_LENGTH - audio.shape[-1]))

    # --- forward ---
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SPEAKER_SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        f_min=0.0,
        f_max=12_000.0,
        n_mels=128,
        power=1.0,
        center=False,
        norm="slaney",
        mel_scale="slaney",
    )
    padding = (WIN_LENGTH - HOP_LENGTH) // 2  # 384
    audio = F.pad(audio.unsqueeze(1), (padding, padding), mode="reflect")
    audio = audio.squeeze(1)
    mel = mel_transform(audio)
    mel = torch.log(torch.clamp(mel, min=1e-5)).transpose(1, 2)  # (1, T, 128)

    return mel.to(dtype=torch.float32).cpu().numpy(), duration_seconds


def main() -> int:
    ap = argparse.ArgumentParser(description="Enroll a reference voice into a .zonos profile.")
    ap.add_argument("--ref", required=True, help="Reference audio (any ffmpeg-readable file).")
    ap.add_argument("--out", required=True, help="Output .zonos profile path.")
    ap.add_argument(
        "--speaker-dir",
        default="weights/zonos2-bf16/speaker_encoder",
        help="ECAPA-TDNN speaker encoder directory.",
    )
    ap.add_argument(
        "--dit",
        default="weights/zonos2-bf16/zonos2-bf16.safetensors",
        help="DiT safetensors (holds speaker_lda_projection).",
    )
    args = ap.parse_args()

    mel_np, duration = _prepare_mel(args.ref)
    mel = mx.array(mel_np)

    enc = EcapaTDNN.from_pretrained(args.speaker_dir)
    lda = SpeakerLDA.from_dit(args.dit)

    emb = enc(mel)              # (1, 2048)
    vec = lda(emb)             # (1, 1024)
    mx.eval(vec)
    vec_np = np.array(vec.astype(mx.float32)).reshape(-1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    SpeakerProfile.save(str(out_path), vec_np, COMPAT)

    print(f"Enrolled {args.ref} ({duration:.2f}s) -> {out_path}")
    print(f"  mel shape {tuple(mel.shape)}  |  LDA vector {vec_np.shape}  |  compat={COMPAT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
