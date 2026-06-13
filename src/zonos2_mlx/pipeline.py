"""End-to-end MLX Zonos2 TTS pipeline: text + speaker -> waveform.

Ties together T1-T6:
  build_prompt (textnorm)  ->  KV-cached AR decode (generate)  ->  DAC decode.

Speaker conditioning is resolved from one of:
  * ``speaker_lda``  -- a ready (1, speaker_lda_dim) LDA vector (the gate path);
  * ``profile``      -- a saved SpeakerProfile (.npz) carrying the LDA vector;
  * ``ref``          -- a precomputed log-mel array to enrol via ECAPA + LDA.

NO torch. mlx + numpy only. The trunk is loaded once per call (8B/~15GB bf16);
callers wanting reuse can pass a preloaded ``model``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_memory_limit(int(45 * (1 << 30)))

from .dac import Dac44k  # noqa: E402
from .generate import SamplingOptions, generate_audio_codes  # noqa: E402
from .model import Zonos2Model  # noqa: E402
from .speaker import SpeakerProfile  # noqa: E402
from .textnorm import build_prompt, normalize_text  # noqa: E402

SAMPLE_RATE = 44100


@dataclass
class SynthesisResult:
    wav: np.ndarray | None        # (1, samples) float32 @ 44100, or None if return_codes
    sample_rate: int
    codes: np.ndarray             # (frames, n_codebooks) int64 RAW (delayed) codes
    eos_frame: int | None
    prompt_len: int


def _resolve_speaker_lda(speaker_lda, profile, ref, speaker_enc_dir, dit_for_lda) -> mx.array:
    """Return the (1, speaker_lda_dim) LDA inject vector as a float32 mx.array.

    Exactly one of ``speaker_lda`` / ``profile`` / ``ref`` must be given. The
    ``ref`` enrolment paths (``speaker_enc_dir``, ``dit_for_lda``) are resolved
    by the caller from the weights/speaker dirs, so it works with a preloaded
    model too.
    """
    provided = [n for n, v in (("speaker_lda", speaker_lda), ("profile", profile),
                               ("ref", ref)) if v is not None]
    if len(provided) > 1:
        raise ValueError(f"synthesize() takes exactly one speaker source; got {provided}.")
    if not provided:
        raise ValueError("synthesize() needs one of speaker_lda=, profile=, or ref=.")

    if speaker_lda is not None:
        arr = np.asarray(speaker_lda, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None]
        return mx.array(arr)

    if profile is not None:
        prof = profile if isinstance(profile, SpeakerProfile) else SpeakerProfile.load(profile)
        return mx.array(prof.lda.reshape(1, -1).astype(np.float32))

    # ref: enrol from a PRECOMPUTED log-mel array: ECAPA -> DiT LDA projection.
    # Raw-wav enrolment (a mel frontend) is out of scope for this module.
    from .speaker import EcapaTDNN, SpeakerLDA  # local import; GPU-only

    mel = np.asarray(ref, dtype=np.float32)
    if mel.ndim != 3:
        raise NotImplementedError(
            "pipeline.synthesize(ref=...) accepts a precomputed log-mel array "
            "(B, T, mel_dim); raw-waveform enrolment has no frontend in this "
            "module. Pass speaker_lda=... or profile=... instead."
        )
    if speaker_enc_dir is None or dit_for_lda is None:
        raise ValueError(
            "ref enrolment needs the speaker encoder dir + a trunk safetensors. "
            "Pass speaker_dir=/weights_dir= (or use a self-contained --model-dir) "
            "so they resolve — required when reusing a preloaded model."
        )
    ecapa = EcapaTDNN.from_pretrained(speaker_enc_dir)
    lda = SpeakerLDA.from_dit(dit_for_lda)
    emb = ecapa(mx.array(mel))            # (B, 2048)
    vec = lda(emb)                        # (B, 1024)
    mx.eval(vec)
    return mx.array(np.asarray(vec, dtype=np.float32)[:1])


def synthesize(
    text: str,
    *,
    speaker_lda=None,
    profile=None,
    ref=None,
    model: Zonos2Model | None = None,
    weights_dir: str = "weights/zonos2-bf16",
    dac_dir: str | None = None,
    speaker_dir: str | None = None,
    greedy: bool = True,
    seed: int = 0,
    max_new_tokens: int = 1024,
    return_codes: bool = False,
    normalize: bool = False,
    speaking_rate_bucket: int = -1,
    quality_buckets=None,
    clean_speaker_background: bool = False,
    accurate_mode: bool = True,
    out_wav: str | None = None,
    **knobs,
) -> SynthesisResult:
    """Synthesize speech from ``text`` conditioned on a speaker.

    Args mirror the oracle defaults. ``greedy=True`` forces temperature 0
    (the parity path). Extra sampling ``**knobs`` (top_k, top_p, min_p,
    repetition_*) override the SamplingOptions defaults.
    """
    wdir = Path(weights_dir)
    if model is None:
        model = Zonos2Model.from_pretrained(str(wdir))

    cfg = model.cfg

    # Resolve the ``ref=`` enrolment assets from the dirs (NOT off the model), so
    # enrolment works whether or not the model was loaded in this call. The LDA
    # tensors live in the trunk under either key (from_dit is key-robust), so the
    # selected tier's safetensors works even quantized; the ECAPA speaker encoder
    # is tier-independent (explicit ``speaker_dir``, else beside the trunk).
    _st = sorted(wdir.glob("*.safetensors"))
    dit_for_lda = str(_st[0]) if _st else None
    _spk = Path(speaker_dir) if speaker_dir else (wdir / "speaker_encoder")
    speaker_enc_dir = str(_spk) if _spk.exists() else None

    spk = _resolve_speaker_lda(speaker_lda, profile, ref, speaker_enc_dir, dit_for_lda)
    has_speaker = True

    norm_text = normalize_text(text, enable=normalize)
    prompt, speaker_position = build_prompt(
        cfg,
        norm_text,
        speaking_rate_bucket=speaking_rate_bucket,
        quality_buckets=quality_buckets,
        has_speaker=has_speaker,
        clean_speaker_background=clean_speaker_background,
        accurate_mode=accurate_mode,
    )

    opt_kwargs = dict(max_new_tokens=int(max_new_tokens), seed=int(seed))
    if greedy:
        opt_kwargs["temperature"] = 0.0
    for key in (
        "temperature", "top_k", "top_p", "min_p",
        "repetition_window", "repetition_penalty", "repetition_codebooks",
    ):
        if key in knobs:
            opt_kwargs[key] = knobs[key]
    options = SamplingOptions(**opt_kwargs)

    codes, eos_frame = generate_audio_codes(
        model,
        mx.array(prompt),
        spk,
        int(speaker_position) if speaker_position is not None else 0,
        options,
    )

    result = SynthesisResult(
        wav=None,
        sample_rate=SAMPLE_RATE,
        codes=codes,
        eos_frame=eos_frame,
        prompt_len=int(prompt.shape[1]),
    )
    if return_codes:
        return result

    dac_path = dac_dir if dac_dir is not None else str(wdir / "dac_44khz")
    dac = Dac44k.from_pretrained(dac_path)
    wav = dac.decode(codes, pad_id=cfg.audio_pad_id, eos_frame=eos_frame)  # (1, samples)
    result.wav = wav

    if out_wav is not None:
        import soundfile as sf

        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_wav, np.asarray(wav).reshape(-1), SAMPLE_RATE)

    return result
