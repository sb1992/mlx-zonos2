"""Dump golden parity fixtures from the plain-PyTorch ZONOS2 oracle.

This drives the vendored `zonos2_ref` (a ComfyUI-free copy of the
Zonos2_TTS-ComfyUI reference) end-to-end on a FIXED text + reference + seed,
and saves per-component fixtures the MLX port is gated against.

Strategy for tractability on Apple Silicon (M5 Max, MPS):
  * The full conditioning+text PREFILL forward pass gives every hidden-state and
    logits fixture with NO autoregressive generation, so we register forward
    hooks on the transformer layers and capture the first (prefill) call.
  * For the audio/codes fixtures we run `generate_audio_codes` with a SMALL cap
    (MAX_NEW_TOKENS frames) so AR generation finishes quickly. This is enough to
    exercise the delayed-code shear, EOS handling, and DAC decode paths. The cap
    is documented below and recorded in config.json.

Device: tries MPS first; falls back to CPU float32 if an op is unsupported.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Vendored reference lives alongside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import zonos2_ref.loader as loader  # noqa: E402
import zonos2_ref.native as native  # noqa: E402
import zonos2_ref.runtime as runtime  # noqa: E402

# --- Fixed, reproducible inputs ------------------------------------------------
TEXT = "The quick brown fox jumps over the lazy dog."
SEED = 0
MAX_NEW_TOKENS = 1024  # generous cap; AR stops at natural EOS for the full short sentence
SPEAKING_RATE_BUCKET = -1  # unconditioned (matches node "default")
QUALITY_BUCKETS = [None] * 6  # all unconditioned (matches node "default")
CLEAN_SPEAKER_BACKGROUND = False
ACCURATE_MODE = True
HIDDEN_LAYERS = [0, 3, 13, 26, 27]  # layer-output hidden states to capture

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "outputs" / "fixtures"
REF_WAV = FIXTURE_DIR / "ref.wav"


def _save(name: str, array) -> None:
    arr = np.asarray(array)
    np.save(FIXTURE_DIR / name, arr)
    print(f"  saved {name:24s} shape={tuple(arr.shape)} dtype={arr.dtype}")


def _pick_device() -> tuple[torch.device, torch.dtype]:
    # bf16 on MPS keeps the 8B DiT at its native checkpoint precision (~16GB
    # vs ~32GB in f32) and runs fast; CPU fallback uses f32 (bf16 CPU is slow
    # and poorly supported). The fork's RMSNorm/SDPA paths are bf16-safe.
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def _build_bundle(device: torch.device, dtype: torch.dtype) -> loader.Zonos2Bundle:
    """Build the bundle directly (bypasses ComfyUI patcher + resolve_device)."""
    checkpoint = loader.model_dir() / "zonos2-bf16.safetensors"
    config = loader.read_bundled_config()
    model = native.build_native_model(config)
    native.load_native_weights(model, checkpoint, device, dtype)
    bundle = loader.Zonos2Bundle(
        model=model,
        config=config,
        model_path=checkpoint,
        device=device,
        torch_dtype=dtype,
        dtype_name="bf16->f32" if dtype == torch.float32 else str(dtype),
        attention="sdpa",
        download_if_missing=False,
        patchers=[],
    )
    runtime.ensure_codec(bundle)
    runtime.ensure_speaker_encoder(bundle)
    return bundle


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    if not REF_WAV.is_file():
        print(f"ERROR: reference audio missing at {REF_WAV}", file=sys.stderr)
        return 1

    torch.manual_seed(SEED)
    device, dtype = _pick_device()
    print(f"Device: {device}  dtype: {dtype}")

    t0 = time.time()
    try:
        bundle = _build_bundle(device, dtype)
    except Exception as exc:  # MPS op fallback
        if device.type == "mps":
            print(f"MPS build failed ({exc!r}); falling back to CPU float32.")
            device, dtype = torch.device("cpu"), torch.float32
            bundle = _build_bundle(device, dtype)
        else:
            raise
    print(f"Bundle loaded in {time.time() - t0:.1f}s on {device} ({dtype}).")

    model = bundle.model
    config = bundle.config

    # --- Reference audio -> speaker embedding -------------------------------
    wav, sr = sf.read(str(REF_WAV), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(wav.T).contiguous()  # (channels, samples)
    reference_audio = {"waveform": waveform.unsqueeze(0), "sample_rate": int(sr)}

    encoder = bundle.speaker_encoder

    # Capture the mel that is actually fed to the ECAPA model. We replicate the
    # encoder's _prepare_audio + mel pipeline (runtime.py) to grab `ecapa_in`.
    prepared = encoder._prepare_audio(reference_audio["waveform"], int(sr))
    padding = (1024 - 256) // 2
    padded = torch.nn.functional.pad(
        prepared.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)
    mel = encoder.mel_transform.to(encoder.device)(padded)
    mel = torch.log(torch.clamp(mel, min=1e-5)).transpose(1, 2)
    _save("ecapa_in.npy", mel.float().cpu().numpy())

    speaker_embedding = runtime.extract_speaker_embedding(bundle, reference_audio)
    _save("ecapa_emb.npy", speaker_embedding.float().cpu().numpy())

    # LDA projection of the embedding (speaker_lda_projection: 2048 -> 1024).
    with torch.inference_mode():
        lda = model.speaker_lda_projection(
            speaker_embedding.to(dtype=model.speaker_lda_projection.weight.dtype)
        )
    _save("lda.npy", lda.float().cpu().numpy())

    # --- Build the conditioning+text prompt ---------------------------------
    prompt, speaker_position = native.build_prompt(
        config,
        text=TEXT,
        speaking_rate_bucket=SPEAKING_RATE_BUCKET,
        quality_buckets=QUALITY_BUCKETS,
        speaker_embedding=speaker_embedding,
        clean_speaker_background=CLEAN_SPEAKER_BACKGROUND,
        accurate_mode=ACCURATE_MODE,
    )
    _save("prompt_ids.npy", prompt.cpu().numpy())
    with open(FIXTURE_DIR / "speaker_position.npy", "wb") as fh:
        np.save(fh, np.asarray(int(speaker_position)))
    print(f"  saved speaker_position.npy value={int(speaker_position)}")

    # --- Hooks to capture prefill hidden states -----------------------------
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_id: int):
        def hook(_module, _inputs, output):
            if layer_id in captured:
                return  # only the first (prefill) call
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_id] = hidden.detach().float().cpu()

        return hook

    handles = []
    for layer_id in HIDDEN_LAYERS:
        handles.append(model.layers[layer_id].register_forward_hook(make_hook(layer_id)))

    # --- Generate a short clip (drives prefill + AR + DAC) ------------------
    options = native.SamplingOptions(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,  # greedy
        seed=SEED,
    )
    t_gen = time.time()
    delayed_codes, eos_frame = native.generate_audio_codes(
        model,
        prompt,
        attention_backend="sdpa",
        options=options,
        speaker_embedding=speaker_embedding,
        speaker_position=speaker_position,
    )
    print(f"  generated {delayed_codes.shape[0]} frames in {time.time() - t_gen:.1f}s "
          f"(eos_frame={eos_frame})")

    for handle in handles:
        handle.remove()

    # Recompute prefill logits cleanly at the last prefill position.
    with torch.inference_mode():
        caches = model.create_kv_cache(
            batch_size=prompt.shape[0],
            max_length=prompt.shape[1] + 1,
            device=device,
            dtype=dtype,
        )
        prefill_logits = model(
            prompt.to(device=device),
            caches,
            "sdpa",
            speaker_embedding=speaker_embedding,
            speaker_position=speaker_position,
        )
    _save("logits.npy", prefill_logits[0].float().cpu().numpy())  # (9, 1026)

    for layer_id in HIDDEN_LAYERS:
        if layer_id not in captured:
            raise RuntimeError(f"Layer {layer_id} hidden state not captured.")
        _save(f"hidden_L{layer_id}.npy", captured[layer_id][0].numpy())  # (T, 2048)

    # --- Delayed codes + EOS + decoded audio --------------------------------
    _save("delayed_codes.npy", delayed_codes.cpu().numpy())
    with open(FIXTURE_DIR / "eos_frame.npy", "wb") as fh:
        np.save(fh, np.asarray(-1 if eos_frame is None else int(eos_frame)))
    print(f"  saved eos_frame.npy value={-1 if eos_frame is None else int(eos_frame)}")

    codec = bundle.codec
    audio = codec.decode(delayed_codes, pad_id=config.audio_pad_id, eos_frame=eos_frame)
    audio_np = audio.float().cpu().numpy()
    _save("audio.npy", audio_np)

    # Listenable 44.1 kHz wav for human sanity-check.
    wav_out = audio_np[0] if audio_np.ndim == 2 else audio_np
    sf.write(str(FIXTURE_DIR / "oracle.wav"), wav_out, runtime.DAC_SAMPLE_RATE)
    print(f"  wrote oracle.wav  samples={wav_out.shape[0]} "
          f"({wav_out.shape[0] / runtime.DAC_SAMPLE_RATE:.2f}s @ {runtime.DAC_SAMPLE_RATE}Hz)")

    # --- Resolved config dump -----------------------------------------------
    cfg_dict = dataclasses.asdict(config)
    cfg_dict["_special_topk_layers"] = {str(k): v for k, v in config.special_topk_layers.items()}
    cfg_dict["special_topk_layers"] = cfg_dict["_special_topk_layers"]
    del cfg_dict["_special_topk_layers"]
    cfg_dict["_oracle"] = {
        "device": str(device),
        "dtype": str(dtype),
        "text": TEXT,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "sampling": "greedy (temperature=0.0)",
        "speaking_rate_bucket": SPEAKING_RATE_BUCKET,
        "quality_buckets": QUALITY_BUCKETS,
        "clean_speaker_background": CLEAN_SPEAKER_BACKGROUND,
        "accurate_mode": ACCURATE_MODE,
        "dac_frame_rate_hz": runtime.DAC_SAMPLE_RATE
        / 512,  # hop_length 512 @ 44100
        "moe_layers": [i for i in range(config.n_layers) if config.is_moe_layer(i)],
        "dense_layers": [
            i for i in range(config.n_layers) if not config.is_moe_layer(i)
        ],
    }
    with open(FIXTURE_DIR / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg_dict, fh, indent=2, default=str)
    print(f"  saved config.json ({len(cfg_dict)} top-level keys)")

    print(f"\nDONE in {time.time() - t0:.1f}s total. Fixtures in {FIXTURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
