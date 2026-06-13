"""UTF-8 byte tokenizer + multi-stream prompt assembly for ZONOS2.

Pure stdlib + numpy. NO torch, NO mlx — this module runs on CPU-only hosts.

This mirrors, byte-for-byte, the oracle prompting logic in
``Zonos2_TTS-ComfyUI/native.py`` (``text_to_byte_ids``, the conditioning-token
offset helpers, ``shear``, and ``build_prompt``). The numpy ``_shear`` is a
faithful port of the torch ``padded.gather(0, row_idx)`` semantics.

``normalize_text`` defaults to passthrough (identity) so the byte-exact gate
stays oracle-faithful; an opt-in minimal English ruleset is available via
``enable=True``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # typing only — no heavy/runtime import
    from zonos2_mlx.config import Zonos2Config

# --- Module constants (mirror native.py top-of-file) ----------------------
BOS_ID = 2
EOS_ID = 3
LEGACY_SYMBOL_VOCAB_SIZE = 192

SILENCE_TOKENS_0_2S = [
    [568, 778, 338, 524, 967, 360, 728, 550, 90],
    [568, 778, 10, 674, 364, 981, 741, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 778, 721, 842, 264, 974, 989, 507, 308],
]


# --- Text -> byte ids -----------------------------------------------------
def text_to_byte_ids(text: str) -> list[int]:
    """UTF-8 byte tokenizer: ``[BOS] + [b + 192 for b in utf8(text)] + [EOS]``."""
    return [
        BOS_ID,
        *(byte + LEGACY_SYMBOL_VOCAB_SIZE for byte in text.encode("utf-8")),
        EOS_ID,
    ]


# --- Conditioning-token offset helpers (mirror native.py) -----------------
def _conditioning_base(config: "Zonos2Config") -> int:
    background = 2 if config.speaker_background_token_enabled else 0
    accurate = 1 if config.accurate_mode_token_enabled and background else 0
    return (
        config.text_vocab
        - config.speaking_rate_num_buckets
        - sum(config.quality_bucket_counts)
        - background
        - accurate
    )


def speaking_rate_token(config: "Zonos2Config", bucket: int) -> int:
    if bucket < 0 or bucket >= config.speaking_rate_num_buckets:
        raise ValueError(
            f"speaking_rate_bucket must be -1 or 0.."
            f"{config.speaking_rate_num_buckets - 1}"
        )
    return _conditioning_base(config) + bucket


def quality_token(config: "Zonos2Config", feature_index: int, bucket: int) -> int:
    count = config.quality_bucket_counts[feature_index]
    if bucket < 0 or bucket >= count:
        raise ValueError(
            f"Quality bucket for {config.quality_features[feature_index]} "
            f"must be 0..{count - 1}"
        )
    return (
        _conditioning_base(config)
        + config.speaking_rate_num_buckets
        + sum(config.quality_bucket_counts[:feature_index])
        + bucket
    )


def speaker_background_token(config: "Zonos2Config", clean: bool) -> int:
    return (
        _conditioning_base(config)
        + config.speaking_rate_num_buckets
        + sum(config.quality_bucket_counts)
        + (0 if clean else 1)
    )


def accurate_mode_token(config: "Zonos2Config") -> int:
    return (
        _conditioning_base(config)
        + config.speaking_rate_num_buckets
        + sum(config.quality_bucket_counts)
        + 2
    )


# --- Causal delay-shift for the silence block -----------------------------
def _shear(codes: np.ndarray, pad_id: int) -> np.ndarray:
    """numpy port of torch ``padded.gather(0, row_idx)``: out[i,j] = padded[rows[i,j], j]."""
    frames, k = codes.shape
    padded = np.full((k - 1 + frames, k), pad_id, dtype=codes.dtype)
    padded[k - 1:] = codes
    rows = (k - 1) + np.arange(frames)[:, None] - np.arange(k)[None, :]
    return padded[rows, np.arange(k)[None, :]]


# --- Prompt assembly ------------------------------------------------------
def build_prompt(
    config: "Zonos2Config",
    text: str,
    speaking_rate_bucket: int = -1,
    quality_buckets: list[int | None] | tuple[int | None, ...] | None = None,
    has_speaker: bool = True,
    clean_speaker_background: bool = False,
    accurate_mode: bool = True,
) -> tuple[np.ndarray, int | None]:
    """Build the conditioning + text prefix prompt, mirroring the oracle.

    Returns ``(prompt, speaker_position)`` where prompt is int32 ``(1, T, 10)``.
    """
    rows: list[list[int]] = []
    audio_padding = [config.audio_pad_id] * config.n_codebooks

    speaker_position: int | None = None
    if has_speaker:
        speaker_position = 0
        rows.append(audio_padding + [config.text_vocab])
        if config.speaker_background_token_enabled:
            rows.append(
                audio_padding
                + [speaker_background_token(config, clean_speaker_background)]
            )
        if config.accurate_mode_token_enabled and accurate_mode:
            rows.append(audio_padding + [accurate_mode_token(config)])

    if speaking_rate_bucket >= 0:
        rows.append(
            audio_padding + [speaking_rate_token(config, speaking_rate_bucket)]
        )

    if quality_buckets is not None:
        if len(quality_buckets) != len(config.quality_features):
            raise ValueError(
                f"Expected {len(config.quality_features)} quality buckets, "
                f"got {len(quality_buckets)}."
            )
        for feature_index, bucket in enumerate(quality_buckets):
            if bucket is None or int(bucket) < 0:
                continue
            rows.append(
                audio_padding + [quality_token(config, feature_index, int(bucket))]
            )

    rows.extend(audio_padding + [token] for token in text_to_byte_ids(text))

    silence = np.array(SILENCE_TOKENS_0_2S, dtype=np.int32)
    silence = _shear(silence[:, : config.n_codebooks], config.audio_pad_id)
    silence_text = np.full((silence.shape[0], 1), config.text_vocab, dtype=np.int32)
    silence_block = np.concatenate([silence, silence_text], axis=1)

    prompt = np.concatenate(
        [np.array(rows, dtype=np.int32), silence_block],
        axis=0,
    )
    return prompt[None, :, :], speaker_position


# --- Minimal optional English text normalization --------------------------
_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def normalize_text(text: str, enable: bool = False) -> str:
    """Oracle-faithful by default (identity). With ``enable=True``, apply a tiny,
    safe English ruleset: collapse whitespace runs, strip ends, and expand ASCII
    digit runs to words digit-by-digit. Identity on the fixture text in both modes.
    """
    if not enable:
        return text
    # Expand ASCII digit runs to spelled words, digit-by-digit. Use an
    # ASCII-only class ([0-9], NOT \d which also matches Unicode digits the
    # map has no entry for) so non-Latin input passes through untouched.
    text = re.sub(
        r"[0-9]+",
        lambda m: " ".join(_DIGIT_WORDS[d] for d in m.group()),
        text,
    )
    # Collapse all whitespace runs to single spaces and strip ends.
    return re.sub(r"\s+", " ", text).strip()
