"""T5 gate: UTF-8 byte tokenizer + build_prompt byte-exact against oracle fixture.

CPU-only (pure stdlib + numpy). No torch, no mlx.
"""

from pathlib import Path

import numpy as np
import pytest

from zonos2_mlx.config import Zonos2Config
from zonos2_mlx.textnorm import (
    SILENCE_TOKENS_0_2S,
    _shear,
    build_prompt,
    normalize_text,
    text_to_byte_ids,
)

_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(
    not (_ROOT / "outputs/fixtures/prompt_ids.npy").exists(),
    reason="needs the oracle build_prompt fixtures (gitignored)",
)
def test_prompt_matches_oracle():
    c = Zonos2Config.load("outputs/fixtures/config.json")
    ids, pos = build_prompt(
        c,
        "The quick brown fox jumps over the lazy dog.",
        speaking_rate_bucket=-1,
        quality_buckets=[None] * len(c.quality_features),
        has_speaker=True,
        clean_speaker_background=False,
        accurate_mode=True,
    )
    ref = np.load("outputs/fixtures/prompt_ids.npy")
    assert np.array(ids).shape == ref.shape  # (1, 66, 10)
    assert np.array_equal(np.array(ids), ref)  # BYTE-EXACT
    assert pos == int(np.load("outputs/fixtures/speaker_position.npy"))  # 0


def test_text_to_byte_ids():
    ids = text_to_byte_ids("Hi")  # 'H'=72->264, 'i'=105->297
    assert ids == [2, 264, 297, 3]


def test_text_to_byte_ids_utf8_multibyte():
    # 'é' = U+00E9 -> UTF-8 bytes 0xC3 0xA9 = 195, 169 -> +192 each
    ids = text_to_byte_ids("é")
    assert ids == [2, 195 + 192, 169 + 192, 3]


def test_normalize_identity_on_fixture():
    t = "The quick brown fox jumps over the lazy dog."
    assert normalize_text(t) == t and normalize_text(t, enable=True) == t


def test_normalize_passthrough_default():
    # Default mode is identity even for messy input.
    messy = "  hello   world  "
    assert normalize_text(messy) == messy


def test_normalize_enabled_collapses_whitespace():
    assert normalize_text("  hello   world  ", enable=True) == "hello world"
    assert normalize_text("a\t\nb", enable=True) == "a b"


def test_normalize_enabled_expands_digits():
    assert normalize_text("I have 3 cats", enable=True) == "I have three cats"
    # Multi-digit runs expand digit-by-digit.
    assert normalize_text("room 12", enable=True) == "room one two"


def test_normalize_enabled_passes_non_ascii_digits():
    # Non-Latin digits (Arabic-Indic, Devanagari) must NOT crash and must pass
    # through untouched — only ASCII [0-9] is expanded.
    assert normalize_text("price ٤٥", enable=True) == "price ٤٥"
    assert normalize_text("कमरा ३", enable=True) == "कमरा ३"


def test_shear_shape_and_spot():
    silence = _shear(np.array(SILENCE_TOKENS_0_2S)[:, :9], 1025)
    assert silence.shape == (17, 9)
    # Causal top-pad: row 0 keeps col 0, rest are pad.
    assert silence[0, 0] == SILENCE_TOKENS_0_2S[0][0]
    assert np.all(silence[0, 1:] == 1025)
    # Row 1: cols 0,1 real, rest pad.
    assert silence[1, 0] == SILENCE_TOKENS_0_2S[1][0]
    assert silence[1, 1] == SILENCE_TOKENS_0_2S[0][1]
    assert np.all(silence[1, 2:] == 1025)


def test_shear_matches_torch_gather_semantics():
    # out[i,j] = padded[rows[i,j], j]
    codes = np.arange(12, dtype=np.int32).reshape(4, 3)
    pad = 99
    frames, k = codes.shape
    padded = np.full((k - 1 + frames, k), pad, dtype=codes.dtype)
    padded[k - 1:] = codes
    rows = (k - 1) + np.arange(frames)[:, None] - np.arange(k)[None, :]
    expected = np.empty((frames, k), dtype=codes.dtype)
    for i in range(frames):
        for j in range(k):
            expected[i, j] = padded[rows[i, j], j]
    assert np.array_equal(_shear(codes, pad), expected)
