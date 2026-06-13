"""Parity + unit tests for the MLX ECAPA-TDNN speaker encoder + LDA.

CPU tests (no GPU / weights needed):
  test_reflect_pad_time      — hand-checked torch-reflect semantics.
  test_speaker_profile_roundtrip — save -> load preserves vector + compat.

GPU parity tests (the T4 gate; require weights + fixtures):
  test_ecapa_embedding_parity — ECAPA x-vector cosine >= 0.999 vs oracle.
  test_lda_parity             — LDA-projected vector cosine >= 0.999 vs oracle.

Follows the convention of tests/test_dac_parity.py and tests/test_trunk_parity.py:
GPU tests are plain functions that load weights/fixtures inline (no marker); they
run when the weights + fixtures are present.
"""

from pathlib import Path

import numpy as np
import mlx.core as mx

from zonos2_mlx.speaker import (
    EcapaTDNN,
    SpeakerLDA,
    SpeakerProfile,
    reflect_pad_time,
)

_R = Path(__file__).resolve().parent.parent
_F = _R / "outputs/fixtures"


def _cos(a, b):
    # Explicit element-wise reductions (no `@`/np.linalg.norm) to avoid the
    # spurious Accelerate-matmul RuntimeWarnings under pytest; pure float64.
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    dot = float(np.sum(a * b))
    na = float(np.sqrt(np.sum(a * a)))
    nb = float(np.sqrt(np.sum(b * b)))
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# CPU tests
# ---------------------------------------------------------------------------

def test_reflect_pad_time():
    # x: (B=1, T=5, C=1) channels-last. Values 1..5 along time.
    x = mx.array(np.arange(1, 6, dtype=np.float32).reshape(1, 5, 1))

    out2 = np.array(reflect_pad_time(x, 2)).flatten().tolist()
    # torch F.pad(mode="reflect", pad=2) on [1,2,3,4,5] -> [3,2,1,2,3,4,5,4,3]
    assert out2 == [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]

    out3 = np.array(reflect_pad_time(x, 3)).flatten().tolist()
    # torch reflect pad=3 -> [4,3,2,1,2,3,4,5,4,3,2]
    assert out3 == [4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]

    # pad=0 is identity
    out0 = np.array(reflect_pad_time(x, 0)).flatten().tolist()
    assert out0 == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_speaker_profile_roundtrip(tmp_path):
    vec = np.random.RandomState(0).randn(1024).astype(np.float32)
    compat = "ecapa-lda-bf16-v1"
    p = tmp_path / "voice.zonos"
    SpeakerProfile.save(str(p), vec, compat)

    loaded = SpeakerProfile.load(str(p))
    assert loaded.lda.shape == (1024,)
    assert loaded.lda.dtype == np.float32
    assert np.array_equal(loaded.lda, vec)
    assert loaded.compat == compat


# ---------------------------------------------------------------------------
# GPU parity tests — the T4 gate
# ---------------------------------------------------------------------------

def test_ecapa_embedding_parity():
    enc = EcapaTDNN.from_pretrained(str(_R / "weights/zonos2-bf16/speaker_encoder"))
    mel = mx.array(np.load(_F / "ecapa_in.npy"))
    emb = enc(mel)
    ref = np.load(_F / "ecapa_emb.npy")
    assert emb.shape == (1, 2048)
    c = _cos(np.array(emb.astype(mx.float32)), ref)
    print(f"ECAPA embedding cosine = {c:.6f}")
    assert c >= 0.999, f"ECAPA embedding cosine {c:.6f} < 0.999"


def test_lda_parity():
    enc = EcapaTDNN.from_pretrained(str(_R / "weights/zonos2-bf16/speaker_encoder"))
    lda = SpeakerLDA.from_dit(str(_R / "weights/zonos2-bf16/zonos2-bf16.safetensors"))
    mel = mx.array(np.load(_F / "ecapa_in.npy"))
    out = lda(enc(mel))
    ref = np.load(_F / "lda.npy")
    assert out.shape == (1, 1024)
    c = _cos(np.array(out.astype(mx.float32)), ref)
    print(f"LDA cosine = {c:.6f}")
    assert c >= 0.999, f"LDA cosine {c:.6f} < 0.999"
