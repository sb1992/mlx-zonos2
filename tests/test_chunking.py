import numpy as np

from zonos2_mlx.chunking import (
    assemble_chunks,
    resolve_max_chars,
    split_for_generation,
)


def test_split_on_sentence_boundaries():
    chunks = split_for_generation("Hello there. How are you? I am fine.", max_chars=240)
    assert chunks == ["Hello there.", "How are you?", "I am fine."]


def test_split_never_breaks_mid_word():
    text = "word " * 200  # 1000 chars, one run-on "sentence"
    chunks = split_for_generation(text.strip(), max_chars=240)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 240
        assert "wor d" not in c and not c.endswith("wor")  # whole words only
        assert all(w == "word" for w in c.split())


def test_resolve_max_chars_script_aware():
    assert resolve_max_chars("hello world") == 240
    assert resolve_max_chars("你好世界") == 120
    assert resolve_max_chars("नमस्ते दुनिया") == 160
    assert resolve_max_chars("hello", language="ZH") == 120


def test_assemble_inserts_gap_between_only():
    a = np.ones(100, dtype=np.float32)
    b = np.ones(50, dtype=np.float32)
    out = assemble_chunks([a, b], sample_rate=1000, gap_ms=10)  # gap = 10 samples
    assert out.shape[0] == 100 + 10 + 50
    assert np.all(out[100:110] == 0.0)


def test_assemble_empty():
    assert assemble_chunks([], 44100, 80).shape[0] == 0
