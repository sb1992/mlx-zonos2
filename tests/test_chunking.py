import numpy as np

from zonos2_mlx.chunking import (
    assemble_chunks,
    estimate_seconds,
    pack_sentences,
    plan_chunks,
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


def test_estimate_monotonic_and_script_rates():
    assert estimate_seconds("ab") < estimate_seconds("abcd")
    # same char count: CJK + Devanagari estimate longer than Latin
    assert estimate_seconds("你好世界") > estimate_seconds("abcd")
    assert estimate_seconds("नमस्ते") > estimate_seconds("abcdef")


def test_estimate_chars_per_sec_override():
    # 10 non-space chars at 5 chars/sec -> 2.0s, regardless of script
    assert estimate_seconds("0123456789", chars_per_sec=5.0) == 2.0


def test_pack_combines_more_than_one_sentence():
    # Ten short sentences; a generous budget must pack several per chunk.
    text = " ".join(["Hi there." for _ in range(10)])
    pieces = split_for_generation(text, max_chars=240)
    assert len(pieces) == 10
    chunks = pack_sentences(pieces, max_seconds=40.0)
    assert len(chunks) < len(pieces)          # the "don't break at every sentence" guarantee
    for c in chunks:
        assert estimate_seconds(c) <= 40.0 or c in pieces


def test_pack_respects_budget_tiny():
    pieces = ["Hello there.", "How are you?", "I am fine."]
    # ~0.07 s/char -> each ~0.8s; a 1.0s budget allows ~1 sentence per chunk
    chunks = pack_sentences(pieces, max_seconds=1.0)
    for c in chunks:
        assert estimate_seconds(c) <= 1.0 or c in pieces


def test_plan_chunks_end_to_end():
    text = "One. Two. Three. Four. Five."
    chunks = plan_chunks(text, max_seconds=40.0)
    assert len(chunks) == 1                    # all five pack into one ~big-budget chunk
    assert "One." in chunks[0] and "Five." in chunks[0]
