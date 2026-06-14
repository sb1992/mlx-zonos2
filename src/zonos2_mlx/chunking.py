"""Word-safe multilingual text chunking for long-form zonos2 TTS.

Pure stdlib + numpy (NO mlx / torch). The primary split is sentence-level; a
length-cap fallback hierarchy (clause -> word -> grapheme) guarantees no chunk
exceeds ``max_chars`` while NEVER breaking mid-word or mid-codepoint/grapheme.
``pack_sentences`` then greedily combines consecutive sentences up to an
estimated-audio-seconds budget (so a chunk is several sentences, not one).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

# Split AFTER a terminal mark, consuming trailing whitespace. Latin .!?  CJK 。！？
# Devanagari danda ।॥.
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？।॥])\s*")
# Clause marks: Latin , ; : —  +  CJK/full-width 、 ， ； ：
_CLAUSE_RE = re.compile(r"(?<=[,;:—、，；：])\s*")

_MAX_CHARS_DEFAULT = 240   # Latin / unknown
_MAX_CHARS_CJK = 120       # CJK/Kana/Hangul pack ~2x model-tokens per char
_MAX_CHARS_DEVANAGARI = 160

_WS_RE = re.compile(r"\s")


def _detect_script(text: str) -> str:
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "devanagari"
        if (0x4E00 <= o <= 0x9FFF) or (0x3040 <= o <= 0x30FF) or (0xAC00 <= o <= 0xD7AF):
            return "cjk"
    return "latin"


def resolve_max_chars(text: str, *, language: str | None = None) -> int:
    """Script-aware safety cap (characters) for the split fallback."""
    code = (language or "").upper()
    if code in ("ZH", "JA", "YUE", "KO"):
        return _MAX_CHARS_CJK
    if code == "HI":
        return _MAX_CHARS_DEVANAGARI
    if code:
        return _MAX_CHARS_DEFAULT
    return {"cjk": _MAX_CHARS_CJK, "devanagari": _MAX_CHARS_DEVANAGARI}.get(
        _detect_script(text), _MAX_CHARS_DEFAULT
    )


def _split_keep(text: str, regex: re.Pattern) -> list[str]:
    return [p for p in (s.strip() for s in regex.split(text)) if p]


def _pack_words(s: str, max_chars: int) -> list[str]:
    out, cur = [], ""
    for word in s.split():
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= max_chars:
            cur += " " + word
        else:
            out.append(cur)
            cur = word
        if len(cur) > max_chars and " " not in cur:
            if _WS_RE.search(cur) is None and _detect_script(cur) == "cjk":
                out.extend(_pack_graphemes(cur, max_chars))
                cur = ""
    if cur:
        out.append(cur)
    return out


def _pack_graphemes(s: str, max_chars: int) -> list[str]:
    """Pack codepoints up to max_chars, never starting a chunk on a combining mark."""
    out, cur = [], ""
    for ch in s:
        if len(cur) >= max_chars and not unicodedata.category(ch).startswith("M"):
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _cap(piece: str, max_chars: int) -> list[str]:
    piece = piece.strip()
    if not piece:
        return []
    if len(piece) <= max_chars:
        return [piece]
    out: list[str] = []
    for clause in _split_keep(piece, _CLAUSE_RE):
        if len(clause) <= max_chars:
            out.append(clause)
        elif _WS_RE.search(clause) is not None:
            out.extend(_pack_words(clause, max_chars))
        else:
            out.extend(_pack_graphemes(clause, max_chars))
    return out


def split_for_generation(text: str, *, max_chars: int, language: str | None = None) -> list[str]:
    """Sentence-first split, then a word-safe length-cap fallback. Stripped, non-empty."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    for sent in _split_keep(text, _SENTENCE_RE):
        chunks.extend(_cap(sent, max_chars))
    return [c for c in (c.strip() for c in chunks) if c]


def assemble_chunks(wavs: list[np.ndarray], sample_rate: int, gap_ms: int) -> np.ndarray:
    """Concatenate mono waveforms with ``gap_ms`` silence BETWEEN chunks (not after the last)."""
    clean = [np.asarray(w, dtype=np.float32).ravel() for w in wavs]
    clean = [w for w in clean if w.size]
    if not clean:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for i, w in enumerate(clean):
        if i:
            pieces.append(gap)
        pieces.append(w)
    return np.concatenate(pieces)


# --- degenerate-chunk health check (WARN-only here) ---
_MIN_SILENCE_STD = 0.01
_MIN_SEC_PER_WORD = 0.18
_MIN_SEC_PER_CHAR_CJK = 0.12


def _count_words(text: str) -> int:
    return len([w for w in _WS_RE.split(text.strip()) if w])


def chunk_health(
    audio: np.ndarray, text: str, sample_rate: int, *, language: str | None = None
) -> bool:
    """Cheap heuristic: is this chunk's audio non-degenerate for ``text``? Never raises."""
    a = np.asarray(audio).ravel()
    if a.size == 0 or not np.all(np.isfinite(a)):
        return False
    if float(np.std(a)) <= _MIN_SILENCE_STD:
        return False
    t = (text or "").strip()
    if not t:
        return True
    audio_s = a.size / float(sample_rate)
    if _detect_script(t) == "cjk":
        n_chars = sum(
            1 for c in t if not c.isspace() and not unicodedata.category(c).startswith("P")
        )
        floor = _MIN_SEC_PER_CHAR_CJK * max(n_chars, 1)
    else:
        floor = _MIN_SEC_PER_WORD * max(_count_words(t), 1)
    return audio_s >= floor


# --- duration estimate + greedy sentence packing (long-form) ---
# sec/char tuned conservative-high so we under-pack rather than overflow 6144.
_SEC_PER_CHAR = {"latin": 0.070, "devanagari": 0.110, "cjk": 0.140}


def _rate_for(text: str, language: str | None) -> float:
    code = (language or "").upper()
    if code in ("ZH", "JA", "YUE", "KO"):
        return _SEC_PER_CHAR["cjk"]
    if code == "HI":
        return _SEC_PER_CHAR["devanagari"]
    if code:
        return _SEC_PER_CHAR["latin"]
    return _SEC_PER_CHAR.get(_detect_script(text), _SEC_PER_CHAR["latin"])


def estimate_seconds(
    text: str, *, language: str | None = None, chars_per_sec: float | None = None
) -> float:
    """Estimated spoken duration of ``text`` (non-whitespace chars × sec/char).

    ``chars_per_sec``, when given, overrides the script-aware rate for ALL scripts
    with ``sec_per_char = 1 / chars_per_sec``.
    """
    rate = (1.0 / float(chars_per_sec)) if chars_per_sec else _rate_for(text, language)
    n = sum(1 for c in text if not c.isspace())
    return n * rate


def pack_sentences(
    pieces: list[str], *, max_seconds: float,
    language: str | None = None, chars_per_sec: float | None = None,
) -> list[str]:
    """Greedily glue consecutive ``pieces`` while the estimate stays <= ``max_seconds``.

    A lone piece already over budget becomes its own chunk (it was already <=
    max_chars from the split). Join with a space for spaced scripts, directly for CJK.
    """
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if not cur:
            cur = p
            continue
        sep = "" if _detect_script(cur + p) == "cjk" else " "
        cand = cur + sep + p
        if estimate_seconds(cand, language=language, chars_per_sec=chars_per_sec) > max_seconds:
            chunks.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks


def plan_chunks(
    text: str, *, max_seconds: float,
    language: str | None = None, chars_per_sec: float | None = None,
) -> list[str]:
    """Split ``text`` on sentence boundaries (word-safe), then pack to ~``max_seconds``."""
    pieces = split_for_generation(
        text, max_chars=resolve_max_chars(text, language=language), language=language
    )
    return pack_sentences(
        pieces, max_seconds=max_seconds, language=language, chars_per_sec=chars_per_sec
    )
