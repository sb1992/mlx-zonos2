"""Seam polishing for long-form (multi-chunk) zonos2 TTS — OPT-IN, additive.

Pure stdlib + numpy (NO mlx / torch). Post-processing helpers that smooth the
joins between independently-generated chunks:

  * ``trim_silence``             — drop per-chunk leading/trailing silence (kills the
                                   dead air a no-EOS chunk leaves, tightens edges).
  * ``release_fade``             — raised-cosine ease-in + a longer ease-OUT so the
                                   final word *tapers to a close* instead of cutting
                                   at full energy.
  * ``boundary_loudness_match``  — match loudness GLOBALLY (each chunk to the median)
                                   then LOCALLY (ramp each chunk's onset to the previous
                                   chunk's tail), removing the "next word louder" step.
  * ``assemble_polished``        — the full recipe: release-fade -> boundary match ->
                                   concatenate with a pause -> peak-normalize.

This is the ear-validated "v1 polish" recipe. It is wired into
``synthesize_long`` behind ``polish=True`` and changes nothing when off.
"""
from __future__ import annotations

import numpy as np


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x**2) + 1e-12)) if x.size else 0.0


def trim_silence(
    audio: np.ndarray, sample_rate: int, *, threshold_db: float = -40.0, pad_ms: int = 30
) -> np.ndarray:
    """Trim leading/trailing silence below ``threshold_db`` (rel. peak), keeping a small
    ``pad_ms`` margin so onsets/offsets are not clipped. Returns 1-D float32. Never raises."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    hop = max(1, int(sample_rate * 0.01))
    n = max(1, x.size // hop)
    peak = float(np.abs(x).max()) + 1e-9
    db = np.array(
        [20.0 * np.log10(_rms(x[i * hop:i * hop + hop]) / peak + 1e-9) for i in range(n)]
    )
    voiced = np.where(db > threshold_db)[0]
    if voiced.size == 0:
        return x
    pad = int(sample_rate * pad_ms / 1000)
    s = max(0, voiced[0] * hop - pad)
    e = min(x.size, (voiced[-1] + 1) * hop + pad)
    return x[s:e]


def release_fade(
    audio: np.ndarray, sample_rate: int, *, release_ms: int = 200, attack_ms: int = 40
) -> np.ndarray:
    """Raised-cosine ease-in (``attack_ms``) + ease-OUT (``release_ms``). The longer
    release lets the final word settle to a close instead of stopping abruptly."""
    y = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    if y.size == 0:
        return y
    a = min(int(sample_rate * attack_ms / 1000), y.size // 2)
    r = min(int(sample_rate * release_ms / 1000), y.size // 2)
    if a > 0:
        y[:a] *= 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, a)))
    if r > 0:
        y[-r:] *= 0.5 * (1.0 + np.cos(np.pi * np.linspace(0.0, 1.0, r)))
    return y


def boundary_loudness_match(
    chunks: list[np.ndarray], sample_rate: int, *,
    tail_s: float = 0.8, ramp_s: float = 0.45, clamp: tuple[float, float] = (0.45, 1.6),
) -> list[np.ndarray]:
    """Match loudness globally (each chunk -> median RMS) then locally: ramp each chunk's
    onset to the previous chunk's tail level over ``ramp_s``, returning to the chunk's
    natural level. Kills the per-seam loudness step. Empty/silent chunks are skipped."""
    cs = [np.asarray(c, dtype=np.float32).reshape(-1).copy() for c in chunks]
    cs = [c for c in cs if c.size]
    if not cs:
        return []
    levels = [_rms(c) for c in cs]
    tgt = float(np.median(levels))
    cs = [c * (tgt / (lvl + 1e-9)) for c, lvl in zip(cs, levels)]
    nt = max(1, int(sample_rate * tail_s))
    for i in range(1, len(cs)):
        head = _rms(cs[i][:nt])
        prev_tail = _rms(cs[i - 1][-nt:])
        if head > 1e-6 and prev_tail > 1e-6:
            g = float(np.clip(prev_tail / head, clamp[0], clamp[1]))
            n = min(int(sample_rate * ramp_s), cs[i].size)
            if n > 0:
                cs[i][:n] *= g + (1.0 - g) * np.linspace(0.0, 1.0, n)
    return cs


def assemble_polished(
    chunks: list[np.ndarray], sample_rate: int, *,
    gap_ms: int = 250, release_ms: int = 200, attack_ms: int = 40, normalize_peak: float = 0.97,
) -> np.ndarray:
    """Full polished assembly: release-fade each chunk -> boundary loudness match ->
    concatenate with ``gap_ms`` silence BETWEEN chunks -> peak-normalize to
    ``normalize_peak``. Returns 1-D float32. Chunks should already be silence-trimmed."""
    cs = [np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks]
    cs = [c for c in cs if c.size]
    if not cs:
        return np.zeros(0, dtype=np.float32)
    shaped = [release_fade(c, sample_rate, release_ms=release_ms, attack_ms=attack_ms) for c in cs]
    matched = boundary_loudness_match(shaped, sample_rate)
    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for i, c in enumerate(matched):
        if i:
            pieces.append(gap)
        pieces.append(c)
    out = np.concatenate(pieces).astype(np.float32)
    if normalize_peak and normalize_peak > 0:
        peak = float(np.abs(out).max())
        if peak > 1e-9:
            out = out * (normalize_peak / peak)
    return np.clip(out, -0.999, 0.999).astype(np.float32)
