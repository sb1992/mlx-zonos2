import numpy as np

from zonos2_mlx.seam_polish import (
    _rms,
    assemble_polished,
    boundary_loudness_match,
    release_fade,
    trim_silence,
)

SR = 8000


def test_trim_silence_removes_dead_air():
    sil = np.zeros(int(SR * 0.2), dtype=np.float32)
    tone = (0.5 * np.sin(2 * np.pi * 200 * np.arange(int(SR * 0.5)) / SR)).astype(np.float32)
    x = np.concatenate([sil, tone, sil])
    out = trim_silence(x, SR)
    assert out.size < x.size                       # dead air gone
    assert np.abs(out).max() > 0.4                 # the speech is kept
    # leading/trailing 30ms pad keeps it from clipping the tone, but well under the 0.2s silence
    assert out.size < tone.size + int(SR * 0.2)


def test_trim_silence_empty_and_all_silent():
    assert trim_silence(np.zeros(0, dtype=np.float32), SR).size == 0
    allsil = np.zeros(SR, dtype=np.float32)
    assert trim_silence(allsil, SR).size == allsil.size   # nothing voiced -> returned as-is


def test_release_fade_tapers_both_ends_keeps_body():
    y = release_fade(np.ones(SR, dtype=np.float32), SR, release_ms=200, attack_ms=40)
    assert y.size == SR                            # fade is in-place, length unchanged
    assert y[0] < 0.05 and y[-1] < 0.05            # both ends faded to ~0
    assert y[SR // 2] > 0.99                        # body untouched
    r = int(SR * 0.2)
    assert y[-r] > y[-1]                            # release region is decreasing


def test_boundary_match_equalizes_levels():
    cs = [np.ones(SR, dtype=np.float32) * 0.2, np.ones(SR, dtype=np.float32) * 0.8]
    m = boundary_loudness_match(cs, SR)
    assert len(m) == 2 and all(c.size == SR for c in m)
    assert abs(_rms(m[0]) - 0.5) < 0.05            # first chunk -> median level
    ramp = int(SR * 0.45)
    assert abs(_rms(m[1][ramp:]) - 0.5) < 0.05     # second chunk body -> median level


def test_boundary_match_reduces_loudness_step():
    c0 = np.ones(SR, dtype=np.float32) * 0.3
    c1 = np.ones(SR, dtype=np.float32) * 0.9
    nt = int(SR * 0.8)
    before = abs(_rms(c0[-nt:]) - _rms(c1[:nt]))
    m = boundary_loudness_match([c0, c1], SR)
    after = abs(_rms(m[0][-nt:]) - _rms(m[1][:nt]))
    assert after < before                          # the seam loudness step shrank


def test_assemble_polished_length_gap_and_peak():
    cs = [np.ones(SR, dtype=np.float32) * 0.5, np.ones(SR, dtype=np.float32) * 0.5]
    gap = int(SR * 0.1)
    out = assemble_polished(cs, SR, gap_ms=100, release_ms=50, attack_ms=10, normalize_peak=0.9)
    assert out.shape[0] == SR + gap + SR           # fades don't change chunk length
    assert abs(np.abs(out).max() - 0.9) < 1e-3     # peak-normalized
    assert np.all(np.abs(out[SR:SR + gap]) < 1e-6)  # the gap is silent


def test_assemble_polished_empty():
    assert assemble_polished([], SR).shape[0] == 0
