"""CPU unit tests for the AR sampling primitives (no GPU weights).

Hand-computed cases for the repetition penalty + greedy sample_codes, mirroring
native.py 1011-1054. These are pure host-side numpy logic.
"""

import numpy as np

from zonos2_mlx.generate import (
    SamplingOptions,
    apply_repetition_penalty,
    sample_codes,
)


def _opts(**kw):
    base = dict(
        temperature=0.0,
        repetition_window=50,
        repetition_penalty=1.2,
        repetition_codebooks=8,
    )
    base.update(kw)
    return SamplingOptions(**base)


def test_rep_penalty_positive_divided_negative_multiplied():
    # 2 codebooks, vocab=5 for a tiny hand check.
    logits = np.array(
        [[2.0, -1.0, 0.5, 3.0, -4.0],   # codebook 0
         [1.0, -2.0, 0.0, 0.0, 0.0]],   # codebook 1
        dtype=np.float32,
    )
    # History: token 3 (positive logit 3.0) and token 1 (negative logit -1.0)
    # seen in codebook 0; token 0 (positive 1.0) seen in codebook 1.
    generated = [np.array([3, 0]), np.array([1, 0])]
    opts = _opts(repetition_penalty=1.2)
    out = apply_repetition_penalty(logits, generated, opts)

    # codebook 0, token 3: positive -> divided by 1.2.
    assert np.isclose(out[0, 3], 3.0 / 1.2, atol=1e-5)
    # codebook 0, token 1: negative -> multiplied by 1.2.
    assert np.isclose(out[0, 1], -1.0 * 1.2, atol=1e-5)
    # codebook 1, token 0: positive -> divided by 1.2.
    assert np.isclose(out[1, 0], 1.0 / 1.2, atol=1e-5)
    # Untouched token (codebook 0, token 0 never in history): unchanged.
    assert np.isclose(out[0, 0], 2.0, atol=1e-5)
    # Original logits not mutated (fresh array returned).
    assert np.isclose(logits[0, 3], 3.0, atol=1e-5)


def test_rep_penalty_skips_when_no_history_or_disabled():
    logits = np.array([[2.0, -1.0, 0.5]], dtype=np.float32)
    opts = _opts()
    # No history -> identity.
    assert np.allclose(apply_repetition_penalty(logits, [], opts), logits)
    # penalty <= 1.0 -> identity.
    assert np.allclose(
        apply_repetition_penalty(logits, [np.array([0])], _opts(repetition_penalty=1.0)),
        logits,
    )
    # window <= 0 -> identity.
    assert np.allclose(
        apply_repetition_penalty(logits, [np.array([0])], _opts(repetition_window=0)),
        logits,
    )


def test_rep_penalty_respects_codebook_count():
    # 9 codebooks; repetition_codebooks=8 means codebook 8 is NEVER penalized.
    vocab = 6
    logits = np.ones((9, vocab), dtype=np.float32) * 2.0
    generated = [np.full(9, 1, dtype=np.int64)]  # token 1 in every codebook
    out = apply_repetition_penalty(logits, generated, _opts(repetition_codebooks=8))
    # codebooks 0..7 token 1 penalized (divided), codebook 8 untouched.
    for cb in range(8):
        assert np.isclose(out[cb, 1], 2.0 / 1.2, atol=1e-5)
    assert np.isclose(out[8, 1], 2.0, atol=1e-5)


def test_sample_codes_greedy_equals_argmax_after_penalty():
    # Build logits where the raw argmax (token 3) differs from the
    # post-penalty argmax once token 3 is penalized below token 0.
    logits = np.array(
        [[2.0, -1.0, 0.5, 2.1, -4.0]],  # raw argmax = token 3 (2.1)
        dtype=np.float32,
    )
    generated = [np.array([3])]  # token 3 recently seen (positive -> /1.2 -> 1.75)
    opts = _opts(repetition_penalty=1.2)

    penalized = apply_repetition_penalty(logits, generated, opts)
    expected = penalized.argmax(axis=-1)  # should now be token 0 (2.0 > 1.75)
    assert expected[0] == 0

    codes = sample_codes(logits, generated, opts)
    assert codes.shape == (1,)
    assert codes[0] == 0
    # And matches raw argmax when there's no history.
    codes0 = sample_codes(logits, [], opts)
    assert codes0[0] == int(logits[0].argmax())


def test_sample_codes_greedy_multi_codebook_shape():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((9, 1026)).astype(np.float32)
    codes = sample_codes(logits, [], _opts())
    assert codes.shape == (9,)
    assert codes.dtype == np.int64
    assert np.array_equal(codes, logits.argmax(axis=-1))
