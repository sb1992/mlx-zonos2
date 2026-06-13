"""GPU module-equivalence gate: KV-cached decode == full non-cached forward.

Proves the KV cache + RoPE offset slice + single-step masking are correct on a
small synthetic prompt BEFORE the expensive e2e parity run. Run SERIALLY (the
trunk is 8B/~15GB bf16).
"""

import numpy as np
import mlx.core as mx
import pytest
from pathlib import Path

from zonos2_mlx.model import Zonos2Model

_R = Path(__file__).resolve().parent.parent


def _last_logits_full(model, ids):
    """Non-cached full forward -> last-position logits (9, 1026) numpy."""
    out = model.forward(ids)
    return np.array(model.head(out.last).astype(mx.float32))


def _cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@pytest.mark.gpu
def test_kv_cache_matches_full_forward():
    model = Zonos2Model.from_pretrained(str(_R / "weights/zonos2-bf16"))
    cfg = model.cfg
    n_in = cfg.n_codebooks + 1  # 9 audio codebooks + 1 text stream

    rng = np.random.default_rng(0)
    prefill_T = 12

    def rand_row():
        row = rng.integers(0, cfg.codebook_size, size=(n_in,)).astype(np.int32)
        row[-1] = cfg.text_vocab  # text stream placeholder
        return row

    prompt = np.stack([rand_row() for _ in range(prefill_T)])[None]  # (1, T, 10)
    decode_rows = [rand_row().reshape(1, 1, n_in) for _ in range(3)]

    # Cached: prefill all T, then 3 single-step decodes feeding the next rows.
    caches = model.make_kv_caches(prefill_T + len(decode_rows))
    logits = model.forward_cached(mx.array(prompt), caches)
    mx.eval(logits)
    cached_last = np.array(logits[0].astype(mx.float32))  # after prefill
    cached_steps = [cached_last]

    seq = prompt.copy()
    for row in decode_rows:
        logits = model.forward_cached(mx.array(row), caches)
        mx.eval(logits, *[c.k for c in caches], *[c.v for c in caches])
        cached_steps.append(np.array(logits[0].astype(mx.float32)))
        seq = np.concatenate([seq, row], axis=1)

    # Full non-cached reference at each matching prefix length.
    full_steps = []
    seq = prompt.copy()
    full_steps.append(_last_logits_full(model, mx.array(seq)))
    for row in decode_rows:
        seq = np.concatenate([seq, row], axis=1)
        full_steps.append(_last_logits_full(model, mx.array(seq)))

    for i, (c, f) in enumerate(zip(cached_steps, full_steps)):
        cos = _cos(c, f)
        max_abs = float(np.abs(c - f).max())
        arg_match = bool((c.argmax(-1) == f.argmax(-1)).all())
        print(f"step {i}: cosine={cos:.6f}  max_abs={max_abs:.4f}  argmax_match={arg_match}")
        assert cos >= 0.9999, f"step {i} cached vs full cosine {cos:.6f} < 0.9999"
        assert arg_match, f"step {i} argmax diverges (cached vs full)"
