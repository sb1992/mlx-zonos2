"""THE e2e parity gate: MLX AR-generated audio codes vs the torch oracle.

Uses the EXACT oracle LDA fixture (NOT a re-enrolled profile) so the gate
isolates the autoregressive loop + KV cache from the speaker encoder's 0.9997
cosine. Run SERIALLY (the trunk is 8B/~15GB bf16).
"""

import numpy as np
from pathlib import Path

from zonos2_mlx.pipeline import synthesize

_F = Path(__file__).resolve().parent.parent / "outputs/fixtures"


def test_e2e_codes_match():
    lda = np.load(_F / "lda.npy")  # exact oracle speaker condition (isolates the AR loop)
    out = synthesize(
        "The quick brown fox jumps over the lazy dog.",
        speaker_lda=lda,
        greedy=True,
        seed=0,
        max_new_tokens=1024,
        return_codes=True,
    )
    ref = np.load(_F / "delayed_codes.npy")  # (262, 9)
    n = min(len(out.codes), len(ref))
    frac = (out.codes[:n] == ref[:n]).mean()
    print("frames:", len(out.codes), "vs", len(ref), "frac_exact:", frac)
    # first divergence (for debugging)
    if frac < 1.0:
        d = np.where((out.codes[:n] != ref[:n]).any(axis=1))[0]
        print("first divergent frame:", int(d[0]) if len(d) else None)
    assert frac >= 0.99
