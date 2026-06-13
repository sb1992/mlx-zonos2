"""GPU parity gate for the MLX Zonos2 MoE transformer trunk.

Loads the 8B bf16 checkpoint and compares per-layer hidden states + the
multi-codebook logits against the torch oracle fixtures (MPS / bf16).
Target: cosine >= 0.999 per captured layer and for the logits.
"""

import numpy as np
import mlx.core as mx
import pytest
from pathlib import Path

from zonos2_mlx.model import Zonos2Model

_R = Path(__file__).resolve().parent.parent


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@pytest.mark.gpu
def test_trunk_parity():
    f = _R / "outputs/fixtures"
    m = Zonos2Model.from_pretrained(str(_R / "weights/zonos2-bf16"))
    ids = mx.array(np.load(f / "prompt_ids.npy"))
    spk = mx.array(np.load(f / "lda.npy"))
    pos = int(np.load(f / "speaker_position.npy"))
    h = m.forward(ids, speaker_lda=spk, speaker_position=pos, capture_layers=[0, 3, 13, 26, 27])
    for L in (0, 3, 13, 26, 27):
        mine = np.array(h.layers[L].astype(mx.float32))
        gold = np.load(f / f"hidden_L{L}.npy")
        c = _cos(mine, gold)
        # Per-row diagnostic: localizes MoE routing-boundary flips (a handful
        # of borderline tokens whose argmax differs from the MPS-bf16 oracle).
        rows = np.array([_cos(mine[i], gold[i]) for i in range(mine.shape[0])])
        bad = np.where(rows < 0.99)[0]
        keep = [i for i in range(mine.shape[0]) if i not in bad]
        excl = _cos(mine[keep], gold[keep]) if keep else float("nan")
        print(f"L{L} cos={c:.5f}  rows>=0.999={int((rows >= 0.999).sum())}/{mine.shape[0]}"
              f"  boundary_rows={bad.tolist()}  cos_excl_boundary={excl:.5f}")
        assert c >= 0.999, f"layer {L} cos {c}"
    logits = m.head(h.last)
    c = _cos(np.array(logits.astype(mx.float32)), np.load(f / "logits.npy"))
    print(f"logits cos={c:.5f}")
    assert c >= 0.999
