"""GPU parity gate for the MLX Zonos2 MoE transformer trunk.

Loads the 8B bf16 checkpoint and compares per-layer hidden states + the
multi-codebook logits against the torch oracle fixtures (MPS / bf16).

The gate is MoE-bf16-aware: whole-tensor cosine is NOT used as the top-level
assertion (see test_trunk_parity docstring for details).
"""

import numpy as np
import mlx.core as mx
from pathlib import Path

from zonos2_mlx.model import Zonos2Model

_R = Path(__file__).resolve().parent.parent
_F = _R / "outputs/fixtures"


def _per_token_cos(a, b):
    """Cosine per token (row); a,b are (T, D)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return num / den


def test_trunk_parity():
    """Parity gate for the 8B MoE trunk vs the torch (MPS bf16) oracle.

    Whole-tensor cosine is NOT used as the gate: in a bf16 MoE a handful of tokens
    sit on routing knife-edges where MLX-bf16 and MPS-bf16 legitimately pick
    different experts (proven: an fp64 router agrees with MLX, not the oracle; fp32
    flips a DIFFERENT set — bf16 is the oracle-matching dtype). Those tokens can
    have per-token cosine as low as ~0.07 at affected layers (L26/27), tanking
    whole-tensor cosine to ~0.94 without affecting the actual decode. A REAL bug,
    by contrast, perturbs ALL tokens. So we gate on:
      (1) the decode-relevant last-position greedy argmax matching the oracle exactly,
      (2) median per-token cosine >= 0.999 per layer (all knife-edge layers still
          hit >= 0.9995 — a real bug tanks median too), and
      (3) >= 90% of tokens having per-token cosine >= 0.99 per layer (catches real
          bugs; knife-edge layers measure ~92% so there is margin above the gate).

    Calibrated thresholds (measured 2026-06-14, 66 tokens, MPS bf16 oracle):
      L0:  median=0.9996  frac>=0.99=1.000
      L3:  median=0.9998  frac>=0.99=0.985
      L13: median=0.9999  frac>=0.99=0.970
      L26: median=0.9999  frac>=0.99=0.924   (5 knife-edge tokens)
      L27: median=0.9999  frac>=0.99=0.924   (same 5 knife-edge tokens)
    """
    m = Zonos2Model.from_pretrained(str(_R / "weights/zonos2-bf16"))
    ids = mx.array(np.load(_F / "prompt_ids.npy"))
    spk = mx.array(np.load(_F / "lda.npy"))
    pos = int(np.load(_F / "speaker_position.npy"))
    h = m.forward(ids, speaker_lda=spk, speaker_position=pos, capture_layers=[0, 3, 13, 26, 27])

    for L in (0, 3, 13, 26, 27):
        got = np.array(h.layers[L].astype(mx.float32)).reshape(-1, 2048)
        ref = np.load(_F / f"hidden_L{L}.npy").reshape(-1, 2048)
        pc = _per_token_cos(got, ref)
        frac = float((pc >= 0.99).mean())
        med = float(np.median(pc))
        print(
            f"L{L}: median_cos={med:.5f}  frac>=0.99={frac:.3f}  min={pc.min():.4f}"
            f"  bad_indices={(pc < 0.99).nonzero()[0].tolist()}"
        )
        assert med >= 0.999, f"layer {L} median per-token cosine {med:.6f} < 0.999 (real bug?)"
        assert frac >= 0.90, f"layer {L} only {frac:.2%} tokens >= 0.99 cosine (real bug?)"

    # THE decode gate: greedy argmax of the last-position multi-codebook logits == oracle.
    logits = np.array(m.head(h.last).astype(mx.float32)).reshape(9, 1026)
    ref_logits = np.load(_F / "logits.npy").reshape(9, 1026)
    got_arg = logits.argmax(-1)
    ref_arg = ref_logits.argmax(-1)
    print(f"last-pos argmax MLX={got_arg.tolist()}  oracle={ref_arg.tolist()}")
    assert (got_arg == ref_arg).all(), "last-position greedy argmax diverges from oracle"
