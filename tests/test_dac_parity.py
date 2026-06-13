"""Tests for the MLX DAC 44.1 kHz decoder.

test_shear_up  — pure CPU / numpy, no GPU required.
test_dac_decode_psnr — loads weights and runs inference; requires GPU + fixtures.
"""

import numpy as np

from zonos2_mlx.dac import shear_up


# ---------------------------------------------------------------------------
# CPU tests
# ---------------------------------------------------------------------------

def test_shear_up():
    x = np.arange(12 * 9, dtype=np.int64).reshape(12, 9)
    out = shear_up(x, pad_id=1025)

    # codebook 0 is un-delayed (no shift)
    assert (out[:, 0] == x[:, 0]).all(), "codebook 0 should be unchanged"

    # codebook 1 shifted up by 1: out[i,1] == x[i+1,1] for i in 0..10
    assert (out[:-1, 1] == x[1:, 1]).all(), "codebook 1 not shifted up by 1"
    assert out[-1, 1] == 1025, "tail of codebook 1 should be pad_id"

    # codebook 8 shifted up by 8: out[i,8] == x[i+8,8] for i in 0..3
    assert (out[:-8, 8] == x[8:, 8]).all(), "codebook 8 not shifted up by 8"
    assert (out[-8:, 8] == 1025).all(), "tail 8 rows of codebook 8 should be pad_id"


def test_shear_up_shape_preserved():
    x = np.zeros((20, 9), dtype=np.int64)
    out = shear_up(x, pad_id=-1)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# GPU parity test
# ---------------------------------------------------------------------------

def test_dac_decode_psnr():
    from pathlib import Path

    _R = Path(__file__).resolve().parent.parent

    from zonos2_mlx.dac import Dac44k

    dac = Dac44k.from_pretrained(str(_R / "weights/zonos2-bf16/dac_44khz"))

    codes = np.load(_R / "outputs/fixtures/delayed_codes.npy")
    eos = int(np.load(_R / "outputs/fixtures/eos_frame.npy"))
    ref = np.load(_R / "outputs/fixtures/audio.npy").squeeze()

    audio = np.asarray(dac.decode(codes, pad_id=1025, eos_frame=eos)).squeeze()

    n = min(len(audio), len(ref))
    a = audio[:n].astype(np.float64)
    r = ref[:n].astype(np.float64)

    mse = np.mean((a - r) ** 2)
    signal_range = r.max() - r.min()
    psnr = 20 * np.log10(signal_range / np.sqrt(mse))

    print(f"DAC PSNR={psnr:.2f} dB  (got {len(audio)} vs ref {len(ref)} samples)")
    print(f"  sample-count match: {len(audio) == len(ref)}")

    assert psnr >= 40.0, f"PSNR {psnr:.2f} dB is below 40 dB gate"
