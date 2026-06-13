# zonos2-mlx

A pure-[MLX](https://github.com/ml-explore/mlx) port of Zyphra's **ZONOS2 8B-MoE**
text-to-speech model, running natively on Apple Silicon. Clone a voice from a few
seconds of reference audio and synthesize speech entirely on-device — no PyTorch
at inference time.

- **8B DiT** with a 16-expert top-1 MoE (layers 3–26), ECAPA-TDNN speaker
  conditioning, and the DAC 44.1 kHz neural codec.
- **Pure-MLX engine** (`src/zonos2_mlx/*` imports no torch). torchaudio is used
  only in `scripts/` for the optional reference-enrolment mel front-end.
- **Three weight tiers** — bf16 (64 GB Macs), int8 (~13 GB), int4-ship (~10.6 GB,
  runs on 16 GB Macs). All three produce full intelligible audio.

---

## Install

The engine deps are MLX + numpy + safetensors + soundfile. The `[oracle]` extra
(torch/torchaudio/transformers) is needed **only** to dump reference fixtures or
to enrol a voice from raw audio via `--ref`.

```bash
# engine only (synthesize from a pre-enrolled .zonos profile)
uv sync

# + the oracle/enrol extras (torchaudio mel for --ref, or fixture regeneration)
uv sync --extra oracle
```

Weights go under `weights/` (see [Weight tiers](#weight-tiers)).

---

## Usage

### Enrol a voice (once)

`--ref` takes any audio clip and caches a 1024-d LDA speaker vector as a `.zonos`
profile (torchaudio mel → MLX ECAPA-TDNN → LDA; identical to the oracle path):

```bash
uv run python scripts/zonos2_enroll.py \
    --ref outputs/fixtures/ref.wav \
    --out outputs/voices/myvoice.zonos
```

### Synthesize

The CLI mirrors the dots/miso flag style. Give it `--text`, exactly one speaker
source (`--ref` to enrol on the fly, or `--profile` for a cached `.zonos`), an
`--out` wav, and a `--quant` tier:

```bash
# enrol + synthesize at int8
uv run python scripts/zonos2_cli.py \
    --text "The quick brown fox jumps over the lazy dog." \
    --ref outputs/fixtures/ref.wav \
    --out outputs/cli/fox.wav \
    --quant int8

# reuse a cached profile at the int4 ship tier (16 GB Macs)
uv run python scripts/zonos2_cli.py \
    --text "Hello there." \
    --profile outputs/voices/myvoice.zonos \
    --out outputs/cli/hello.wav \
    --quant int4

# full-precision bf16 (64 GB Macs)
uv run python scripts/zonos2_cli.py \
    --text "Top quality, please." \
    --profile outputs/voices/myvoice.zonos \
    --out outputs/cli/top.wav \
    --quant bf16
```

Useful flags: `--speaking-rate <0..7|-1>`, `--accurate-mode` / `--expressive`,
`--seed`, `--max-new-tokens`, `--sample` (stochastic; default is greedy/parity).

Programmatic use:

```python
from zonos2_mlx.pipeline import synthesize

res = synthesize(
    "The quick brown fox jumps over the lazy dog.",
    profile="outputs/voices/myvoice.zonos",
    weights_dir="weights/zonos2-int8",
    dac_dir="weights/zonos2-bf16/dac_44khz",  # DAC is tier-independent
    out_wav="outputs/cli/fox.wav",
)
print(res.wav.shape, res.eos_frame)
```

---

## Weight tiers

The quantized dirs ship only the trunk safetensors; the **DAC codec** and
**speaker encoder** are tier-independent and always come from the bf16 dir (the
CLI wires this for you).

| Tier | dir | size | peak RAM | target Macs | HF location |
|---|---|---|---|---|---|
| bf16 | `weights/zonos2-bf16` | ~15 GB | ~44 GB | 64 GB | `shraey/zonos2-mlx` → `zonos2-bf16/` |
| int8 | `weights/zonos2-int8` | ~8 GB | ~13 GB | 32 GB | `shraey/zonos2-mlx` → `zonos2-int8/` |
| int4-ship | `weights/zonos2-int4-ship` | ~5.8 GB | ~10.6 GB | 16 GB | `shraey/zonos2-mlx` → `zonos2-int4-ship/` |

Download (after `hf auth login`):

```bash
hf download shraey/zonos2-mlx --repo-type model --local-dir weights/
```

Uploading the weights (maintainers) is staged in `scripts/zonos2_hf_upload.sh`.

---

## Parity methodology

The port is validated **component-by-component** against a vendored plain-PyTorch
reference (`scripts/zonos2_oracle/`), dumped on MPS/bf16. Each task has a gate:

- **MoE trunk (T2):** per-token median cosine ≥0.999 vs the oracle + last-position
  multi-codebook argmax byte-exact.
- **DAC decoder (T3):** waveform PSNR **73.07 dB** (sample-exact).
- **Speaker (T4):** ECAPA x-vector cosine ≈1.0, LDA projection 0.9997.
- **Text frontend (T5):** `build_prompt` tensor byte-exact.
- **KV cache (T6):** cached vs full-forward cosine ≥0.9999, argmax exact.

**The AR trunk is gated teacher-forced + by-ear**, not by free-running
cross-framework token-match. On this 8B **top-1** MoE, a few tokens sit on a
router knife-edge where MPS-bf16 and Metal-bf16 round the gate logits just
differently enough to pick a different expert; greedy decoding compounds each flip,
so the free-running cross-framework code-match is inherently ~0.03 — a property of
the model class, **not** a port bug (the KV cache, argmax, and per-component parity
are all proven independently, and fp32 doesn't help — bf16 is the
oracle-matching dtype). Teacher forcing removes the divergence confound and gives
**frame-0 all-9 argmax exact** + per-codebook agreement **0.872**. The free-running
render was approved by ear (a good clone — same words, same voice).

Full numbers: [`docs/research/01-port-results.md`](docs/research/01-port-results.md).
Oracle notes: [`docs/research/00-oracle-notes.md`](docs/research/00-oracle-notes.md).
MoE-quant recipe: [`docs/research/02-moe-quant-research.md`](docs/research/02-moe-quant-research.md).

### Tests

```bash
uv run pytest -q -m "not gpu"   # CPU suite (fast; no weights needed)
uv run pytest -q -m gpu         # GPU parity gates (need weights + fixtures, run serial)
```

---

## Quantization

The int8 and int4-ship tiers use a **MoE-aware recipe** (the expert quantization
matches mlx-lm's `QuantizedSwitchLinear`/`gather_qmm` exactly; the non-expert path
is the sensitive one). Highlights:

- Router (down_proj, mlp, balancing biases, EDA states-scale/rmsnorm): **bf16** —
  a top-1 router is a knife-edge.
- Attention + dense-FFN linears: **int8** (they feed the recurrent router state).
- Experts gate/up (the memory bulk): **int4**; expert down-proj + `lm_head`:
  **int8** (protects the EOA logit + the residual write-back).
- Embeddings + all RMSNorm: **bf16**.

Tiers are gated on **per-layer hidden cosine + teacher-forced logit-KL +
finite/non-silent audio + ear**, not on sampled-token agreement (which is a
top-1 knife-edge with a ~0.85 noise floor). Measured: int8 KL 0.070, int4-ship KL
0.202. See [`docs/research/02-moe-quant-research.md`](docs/research/02-moe-quant-research.md).

---

## License & attribution

**Apache-2.0**, with attribution to **[Zyphra](https://www.zyphra.com/)** for the
ZONOS2 model and weights. This is an independent MLX port.

- **Model & weights:** ZONOS2 by Zyphra.
- **Porting reference:** the [Zonos2_TTS-ComfyUI](https://github.com/Saganaki22/Zonos2_TTS-ComfyUI)
  fork by **Saganaki22**, used as the op-for-op PyTorch reference for this port.
- **MLX:** Apple's [ml-explore/mlx](https://github.com/ml-explore/mlx).

Use of the underlying ZONOS2 model is subject to Zyphra's model license; comply
with their terms in addition to this repository's Apache-2.0 license.
