# zonos2-mlx

A pure-[MLX](https://github.com/ml-explore/mlx) port of Zyphra's [**ZONOS2**](https://github.com/Zyphra/ZONOS2) — an **8B-parameter Mixture-of-Experts** autoregressive text-to-speech model — running natively on Apple Silicon.

ZONOS2 is a **16-expert, top-1 MoE** AR model (layer 26 routes top-2), paired with a DAC 44.1 kHz neural codec and an ECAPA-TDNN speaker encoder. It clones a voice from a few seconds of reference audio and synthesizes 44.1 kHz speech. This repo is a clean-room MLX reimplementation of the **inference runtime** — no PyTorch in the inference path — gated per-stage against the original model.

> **Ready-to-run weights** are published on Hugging Face — [`shraey/zonos2-mlx`](https://huggingface.co/shraey/zonos2-mlx). **Download and run — no PyTorch and no conversion step.** Three precision tiers ship (`bf16` / `int8` / `int4`); the quantized builds shrink the 14 GB bf16 trunk to **7.6 GB (int8)** or **5.4 GB (int4)**. `uv sync` the runtime, `hf download` the weights, and go — see [Weights](#weights).

## What it is

- **Architecture:** an 8B autoregressive MoE trunk (16 experts, top-1; layer 26 top-2) → multi-codebook audio tokens → the **DAC 44.1 kHz** neural codec for the waveform. An **ECAPA-TDNN** speaker encoder (+ an LDA projection) conditions the speaker identity.
- **Voice cloning:** one reference clip clones the voice — enroll it once into a `.zonos` profile, or pass `--ref` to enroll on the fly.
- **Pure-MLX runtime:** nothing in `src/zonos2_mlx/*` imports torch; the inference math is all MLX. `torchaudio` appears only in `scripts/` for the reference-enrollment mel front-end (and the dev parity oracle).
- **CLI + Python API:** a `scripts/zonos2_cli.py` front-end and `from zonos2_mlx import synthesize`.

## Scope — what this is / isn't

This is a **converted-weight MLX inference runtime** for ZONOS2. It deliberately does **not** replicate upstream's full surface. It **is**:

- a from-scratch MLX port of the ZONOS2 inference math, numerically gated against the original PyTorch model (the porting oracle is the clean plain-torch ComfyUI fork, vendored read-only under `scripts/zonos2_oracle/zonos2_ref/`);
- a CLI + Python API that synthesizes from a **local, already-converted** weights directory;
- a small **runtime addition not present upstream**: [enroll a voice once](#enroll-once-reuse-a-voice) into a `.zonos` profile and reuse it, so the reference encode is paid once.

It is **not** a drop-in replacement for the upstream package. In particular, this runtime:

- **points at a local converted directory** — there is no HF hub fetch baked into the runtime (you `hf download` the weights yourself, then point `--model-dir` at the downloaded tier folder, or use `--quant` against the in-repo `weights/zonos2-*` dev layout);
- **decodes greedily by default** — greedy (temperature 0) is the parity path; `--sample` selects the stochastic oracle sampler;
- **does inference only** — no fine-tuning or training.

If you need anything outside that, use the upstream project: [Zyphra/ZONOS2](https://github.com/Zyphra/ZONOS2).

## Install

Requires Python ≥ 3.12 on Apple Silicon (MLX is Metal-only). This repo uses [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/sb1992/mlx-zonos2.git
cd mlx-zonos2

# engine only — synthesize from a cached .zonos profile
uv sync

# + the [oracle] extra — torch/torchaudio/transformers, needed ONLY to enroll a
# voice from raw audio (--ref) or to regenerate the dev parity fixtures
uv sync --extra oracle
```

The runtime deps are `mlx`, `numpy`, `safetensors`, `soundfile`. The `[oracle]` extra (`torch`, `torchaudio`, `transformers`, …) is needed **only** for two things — don't conflate them:

- **Enrolling from raw audio** (`--ref` on the CLI, or `scripts/zonos2_enroll.py`) needs `torchaudio` for the mel front-end. Once a voice is enrolled into a `.zonos` profile, generation from `--profile` is **pure-MLX** and needs no extra.
- **Regenerating parity fixtures** (the dev oracle under `scripts/zonos2_oracle/`) needs the full extra to dump reference tensors from the original PyTorch model.

## Weights

Two ways to get runnable MLX weights — **most people want Option A.**

### Option A — download ready MLX weights (recommended)

Pre-converted, pre-quantized MLX weights are published at [`shraey/zonos2-mlx`](https://huggingface.co/shraey/zonos2-mlx). **No PyTorch, no conversion.** Each tier folder is **self-contained** (trunk + the DAC codec + the speaker encoder), so download one tier and point `--model-dir` at it:

```bash
# int8 (~7.9 GB) — the balanced tier
hf download shraey/zonos2-mlx --include "int8/*" --local-dir ./zonos2-mlx-weights
python scripts/zonos2_cli.py --model-dir ./zonos2-mlx-weights/int8 \
    --text "The quick brown fox jumps over the lazy dog." \
    --ref ref.wav --out out.wav

# int4 (~5.7 GB) — fits 16 GB Macs; same flow, swap the folder:
hf download shraey/zonos2-mlx --include "int4/*" --local-dir ./zonos2-mlx-weights
python scripts/zonos2_cli.py --model-dir ./zonos2-mlx-weights/int4 \
    --text "Hello there." --ref ref.wav --out out.wav
```

`--ref` enrolment needs the `[oracle]` extra (torchaudio for the mel); a cached `--profile` does not. To grab all tiers at once, drop the `--include` filter.

**Tiers** (each HF folder `bf16/`, `int8/`, `int4/` is self-contained — trunk + `dac_44khz/` + `speaker_encoder/`):

| Folder | what's quantized | folder size | peak RAM | target Macs |
|---|---|---|---|---|
| `bf16/` | nothing (reference) | ~14 GB | ~44 GB | 64 GB |
| `int8/` | attention/FFN/lm_head + experts int8; router/embeddings/norms bf16 | ~7.9 GB | ~13 GB | 32 GB |
| `int4/` | attention/FFN/lm_head int8; experts gate/up int4, down int8; router/embeddings/norms bf16 | ~5.7 GB | ~10.6 GB | 16 GB |

<sub>Folder size includes the bundled ~315 MB DAC codec + ECAPA speaker encoder (shared, identical across tiers — Hugging Face Xet de-dups them).</sub>

The MoE experts (the bulk of the 8B) carry the int4; the **router/gate**, the **`lm_head`**, and the sensitive expert **`down`** projection stay int8/bf16 — the MoE-quant recipe that keeps the model intact. All three tiers produce **full, intelligible audio**. They're equal options — pick by the RAM you have.

### Option B — convert / quantize from source (advanced)

For reproducibility, re-quantizing, or auditing, build the tiers yourself from the bf16 trunk (the quantizer needs only `mlx` — no torch):

```bash
# int8 — everything int8 (the conservative reference tier)
python -m zonos2_mlx.quantize --weights-dir weights/zonos2-bf16 \
    --tier int8 --out weights/zonos2-int8/zonos2-int8.safetensors

# int4 — int8 linears + int8 expert-down, int4 expert gate/up
python -m zonos2_mlx.quantize --weights-dir weights/zonos2-bf16 \
    --tier int4 --out weights/zonos2-int4/zonos2-int4.safetensors
```

`--tier` writes a `quant_config.json` sidecar beside the safetensors; the loader globs `*.safetensors` and reads the sidecar, so nothing changes at the CLI/API level. (The per-projection knobs `--bits` / `--expert-gate-up-bits` / `--expert-down-bits` are exposed for experiments.) Staging the three-tier upload tree is in `scripts/zonos2_hf_upload.sh`.

## CLI usage

```bash
python scripts/zonos2_cli.py \
    --text "The quick brown fox jumps over the lazy dog." \
    --ref ref.wav \
    --out outputs/cli/fox.wav \
    --quant int8
# -> outputs/cli/fox.wav  (44.1 kHz)
```

Key flags:

- `--text` — the text to synthesize (required).
- `--ref <audio>` **or** `--profile <voice.zonos>` (exactly one) — `--ref` enrolls a reference clip on the fly (needs the `[oracle]` extra for the mel); `--profile` reuses a cached `.zonos` voice (pure-MLX).
- `--out <wav>` — output path (required).
- `--quant {bf16,int8,int4}` — precision tier (default `int8`) against the in-repo `weights/zonos2-*` dev layout.
- `--model-dir <dir>` — a self-contained MLX weights folder (e.g. one tier of an `hf download`, containing the trunk + `dac_44khz/` + `speaker_encoder/`). Overrides `--quant`; this is the path for downloaded weights.
- `--speaking-rate <0..7|-1>` — speaking-rate bucket, or `-1` (unset, the default).
- `--accurate-mode` (default) / `--expressive` — accurate-mode token on/off.
- `--seed <n>` — sampling seed (only affects `--sample`; greedy is deterministic).
- `--max-new-tokens <n>` — AR decode cap in frames (default 1024).
- `--sample` — stochastic sampling (the oracle's `SamplingOptions`); default is greedy (the parity path).

### Long-form (longer than one pass)

A single pass is capped at the model's ~60s context. For longer text, `--long`
splits on sentence boundaries, **packs** several sentences into ~40s chunks,
generates each with the same voice, and concatenates them:

```bash
python scripts/zonos2_cli.py --long \
    --text "$(cat article.txt)" \
    --profile outputs/voices/alice.zonos \
    --out outputs/cli/article.wav \
    --max-seconds 40 --gap-ms 80
```

`--max-seconds` is the per-chunk target (estimated audio seconds; default 40,
kept safely under the ~60s ceiling), `--gap-ms` the silence between chunks
(default 80). Chunks are independent — the same speaker vector conditions each —
so memory stays at the single-pass footprint regardless of total length. For
non-Latin text pass `--language ZH`/`HI`/… (or tune `--chars-per-sec`) so the
chunk-size estimate matches the script.

## Python API

```python
from zonos2_mlx import synthesize

res = synthesize(
    "The quick brown fox jumps over the lazy dog.",
    profile="outputs/voices/myvoice.zonos",
    weights_dir="weights/zonos2-int8",
    dac_dir="weights/zonos2-bf16/dac_44khz",   # DAC + speaker encoder are tier-independent
    out_wav="outputs/cli/fox.wav",
)

print(res.wav.shape, res.sample_rate, res.eos_frame)   # (1, samples), 44100, <eos frame>
```

`synthesize(...)` takes exactly one speaker source — `profile=` (a cached `.zonos`), `ref=` (a precomputed log-mel array), or `speaker_lda=` (a 1024-d vector) — plus `greedy=` (default `True`, the parity path), `seed=`, `max_new_tokens=`, and the oracle sampling `**knobs`.

## Enroll once, reuse a voice

Compute a voice's speaker conditioning **once**, save it to a small `.zonos` profile, and reuse it for every later generation — so you never re-pass (or re-encode) the reference. The enroll path runs the torchaudio mel → MLX ECAPA-TDNN → LDA projection (identical to the parity oracle's path); generation from the profile is then pure-MLX.

```bash
# 1. enroll a reference voice -> a reusable .zonos profile
python scripts/zonos2_enroll.py --ref ref.wav --out outputs/voices/alice.zonos

# 2. generate from the profile — no --ref (and no torchaudio) needed
python scripts/zonos2_cli.py --profile outputs/voices/alice.zonos \
    --text "Hello from the enrolled voice." --out outputs/cli/hello.wav --quant int8
```

```python
from zonos2_mlx.speaker import SpeakerProfile
profile = SpeakerProfile.load("outputs/voices/alice.zonos")
res = synthesize("Hello from the enrolled voice.", profile=profile,
                 weights_dir="weights/zonos2-int8",
                 dac_dir="weights/zonos2-bf16/dac_44khz", out_wav="hello.wav")
```

A profile carries the 1024-d LDA speaker vector and is **portable across tiers** — enroll once, generate at bf16 / int8 / int4.

## How it was ported / parity

Every stage was gated numerically against the original PyTorch model (a dev-only oracle under `scripts/zonos2_oracle/` dumps reference fixtures on MPS/bf16) before any behavioral test:

| Stage | Metric | Result |
|---|---|---|
| MoE trunk (per-layer hidden) | cosine vs torch oracle | ≥0.999 |
| MoE trunk (final logits, last pos) | greedy argmax | byte-exact |
| DAC 44.1 kHz decode | PSNR vs torch | 73.07 dB |
| ECAPA speaker embedding | cosine | 1.0000 |
| Speaker LDA projection | cosine | 0.9997 |
| Text `build_prompt` | token ids | byte-exact |
| KV-cache (prefill vs step) | cosine | ≥0.9999 |

**Why the end-to-end gate is teacher-forced + by-ear, not free-running token-match.** The AR trunk is gated **teacher-forced and by ear**, not by free-running cross-framework code-match. On an 8B **top-1** MoE, a few tokens sit on a routing knife-edge where MLX-Metal-bf16 and the PyTorch-MPS-bf16 oracle legitimately round the gate logits differently enough to pick a *different* expert. Under free-running greedy decoding each such flip compounds, so the end-to-end code-match is inherently low (~0.03) — even though the port is numerically correct: **frame-0 all-9 argmax is exact**, and **teacher-forced per-codebook agreement is 0.872**. This is a cross-framework runtime-precision property of the model class, **not** a port bug (fp32 doesn't fix it — bf16 is the oracle-matching dtype; the KV cache, argmax, and per-component parity are all proven independently). The component gates prove the math; teacher-forced + by-ear prove the product.

Tests live in `tests/` (pytest). The CPU set runs without weights; the GPU parity gates need the trunk weights + fixtures and run serially:

```bash
uv run pytest -q -m "not gpu"   # CPU suite (fast; no GPU, no 8B inference)
uv run pytest -q -m gpu         # GPU parity gates (need weights + fixtures)
```

## Requirements & notes

These are specifics of *this MLX port* — not limitations of the model itself, which behaves the same as upstream ZONOS2.

- **Apple Silicon only** — MLX is Metal-only (the upstream PyTorch model targets CUDA/MPS).
- **Speed:** ~**1.06 s per audio-second** at int8 (greedy, M5 Max).
- **Footprint:** peak RAM is tier-dependent — ~44 GB (bf16, 64 GB Macs), ~13 GB (int8, 32 GB Macs), ~10.6 GB (int4, 16 GB Macs). All scripts and the pipeline set `mx.set_memory_limit(45 GB)` as a hard ceiling.
- **Shared assets:** the DAC codec and speaker encoder are tier-independent and live in the bf16 folder; the CLI wires them automatically (see [Weights](#weights)).

## Responsible use

This runtime performs **voice cloning** — it can reproduce a person's voice from a few seconds of reference audio. That capability carries real risk of misuse. By using this software you agree to use it responsibly:

- **No impersonation, fraud, or disinformation.** Do not use cloned voices to impersonate real people without authorization, to commit fraud or social engineering, to evade voice-based authentication, or to produce misleading or deceptive content.
- **Consent for reference audio.** Only clone a voice you own or for which you have the speaker's explicit, informed consent. Respect applicable privacy / publicity / data-protection laws in your jurisdiction.
- **Disclose AI-generated audio.** Clearly label synthesized speech as AI-generated wherever it is published or shared, so listeners are never misled about its origin.
- **Watermark + detect downstream.** You are encouraged to apply audio watermarking to generated output and to deploy synthetic-speech detection in any pipeline that ingests it, to support provenance and abuse mitigation.

The authors and contributors disclaim responsibility for misuse. Comply with all applicable laws and with the upstream ZONOS2 / Zyphra usage terms.

## Attribution + licenses

This is a derivative port. The original model and the components it builds on are each independently licensed:

- **ZONOS2** — **Apache-2.0**, © **[Zyphra](https://www.zyphra.com/)**. The 8B-MoE model, the DAC 44.1 kHz codec, and the speaker encoder are by Zyphra. [Code](https://github.com/Zyphra/ZONOS2)
- **Released checkpoint** — this port converts the **`drbaph/ZONOS2-BF16`** release, whose speaker encoder is an **ECAPA-TDNN** (2048-d). Note: Zyphra's current upstream HEAD wraps a different (Qwen3-style) speaker path — so this port tracks *that BF16 release*, not necessarily HEAD. The trunk, DAC codec, and AR math are the same model.
- **Porting oracle** — the clean plain-torch [Zonos2_TTS-ComfyUI](https://github.com/Saganaki22/Zonos2_TTS-ComfyUI) fork by **Saganaki22** (Apache-2.0), vendored read-only under `scripts/zonos2_oracle/zonos2_ref/` and used as the op-for-op reference for this port.
- **MLX** — Apple's [ml-explore/mlx](https://github.com/ml-explore/mlx).

The MLX port code in this repository is licensed **Apache-2.0** (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). You must comply with the upstream ZONOS2 license and usage terms for the model weights and any redistributed components.

### Credit

Full credit to **Zyphra** for the ZONOS2 model, its training, and the open release. This repo only re-expresses their runtime in MLX; the research, training, and weights are theirs.
