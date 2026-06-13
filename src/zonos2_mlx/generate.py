"""Autoregressive audio-code generation for the MLX Zonos2 trunk.

Mirrors ``Zonos2_TTS-ComfyUI/native.py`` op-for-op:
  * ``SamplingOptions``                 (native.py 998-1008)
  * ``_apply_repetition_penalty``        (native.py 1011-1043)
  * ``sample_codes``                     (native.py 1046-1097)
  * ``generate_audio_codes`` (AR loop)   (native.py 1101-1206)

The trunk forward runs on-GPU (bf16, ~15 GB); the sampling head-work is plain
host-side scalar/numpy on the small ``(9, 1026)`` logits + the int history.
Per-step ``mx.eval`` keeps the lazy graph + the KV cache from growing unbounded.

NO torch. numpy is fine (host-side sampling only).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

mx.set_memory_limit(int(45 * (1 << 30)))


# ---------------------------------------------------------------------------
# Sampling options (native.py SamplingOptions 998-1008)
# ---------------------------------------------------------------------------
@dataclass
class SamplingOptions:
    max_new_tokens: int = 1024
    temperature: float = 1.15
    top_k: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    repetition_codebooks: int = 8
    seed: int = 0


# ---------------------------------------------------------------------------
# Repetition penalty (native.py _apply_repetition_penalty 1011-1043)
# ---------------------------------------------------------------------------
def apply_repetition_penalty(
    logits: np.ndarray,
    generated: list[np.ndarray],
    options: SamplingOptions,
) -> np.ndarray:
    """Divide-if-positive / multiply-if-negative the logits of recently seen
    tokens, per codebook. Operates on a numpy ``(1, n_codebooks, vocab)`` array
    (or ``(n_codebooks, vocab)`` — both accepted) and returns a fresh array.

    ``generated`` is the host-side list of per-step ``(n_codebooks,)`` int codes.
    The penalty is applied BEFORE the greedy argmax, so it shifts argmax even at
    temperature 0 (the critical detail — native.py 1052).
    """
    if (
        options.repetition_window <= 0
        or options.repetition_penalty <= 1.0
        or not generated
    ):
        # Disabled / no history: still honour the "returns a fresh array" contract.
        return np.array(logits, dtype=np.float32, copy=True)

    result = np.array(logits, dtype=np.float32, copy=True)
    # Normalise to (1, n_codebooks, vocab) for indexing parity with the oracle.
    squeezed = result.ndim == 2
    if squeezed:
        result = result[None]

    n_codebooks = result.shape[1]
    vocab = result.shape[-1]
    cb_count = (
        n_codebooks
        if options.repetition_codebooks < 0
        else min(n_codebooks, options.repetition_codebooks)
    )

    recent = np.stack(generated[-options.repetition_window:]).astype(np.int64)  # (W, n_cb)
    penalty = float(options.repetition_penalty)
    for codebook in range(cb_count):
        ids = np.unique(recent[:, codebook])
        ids = ids[(ids >= 0) & (ids < vocab)]
        if ids.size == 0:
            continue
        values = result[0, codebook, ids]
        adjusted = np.where(values > 0, values / penalty, values * penalty)
        result[0, codebook, ids] = adjusted

    return result[0] if squeezed else result


# ---------------------------------------------------------------------------
# sample_codes (native.py 1046-1097)
# ---------------------------------------------------------------------------
def sample_codes(
    logits,
    generated: list[np.ndarray],
    options: SamplingOptions,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample one token per codebook from the last-position logits.

    ``logits`` is the trunk head output, ``(1, n_codebooks, vocab)`` or
    ``(n_codebooks, vocab)`` — an mx.array or numpy array. The repetition penalty
    is ALWAYS applied first (even greedy). Returns ``(n_codebooks,)`` int64.

    Greedy (``temperature <= 1e-5``) is the gate path: argmax after penalty.
    The stochastic path mirrors native.py 1056-1097 (top_k -> softmax -> top_p
    -> min_p -> renorm -> multinomial); it is NOT exercised by the parity gate.
    """
    if isinstance(logits, mx.array):
        # bf16 (and other non-numpy dtypes) must be cast to float32 first —
        # numpy has no bf16, so a direct buffer conversion errors.
        arr = np.array(logits.astype(mx.float32))
    else:
        arr = np.asarray(logits)
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    arr = apply_repetition_penalty(arr, generated, options)  # (1, n_cb, vocab)

    if options.temperature <= 1e-5:
        return arr.argmax(axis=-1)[0].astype(np.int64)  # (n_codebooks,)

    logits2 = arr / max(options.temperature, 1e-8)
    vocab = logits2.shape[-1]
    work = logits2[0]  # (n_cb, vocab)

    if 0 < options.top_k < vocab:
        kth = np.partition(work, vocab - options.top_k, axis=-1)[:, vocab - options.top_k]
        work = np.where(work < kth[:, None], -np.inf, work)

    work = work - work.max(axis=-1, keepdims=True)
    probs = np.exp(work)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    if 0.0 < options.top_p < 1.0:
        order = np.argsort(-probs, axis=-1)
        sorted_probs = np.take_along_axis(probs, order, axis=-1)
        cumulative = np.cumsum(sorted_probs, axis=-1)
        remove = (cumulative - sorted_probs) > options.top_p
        sorted_probs = np.where(remove, 0.0, sorted_probs)
        new_probs = np.zeros_like(probs)
        np.put_along_axis(new_probs, order, sorted_probs, axis=-1)
        probs = new_probs

    if options.min_p > 0:
        maximum = probs.max(axis=-1, keepdims=True)
        probs = np.where(probs < maximum * options.min_p, 0.0, probs)

    sums = probs.sum(axis=-1, keepdims=True)
    invalid = sums <= 0
    probs = probs / np.clip(sums, 1e-8, None)
    if invalid.any():
        greedy = logits2[0].argmax(axis=-1)
        fallback = np.zeros_like(probs)
        fallback[np.arange(probs.shape[0]), greedy] = 1.0
        probs = np.where(invalid, fallback, probs)

    gen = rng if rng is not None else np.random.default_rng(options.seed)
    out = np.empty(probs.shape[0], dtype=np.int64)
    for cb in range(probs.shape[0]):
        out[cb] = gen.choice(vocab, p=probs[cb])
    return out


# ---------------------------------------------------------------------------
# AR loop (native.py generate_audio_codes 1101-1206)
# ---------------------------------------------------------------------------
def generate_audio_codes(
    model,
    prompt,
    speaker_lda,
    speaker_position: int,
    options: SamplingOptions,
    progress_every: int = 0,
) -> tuple[np.ndarray, int | None]:
    """Run the cached-KV autoregressive decode.

    Args:
        model:            Zonos2Model (with make_kv_caches + forward_cached).
        prompt:           int ids, mx.array or numpy ``(1, T, 10)``.
        speaker_lda:      ``(1, speaker_lda_dim)`` mx.array/numpy — injected at
                          prefill only.
        speaker_position: prompt position to inject the speaker at.
        options:          SamplingOptions (gate uses temperature=0.0).
        progress_every:   if >0, print step progress every N steps.

    Returns:
        (codes, eos_frame):
          codes      -- ``(frames, n_codebooks)`` int64 RAW (delayed) codes.
          eos_frame  -- int or None.
    """
    cfg = model.cfg
    prompt = mx.array(np.asarray(prompt)) if not isinstance(prompt, mx.array) else prompt
    if speaker_lda is not None and not isinstance(speaker_lda, mx.array):
        speaker_lda = mx.array(np.asarray(speaker_lda, dtype=np.float32))

    prompt_len = prompt.shape[1]
    total_length = prompt_len + int(options.max_new_tokens)
    if total_length > cfg.max_seqlen:
        raise ValueError(
            f"Prompt ({prompt_len}) + max_new_tokens ({options.max_new_tokens}) "
            f"exceeds Zonos2 max sequence length {cfg.max_seqlen}."
        )

    caches = model.make_kv_caches(total_length)

    # Prefill: speaker injected ONLY here (native.py 1126-1132).
    logits = model.forward_cached(
        prompt,
        caches,
        speaker_lda=speaker_lda,
        speaker_position=speaker_position,
    )
    mx.eval(logits)

    n_cb = cfg.n_codebooks
    eoa_id = cfg.eoa_id
    text_vocab = cfg.text_vocab

    generated: list[np.ndarray] = []
    eos_frame: int | None = None
    eos_countdown = 0
    # ONE persistent generator threaded through every step so the RNG state
    # advances across frames. seed>0 -> reproducible; seed<=0 -> fresh entropy
    # (matches the oracle: seed>0 manual-seeds, else the global RNG). Building a
    # fresh default_rng(seed) per step would reset the state and degenerate the
    # stochastic (temperature>0) path. Greedy never touches rng.
    rng = np.random.default_rng(options.seed if options.seed > 0 else None)

    for step in range(int(options.max_new_tokens)):
        codes = sample_codes(logits, generated, options, rng)  # (n_cb,) int64 host
        generated.append(codes)

        if progress_every and (step == 0 or (step + 1) % progress_every == 0):
            print(f"  zonos2 audio tokens {step + 1}/{options.max_new_tokens}")

        # EOS handling (native.py 1169-1183).
        if eos_frame is None:
            eos_codebooks = np.nonzero(codes == eoa_id)[0]
            if eos_codebooks.size > 0:
                eos_frame = max(0, step - int(eos_codebooks.max()))
                eos_countdown = n_cb + 1
        if eos_frame is not None:
            eos_countdown -= 1
            if eos_countdown <= 0:
                break

        # Next-row input: the 9 sampled codes + the text-vocab placeholder.
        next_row = mx.array(
            np.concatenate([codes.astype(np.int32), [text_vocab]]).reshape(1, 1, n_cb + 1)
        )
        logits = model.forward_cached(next_row, caches)

        # Per-step eval: realise the new logits + the cache tensors so the lazy
        # graph and the cache buffers do not accumulate across hundreds of steps.
        mx.eval(logits, *[c.k for c in caches], *[c.v for c in caches])

    if not generated:
        raise RuntimeError("Zonos2 generated no audio token frames.")

    codes_np = np.stack(generated).astype(np.int64)  # (frames, n_codebooks)
    return codes_np, eos_frame
