"""Zonos2 model configuration.

Pure stdlib — no mlx/torch imports. Safe to load on CPU-only hosts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Zonos2Config:
    # Core transformer dims
    n_layers: int = 28
    dim: int = 2048
    head_dim: int = 128
    n_heads: int = 16
    n_kv_heads: int = 4
    intermediate_size: int = 3072

    # Normalisation
    norm_eps: float = 1e-5

    # Positional encoding
    rope_theta: float = 10000.0
    max_seqlen: int = 6144

    # Audio codec
    n_codebooks: int = 9
    codebook_size: int = 1024
    eoa_id: int = 1024
    audio_pad_id: int = 1025

    # Vocabulary
    text_vocab: int = 519

    # Training detail
    loss_softcap: float = 15.0

    # Speaker conditioning
    speaker_enabled: bool = True
    speaker_embedding_dim: int = 2048
    speaker_lda_dim: int = 1024
    speaker_background_token_enabled: bool = True
    accurate_mode_token_enabled: bool = True

    # Speaking-rate
    speaking_rate_num_buckets: int = 8
    speaking_rate_buckets: list[str] = field(default_factory=list)

    # Quality conditioning
    quality_features: list[str] = field(default_factory=list)
    quality_bucket_counts: list[int] = field(default_factory=list)

    # MoE
    moe_n_experts: int = 16
    moe_router_topk: int = 1
    special_topk_layers: dict[str, int] = field(default_factory=dict)
    moe_router_dim: int = 128
    moe_start_from_layer: int = 3
    moe_end_from_layer: int = 1

    # Oracle metadata (optional; kept for traceability)
    _oracle: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, json_path: str | Path) -> "Zonos2Config":
        """Load from the config.json dumped by the oracle."""
        with open(json_path) as f:
            raw: dict[str, Any] = json.load(f)

        # Build kwargs matching our field names (the oracle JSON already uses
        # the same names, so this is mostly a direct pass-through).
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        for k, v in raw.items():
            if k in known:
                kwargs[k] = v

        return cls(**kwargs)
