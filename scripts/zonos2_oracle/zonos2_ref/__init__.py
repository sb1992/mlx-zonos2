"""Vendored, read-only copy of the plain-PyTorch ZONOS2 reference implementation.

Source: ~/Zonos2_TTS-ComfyUI (a ComfyUI custom-node fork that runs without
flashinfer/CUDA via PyTorch SDPA). These files are imported directly (NOT via
ComfyUI) to act as the parity oracle for the zonos2-mlx port.

Only `loader.py` was modified vs upstream (see its header). `native.py` and
`runtime.py` are verbatim. The upstream package `__init__.py` (which imports the
ComfyUI `nodes` module) is intentionally NOT vendored; this minimal __init__
exposes only the inference building blocks.
"""

from __future__ import annotations

__all__ = ["native", "loader", "runtime"]
