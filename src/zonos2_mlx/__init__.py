"""zonos2-mlx — pure-MLX port of Zyphra ZONOS2 8B-MoE TTS for Apple Silicon."""

from .config import Zonos2Config
from .pipeline import SynthesisResult, synthesize

__version__ = "0.1.0"

__all__ = ["Zonos2Config", "SynthesisResult", "synthesize", "__version__"]
