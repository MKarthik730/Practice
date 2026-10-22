"""Small, original decoder-only language-model training toolkit."""

from .config import ModelConfig, TrainConfig
from .model import DecoderLM
from .tokenizer import ByteTokenizer

__all__ = ["ByteTokenizer", "DecoderLM", "ModelConfig", "TrainConfig"]
