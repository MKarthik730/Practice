from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 260
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size < 260:
            raise ValueError("vocab_size must include all byte and special tokens")
        if self.max_seq_len <= 0 or self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("model dimensions and layer count must be positive")
        if self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by a positive n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 20
    grad_clip: float = 1.0
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.steps <= 0:
            raise ValueError("batch_size and steps must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate and grad_clip must be positive")
        if self.eval_interval <= 0 or self.eval_batches <= 0:
            raise ValueError("evaluation intervals and batches must be positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
