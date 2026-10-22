from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .model import DecoderLM


def save_checkpoint(
    path: str | Path,
    model: DecoderLM,
    *,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "step": step,
            "extra": extra or {},
        },
        temporary,
    )
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[DecoderLM, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = DecoderLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
