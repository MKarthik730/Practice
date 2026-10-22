import random
from collections.abc import Callable

import torch

from .config import TrainConfig
from .model import DecoderLM

BatchSampler = Callable[[], tuple[torch.Tensor, torch.Tensor]]


@torch.no_grad()
def estimate_loss(
    model: DecoderLM,
    sampler: BatchSampler,
    *,
    batches: int,
) -> float:
    if batches <= 0:
        raise ValueError("batches must be positive")
    was_training = model.training
    model.eval()
    losses = []
    try:
        for _ in range(batches):
            inputs, targets = sampler()
            _, loss = model(inputs, targets)
            if loss is not None and torch.isfinite(loss):
                losses.append(float(loss.item()))
    finally:
        model.train(was_training)
    if not losses:
        raise ValueError("could not estimate a finite loss")
    return sum(losses) / len(losses)


def train(
    model: DecoderLM,
    train_sampler: BatchSampler,
    validation_sampler: BatchSampler,
    config: TrainConfig,
    *,
    device: torch.device,
    on_step: Callable[[int, float, float | None], None] | None = None,
) -> list[dict[str, float]]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    for step in range(1, config.steps + 1):
        model.train()
        inputs, targets = train_sampler()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        validation_loss = None
        if step % config.eval_interval == 0 or step == config.steps:
            validation_loss = estimate_loss(
                model, validation_sampler, batches=config.eval_batches
            )
        row = {"step": float(step), "train_loss": float(loss.item())}
        if validation_loss is not None:
            row["validation_loss"] = validation_loss
        history.append(row)
        if on_step is not None:
            on_step(step, float(loss.item()), validation_loss)
    return history
