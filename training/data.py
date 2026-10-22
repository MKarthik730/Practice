import json
import random
from pathlib import Path
from typing import Any

import torch

from .tokenizer import ByteTokenizer


def load_pretraining_tokens(
    path: str | Path, tokenizer: ByteTokenizer | None = None
) -> torch.Tensor:
    text = Path(path).read_text(encoding="utf-8")
    token_ids = (tokenizer or ByteTokenizer()).encode(text, add_eos=True)
    if len(token_ids) < 2:
        raise ValueError("pretraining text must contain at least two tokens")
    return torch.tensor(token_ids, dtype=torch.long)


def split_tokens(
    tokens: torch.Tensor, validation_fraction: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    split_at = int(len(tokens) * (1.0 - validation_fraction))
    if split_at < 2 or len(tokens) - split_at < 2:
        raise ValueError("token sequence is too short for a train/validation split")
    return tokens[:split_at], tokens[split_at:]


def sample_language_model_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    if len(tokens) <= sequence_length:
        raise ValueError("token split must be longer than the sequence length")
    starts = torch.randint(
        len(tokens) - sequence_length,
        (batch_size,),
        generator=generator,
    )
    inputs = torch.stack([tokens[start : start + sequence_length] for start in starts])
    targets = torch.stack(
        [tokens[start + 1 : start + sequence_length + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)


def load_sft_records(path: str | Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row: Any = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            prompt, response = row.get("prompt"), row.get("response")
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError(
                    f"line {line_number}: each record needs string prompt and response"
                )
            if not prompt.strip() or not response.strip():
                raise ValueError(f"line {line_number}: prompt and response cannot be blank")
            records.append({"prompt": prompt, "response": response})
    if not records:
        raise ValueError("SFT file contains no usable records")
    return records


def encode_sft_record(
    record: dict[str, str],
    tokenizer: ByteTokenizer,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    prompt_ids = tokenizer.encode(record["prompt"], add_bos=True)
    response_ids = tokenizer.encode(record["response"], add_eos=True)
    token_budget = sequence_length + 1
    prompt_budget = token_budget - 1
    if len(prompt_ids) > prompt_budget:
        prompt_ids = [prompt_ids[0], *prompt_ids[-(prompt_budget - 1) :]]
    response_ids = response_ids[: token_budget - len(prompt_ids)]
    tokens = prompt_ids + response_ids
    if len(tokens) < 2:
        raise ValueError("encoded SFT example is too short")
    inputs = torch.tensor(tokens[:-1], dtype=torch.long)
    targets = torch.tensor(tokens[1:], dtype=torch.long)
    first_response_target = max(0, len(prompt_ids) - 1)
    targets[:first_response_target] = -100
    return inputs, targets


def split_sft_records(
    records: list[dict[str, str]],
    *,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("SFT needs at least two records for a train/validation split")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def sample_sft_batch(
    records: list[dict[str, str]],
    tokenizer: ByteTokenizer,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("records cannot be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selected = [
        encode_sft_record(rng.choice(records), tokenizer, sequence_length)
        for _ in range(batch_size)
    ]
    inputs = torch.full(
        (batch_size, sequence_length), tokenizer.pad_id, dtype=torch.long
    )
    targets = torch.full((batch_size, sequence_length), -100, dtype=torch.long)
    for index, (input_ids, target_ids) in enumerate(selected):
        inputs[index, : len(input_ids)] = input_ids
        targets[index, : len(target_ids)] = target_ids
    return inputs.to(device), targets.to(device)
