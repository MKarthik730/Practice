import argparse
import random
from pathlib import Path

import torch

from training.checkpoint import save_checkpoint
from training.config import ModelConfig, TrainConfig
from training.data import load_pretraining_tokens, sample_language_model_batch, split_tokens
from training.engine import train
from training.model import DecoderLM
from training.scripts.common import add_common_arguments, read_config
from training.tokenizer import ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a small decoder LM on plain text")
    add_common_arguments(parser)
    args = parser.parse_args()
    settings = read_config(args.config)
    model_config = ModelConfig.from_dict(settings.get("model", {}))
    train_values = dict(settings.get("training", {}))
    if args.steps is not None:
        train_values["steps"] = args.steps
    if args.batch_size is not None:
        train_values["batch_size"] = args.batch_size
    train_config = TrainConfig.from_dict(train_values)
    sequence_length = args.sequence_length or model_config.max_seq_len
    if sequence_length > model_config.max_seq_len:
        parser.error("sequence length cannot exceed model.max_seq_len")
    tokenizer = ByteTokenizer()
    if model_config.vocab_size != tokenizer.vocab_size:
        parser.error(
            f"model.vocab_size must match byte tokenizer vocab size {tokenizer.vocab_size}"
        )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokens = load_pretraining_tokens(args.data, tokenizer)
    train_tokens, validation_tokens = split_tokens(tokens)
    rng = torch.Generator().manual_seed(train_config.seed)

    def sampler(split: torch.Tensor):
        return lambda: sample_language_model_batch(
            split,
            batch_size=train_config.batch_size,
            sequence_length=sequence_length,
            device=device,
            generator=rng,
        )

    model = DecoderLM(model_config).to(device)
    history = train(
        model,
        sampler(train_tokens),
        sampler(validation_tokens),
        train_config,
        device=device,
        on_step=lambda step, loss, val: print(
            f"step={step} train_loss={loss:.4f}"
            + (f" validation_loss={val:.4f}" if val is not None else "")
        ),
    )
    save_checkpoint(
        args.output,
        model,
        step=int(history[-1]["step"]),
        extra={"stage": "pretraining", "tokenizer": "utf8-byte-v1"},
    )
    print(f"Saved checkpoint to {Path(args.output)}")


if __name__ == "__main__":
    main()
