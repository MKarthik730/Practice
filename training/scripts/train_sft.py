import argparse
import random
from pathlib import Path

import torch

from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import TrainConfig
from training.data import load_sft_records, sample_sft_batch, split_sft_records
from training.engine import train
from training.scripts.common import add_common_arguments, read_config
from training.tokenizer import ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tune a decoder LM")
    add_common_arguments(parser)
    parser.add_argument("--init-checkpoint", required=True, help="Base model checkpoint")
    args = parser.parse_args()
    settings = read_config(args.config)
    train_values = dict(settings.get("training", {}))
    if args.steps is not None:
        train_values["steps"] = args.steps
    if args.batch_size is not None:
        train_values["batch_size"] = args.batch_size
    train_config = TrainConfig.from_dict(train_values)
    sequence_length = args.sequence_length or settings.get("model", {}).get(
        "max_seq_len", 256
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.init_checkpoint, device=device)
    if sequence_length > model.config.max_seq_len:
        parser.error("sequence length cannot exceed the checkpoint model's max_seq_len")
    records = load_sft_records(args.data)
    train_records, validation_records = split_sft_records(
        records, seed=train_config.seed
    )
    tokenizer = ByteTokenizer()
    train_rng = random.Random(train_config.seed)
    validation_rng = random.Random(train_config.seed + 1)

    def make_batch_sampler(
        selected_records: list[dict[str, str]], rng: random.Random
    ):
        def batch():
            return sample_sft_batch(
                selected_records,
                tokenizer,
                batch_size=train_config.batch_size,
                sequence_length=sequence_length,
                device=device,
                rng=rng,
            )

        return batch

    train_sampler = make_batch_sampler(train_records, train_rng)
    validation_sampler = make_batch_sampler(validation_records, validation_rng)

    train(
        model,
        train_sampler,
        validation_sampler,
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
        step=int(checkpoint["step"]) + train_config.steps,
        extra={"stage": "sft", "base_checkpoint": str(args.init_checkpoint)},
    )
    print(f"Saved SFT checkpoint to {Path(args.output)}")


if __name__ == "__main__":
    main()
