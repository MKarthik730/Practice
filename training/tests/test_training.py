import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import ModelConfig
from training.data import (
    encode_sft_record,
    load_pretraining_tokens,
    load_sft_records,
    sample_language_model_batch,
    sample_sft_batch,
    split_sft_records,
    split_tokens,
)
from training.model import DecoderLM
from training.tokenizer import ByteTokenizer


class TokenizerTests(unittest.TestCase):
    def test_utf8_round_trip(self) -> None:
        tokenizer = ByteTokenizer()
        text = "hello, 世界 🌍"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_special_tokens(self) -> None:
        tokenizer = ByteTokenizer()
        self.assertEqual(
            tokenizer.encode("x", add_bos=True, add_eos=True),
            [tokenizer.bos_id, ord("x") + tokenizer.byte_offset, tokenizer.eos_id],
        )


class DataTests(unittest.TestCase):
    def test_pretraining_batches_and_split(self) -> None:
        tokens = torch.arange(30)
        train_tokens, validation_tokens = split_tokens(tokens, 0.2)
        self.assertEqual(len(train_tokens), 24)
        self.assertEqual(len(validation_tokens), 6)
        inputs, targets = sample_language_model_batch(
            train_tokens,
            batch_size=3,
            sequence_length=4,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(5),
        )
        self.assertEqual(inputs.shape, (3, 4))
        torch.testing.assert_close(targets, inputs + 1)

    def test_sft_only_supervises_response_tokens(self) -> None:
        tokenizer = ByteTokenizer()
        record = {"prompt": "Q:", "response": "A"}
        inputs, targets = encode_sft_record(record, tokenizer, sequence_length=8)
        response_start = len(tokenizer.encode("Q:", add_bos=True)) - 1
        self.assertTrue(torch.all(targets[:response_start] == -100))
        self.assertTrue(torch.all(targets[response_start:] != -100))
        self.assertEqual(inputs.shape, targets.shape)

    def test_long_prompt_keeps_response_supervision(self) -> None:
        inputs, targets = encode_sft_record(
            {"prompt": "p" * 100, "response": "answer"},
            ByteTokenizer(),
            sequence_length=8,
        )
        self.assertEqual(len(inputs), 8)
        self.assertTrue(torch.any(targets != -100))

    def test_sft_split_is_reproducible_and_disjoint(self) -> None:
        records = [
            {"prompt": f"q{i}", "response": f"a{i}"} for i in range(10)
        ]
        train, validation = split_sft_records(records, seed=8)
        train_again, validation_again = split_sft_records(records, seed=8)
        self.assertEqual(train, train_again)
        self.assertEqual(validation, validation_again)
        self.assertFalse(set(map(json.dumps, train)) & set(map(json.dumps, validation)))
        with self.assertRaises(ValueError):
            split_sft_records(records[:1])

    def test_sft_jsonl_validation_and_batch_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text(
                json.dumps({"prompt": "Question", "response": "Answer"}) + "\n",
                encoding="utf-8",
            )
            records = load_sft_records(path)
            inputs, targets = sample_sft_batch(
                records,
                ByteTokenizer(),
                batch_size=2,
                sequence_length=24,
                device=torch.device("cpu"),
                rng=random.Random(1),
            )
            self.assertEqual(inputs.shape, (2, 24))
            self.assertTrue(torch.any(targets == -100))

    def test_short_pretraining_text_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.txt"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_pretraining_tokens(path)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.config = ModelConfig(
            max_seq_len=16, d_model=16, n_heads=4, n_layers=2, dropout=0
        )
        self.model = DecoderLM(self.config)

    def test_forward_loss_and_generation(self) -> None:
        tokens = torch.randint(0, self.config.vocab_size, (2, 8))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(logits.shape, (2, 8, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        generated = self.model.generate(tokens[:1, :3], max_new_tokens=2, top_k=4)
        self.assertEqual(generated.shape, (1, 5))

    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "model.pt"
            save_checkpoint(path, self.model, step=7, extra={"stage": "test"})
            restored, checkpoint = load_checkpoint(path)
            self.assertEqual(checkpoint["step"], 7)
            self.assertEqual(checkpoint["extra"]["stage"], "test")
            self.assertEqual(restored.config, self.config)
            inputs = torch.randint(0, self.config.vocab_size, (1, 5))
            torch.testing.assert_close(self.model(inputs)[0], restored(inputs)[0])


if __name__ == "__main__":
    unittest.main()
