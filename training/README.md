# Training from scratch: pretraining and SFT

This is an original, compact PyTorch training path inspired by staged LLM
training projects. It implements a decoder-only Transformer from scratch,
plain-text next-token pretraining, and supervised fine-tuning (SFT) on
`prompt`/`response` JSONL records. It does not implement a billion-parameter
model, a production tokenizer, distributed training, reward modeling, PPO,
DPO, or GRPO.

## Setup

From the repository root, install PyTorch appropriate for your machine:

```powershell
python -m pip install -r training/requirements.txt
```

Start with `training/configs/small.json`; the default configuration is intended
for a modest educational run and can also run on CPU. Increase context, width,
depth, or batch size only when the available memory permits.

## Pretraining

Put UTF-8 plain text in `data/train.txt`, then run:

```powershell
python -m training.scripts.train_pretrain --data data/train.txt --output artifacts/base.pt --config training/configs/small.json
```

The model learns next-byte prediction from the text. The included UTF-8
byte-level tokenizer is intentionally simple and fixed at 260 IDs (four
special tokens plus 256 bytes). It is useful for experiments, not a
production-quality subword tokenizer.

## Supervised fine-tuning

Create a UTF-8 JSONL file with one object per line:

```json
{"prompt":"What does a decoder model predict?","response":"It predicts the next token from the preceding context."}
{"prompt":"Explain attention briefly.","response":"Attention lets each position use information from earlier positions."}
```

Continue training the base checkpoint:

```powershell
python -m training.scripts.train_sft --init-checkpoint artifacts/base.pt --data data/sft.jsonl --output artifacts/sft.pt --config training/configs/small.json --steps 100
```

SFT loss is applied to response tokens, not prompt tokens. Records are split
into a reproducible 90/10 training/validation partition (with at least one
record in each when possible). The small example loader samples each partition
with replacement; evaluation is intended as a basic training signal, not a
benchmark.

## Generate text

```powershell
python -m training.scripts.generate --checkpoint artifacts/sft.pt --prompt "Explain language models:" --max-new-tokens 100
```

## Package layout

- `config.py` — validated model and optimizer-loop configuration
- `tokenizer.py` — fixed UTF-8 byte tokenizer
- `model.py` — causal decoder Transformer and sampling
- `data.py` — text batches and prompt/response SFT batches
- `engine.py` — training loop, gradient clipping, and loss evaluation
- `checkpoint.py` — model checkpoint save/load
- `scripts/` — pretraining, SFT, and generation commands
- `configs/small.json` — small runnable configuration
- `notebooks/` — Kaggle/Colab walkthroughs for pretraining and SFT
- `tests/` — CPU tests for model/data/checkpoint behavior

Run tests from the repository root:

```powershell
python -m unittest discover -s training/tests -v
```
