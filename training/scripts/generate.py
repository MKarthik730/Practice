import argparse

import torch

from training.checkpoint import load_checkpoint
from training.tokenizer import ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a training checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_checkpoint(args.checkpoint, device=device)
    tokenizer = ByteTokenizer()
    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device
    )
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_id,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
