import argparse
import json
from pathlib import Path
from typing import Any


def read_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("config file must contain a JSON object")
    return values


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Optional JSON configuration file")
    parser.add_argument("--data", required=True, help="Training data file path")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--steps", type=int, help="Override training steps")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--sequence-length", type=int, help="Override sequence length")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto (default)")
