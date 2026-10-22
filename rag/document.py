from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    id: int
    text: str
    metadata: dict[str, Any] | None = None
