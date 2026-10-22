import faiss
import numpy as np
from pathlib import Path


class VectorStore:
    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def _prepare(self, vectors: np.ndarray) -> np.ndarray:
        prepared = np.asarray(vectors, dtype=np.float32)
        if prepared.ndim == 1:
            prepared = prepared.reshape(1, -1)
        if prepared.ndim != 2 or prepared.shape[1] != self.dimension:
            raise ValueError(
                f"expected vectors with shape (n, {self.dimension}), "
                f"got {prepared.shape}"
            )
        norms = np.linalg.norm(prepared, axis=1, keepdims=True)
        return np.divide(
            prepared,
            norms,
            out=np.zeros_like(prepared),
            where=norms != 0,
        )

    def add(self, embeddings: np.ndarray) -> None:
        prepared = self._prepare(embeddings)
        if prepared.shape[0]:
            self.index.add(prepared)

    def search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        queries = self._prepare(query_embedding)
        count = min(top_k, self.index.ntotal)
        if count == 0:
            return (
                np.empty((queries.shape[0], 0), dtype=np.float32),
                np.empty((queries.shape[0], 0), dtype=np.int64),
            )
        return self.index.search(queries, count)

    def save(self, path: str | Path) -> None:
        faiss.write_index(self.index, f"{path}.index")

    def load(self, path: str | Path) -> None:
        index = faiss.read_index(f"{path}.index")
        if index.d != self.dimension:
            raise ValueError(
                f"index dimension {index.d} does not match {self.dimension}"
            )
        self.index = index

    def size(self) -> int:
        return self.index.ntotal
