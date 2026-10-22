from collections.abc import Sequence
import json
from typing import Any
from pathlib import Path

from .document import Document
from .embeddings import EmbeddingModel
from .sparse_retriever import SparseRetriever
from .vector_store import VectorStore


class RAGModel:
    _METHODS = {"dense", "bm25", "tfidf", "hybrid"}

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        *,
        reranker_model: str | None = None,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        self.embedding_model = EmbeddingModel(embedding_model)
        self.vector_store = VectorStore(self.embedding_model.get_dimension())
        self.documents: list[Document] = []
        self.sparse_retriever = SparseRetriever()
        self.rrf_k = rrf_k
        self.reranker = None
        if reranker_model:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers to use a cross-encoder reranker"
                ) from exc
            self.reranker = CrossEncoder(reranker_model)

    def add_documents(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        texts = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("documents must be non-empty strings")
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("metadatas must have the same length as texts")
        if any(not isinstance(metadata, dict) for metadata in metadatas):
            raise ValueError("each metadata entry must be a dictionary")
        if not texts:
            return

        start_id = len(self.documents)
        new_documents = [
            Document(id=start_id + i, text=text, metadata=dict(metadata))
            for i, (text, metadata) in enumerate(zip(texts, metadatas))
        ]
        embeddings = self.embedding_model.encode(texts)
        self.vector_store.add(embeddings)
        self.documents.extend(new_documents)
        self.sparse_retriever.set_documents(
            [document.text for document in self.documents]
        )

    def _dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self.documents or top_k == 0:
            return []
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.vector_store.search(query_embedding, top_k)
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if index >= 0
        ]

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 3,
        *,
        method: str = "hybrid",
        rerank: bool = False,
    ) -> list[tuple[Document, float]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {sorted(self._METHODS)}")
        if rerank and self.reranker is None:
            raise ValueError(
                "reranking requires configuring reranker_model in RAGModel"
            )
        if not self.documents or top_k == 0:
            return []

        candidate_k = min(len(self.documents), max(top_k * 4, top_k))
        if method == "dense":
            ranked = self._dense_search(query, candidate_k)
        elif method == "bm25":
            ranked = self.sparse_retriever.bm25(query, candidate_k)
        elif method == "tfidf":
            ranked = self.sparse_retriever.tfidf(query, candidate_k)
        else:
            dense = self._dense_search(query, candidate_k)
            sparse = self.sparse_retriever.bm25(query, candidate_k)
            fused: dict[int, float] = {}
            for results in (dense, sparse):
                for rank, (document_id, _) in enumerate(results, start=1):
                    fused[document_id] = fused.get(document_id, 0.0) + (
                        1.0 / (self.rrf_k + rank)
                    )
            ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))

        if rerank:
            if self.reranker is None:
                raise RuntimeError("reranker was not initialized")
            candidates = ranked[:candidate_k]
            scores = self.reranker.predict(
                [
                    [query, self.documents[document_id].text]
                    for document_id, _ in candidates
                ]
            )
            ranked = sorted(
                (
                    (document_id, float(score))
                    for (document_id, _), score in zip(candidates, scores)
                ),
                key=lambda item: (-item[1], item[0]),
            )
        return [
            (self.documents[document_id], score)
            for document_id, score in ranked[:top_k]
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        *,
        method: str = "hybrid",
        rerank: bool = False,
    ) -> list[Document]:
        return [
            document
            for document, _ in self.retrieve_with_scores(
                query, top_k, method=method, rerank=rerank
            )
        ]

    def generate_prompt(self, query: str, retrieved_docs: Sequence[Document]) -> str:
        context = "\n\n".join(
            f"Document {index} (id={doc.id}):\n{doc.text}"
            for index, doc in enumerate(retrieved_docs, start=1)
        )
        return f"""Answer the question using the supplied context. If the context does not contain the answer, say so.

Context:
{context}

Question: {query}

Answer:"""

    def query(
        self,
        question: str,
        top_k: int = 3,
        *,
        method: str = "hybrid",
        rerank: bool = False,
    ) -> dict[str, Any]:
        scored_documents = self.retrieve_with_scores(
            question, top_k=top_k, method=method, rerank=rerank
        )
        retrieved_docs = [document for document, _ in scored_documents]
        return {
            "retrieved_documents": retrieved_docs,
            "scores": [score for _, score in scored_documents],
            "prompt": self.generate_prompt(question, retrieved_docs),
            "sources": [
                {**(doc.metadata or {}), "id": doc.id} for doc in retrieved_docs
            ],
            "method": method,
        }

    def save_index(self, path: str) -> None:
        """Save the FAISS index and JSON-serializable document metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        document_path = Path(f"{path}.documents.json")
        self.vector_store.save(path)
        with document_path.open("w", encoding="utf-8") as file:
            json.dump(
                [
                    {"id": doc.id, "text": doc.text, "metadata": doc.metadata or {}}
                    for doc in self.documents
                ],
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load_index(self, path: str) -> None:
        """Load an index previously saved with save_index."""
        document_path = Path(f"{path}.documents.json")
        with document_path.open(encoding="utf-8") as file:
            rows = json.load(file)
        if not isinstance(rows, list):
            raise ValueError("saved documents must be a JSON list")
        documents = [
            Document(
                id=int(row["id"]),
                text=str(row["text"]),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in rows
        ]
        self.vector_store.load(path)
        if self.vector_store.size() != len(documents):
            raise ValueError("saved document count does not match the FAISS index")
        self.documents = documents
        self.sparse_retriever.set_documents([doc.text for doc in documents])
