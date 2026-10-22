import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rag.rag_model import RAGModel
from rag.sparse_retriever import SparseRetriever
from rag.vector_store import VectorStore


class FakeEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.dimension = 3

    def get_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vectors.append(
                [
                    float("python" in lowered or "programming" in lowered),
                    float("cat" in lowered or "feline" in lowered),
                    float("search" in lowered or "retrieval" in lowered),
                ]
            )
        return np.asarray(vectors, dtype=np.float32).reshape(-1, self.dimension)


class SparseRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = SparseRetriever(
            [
                "python programming language",
                "feline cat animal",
                "hybrid retrieval combines search",
            ]
        )

    def test_bm25_ranks_exact_term_match(self) -> None:
        results = self.retriever.bm25("python")
        self.assertEqual(results[0][0], 0)
        self.assertGreater(results[0][1], 0)

    def test_tfidf_ranks_exact_term_match(self) -> None:
        results = self.retriever.tfidf("feline")
        self.assertEqual(results[0][0], 1)
        self.assertGreater(results[0][1], 0)

    def test_empty_query_and_invalid_arguments(self) -> None:
        self.assertEqual(self.retriever.bm25("unknown term"), [])
        with self.assertRaises(ValueError):
            self.retriever.bm25("python", k1=0)
        with self.assertRaises(ValueError):
            self.retriever.tfidf("python", top_k=-1)


class VectorStoreTests(unittest.TestCase):
    def test_cosine_search_and_empty_index(self) -> None:
        store = VectorStore(2)
        scores, indices = store.search(np.array([[1, 0]], dtype=np.float32), 3)
        self.assertEqual(scores.shape, (1, 0))
        self.assertEqual(indices.shape, (1, 0))
        store.add(np.array([[1, 0], [0, 2]], dtype=np.float32))
        scores, indices = store.search(np.array([[2, 0]], dtype=np.float32), 5)
        self.assertEqual(indices.tolist(), [[0, 1]])
        np.testing.assert_allclose(scores, [[1.0, 0.0]])

    def test_rejects_wrong_dimensions(self) -> None:
        store = VectorStore(2)
        with self.assertRaises(ValueError):
            store.add(np.ones((1, 3), dtype=np.float32))


class RAGModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_patch = patch("rag.rag_model.EmbeddingModel", FakeEmbeddingModel)
        self.embedding_patch.start()
        self.addCleanup(self.embedding_patch.stop)
        self.rag = RAGModel()
        self.rag.add_documents(
            [
                "Python is a programming language",
                "A cat is a feline animal",
                "Hybrid retrieval combines search methods",
            ],
            [{"source": "python"}, {"source": "cat"}, {"source": "search"}],
        )

    def test_all_retrieval_methods_return_documents(self) -> None:
        for method in ("bm25", "tfidf", "dense", "hybrid"):
            with self.subTest(method=method):
                results = self.rag.retrieve("python", top_k=2, method=method)
                self.assertGreaterEqual(len(results), 1)
                self.assertEqual(results[0].id, 0)

    def test_query_shapes_sources_and_scores(self) -> None:
        result = self.rag.query("python", method="hybrid", top_k=2)
        self.assertEqual(result["method"], "hybrid")
        self.assertEqual(len(result["retrieved_documents"]), len(result["scores"]))
        self.assertEqual(result["sources"][0], {"id": 0, "source": "python"})
        self.assertIn("Question: python", result["prompt"])

    def test_validates_documents_and_query(self) -> None:
        with self.assertRaises(ValueError):
            self.rag.add_documents(["one", "two"], metadatas=[{}])
        with self.assertRaises(ValueError):
            self.rag.add_documents(["one"], metadatas=[None])
        with self.assertRaises(ValueError):
            self.rag.add_documents([""])
        with self.assertRaises(ValueError):
            self.rag.retrieve("python", method="unknown")
        with self.assertRaises(ValueError):
            self.rag.retrieve(" ")

    def test_save_and_load_restores_documents_and_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "index")
            self.rag.save_index(prefix)
            restored = RAGModel()
            restored.load_index(prefix)
            self.assertEqual(len(restored.documents), 3)
            self.assertEqual(restored.documents[0].metadata, {"source": "python"})
            self.assertEqual(
                restored.retrieve("python", method="hybrid", top_k=1)[0].id, 0
            )


if __name__ == "__main__":
    unittest.main()
