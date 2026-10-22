import math
import re
from collections import Counter


_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class SparseRetriever:
    """Exact BM25 and TF-IDF search over a small in-memory corpus."""

    def __init__(self, documents: list[str] | None = None) -> None:
        self.set_documents(documents or [])

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text.casefold())

    def set_documents(self, documents: list[str]) -> None:
        self.documents = list(documents)
        self.tokens = [self.tokenize(text) for text in self.documents]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.document_frequencies: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequencies.update(frequencies.keys())
        self.average_length = (
            sum(map(len, self.tokens)) / len(self.tokens) if self.tokens else 0.0
        )

    def _rank(self, scores: list[float], top_k: int) -> list[tuple[int, float]]:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        return sorted(
            (
                (index, score)
                for index, score in enumerate(scores)
                if score > 0.0
            ),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]

    def bm25(
        self, query: str, top_k: int = 5, *, k1: float = 1.5, b: float = 0.75
    ) -> list[tuple[int, float]]:
        if k1 <= 0 or not 0.0 <= b <= 1.0:
            raise ValueError("k1 must be positive and b must be between 0 and 1")
        query_terms = set(self.tokenize(query))
        count = len(self.documents)
        if not query_terms or not count:
            return []

        scores: list[float] = []
        for tokens, frequencies in zip(self.tokens, self.term_frequencies):
            score = 0.0
            length_factor = (
                1.0 - b + b * len(tokens) / self.average_length
                if self.average_length
                else 1.0
            )
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                doc_frequency = self.document_frequencies[term]
                inverse_frequency = math.log(
                    1.0 + (count - doc_frequency + 0.5) / (doc_frequency + 0.5)
                )
                score += inverse_frequency * (
                    frequency * (k1 + 1.0)
                    / (frequency + k1 * length_factor)
                )
            scores.append(score)
        return self._rank(scores, top_k)

    def tfidf(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_terms = Counter(self.tokenize(query))
        if not query_terms or not self.documents:
            return []

        count = len(self.documents)

        def weighted_vector(
            frequencies: Counter[str], length: int
        ) -> dict[str, float]:
            vector: dict[str, float] = {}
            for term, frequency in frequencies.items():
                doc_frequency = self.document_frequencies[term]
                inverse_frequency = math.log(
                    (1.0 + count) / (1.0 + doc_frequency)
                ) + 1.0
                vector[term] = (frequency / max(length, 1)) * inverse_frequency
            return vector

        query_vector = weighted_vector(query_terms, sum(query_terms.values()))
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        scores = []
        for frequencies, tokens in zip(self.term_frequencies, self.tokens):
            doc_vector = weighted_vector(frequencies, len(tokens))
            doc_norm = math.sqrt(sum(value * value for value in doc_vector.values()))
            dot_product = sum(
                query_vector.get(term, 0.0) * value
                for term, value in doc_vector.items()
            )
            scores.append(
                dot_product / (query_norm * doc_norm)
                if query_norm and doc_norm
                else 0.0
            )
        return self._rank(scores, top_k)
