# Retrieval-Augmented Generation

This package provides selectable sparse, dense, and hybrid document retrieval.
The default `hybrid` path combines BM25 keyword ranking and dense embedding
ranking with reciprocal rank fusion (RRF), so the scores from the two
retrievers do not need to be calibrated against one another.

## Install

From the repository root:

```bash
python -m pip install -r rag/requirements.txt
```

The first use of `RAGModel` downloads the default
`sentence-transformers/all-MiniLM-L6-v2` embedding model from Hugging Face.
Choose an embedding model compatible with Sentence Transformers if you need to
change it.

## Quick start

```python
from rag import RAGModel

rag = RAGModel()
rag.add_documents(
    [
        "FAISS supports efficient similarity search over dense vectors.",
        "BM25 retrieves documents using lexical term statistics.",
    ],
    metadatas=[{"source": "vectors.txt"}, {"source": "search.txt"}],
)

result = rag.query("How does keyword search work?", method="hybrid", top_k=2)
print(result["prompt"])
print(result["sources"])
```

`query` returns the retrieved `Document` objects, the retriever scores, the
prompt text, source metadata, and the selected retrieval method. The RAG
package prepares context; it does not call a text-generation model.

## Retrieval methods

```python
rag.retrieve(question, method="bm25")   # Sparse BM25 keyword ranking
rag.retrieve(question, method="tfidf")  # Sparse TF-IDF cosine similarity
rag.retrieve(question, method="dense")  # Dense embedding cosine similarity
rag.retrieve(question, method="hybrid") # BM25 + dense, fused with RRF
```

Optional cross-encoder reranking uses Sentence Transformers:

```python
rag = RAGModel(reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
ranked = rag.retrieve(question, method="hybrid", rerank=True, top_k=3)
```

Reranking is opt-in because it adds model download, memory, and inference
requirements. The reranker reorders the retrieved candidate set; it does not
replace retrieval.

## Persistence

`save_index("artifacts/my-index")` writes a FAISS index to
`artifacts/my-index.index` and document text/metadata to
`artifacts/my-index.documents.json`. Metadata must be JSON-serializable.
`load_index` expects both files and an embedding model with the same vector
dimension used to create the index.

## Limitations

- This implementation is intended for learning and small local corpora. The
  sparse index is rebuilt in memory when documents are added.
- Dense search uses an exact FAISS inner-product index over normalized
  embeddings; it is not an approximate or distributed vector database.
- BM25 and TF-IDF use simple Unicode word tokenization without stemming.
- The default prompt is a context template; connect its output to a compatible
  chat or text-generation model yourself.
