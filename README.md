# large-language-models

Learning how modern LLM systems work by building the pieces from scratch: tokenizers, transformers, embeddings, vector search, and a full RAG pipeline with a chat UI.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c) ![FAISS](https://img.shields.io/badge/FAISS-vector%20search-green) ![ChromaDB](https://img.shields.io/badge/ChromaDB-persistent%20store-orange) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

## What's inside

| Area | What it covers |
|---|---|
| **Transformer fundamentals** | Attention, positional encoding, encoder-decoder, forward/backward propagation, written without high-level APIs |
| **Tokenization & cleaning** | Custom tokenizer and text preprocessing pipeline |
| **RAG (library)** | Document store, sentence-transformer embeddings, FAISS index, prompt builder |
| **RAG-GPT (app)** | FastAPI backend + single-page chat UI, upload docs and chat with them, persisted in ChromaDB |
| **Local LLM** | TinyLlama 1.1B for fully local generation |

## Repo layout

```
large-language-models/
├── part-one.py            # Transformer fundamentals
├── encoder-decoder.py     # Seq2seq / encoder-decoder transformer
├── propagation.py         # Forward & backward propagation from scratch
├── tokenizer.py           # Custom tokenizer
├── cleaning.py            # Text preprocessing
├── rag/
│   ├── document.py        # Document data structures
│   ├── embeddings.py      # Sentence-transformer embeddings (384-dim)
│   ├── vector_store.py    # FAISS vector index
│   ├── rag_model.py       # End-to-end RAG pipeline
│   ├── main.py            # Usage example
│   ├── requirements.txt
│   └── rag-gpt/
│       ├── main.py        # FastAPI backend
│       ├── index.html     # Chat frontend
│       ├── uploaded_docs/ # Your uploaded documents
│       └── vector_db/     # ChromaDB storage (auto-created)
└── tiny-llama-1.1B/       # Local model files
```

## Quickstart

```bash
git clone https://github.com/MKarthik730/large-language-models.git
cd large-language-models
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r rag/requirements.txt
```

**RAG library example**

```bash
cd rag
python main.py
```

**RAG-GPT chat app**

```bash
cd rag/rag-gpt
uvicorn main:app --reload
# open http://127.0.0.1:8000
```

## Usage

```python
from rag import RAGModel

rag = RAGModel()
rag.add_documents([
    "Python is a high-level programming language",
    "Machine learning models learn patterns from data",
    "Transformers use attention mechanisms for NLP",
])

result = rag.query("What is machine learning?", top_k=2)

print(result["prompt"])                      # prompt ready for an LLM
for doc in result["retrieved_documents"]:    # sources used
    print("-", doc.text)
```

## How the RAG pipeline works

```
documents ─► embed (384-d) ─► vector index ─┐
                                            ├─► top-k similar chunks ─► prompt ─► LLM ─► answer
question  ─► embed (same model) ────────────┘
```

1. **Index**: embed documents with a sentence-transformer
2. **Store**: keep vectors in FAISS (library) or ChromaDB (RAG-GPT app, persistent)
3. **Retrieve**: embed the query with the same model, rank by cosine similarity
4. **Generate**: inject retrieved context into the prompt and send it to the LLM

## Roadmap

- [ ] Chunking strategies for long documents
- [ ] Hybrid search (BM25 + semantic)
- [ ] Re-ranking (cross-encoder)
- [ ] Retrieval evaluation (recall@k, MRR)
- [ ] Multiple embedding models, swappable via config
- [ ] LLM integration examples (Ollama, TinyLlama)
- [ ] Tests + CI (GitHub Actions)

## Contributing

Learning project, but issues and PRs are welcome: bug reports, alternative implementations, docs.

## License

MIT
