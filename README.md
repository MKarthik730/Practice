# Large Language Models

An educational repository for understanding language models by building and
running their components: transformer fundamentals, a small from-scratch
training pipeline, retrieval-augmented generation (RAG), and hosted inference
notebooks.

## What’s here

| Area | Contents |
| --- | --- |
| From-scratch training | A small decoder-only Transformer, UTF-8 byte tokenizer, next-token pretraining, supervised fine-tuning (SFT), checkpoints, and text generation in [`training/`](training/README.md). |
| Kaggle and Colab | Notebooks for Hugging Face/Ollama inference and tunnels, Hugging Face LoRA/QLoRA fine-tuning, RAG retrieval, from-scratch pretraining, and SFT. |
| RAG | Sparse BM25 and TF-IDF, dense FAISS search, hybrid reciprocal-rank fusion, optional cross-encoder reranking, and persistence. See [`rag/README.md`](rag/README.md). |
| Fundamentals | Transformer and encoder-decoder examples, a tokenizer example, data cleaning, and forward/backward propagation. |
| RAG-GPT app | A separate FastAPI-based document chat example in `rag/rag-gpt/`. |

## Repository layout

```text
.
├── notebooks/
│   ├── 01-kaggle-colab-model-inference.ipynb
│   ├── 02-kaggle-colab-lora-finetuning.ipynb
│   └── 03-rag-retrieval-techniques.ipynb
├── training/
│   ├── configs/small.json
│   ├── notebooks/                 # from-scratch pretraining and SFT
│   ├── scripts/                   # train, fine-tune, generate
│   ├── tests/
│   └── model.py, data.py, engine.py, tokenizer.py, checkpoint.py
├── rag/
│   ├── rag_model.py
│   ├── sparse_retriever.py
│   ├── vector_store.py
│   └── rag-gpt/                   # separate web app
├── tests/
├── cleaning.py
├── encoder-decoder.py
├── part-one.py
├── propagation.py
└── tokenizer.py
```

## Quick start

Use Python 3.10+ and install the dependencies for the feature you want. PyTorch
installation can vary by operating system and CUDA version; choose the
appropriate PyTorch build for your environment.

### From-scratch pretraining and SFT

Install PyTorch as appropriate for your machine, then place UTF-8 training
text in `data/train.txt` and run from the repository root:

```bash
python -m training.scripts.train_pretrain --data data/train.txt --output artifacts/base.pt --config training/configs/small.json
```

Fine-tune the checkpoint on JSONL records containing `prompt` and `response`
strings:

```bash
python -m training.scripts.train_sft --init-checkpoint artifacts/base.pt --data data/sft.jsonl --output artifacts/sft.pt --config training/configs/small.json
```

Generate a sample:

```bash
python -m training.scripts.generate --checkpoint artifacts/sft.pt --prompt "Explain attention:" --max-new-tokens 100
```

See [`training/README.md`](training/README.md) for data format, notebook
instructions, and limitations. This is a small educational byte-token model;
it does not train billion-parameter models or fine-tune arbitrary Hugging Face
or Ollama checkpoints.

### RAG

Install the RAG dependencies and run the example:

```bash
python -m pip install -r rag/requirements.txt
python -m rag.main
```

The first dense search downloads the configured Sentence Transformers
embedding model. The package prepares retrieved context and a prompt; connect
that prompt to a generation model separately. See [`rag/README.md`](rag/README.md)
for retrieval methods, reranking, and persistence.

### Kaggle and Colab notebooks

Open notebooks from `notebooks/` or `training/notebooks/` in Kaggle or Google
Colab and run cells in order. Select a GPU runtime when available. The
notebooks document their input paths, model setup, and checkpoint handling.
Save outputs to persistent storage before the hosted runtime expires.

The inference notebook supports Hugging Face repositories compatible with
`AutoModelForCausalLM` and a reachable Ollama server/model; these are different
formats and not every model is compatible. The LoRA/QLoRA notebook fine-tunes
compatible Hugging Face causal LMs. Gated Hugging Face models require a token
stored in notebook secrets. Public tunnel cells are optional: protect the
endpoint with its bearer token, keep URLs and credentials private, and stop
the tunnel when finished.

## Tests

Run the RAG and from-scratch training unit tests from the repository root:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s training/tests -v
```

## Reference

The staged learning path is informed by
[train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch).
The `training/` implementation in this repository is original and is not a
copy of that project's source.
