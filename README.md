# Large Language Models: Learning and Practical Workflows

This repository combines small educational implementations with reusable
Kaggle/Colab notebooks for model inference, fine-tuning, and retrieval.

## Contents

- `notebooks/01-kaggle-colab-model-inference.ipynb` — load a Hugging Face
  causal language model or call an Ollama model, test generation, and
  optionally expose a bearer-token-protected API through ngrok or Cloudflare
  Tunnel.
- `notebooks/02-kaggle-colab-lora-finetuning.ipynb` — supervised LoRA/QLoRA
  fine-tuning for compatible Hugging Face causal language models.
- `notebooks/03-rag-retrieval-techniques.ipynb` — compare BM25, TF-IDF, dense,
  and hybrid retrieval in the local RAG package.
- `training/` — original small decoder-only Transformer with plain-text
  pretraining, prompt/response SFT, checkpointing, generation, CPU tests, and
  Kaggle/Colab notebooks.
- `rag/` — sparse and dense retrieval, hybrid RRF ranking, optional
  cross-encoder reranking, and index persistence.
- `part-one.py`, `encoder-decoder.py`, `tokenizer.py`, and `cleaning.py` —
  standalone learning examples.

## Notebook use

Open the notebooks in Kaggle or Google Colab, select a GPU runtime when
available, then run the cells in order. For gated Hugging Face repositories,
add a read-only Hugging Face token to the platform's notebook secrets; never
write credentials into notebook cells or outputs.

The inference notebook accepts a Hugging Face model repository ID or standard
`huggingface.co/owner/repository` URL for `AutoModelForCausalLM`-compatible
text-generation models, and can call an Ollama server/model. These formats are not
interchangeable: an Ollama model or arbitrary model link may not be loadable
by Transformers. Check the model card, license, tokenizer, architecture, and
runtime memory requirements before downloading or fine-tuning.

The public tunnel cells are optional. Use the API bearer token, keep tunnel
URLs private, stop the server when finished, and do not expose an unauthenticated
model endpoint. Tunnel addresses are temporary and may be publicly reachable.

Fine-tuning uses adapter-based supervised tuning to reduce memory use. It is
not a guarantee that every architecture, dataset, or free-tier GPU fits.
QLoRA's 4-bit path requires a compatible CUDA GPU and `bitsandbytes`; the
notebook includes a non-quantized LoRA option. Follow the model and dataset
licenses and Hugging Face terms.

## Run the RAG example

From the repository root:

```bash
python -m pip install -r rag/requirements.txt
python -m rag.main
```

RAG usage, retrieval options, and persistence details are in [rag/README.md](rag/README.md).

## Train a small model from scratch

See [training/README.md](training/README.md) for setup, pretraining, SFT, and
generation commands. This educational byte-token model is a separate
from-scratch path; it does not fine-tune arbitrary Hugging Face/Ollama models.

## Tests

Run the lightweight retrieval tests from the repository root:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s training/tests -v
```

## Reference

The notebook workflow and staged model-training concepts were informed by
[train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch).
This repository implements its own workflows rather than copying that
project's source files.
