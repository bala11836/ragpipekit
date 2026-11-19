# RAGPipeKit

> Lightweight, developer-first toolkit for Retrieval-Augmented Generation in Python.

Build robust, local, or cloud-backed RAG workflows in 3 lines of code without heavy dependencies or steep configuration learning curves.

---

## Highlights

- **Zero-Setup Quickstart**: Instant in-memory search with zero external services or API keys required.
- **Multiple Embedding Backends**: Hash-based, TF-IDF, Sentence Transformers, or OpenAI embeddings.
- **Smart Chunking**: Recursive text chunking preserving paragraph boundaries and sentence flow.
- **Pluggable Vector Store**: Native FAISS integration with automatic fallback to high-speed numpy cosine similarity.
- **Production-Ready**: Save and reload indexes directly to disk.

---

## Installation

```bash
pip install ragpipekit
```

To install with optional vector engine and model backends:

```bash
pip install "ragpipekit[all]"
```

---

## Quickstart

```python
from ragpipekit import RAGPipeline

# 1. Initialize
rag = RAGPipeline()

# 2. Ingest documents or raw text
rag.ingest([
    "FastAPI is a modern web framework for Python APIs.",
    "LangGraph facilitates stateful multi-agent workflows.",
    "Azure Databricks enables scalable cloud analytics."
])

# 3. Retrieve relevant context
results = rag.query("How to design agent workflows?", top_k=2)

for r in results:
    print(f"Score: {r.score:.3f} | {r.content}")
```

---

## Context Injection for LLMs

```python
prompt = rag.build_prompt("What is LangGraph?")
print(prompt)
```

---

## Author

- **Bala Sai Boppana**
- GitHub: [@bala11836](https://github.com/bala11836)
- Email: saiboppana3@gmail.com
