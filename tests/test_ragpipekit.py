"""Unit tests for RAGPipeKit."""

import pytest
from pathlib import Path
from ragpipekit.loader import Document, DocumentLoader
from ragpipekit.chunker import TextChunker, RecursiveChunker
from ragpipekit.embedder import HashEmbedder, get_embedder
from ragpipekit.store import MemoryStore, FAISSStore
from ragpipekit.retriever import Retriever
from ragpipekit.pipeline import RAGPipeline


def test_document_loader_text():
    loader = DocumentLoader()
    docs = loader.load_text("Hello world this is a test document.")
    assert len(docs) == 1
    assert docs[0].content == "Hello world this is a test document."
    assert docs[0].metadata["type"] == "text"


def test_document_loader_files(tmp_path: Path):
    loader = DocumentLoader()
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Testing file loading inside temporary directory.", encoding="utf-8")

    docs = loader.load(str(sample_file))
    assert len(docs) == 1
    assert "Testing file loading" in docs[0].content
    assert docs[0].metadata["filename"] == "sample.txt"


def test_text_chunker():
    chunker = TextChunker(chunk_size=20, overlap=5)
    doc = Document(content="12345678901234567890abcdefghij", metadata={"source": "test"})
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    assert chunks[0].content == "12345678901234567890"


def test_recursive_chunker():
    chunker = RecursiveChunker(chunk_size=30, overlap=5)
    text = "First paragraph here.\n\nSecond paragraph here with more details."
    doc = Document(content=text, metadata={"source": "para"})
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    assert "First paragraph here." in chunks[0].content


def test_hash_embedder():
    embedder = HashEmbedder(dimension=128)
    assert embedder.dimension == 128
    vecs = embedder.embed(["hello world", "artificial intelligence"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 128


def test_memory_store_and_retriever():
    embedder = HashEmbedder(dimension=64)
    store = MemoryStore(dimension=64)
    retriever = Retriever(embedder=embedder, store=store)

    doc1 = Document(content="Python is a popular programming language.", metadata={"topic": "python"})
    doc2 = Document(content="The Eiffel Tower is located in Paris.", metadata={"topic": "paris"})

    chunker = TextChunker(chunk_size=100)
    chunks = chunker.chunk_many([doc1, doc2])
    vectors = embedder.embed_chunks(chunks)
    store.add(vectors, chunks)

    assert store.count() == 2

    results = retriever.retrieve("Python programming", top_k=1)
    assert len(results) == 1
    assert "Python" in results[0].content


def test_rag_pipeline_end_to_end(tmp_path: Path):
    rag = RAGPipeline(embedder="hash", chunk_size=150, store="memory")

    texts = [
        "FastAPI is a modern, fast web framework for building APIs with Python.",
        "LangGraph enables complex cyclical workflows for LLM multi-agent systems.",
        "PostgreSQL is a powerful, open-source object-relational database system.",
    ]

    count = rag.ingest(texts)
    assert count == 3
    assert rag.total_chunks == 3

    # Query
    results = rag.query("How to build agent workflows?")
    assert len(results) > 0
    assert any("LangGraph" in r.content for r in results)

    # Prompt generation
    prompt = rag.build_prompt("What is LangGraph?")
    assert "LangGraph" in prompt
    assert "=== CONTEXT ===" in prompt

    # Save and Load
    save_dir = tmp_path / "rag_index"
    rag.save(save_dir)

    new_rag = RAGPipeline(embedder="hash", chunk_size=150, store="memory")
    new_rag.load(save_dir)
    assert new_rag.total_chunks == 3
    new_results = new_rag.query("How to build agent workflows?")
    assert len(new_results) > 0
