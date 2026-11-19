"""
RAGPipeKit — Zero-config RAG pipelines in 3 lines of Python.

A lightweight, production-ready toolkit for building Retrieval-Augmented
Generation (RAG) pipelines. Ingest documents, chunk text, generate embeddings,
store in FAISS, and retrieve relevant context — all with minimal setup.

Usage:
    from ragpipekit import RAGPipeline

    rag = RAGPipeline()
    rag.ingest("path/to/docs/")
    results = rag.query("What is the main topic?")
"""

__version__ = "0.1.0"
__author__ = "Bala Sai Boppana"
__license__ = "MIT"

from ragpipekit.pipeline import RAGPipeline
from ragpipekit.chunker import TextChunker, RecursiveChunker
from ragpipekit.loader import DocumentLoader
from ragpipekit.store import FAISSStore
from ragpipekit.retriever import Retriever

__all__ = [
    "RAGPipeline",
    "TextChunker",
    "RecursiveChunker",
    "DocumentLoader",
    "FAISSStore",
    "Retriever",
]
