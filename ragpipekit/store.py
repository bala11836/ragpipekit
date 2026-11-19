"""Vector store implementations for RAGPipeKit.

Provides FAISSStore (using faiss-cpu if available) and MemoryStore
(pure Python + numpy fallback with zero external C-dependencies).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ragpipekit.chunker import Chunk


class BaseVectorStore(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    def add(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """Add vectors and their associated chunks to the store."""
        ...

    @abstractmethod
    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """Search for top_k nearest chunks given a query vector.

        Returns list of (Chunk, score) tuples.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors in store."""
        ...

    @abstractmethod
    def save(self, directory: str | Path) -> None:
        """Persist index and metadata to disk."""
        ...

    @abstractmethod
    def load(self, directory: str | Path) -> None:
        """Load index and metadata from disk."""
        ...


class MemoryStore(BaseVectorStore):
    """In-memory vector store using cosine similarity via numpy.

    Lightweight, deterministic, and requires no compiled binary libraries.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.chunks: List[Chunk] = []

    def add(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("Number of vectors must match number of chunks.")
        if not vectors:
            return

        new_vecs = np.array(vectors, dtype=np.float32)
        # Normalize vectors for cosine similarity via dot product
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = new_vecs / norms

        if self.vectors.shape[0] == 0:
            self.vectors = normalized
        else:
            self.vectors = np.vstack([self.vectors, normalized])

        self.chunks.extend(chunks)

    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        if self.count() == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        scores = np.dot(self.vectors, q)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results

    def count(self) -> int:
        return len(self.chunks)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "vectors.npy", self.vectors)
        chunks_data = [
            {"content": c.content, "metadata": c.metadata, "index": c.index}
            for c in self.chunks
        ]
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

    def load(self, directory: str | Path) -> None:
        path = Path(directory)
        self.vectors = np.load(path / "vectors.npy")
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self.chunks = [
            Chunk(content=item["content"], metadata=item["metadata"], index=item["index"])
            for item in chunks_data
        ]


class FAISSStore(BaseVectorStore):
    """FAISS-powered vector store with automatic fallback to MemoryStore."""

    def __init__(self, dimension: int, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.chunks: List[Chunk] = []
        self._use_faiss = False

        try:
            import faiss
            self.faiss = faiss
            if index_type == "flat":
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
            else:
                self.index = faiss.IndexFlatL2(dimension)
            self._use_faiss = True
        except ImportError:
            # Fallback to MemoryStore if faiss is not installed
            self._fallback_store = MemoryStore(dimension)

    def add(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        if not self._use_faiss:
            self._fallback_store.add(vectors, chunks)
            return

        if len(vectors) != len(chunks):
            raise ValueError("Number of vectors must match number of chunks.")
        if not vectors:
            return

        vec_arr = np.array(vectors, dtype=np.float32)
        # Normalize for Inner Product (cosine similarity)
        self.faiss.normalize_L2(vec_arr)
        self.index.add(vec_arr)
        self.chunks.extend(chunks)

    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        if not self._use_faiss:
            return self._fallback_store.search(query_vector, top_k=top_k)

        if self.count() == 0:
            return []

        q = np.array([query_vector], dtype=np.float32)
        self.faiss.normalize_L2(q)

        scores, indices = self.index.search(q, min(top_k, self.count()))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def count(self) -> int:
        if not self._use_faiss:
            return self._fallback_store.count()
        return self.index.ntotal

    def save(self, directory: str | Path) -> None:
        if not self._use_faiss:
            self._fallback_store.save(directory)
            return

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(path / "faiss.index"))

        chunks_data = [
            {"content": c.content, "metadata": c.metadata, "index": c.index}
            for c in self.chunks
        ]
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

    def load(self, directory: str | Path) -> None:
        if not self._use_faiss:
            self._fallback_store.load(directory)
            return

        path = Path(directory)
        self.index = self.faiss.read_index(str(path / "faiss.index"))
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self.chunks = [
            Chunk(content=item["content"], metadata=item["metadata"], index=item["index"])
            for item in chunks_data
        ]
