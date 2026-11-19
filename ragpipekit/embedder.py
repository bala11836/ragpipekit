"""Embedding generation using multiple backends."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from ragpipekit.chunker import Chunk


class BaseEmbedder(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate embeddings for Chunk objects.

        Args:
            chunks: List of Chunk objects.

        Returns:
            List of embedding vectors.
        """
        texts = [c.content for c in chunks]
        return self.embed(texts)


class TFIDFEmbedder(BaseEmbedder):
    """TF-IDF based embedder — no external API or GPU required.

    Uses scikit-learn's TfidfVectorizer to create sparse-to-dense embeddings.
    Great for local, offline, zero-cost RAG pipelines.

    Usage:
        embedder = TFIDFEmbedder(max_features=512)
        vectors = embedder.embed(["hello world", "foo bar"])
    """

    def __init__(self, max_features: int = 512):
        """Initialize TF-IDF embedder.

        Args:
            max_features: Dimensionality of the output vectors.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError(
                "scikit-learn is required for TFIDFEmbedder. "
                "Install it with: pip install ragpipekit[tfidf]"
            )

        self.max_features = max_features
        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    @property
    def dimension(self) -> int:
        return self.max_features

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate TF-IDF embeddings."""
        if not self._fitted:
            matrix = self._vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            matrix = self._vectorizer.transform(texts)

        return matrix.toarray().tolist()

    def fit(self, texts: List[str]) -> None:
        """Pre-fit the vectorizer on a corpus."""
        self._vectorizer.fit(texts)
        self._fitted = True


class HashEmbedder(BaseEmbedder):
    """Deterministic hash-based embedder — zero dependencies, instant.

    Uses feature hashing to create fixed-dimension vectors from text.
    Not as semantically meaningful as learned embeddings, but useful for
    prototyping, testing, and environments with no ML libraries.

    Usage:
        embedder = HashEmbedder(dimension=256)
        vectors = embedder.embed(["hello", "world"])
    """

    def __init__(self, dimension: int = 256):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate hash-based embeddings."""
        return [self._hash_text(t) for t in texts]

    def _hash_text(self, text: str) -> List[float]:
        """Create a deterministic embedding via feature hashing."""
        words = text.lower().split()
        vector = [0.0] * self._dimension

        for word in words:
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            index = h % self._dimension
            sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
            vector[index] += sign

        # L2 normalize
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence-Transformers embedder — high-quality semantic embeddings.

    Requires the sentence-transformers library and downloads a model on
    first use. Produces the best retrieval quality.

    Usage:
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        vectors = embedder.embed(["hello world"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install ragpipekit[transformers]"
            )

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate semantic embeddings using Sentence Transformers."""
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI API embedder — uses text-embedding-3-small by default.

    Requires an OpenAI API key set as OPENAI_API_KEY environment variable
    or passed directly.

    Usage:
        embedder = OpenAIEmbedder(api_key="sk-...")
        vectors = embedder.embed(["hello world"])
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimension: int = 1536,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required. Install it with: pip install ragpipekit[openai]"
            )
        import os

        self.model = model
        self._dimension = dimension
        self._client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via OpenAI API."""
        response = self._client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


def get_embedder(backend: str = "hash", **kwargs) -> BaseEmbedder:
    """Factory function to create an embedder by name.

    Args:
        backend: One of 'hash', 'tfidf', 'sentence-transformers', 'openai'.

    Returns:
        An initialized embedder instance.
    """
    backends = {
        "hash": HashEmbedder,
        "tfidf": TFIDFEmbedder,
        "sentence-transformers": SentenceTransformerEmbedder,
        "openai": OpenAIEmbedder,
    }

    if backend not in backends:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(backends.keys())}")

    return backends[backend](**kwargs)
