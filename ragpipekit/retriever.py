"""Retriever module for semantic search and context assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ragpipekit.chunker import Chunk
from ragpipekit.embedder import BaseEmbedder
from ragpipekit.store import BaseVectorStore


@dataclass
class RetrievalResult:
    """A scored retrieval result."""

    chunk: Chunk
    score: float

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def metadata(self) -> dict:
        return self.chunk.metadata

    def __repr__(self) -> str:
        preview = self.content[:50].replace("\n", " ")
        return f'RetrievalResult(score={self.score:.4f}, preview="{preview}...")'


class Retriever:
    """Coordinates vector search, filtering, and prompt context building.

    Usage:
        retriever = Retriever(embedder=embedder, store=store)
        results = retriever.retrieve("What is RAG?", top_k=3)
        context_str = retriever.get_context("What is RAG?", top_k=3)
    """

    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_fn: Optional[Callable[[dict], bool]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve most relevant chunks for a query string.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            score_threshold: Minimum similarity score cutoff.
            filter_fn: Optional callable to filter on chunk.metadata.

        Returns:
            List of RetrievalResult objects sorted by descending similarity score.
        """
        query_vector = self.embedder.embed([query])[0]
        # Query slightly more items if filtering is active
        fetch_k = top_k * 2 if filter_fn else top_k
        raw_results = self.store.search(query_vector, top_k=fetch_k)

        filtered: List[RetrievalResult] = []
        for chunk, score in raw_results:
            if score_threshold is not None and score < score_threshold:
                continue
            if filter_fn and not filter_fn(chunk.metadata):
                continue
            filtered.append(RetrievalResult(chunk=chunk, score=score))
            if len(filtered) >= top_k:
                break

        return filtered

    def get_context(
        self,
        query: str,
        top_k: int = 3,
        separator: str = "\n\n---\n\n",
        score_threshold: Optional[float] = None,
    ) -> str:
        """Convenience method to retrieve chunks and join into a formatted context block.

        Args:
            query: The user query.
            top_k: Number of chunks to include.
            separator: Delimiter between chunks.
            score_threshold: Minimum similarity cutoff.

        Returns:
            Combined string ready for LLM prompt injection.
        """
        results = self.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, start=1):
            source = r.metadata.get("filename") or r.metadata.get("source", f"Chunk {i}")
            parts.append(f"[Source: {source}]\n{r.content}")

        return separator.join(parts)
