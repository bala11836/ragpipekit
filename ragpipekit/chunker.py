"""Text chunking strategies for splitting documents into retrieval-sized pieces."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ragpipekit.loader import Document


@dataclass
class Chunk:
    """A chunk of text with provenance metadata."""

    content: str
    metadata: dict = field(default_factory=dict)
    index: int = 0

    @property
    def char_count(self) -> int:
        return len(self.content)

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f'Chunk(index={self.index}, chars={self.char_count}, preview="{preview}...")'


class TextChunker:
    """Split text into fixed-size chunks with configurable overlap.

    Usage:
        chunker = TextChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk(document)
        chunks = chunker.chunk_text("some long text...")
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """Initialize the chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            overlap: Number of overlapping characters between consecutive chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        """Split a Document into chunks.

        Args:
            document: A Document object to chunk.

        Returns:
            List of Chunk objects with inherited metadata.
        """
        raw_chunks = self._split(document.content)
        return [
            Chunk(
                content=text,
                metadata={**document.metadata, "chunk_index": i, "total_chunks": len(raw_chunks)},
                index=i,
            )
            for i, text in enumerate(raw_chunks)
        ]

    def chunk_text(self, text: str, source: str = "inline") -> List[Chunk]:
        """Split raw text into chunks.

        Args:
            text: Raw text to chunk.
            source: Source label for metadata.

        Returns:
            List of Chunk objects.
        """
        doc = Document(content=text, metadata={"source": source})
        return self.chunk(doc)

    def chunk_many(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents, preserving global index order.

        Args:
            documents: List of Document objects.

        Returns:
            List of Chunk objects with continuous indexing.
        """
        all_chunks = []
        global_index = 0
        for doc in documents:
            chunks = self.chunk(doc)
            for c in chunks:
                c.index = global_index
                c.metadata["global_index"] = global_index
                global_index += 1
            all_chunks.extend(chunks)
        return all_chunks

    def _split(self, text: str) -> List[str]:
        """Core splitting logic."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks


class RecursiveChunker:
    """Split text using a hierarchy of separators for more natural boundaries.

    Tries to split on paragraphs first, then sentences, then words,
    falling back to character-level splits only when necessary.

    Usage:
        chunker = RecursiveChunker(chunk_size=500)
        chunks = chunker.chunk(document)
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        """Initialize the recursive chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            overlap: Number of overlapping characters between chunks.
            separators: List of separators to try, in order of preference.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> List[Chunk]:
        """Split a Document into chunks using recursive separators.

        Args:
            document: A Document object to chunk.

        Returns:
            List of Chunk objects.
        """
        raw_chunks = self._recursive_split(document.content, self.separators)
        merged = self._merge_chunks(raw_chunks)

        return [
            Chunk(
                content=text,
                metadata={**document.metadata, "chunk_index": i, "total_chunks": len(merged)},
                index=i,
            )
            for i, text in enumerate(merged)
        ]

    def chunk_text(self, text: str, source: str = "inline") -> List[Chunk]:
        """Split raw text into chunks."""
        doc = Document(content=text, metadata={"source": source})
        return self.chunk(doc)

    def chunk_many(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents with continuous indexing."""
        all_chunks = []
        global_index = 0
        for doc in documents:
            chunks = self.chunk(doc)
            for c in chunks:
                c.index = global_index
                c.metadata["global_index"] = global_index
                global_index += 1
            all_chunks.extend(chunks)
        return all_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators in priority order."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Fallback: hard split
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.overlap)]

        separator = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(separator)
        result = []

        for part in parts:
            if len(part) <= self.chunk_size:
                if part.strip():
                    result.append(part)
            else:
                result.extend(self._recursive_split(part, remaining_seps))

        return result

    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small chunks together up to chunk_size."""
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for chunk in chunks[1:]:
            combined = current + " " + chunk
            if len(combined) <= self.chunk_size:
                current = combined
            else:
                if current.strip():
                    merged.append(current.strip())
                current = chunk

        if current.strip():
            merged.append(current.strip())

        return merged
