"""End-to-end RAG Pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ragpipekit.chunker import RecursiveChunker, TextChunker
from ragpipekit.embedder import BaseEmbedder, HashEmbedder, get_embedder
from ragpipekit.loader import Document, DocumentLoader
from ragpipekit.retriever import RetrievalResult, Retriever
from ragpipekit.store import BaseVectorStore, FAISSStore, MemoryStore


class RAGPipeline:
    """All-in-one RAG pipeline with 3-line simplicity.

    Usage:
        from ragpipekit import RAGPipeline

        # 1. Initialize
        rag = RAGPipeline()

        # 2. Ingest documents, text files, or direct strings
        rag.ingest("docs/")
        # or: rag.ingest(["Python is an interpreted language...", "Rust offers memory safety..."])

        # 3. Retrieve relevant context
        results = rag.query("Tell me about Python", top_k=3)
        for r in results:
            print(r.score, r.content)

        # 4. Generate LLM prompt
        prompt = rag.build_prompt("Tell me about Python")
    """

    def __init__(
        self,
        embedder: Optional[Union[str, BaseEmbedder]] = "hash",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        store: Optional[Union[str, BaseVectorStore]] = "faiss",
        **embedder_kwargs: Any,
    ):
        """Initialize the pipeline.

        Args:
            embedder: Embedder name ('hash', 'tfidf', 'sentence-transformers', 'openai')
                      or a custom BaseEmbedder instance.
            chunk_size: Maximum characters per text chunk.
            chunk_overlap: Overlap between consecutive chunks.
            store: 'faiss', 'memory', or a custom BaseVectorStore instance.
            **embedder_kwargs: Arguments forwarded to get_embedder().
        """
        # 1. Setup Embedder
        if isinstance(embedder, BaseEmbedder):
            self.embedder = embedder
        elif isinstance(embedder, str):
            self.embedder = get_embedder(embedder, **embedder_kwargs)
        else:
            self.embedder = HashEmbedder()

        # 2. Setup Chunker & Loader
        self.chunker = RecursiveChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.loader = DocumentLoader()

        # 3. Setup Store
        if isinstance(store, BaseVectorStore):
            self.store = store
        elif store == "memory":
            self.store = MemoryStore(dimension=self.embedder.dimension)
        else:
            # Default to FAISSStore (auto-falls back to MemoryStore if faiss not installed)
            self.store = FAISSStore(dimension=self.embedder.dimension)

        # 4. Setup Retriever
        self.retriever = Retriever(embedder=self.embedder, store=self.store)

    def ingest(
        self,
        target: Union[str, Path, List[str], List[Document]],
        recursive: bool = True,
    ) -> int:
        """Ingest text, files, directories, or existing Document objects.

        Args:
            target: File path, directory path, list of text strings, or list of Documents.
            recursive: Whether to walk directories recursively.

        Returns:
            Number of chunks indexed into vector store.
        """
        docs: List[Document] = []

        if isinstance(target, (str, Path)):
            path_obj = Path(target)
            if path_obj.exists():
                docs = self.loader.load(str(path_obj), recursive=recursive)
            else:
                # Treat single non-existent string as raw text content
                docs = self.loader.load_text(str(target))
        elif isinstance(target, list):
            if not target:
                return 0
            if isinstance(target[0], Document):
                docs = target  # type: ignore
            elif isinstance(target[0], str):
                docs = self.loader.load_texts(target)  # type: ignore
            else:
                raise TypeError("List items must be str or Document")

        if not docs:
            return 0

        chunks = self.chunker.chunk_many(docs)
        if not chunks:
            return 0

        vectors = self.embedder.embed_chunks(chunks)
        self.store.add(vectors, chunks)
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_fn: Optional[Callable[[dict], bool]] = None,
    ) -> List[RetrievalResult]:
        """Query the pipeline for the most relevant context chunks.

        Args:
            query_text: User question or search phrase.
            top_k: Number of chunks to retrieve.
            score_threshold: Minimum similarity cutoff.
            filter_fn: Metadata filter callable.

        Returns:
            List of RetrievalResult objects.
        """
        return self.retriever.retrieve(
            query=query_text,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_fn=filter_fn,
        )

    def get_context(self, query_text: str, top_k: int = 3) -> str:
        """Get combined context string for prompt injection."""
        return self.retriever.get_context(query=query_text, top_k=top_k)

    def build_prompt(
        self,
        query_text: str,
        system_instruction: str = "You are a helpful assistant. Use the provided context to answer the question accurately.",
        top_k: int = 3,
    ) -> str:
        """Build a complete prompt containing retrieved context and query."""
        context = self.get_context(query_text, top_k=top_k)
        if not context:
            return f"{system_instruction}\n\nQuestion: {query_text}\nAnswer:"

        return (
            f"{system_instruction}\n\n"
            f"=== CONTEXT ===\n{context}\n===============\n\n"
            f"Question: {query_text}\n"
            f"Answer:"
        )

    def ask(
        self,
        query_text: str,
        llm_fn: Callable[[str], str],
        top_k: int = 3,
    ) -> str:
        """Retrieve context, build prompt, and call custom LLM function."""
        prompt = self.build_prompt(query_text, top_k=top_k)
        return llm_fn(prompt)

    def save(self, directory: Union[str, Path]) -> None:
        """Save vector store index to disk."""
        self.store.save(directory)

    def load(self, directory: Union[str, Path]) -> None:
        """Load vector store index from disk."""
        self.store.load(directory)

    @property
    def total_chunks(self) -> int:
        """Return total chunks currently indexed."""
        return self.store.count()
