"""Document loaders for various file formats."""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Document:
    """A single document with content and metadata."""

    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def char_count(self) -> int:
        return len(self.content)

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f'Document(source="{self.source}", chars={self.char_count}, preview="{preview}...")'


class DocumentLoader:
    """Load documents from files and directories.

    Supports .txt, .md, .csv, .json, .py, .html, and .log files.
    Automatically walks directories and loads all supported files.

    Usage:
        loader = DocumentLoader()
        docs = loader.load("path/to/folder")
        docs = loader.load("single_file.txt")
        docs = loader.load_text("Raw string content here")
    """

    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".csv", ".json", ".py", ".js",
        ".html", ".htm", ".log", ".yaml", ".yml",
        ".rst", ".xml", ".sql", ".sh", ".bat", ".cfg",
        ".ini", ".toml", ".env", ".gitignore",
    }

    def __init__(self, encoding: str = "utf-8", errors: str = "ignore"):
        self.encoding = encoding
        self.errors = errors

    def load(self, path: str, recursive: bool = True) -> List[Document]:
        """Load documents from a file or directory.

        Args:
            path: Path to a file or directory.
            recursive: Whether to walk subdirectories.

        Returns:
            List of Document objects.
        """
        target = Path(path)

        if target.is_file():
            return self._load_file(target)
        elif target.is_dir():
            return self._load_directory(target, recursive)
        else:
            # Try glob pattern
            matched_files = glob.glob(path, recursive=recursive)
            docs = []
            for f in matched_files:
                fp = Path(f)
                if fp.is_file():
                    docs.extend(self._load_file(fp))
            if not docs:
                raise FileNotFoundError(f"No files found at: {path}")
            return docs

    def load_text(self, text: str, source: str = "inline") -> List[Document]:
        """Create a Document from raw text.

        Args:
            text: Raw text content.
            source: Source label for metadata.

        Returns:
            List containing one Document.
        """
        return [Document(content=text, metadata={"source": source, "type": "text"})]

    def load_texts(self, texts: List[str], source_prefix: str = "text") -> List[Document]:
        """Create Documents from a list of strings.

        Args:
            texts: List of text strings.
            source_prefix: Prefix for source labels.

        Returns:
            List of Document objects.
        """
        return [
            Document(
                content=t,
                metadata={"source": f"{source_prefix}_{i}", "type": "text"},
            )
            for i, t in enumerate(texts)
        ]

    def _load_file(self, path: Path) -> List[Document]:
        """Load a single file into a Document."""
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return []

        try:
            content = path.read_text(encoding=self.encoding, errors=self.errors)
        except Exception as e:
            print(f"[ragpipekit] Warning: Could not read {path}: {e}")
            return []

        if not content.strip():
            return []

        metadata = {
            "source": str(path.resolve()),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "type": "file",
        }

        return [Document(content=content, metadata=metadata)]

    def _load_directory(self, directory: Path, recursive: bool) -> List[Document]:
        """Load all supported files from a directory."""
        docs = []
        pattern = "**/*" if recursive else "*"

        for file_path in sorted(directory.glob(pattern)):
            if file_path.is_file():
                docs.extend(self._load_file(file_path))

        if not docs:
            print(f"[ragpipekit] Warning: No supported files found in {directory}")

        return docs
