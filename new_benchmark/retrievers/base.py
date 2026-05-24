"""Base classes and shared utilities for retrievers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


@dataclass
class DatabaseEntry:
    """One record in the database — either a (file, version) pair or a chapter."""
    doc_id: int
    source: str  # "github" or "narrative"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Tokenize with basic punctuation/symbol removal."""
    cleaned = re.sub(r'[\[\]@(){}<>,;:!?"\'\\]', ' ', text.lower())
    tokens = re.split(r'[/\s_\.]+', cleaned)
    return [t for t in tokens if len(t) > 1]


class Retriever(ABC):
    @abstractmethod
    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        """Add documents. System decides how to process internally."""

    @abstractmethod
    def find(self, query: str, top_k: int = 10) -> list[int]:
        """Find relevant docs. Returns ordered doc_ids by relevance."""

    @property
    def is_ranked(self) -> bool:
        """True if results are relevance-ranked. False if all equal (FullContext)."""
        return True

    @property
    def supported_sources(self) -> list[str] | None:
        """Sources this retriever can evaluate. None = all sources."""
        return None
