"""Base interface for memory systems under evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryResult:
    """Result from a memory system query."""
    answer: str = ""
    retrieved_context: str = ""
    retrieved_facts: List[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemorySystem(ABC):
    """Abstract interface for a memory system under evaluation."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def remember(
        self,
        text: str,
    ) -> str:
        """Store one memory text.

        Args:
            text: The natural language content to remember.

        Returns a write_id or task identifier.
        """
        ...

    @abstractmethod
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """Query the memory system.

        Args:
            question: Natural language question
            top_k: Optional retrieval budget override for the current query.

        Returns:
            QueryResult with answer and context
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memories."""
        ...

    def ingest_scenario(self, scenario) -> None:
        """Ingest all memory writes from a scenario, ordered by record_time."""
        writes = sorted(scenario.memory_writes, key=lambda w: w.record_time)
        for write in writes:
            self.remember(text=write.text)

    def remember_many(self, texts: List[str]) -> List[str]:
        """Optional bulk-ingest hook.

        Systems with native batch ingestion can override this to reduce overhead.
        The default implementation preserves the one-text-per-memory-write protocol.
        """
        return [self.remember(text=text) for text in texts]

    def remember_chain(self, chain_id: str, node_ids: List[str], texts: List[str]) -> List[str]:
        """Optional chain-aware ingest hook.

        Systems that preserve source chain/node provenance can override this to align
        retrieval metadata with benchmark node ids while still ingesting one chain
        sequentially or in bulk.
        """
        del chain_id, node_ids
        return self.remember_many(texts)
