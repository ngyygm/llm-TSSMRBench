"""Full-context baseline for upper-bound reading with the complete event chain."""

from __future__ import annotations

import time
from typing import List, Optional

from .base import MemorySystem, QueryResult


class FullContextBaseline(MemorySystem):
    """Returns the full ingested chain context for every query."""

    def __init__(self, name: str = "Full Context"):
        super().__init__(name)
        self.texts: List[str] = []

    def remember(
        self,
        text: str,
    ) -> str:
        self.texts.append(text)
        return f"full_context_{len(self.texts)}"

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        del question, top_k
        start = time.time()
        context = "\n".join(self.texts)
        return QueryResult(
            answer=context,
            retrieved_context=context,
            retrieved_facts=list(self.texts),
            confidence=1.0 if self.texts else 0.0,
            latency_ms=(time.time() - start) * 1000,
            metadata={"baseline": "full_context", "num_results": len(self.texts)},
        )

    def reset(self) -> None:
        self.texts.clear()
