"""Naive RAG baseline — text concatenation + simple keyword retrieval."""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Set

from .base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)


class NaiveRAGBaseline(MemorySystem):
    """Naive RAG: concatenate all texts, retrieve by keyword overlap."""

    def __init__(self, name: str = "Naive RAG"):
        super().__init__(name)
        self.texts: List[str] = []
        self.top_k: int = 5

    def remember(self, text: str) -> str:
        # 中文注释：Naive RAG 主实验只保留原始文本。
        self.texts.append(text)
        return f"naive_{len(self.texts)}"

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        start = time.time()
        if not self.texts:
            return QueryResult(answer="", latency_ms=(time.time() - start) * 1000)

        # Simple keyword overlap scoring
        q_tokens: Set[str] = set(question)
        # Chinese character bigrams
        q_bigrams = {question[i:i+2] for i in range(len(question) - 1)}
        q_features = q_tokens | q_bigrams

        scored = []
        for i, text in enumerate(self.texts):
            t_bigrams = {text[j:j+2] for j in range(len(text) - 1)}
            t_features = set(text) | t_bigrams
            overlap = len(q_features & t_features)
            scored.append((overlap, i, text))

        scored.sort(key=lambda x: (-x[0], x[1]))
        effective_top_k = top_k if top_k is not None else self.top_k
        retrieved = [s[2] for s in scored[:effective_top_k] if s[0] > 0]
        best_overlap = scored[0][0] if scored and scored[0][0] > 0 else 0

        context = "\n".join(retrieved)
        return QueryResult(
            answer=context,
            retrieved_context=context,
            retrieved_facts=retrieved,
            confidence=best_overlap / max(len(q_features), 1) if best_overlap else 0.0,
            latency_ms=(time.time() - start) * 1000,
        )

    def reset(self) -> None:
        self.texts.clear()
