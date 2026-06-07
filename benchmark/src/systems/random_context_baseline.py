"""Random-context baseline for unrelated-memory control experiments."""

from __future__ import annotations

import hashlib
import random
import time
from typing import Iterable, Optional

from .base import MemorySystem, QueryResult


class RandomContextBaseline(MemorySystem):
    """Returns random non-target-chain memories as retrieved context.

    The evaluation runner provides the global pool and per-question exclusion
    list so that this baseline can simulate unrelated memory retrieval while
    preserving the same top-k budget as the real retrievers.
    """

    def __init__(self, name: str = "Random Context"):
        super().__init__(name)
        self.pool_texts: list[str] = []

    def remember(self, text: str) -> str:
        self.pool_texts.append(text)
        return f"random_context_{len(self.pool_texts)}"

    def remember_many(self, texts: list[str]) -> list[str]:
        start_index = len(self.pool_texts)
        self.pool_texts.extend(texts)
        return [f"random_context_{start_index + idx + 1}" for idx in range(len(texts))]

    def build_random_query_result(
        self,
        *,
        question_id: str,
        top_k: int,
        excluded_texts: Iterable[str],
    ) -> QueryResult:
        start = time.time()
        excluded = set(excluded_texts)
        candidates = [text for text in self.pool_texts if text not in excluded]

        seed = int(hashlib.md5(question_id.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        rng.shuffle(candidates)
        facts = candidates[: max(int(top_k or 0), 0)]
        context = "\n".join(facts)

        return QueryResult(
            answer=context,
            retrieved_context=context,
            retrieved_facts=facts,
            confidence=1.0 if facts else 0.0,
            latency_ms=(time.time() - start) * 1000,
            metadata={
                "baseline": "random_context",
                "num_results": len(facts),
                "pool_size": len(self.pool_texts),
                "excluded_count": len(excluded),
                "sampling_seed": seed,
            },
        )

    def query(self, question: str, top_k: Optional[int] = None) -> QueryResult:
        del question, top_k
        raise RuntimeError(
            "RandomContextBaseline requires runner-provided exclusions; "
            "use build_random_query_result() instead of query()."
        )

    def reset(self) -> None:
        self.pool_texts.clear()
