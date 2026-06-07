"""No-context baseline for measuring LLM prior without retrieved memory."""

from __future__ import annotations

import time
from typing import Optional

from .base import MemorySystem, QueryResult


class NoContextBaseline(MemorySystem):
    """不返回任何检索记忆的对照后端。

    中文注释：该 baseline 仍然走统一的 AnswerGenerator / Judge 流程，
    但 query() 永远返回空 retrieval_context，用于测量 LLM 在无记忆条件下的先验和猜测能力。
    """

    def __init__(self, name: str = "No Context"):
        super().__init__(name)

    def remember(
        self,
        text: str,
    ) -> str:
        # 中文注释：No-Context baseline 故意忽略所有写入内容。
        del text
        return "no_context_ignored"

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        # 中文注释：只保留统一接口，不向答案生成器提供任何检索上下文。
        del question
        del top_k
        start = time.time()
        return QueryResult(
            answer="",
            retrieved_context="",
            retrieved_facts=[],
            confidence=0.0,
            latency_ms=(time.time() - start) * 1000,
            metadata={"baseline": "no_context"},
        )

    def reset(self) -> None:
        # 中文注释：该 baseline 无状态，无需清理。
        return None
