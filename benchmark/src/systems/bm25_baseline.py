"""BM25 baseline with optional rank_bm25 dependency and pure-Python fallback."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from typing import List, Optional

from .base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi as _RankBM25Okapi
except ImportError:
    _RankBM25Okapi = None
    logger.warning("rank_bm25 not installed; BM25 baseline will use the built-in fallback implementation")

try:
    import jieba

    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False
    logger.warning("jieba not installed, falling back to character-level tokenization for CJK text")


class _FallbackBM25:
    """Simple BM25Okapi-compatible scorer."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.corpus_size = len(corpus)
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        n_q: Counter[str] = Counter()
        for doc in self.corpus:
            for token in set(doc):
                n_q[token] += 1
        idf: dict[str, float] = {}
        for token, freq in n_q.items():
            idf[token] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
        return idf

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        if not self.corpus:
            return []
        scores: List[float] = []
        for doc_tf, doc_len in zip(self.doc_freqs, self.doc_lens):
            score = 0.0
            for token in query_tokens:
                tf = doc_tf.get(token, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(token, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl if self.avgdl else 0.0))
                if denom == 0:
                    continue
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


class BM25Baseline(MemorySystem):
    """Sparse lexical retrieval baseline."""

    def __init__(self, name: str = "BM25"):
        super().__init__(name)
        self.raw_texts: List[str] = []
        self.tokenized_texts: List[List[str]] = []
        self.bm25: Optional[object] = None
        self.top_k: int = 5

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
        if has_cjk:
            if _JIEBA_AVAILABLE:
                return list(jieba.cut(text))
            chars = list(text)
            bigrams = [text[i : i + 2] for i in range(len(text) - 1)]
            return chars + bigrams
        return text.split()

    def _build_index(self) -> None:
        if not self.tokenized_texts:
            self.bm25 = None
            return
        if _RankBM25Okapi is not None:
            self.bm25 = _RankBM25Okapi(self.tokenized_texts)
        else:
            self.bm25 = _FallbackBM25(self.tokenized_texts)

    def remember(self, text: str) -> str:
        self.raw_texts.append(text)
        self.tokenized_texts.append(self._tokenize(text))
        self._build_index()
        return f"bm25_{len(self.raw_texts)}"

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        start = time.time()
        if not self.bm25 or not self.raw_texts:
            return QueryResult(answer="", latency_ms=(time.time() - start) * 1000)

        import numpy as np

        query_tokens = self._tokenize(question)
        scores = self.bm25.get_scores(query_tokens)
        effective_top_k = top_k if top_k is not None else self.top_k
        top_indices = np.argsort(scores)[::-1][:effective_top_k]
        retrieved = [self.raw_texts[i] for i in top_indices if scores[i] > 0]

        context = "\n".join(retrieved)
        positive_scores = [float(scores[i]) for i in top_indices if scores[i] > 0]
        max_score = positive_scores[0] if positive_scores else 0.0
        confidence = min(max_score / 10.0, 1.0)

        return QueryResult(
            answer=context,
            retrieved_context=context,
            retrieved_facts=retrieved,
            confidence=confidence,
            latency_ms=(time.time() - start) * 1000,
            metadata={
                "implementation": "rank_bm25" if _RankBM25Okapi is not None else "fallback_python_bm25",
            },
        )

    def reset(self) -> None:
        self.raw_texts.clear()
        self.tokenized_texts.clear()
        self.bm25 = None
