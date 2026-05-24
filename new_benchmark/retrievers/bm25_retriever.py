"""BM25 (Okapi BM25) retriever."""

from __future__ import annotations

import math
from collections import Counter

from .base import DatabaseEntry, Retriever, tokenize


class BM25:
    """Pure-Python BM25Okapi."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.doc_freqs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}

    def add_doc(self, tokens: list[str]) -> None:
        self.corpus.append(tokens)
        self.doc_freqs.append(Counter(tokens))
        self.doc_lens.append(len(tokens))

    def finalize(self) -> None:
        n_q: Counter[str] = Counter()
        for doc in self.corpus:
            for token in set(doc):
                n_q[token] += 1
        self.idf = {
            token: math.log(1 + (len(self.corpus) - freq + 0.5) / (freq + 0.5))
            for token, freq in n_q.items()
        }
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0

    def query(self, query_tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        scores = []
        for i, (doc_tf, doc_len) in enumerate(zip(self.doc_freqs, self.doc_lens)):
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
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class BM25Retriever(Retriever):
    """Keyword retrieval via BM25Okapi."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._bm25 = BM25(k1=k1, b=b)
        self._doc_ids: list[int] = []

    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        for doc in docs:
            self._doc_ids.append(doc.doc_id)
            self._bm25.add_doc(tokenize(doc.text))
        self._bm25.finalize()

    def find(self, query: str, top_k: int = 10) -> list[int]:
        tokens = tokenize(query)
        scored = self._bm25.query(tokens, top_k=top_k)
        return [self._doc_ids[idx] for idx, _ in scored]

    def find_scored(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_id, score) pairs."""
        tokens = tokenize(query)
        scored = self._bm25.query(tokens, top_k=top_k)
        return [(self._doc_ids[idx], score) for idx, score in scored]
