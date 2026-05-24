"""TF-IDF retriever with cosine similarity."""

from __future__ import annotations

import math
from collections import Counter

from .base import DatabaseEntry, Retriever, tokenize


class TFIDFRetriever(Retriever):
    """Sparse retrieval via TF-IDF cosine similarity."""

    def __init__(self):
        self._doc_ids: list[int] = []
        self._doc_tfs: list[Counter] = []
        self._df: Counter = Counter()
        self._n_docs: int = 0
        self._doc_norms: list[float] = []

    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        for doc in docs:
            tokens = tokenize(doc.text)
            self._doc_ids.append(doc.doc_id)
            tf = Counter(tokens)
            self._doc_tfs.append(tf)
            for token in set(tokens):
                self._df[token] += 1
        self._n_docs = len(self._doc_ids)

        # Precompute TF-IDF norms for each doc
        self._doc_norms = []
        for tf in self._doc_tfs:
            norm_sq = sum((count * self._idf(t)) ** 2 for t, count in tf.items())
            self._doc_norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

    def _idf(self, token: str) -> float:
        return math.log(1 + self._n_docs / (1 + self._df.get(token, 0)))

    def find(self, query: str, top_k: int = 10) -> list[int]:
        query_tf = Counter(tokenize(query))
        # Query TF-IDF vector
        query_vec = {t: count * self._idf(t) for t, count in query_tf.items()}
        query_norm = math.sqrt(sum(v ** 2 for v in query_vec.values())) or 1.0

        scores = []
        for i, doc_tf in enumerate(self._doc_tfs):
            dot = sum(
                query_vec[t] * doc_tf.get(t, 0) * self._idf(t)
                for t in query_vec if t in doc_tf
            )
            score = dot / (query_norm * self._doc_norms[i]) if self._doc_norms[i] > 0 else 0.0
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [self._doc_ids[idx] for idx, _ in scores[:top_k]]
