"""Random retriever — lower-bound baseline."""

from __future__ import annotations

import random

from .base import DatabaseEntry, Retriever


class RandomRetriever(Retriever):
    """Random retrieval baseline. Returns a random subset of doc_ids."""

    def __init__(self, seed: int = 42):
        self._seed = seed

    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        self._doc_ids = [doc.doc_id for doc in docs]
        self._rng = random.Random(self._seed)

    def find(self, query: str, top_k: int = 10) -> list[int]:
        k = min(top_k, len(self._doc_ids))
        return self._rng.sample(self._doc_ids, k)
