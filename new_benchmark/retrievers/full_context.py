"""Full context baseline — returns all documents (perfect retrieval)."""

from __future__ import annotations

from .base import DatabaseEntry, Retriever


class FullContextRetriever(Retriever):
    """Perfect-retrieval baseline: returns all doc_ids regardless of query."""

    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        self._doc_ids = [doc.doc_id for doc in docs]

    def find(self, query: str, top_k: int = 10) -> list[int]:
        return list(self._doc_ids)

    @property
    def is_ranked(self) -> bool:
        return False
