"""Hybrid retriever: Reciprocal Rank Fusion of BM25 + dense retrieval."""

from __future__ import annotations

from .base import DatabaseEntry, Retriever
from .bm25_retriever import BM25Retriever
from .faiss_retriever import FAISSRetriever


class HybridRetriever(Retriever):
    """Reciprocal Rank Fusion (RRF) of BM25 and FAISS results.

    RRF score = sum(1 / (k + rank)) for each component retriever.
    Robust to different score scales — only uses rank positions.
    """

    def __init__(
        self,
        rrf_k: int = 60,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        overlap: int = 64,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ):
        self._bm25 = BM25Retriever()
        self._faiss = FAISSRetriever(
            model_name=model_name, chunk_size=chunk_size, overlap=overlap,
        )
        self._rrf_k = rrf_k
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight

    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        print("  [Hybrid] Building BM25 index...")
        self._bm25.add_documents(docs)
        print("  [Hybrid] Building FAISS index...")
        self._faiss.add_documents(docs)

    def find(self, query: str, top_k: int = 10) -> list[int]:
        expand = max(top_k * 5, 100)

        bm25_ids = self._bm25.find(query, top_k=expand)
        faiss_ids = self._faiss.find(query, top_k=expand)

        # RRF fusion with weights
        scores: dict[int, float] = {}
        for rank, doc_id in enumerate(bm25_ids, 1):
            scores[doc_id] = scores.get(doc_id, 0) + self._bm25_weight / (self._rrf_k + rank)
        for rank, doc_id in enumerate(faiss_ids, 1):
            scores[doc_id] = scores.get(doc_id, 0) + self._dense_weight / (self._rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_k]]
