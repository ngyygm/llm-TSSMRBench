"""Cross-encoder re-ranking retriever: BM25 first stage + cross-encoder re-rank."""

from __future__ import annotations

from .base import DatabaseEntry, Retriever
from .bm25_retriever import BM25Retriever


class CrossEncoderRetriever(Retriever):
    """Two-stage pipeline: BM25 retrieves candidates, cross-encoder re-ranks.

    Uses sentence-transformers CrossEncoder for re-ranking.
    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        n_candidates: int = 100,
        max_text_length: int = 512,
    ):
        self._model_name = model_name
        self._n_candidates = n_candidates
        self._max_text_length = max_text_length
        self._bm25 = BM25Retriever()
        self._model = None
        self._docs: dict[int, str] = {}

    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        self._docs = {doc.doc_id: doc.text for doc in docs}
        self._bm25.add_documents(docs)

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"  Loading cross-encoder: {self._model_name}")
            self._model = CrossEncoder(self._model_name)

    def find(self, query: str, top_k: int = 10) -> list[int]:
        self._load_model()

        # Stage 1: BM25 retrieves candidates
        n_cand = max(self._n_candidates, top_k * 3)
        candidates = self._bm25.find(query, top_k=n_cand)

        # Stage 2: Cross-encoder re-ranks
        pairs = [
            (query, self._docs[doc_id][:self._max_text_length])
            for doc_id in candidates
        ]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_k]]
