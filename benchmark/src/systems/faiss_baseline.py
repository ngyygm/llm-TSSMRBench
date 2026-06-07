"""FAISS vector store baseline with remote embedding API support."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import httpx
import numpy as np
from openai import OpenAI

from .base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)


class FAISSBaseline(MemorySystem):
    """FAISS-based dense retrieval baseline."""

    def __init__(
        self,
        name: str = "FAISS Vector Store",
        embedding_model: str = "BAAI/bge-m3",
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        top_k: int = 5,
    ):
        super().__init__(name)
        self.embedding_model = embedding_model
        self.embedding_base_url = self._normalize_base_url(embedding_base_url or "https://api.siliconflow.cn/v1")
        self.embedding_api_key = embedding_api_key or ""
        http_client = httpx.Client(timeout=60, trust_env=False)
        self.client = OpenAI(
            base_url=self.embedding_base_url,
            api_key=self.embedding_api_key,
            timeout=60,
            http_client=http_client,
        )
        self.top_k = top_k
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None

        try:
            import faiss

            self.faiss = faiss
            self._faiss_available = True
        except ImportError:
            self.faiss = None
            self._faiss_available = False
            logger.warning("faiss not installed; FAISS baseline will use NumPy inner-product fallback")

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        cleaned = base_url.rstrip("/")
        if cleaned.endswith("/embeddings"):
            cleaned = cleaned[: -len("/embeddings")]
        return cleaned

    def _embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
            encoding_format="float",
        )
        matrix = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def remember(self, text: str) -> str:
        self.texts.append(text)
        emb = self._embed([text])
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        self._rebuild_index()
        return f"faiss_{len(self.texts)}"

    def _rebuild_index(self) -> None:
        if self.embeddings is None:
            self.index = None
            return
        if self._faiss_available:
            dim = self.embeddings.shape[1]
            self.index = self.faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings.astype(np.float32))
        else:
            self.index = self.embeddings.astype(np.float32)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        start = time.time()
        if self.index is None or not self.texts:
            return QueryResult(answer="", latency_ms=(time.time() - start) * 1000)

        q_emb = self._embed([question])
        effective_top_k = top_k if top_k is not None else self.top_k
        if self._faiss_available:
            scores, indices = self.index.search(q_emb.astype(np.float32), min(effective_top_k, len(self.texts)))
        else:
            flat_scores = (self.index @ q_emb[0].astype(np.float32)).astype(np.float32)
            order = np.argsort(flat_scores)[::-1][: min(effective_top_k, len(self.texts))]
            scores = np.array([flat_scores[order]], dtype=np.float32)
            indices = np.array([order], dtype=np.int64)

        retrieved = []
        candidate_scores = []
        for rank, idx in enumerate(indices[0]):
            score = float(scores[0][rank])
            if idx >= 0 and score > 0:
                retrieved.append(self.texts[idx])
                candidate_scores.append(score)

        max_score = max(candidate_scores) if candidate_scores else 0.0

        return QueryResult(
            answer="\n".join(retrieved),
            retrieved_context="\n".join(retrieved),
            retrieved_facts=retrieved,
            confidence=max_score if candidate_scores else 0.0,
            latency_ms=(time.time() - start) * 1000,
            metadata={
                "embedding_model": self.embedding_model,
                "embedding_base_url": self.embedding_base_url,
                "implementation": "faiss" if self._faiss_available else "numpy_inner_product_fallback",
            },
        )

    def reset(self) -> None:
        self.texts = []
        self.embeddings = None
        self.index = None
