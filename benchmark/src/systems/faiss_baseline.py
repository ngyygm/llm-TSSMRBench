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
        # Parallel provenance lists, aligned with self.texts by position, so that
        # coverage/CSR can be scored by exact node identity (consumed by
        # evaluation._explicit_retrieved_pairs_from_metadata) instead of fuzzy
        # text matching.
        self.node_ids: List[str] = []
        self.chain_ids: List[str] = []
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
        self.node_ids.append(f"faiss_{len(self.texts)}")
        self.chain_ids.append("")
        emb = self._embed([text])
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        self._rebuild_index()
        return f"faiss_{len(self.texts)}"

    def remember_chain(self, chain_id: str, node_ids: List[str], texts: List[str]) -> List[str]:
        """Ingest one chain while preserving per-node source identity.

        Embeds in a single batch to keep ingestion efficient.
        """
        if not texts:
            return []
        ids: List[str] = []
        for index, text in enumerate(texts):
            node_id = node_ids[index] if index < len(node_ids) else f"{chain_id}_node_{index + 1:04d}"
            self.texts.append(text)
            self.node_ids.append(node_id)
            self.chain_ids.append(chain_id)
            ids.append(node_id)
        emb = self._embed(texts)
        self.embeddings = emb if self.embeddings is None else np.vstack([self.embeddings, emb])
        self._rebuild_index()
        return ids

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
        retrieved_node_ids: List[str] = []
        retrieved_chain_ids: List[str] = []
        has_provenance = len(self.node_ids) == len(self.texts) and len(self.chain_ids) == len(self.texts)
        for rank, idx in enumerate(indices[0]):
            score = float(scores[0][rank])
            if idx >= 0 and score > 0:
                retrieved.append(self.texts[idx])
                candidate_scores.append(score)
                if has_provenance:
                    retrieved_node_ids.append(self.node_ids[idx])
                    retrieved_chain_ids.append(self.chain_ids[idx])

        max_score = max(candidate_scores) if candidate_scores else 0.0

        metadata = {
            "embedding_model": self.embedding_model,
            "embedding_base_url": self.embedding_base_url,
            "implementation": "faiss" if self._faiss_available else "numpy_inner_product_fallback",
        }
        # Emit exact source node ids so evaluation scores coverage by identity.
        if has_provenance:
            metadata["retrieved_source_node_ids"] = retrieved_node_ids
            metadata["retrieved_source_chain_ids"] = retrieved_chain_ids

        return QueryResult(
            answer="\n".join(retrieved),
            retrieved_context="\n".join(retrieved),
            retrieved_facts=retrieved,
            confidence=max_score if candidate_scores else 0.0,
            latency_ms=(time.time() - start) * 1000,
            metadata=metadata,
        )

    def reset(self) -> None:
        self.texts = []
        self.node_ids = []
        self.chain_ids = []
        self.embeddings = None
        self.index = None
