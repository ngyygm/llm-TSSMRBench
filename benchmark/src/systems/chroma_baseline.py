"""ChromaDB baseline：语义检索后端。"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from .base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)


class ChromaBaseline(MemorySystem):
    """ChromaDB semantic retrieval baseline."""

    def __init__(
        self,
        name: str = "ChromaDB",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 5,
        expose_metadata_in_context: bool = False,
    ):
        super().__init__(name)
        self.encoder = SentenceTransformer(embedding_model)
        self.top_k = top_k
        # 中文注释：主实验默认不把命中结果的 metadata 手动拼回上下文。
        self.expose_metadata_in_context = expose_metadata_in_context

        import chromadb

        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="bitpqa_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self._all_docs: List[str] = []

    def remember(
        self,
        text: str,
    ) -> str:
        write_id = f"chroma_{self.collection.count()}"
        emb = self.encoder.encode([text]).tolist()
        self.collection.add(
            ids=[write_id],
            documents=[text],
            embeddings=emb,
        )
        self._all_docs.append(text)
        return write_id

    @staticmethod
    def _format_with_metadata(doc: str, metadata: Optional[dict]) -> str:
        """构造 metadata-exposed 补充实验使用的检索文本。"""
        metadata = metadata or {}
        parts = [doc]
        if metadata.get("event_time"):
            parts.append(f"event_time={metadata['event_time']}")
        if metadata.get("record_time"):
            parts.append(f"record_time={metadata['record_time']}")
        if metadata.get("source"):
            parts.append(f"source={metadata['source']}")
        return " | ".join(parts)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        start = time.time()

        effective_top_k = top_k if top_k is not None else self.top_k
        kwargs = {
            "query_embeddings": self.encoder.encode([question]).tolist(),
            "n_results": min(effective_top_k, self.collection.count() or 1),
        }

        try:
            results = self.collection.query(**kwargs)
            docs = results["documents"][0] if results["documents"] else []
        except Exception as e:
            logger.warning("ChromaDB query failed: %s", e)
            docs = []

        facts = docs

        context = "\n".join(facts)
        return QueryResult(
            answer=context,
            retrieved_context=context,
            retrieved_facts=facts,
            latency_ms=(time.time() - start) * 1000,
            metadata={
                "num_results": len(docs),
                "metadata_exposed": False,
            },
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection("bitpqa_memory")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="bitpqa_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self._all_docs = []
