"""ChromaDB retriever: semantic vector search with sentence-transformers."""

from __future__ import annotations

from .base import DatabaseEntry, Retriever


class ChromaDBRetriever(Retriever):
    """ChromaDB with HNSW cosine space for semantic retrieval.

    No external services needed beyond pip install chromadb sentence-transformers.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self._model_name = embedding_model
        self._encoder = None
        self._client = None
        self._collection = None
        self._doc_ids: list[int] = []

    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        print(f"  Loading encoder: {self._model_name}")
        self._encoder = SentenceTransformer(self._model_name)

        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name="bitempqa",
            metadata={"hnsw:space": "cosine"},
        )

        self._doc_ids = [doc.doc_id for doc in docs]
        texts = [doc.text for doc in docs]

        # Batch encode and insert
        batch_size = 256
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = [str(did) for did in self._doc_ids[i:i + batch_size]]
            batch_embs = self._encoder.encode(batch_texts).tolist()
            self._collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embs,
            )
        print(f"  Indexed {len(docs)} docs in ChromaDB")

    def find(self, query: str, top_k: int = 10) -> list[int]:
        if self._collection is None or self._collection.count() == 0:
            return []

        n_results = min(top_k, self._collection.count())
        results = self._collection.query(
            query_embeddings=self._encoder.encode([query]).tolist(),
            n_results=n_results,
        )

        ids = results["ids"][0] if results["ids"] else []
        return [int(doc_id) for doc_id in ids]
