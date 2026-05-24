"""Mem0 retriever: memory-based retrieval with LLM-powered fact extraction.

Requires external services:
  - Qdrant running on localhost:6333
  - LLM API (default: BigModel GLM-4-Flash via SiliconFlow)
  - Embedding API (default: BAAI/bge-m3 via SiliconFlow)

Install: pip install mem0ai qdrant-client
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from typing import Any

from .base import DatabaseEntry, Retriever

logger = logging.getLogger(__name__)

STATE_VERSION_FACT_EXTRACTION_PROMPT = """You extract memory facts for a temporal state-version benchmark.

The input is a single memory text describing one time-specific state of an event.
Extract factual statements that preserve:
- the event identity,
- the main actors or entities,
- the time phrase or time anchor if present,
- the state or version status described in the text.
- concrete evidence details that may later decide question answering, including:
  - numeric values,
  - explicit status phrases,
  - failure symptoms,
  - acceptance or rejection outcomes,
  - configuration names,
  - named actors,
  - causal clues.

Do not use outside knowledge.
Do not merge different event lines.
Do not drop explicit time distinctions.
Do not rewrite a detailed fact into a vague summary.
Stay close to the original wording and keep each fact self-contained.

Prefer 1 to 3 detailed facts over one short generic paraphrase.
If the memory text says uploads were stuck at 0% progress, keep that exact symptom.
If the memory text names a setting or branch, keep that exact name.

Return JSON with a top-level key `facts` whose value is a list of strings."""

STATE_VERSION_UPDATE_MEMORY_PROMPT = """You are maintaining long-term memories for a temporal state-version benchmark.

Each new fact may describe:
- a new version of the same event state,
- a revision of an older state,
- or a different nearby event line involving similar entities.

Rules:
1. Only UPDATE when the new fact clearly revises or settles the same event line as an old memory.
2. If two memories involve similar entities but different event lines or different branches, keep them separate with ADD.
3. Preserve time anchors and version distinctions whenever they are present.
4. Do not collapse multiple temporal versions into one vague summary unless the new fact explicitly replaces the older one.
5. Prefer ADD over UPDATE when uncertain.
6. Never replace a detailed memory with a shorter, vaguer paraphrase.
7. An UPDATED memory must keep the concrete evidence needed for downstream QA, such as:
   - numeric states like 0% progress,
   - explicit resolved/failed/reopened status,
   - named configuration keys,
   - who said what,
   - whether a problem persisted or was fixed.
8. If an older detailed memory and a newer detailed memory describe different stages of the same event, keep both with ADD unless the newer one explicitly supersedes the older wording.

Return JSON only in the format expected by the memory action schema."""


class Mem0Retriever(Retriever):
    """Mem0 memory-based retriever.

    Ingests documents via Mem0's add() with LLM fact extraction,
    retrieves via Mem0's search() with grouped results mapped back to doc_ids.
    Only ingests narrative docs — GitHub code is not meaningful for fact extraction.
    """

    @property
    def supported_sources(self) -> list[str] | None:
        return ["narrative"]

    def __init__(
        self,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        embedder_model: str | None = None,
        embedder_base_url: str | None = None,
        embedder_api_key: str | None = None,
    ):
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._llm_api_key = llm_api_key
        self._embedder_model = embedder_model
        self._embedder_base_url = embedder_base_url
        self._embedder_api_key = embedder_api_key
        self._memory = None
        self._docs: dict[int, str] = {}
        self._text_to_doc_id: dict[str, int] = {}
        self._doc_word_sets: dict[int, set[str]] = {}  # precomputed for fast overlap

    def _init_mem0(self):
        if self._memory is not None:
            return

        import contextlib

        @contextlib.contextmanager
        def _proxy_env_disabled():
            proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
            old_env = {key: os.environ.get(key) for key in proxy_keys}
            try:
                for key in proxy_keys:
                    os.environ.pop(key, None)
                yield
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        import httpx
        import mem0.embeddings.openai as mem0_openai_embedding_module
        import mem0.llms.openai as mem0_openai_llm_module
        from mem0 import Memory
        from openai import OpenAI

        original_embedding_openai = mem0_openai_embedding_module.OpenAI
        original_llm_openai = mem0_openai_llm_module.OpenAI

        def _openai_factory(*args, **kwargs):
            kwargs.setdefault("http_client", httpx.Client(timeout=60.0, trust_env=False))
            return OpenAI(*args, **kwargs)

        mem0_openai_embedding_module.OpenAI = _openai_factory
        mem0_openai_llm_module.OpenAI = _openai_factory
        try:
            with _proxy_env_disabled():
                self._memory = Memory.from_config(self._build_config())
        finally:
            mem0_openai_embedding_module.OpenAI = original_embedding_openai
            mem0_openai_llm_module.OpenAI = original_llm_openai

    @staticmethod
    def _checkpoint_path() -> str:
        runtime_dir = os.path.join(tempfile.gettempdir(), "bitempqa_mem0_runtime")
        os.makedirs(runtime_dir, exist_ok=True)
        return os.path.join(runtime_dir, "checkpoint.txt")

    def _load_checkpoint(self) -> set[int]:
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return set()
        with open(path) as f:
            return {int(line.strip()) for line in f if line.strip().isdigit()}

    def _save_checkpoint(self, doc_ids: list[int]) -> None:
        path = self._checkpoint_path()
        with open(path, "a") as f:
            for doc_id in doc_ids:
                f.write(f"{doc_id}\n")

    def _build_config(self) -> dict:
        # Fixed names so checkpoint resume can reuse the same collection
        self._user_id = "benchmark_user_persistent"
        self._collection_name = "mem0_benchmark"

        llm_api_key = self._llm_api_key or os.environ.get("OPENAI_API_KEY", "not-ollama")
        llm_base_url = self._llm_base_url or "http://localhost:9997/v1"
        llm_model = self._llm_model or "gemma4-26b-32k"

        # Default: local HuggingFace sentence-transformers (no API needed)
        embedder_model = self._embedder_model or "all-MiniLM-L6-v2"
        embedder_dims = 384  # all-MiniLM-L6-v2 dimension

        os.environ["OPENAI_API_KEY"] = llm_api_key
        os.environ["OPENAI_BASE_URL"] = llm_base_url

        runtime_dir = os.path.join(tempfile.gettempdir(), "bitempqa_mem0_runtime")
        os.makedirs(runtime_dir, exist_ok=True)
        history_db_path = os.path.join(runtime_dir, "history_persistent.db")

        embedder_config: dict[str, Any] = {
            "provider": "huggingface",
            "config": {
                "model": embedder_model,
            },
        }
        # If external embedder API is explicitly configured, use OpenAI provider instead
        if self._embedder_base_url:
            embedder_config = {
                "provider": "openai",
                "config": {
                    "model": self._embedder_model or "Pro/BAAI/bge-m3",
                    "openai_base_url": self._embedder_base_url,
                    "api_key": self._embedder_api_key or llm_api_key,
                },
            }
            embedder_dims = 1024

        return {
            "custom_fact_extraction_prompt": STATE_VERSION_FACT_EXTRACTION_PROMPT,
            "custom_update_memory_prompt": STATE_VERSION_UPDATE_MEMORY_PROMPT,
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "openai_base_url": llm_base_url,
                    "api_key": llm_api_key,
                    "temperature": 0.0,
                },
            },
            "embedder": embedder_config,
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "embedding_model_dims": embedder_dims,
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": self._collection_name,
                },
            },
            "history_db_path": history_db_path,
            "version": "v1.1",
        }

    def add_documents(self, docs: list[DatabaseEntry], max_workers: int = 3) -> None:
        print("  Initializing Mem0...")
        self._init_mem0()

        # Store all docs for lookup (needed for find() doc matching)
        for doc in docs:
            self._docs[doc.doc_id] = doc.text
            self._text_to_doc_id[doc.text[:200]] = doc.doc_id

        # Precompute word sets only for narrative docs (the ones actually ingested)
        for doc in docs:
            if doc.source == "narrative":
                self._doc_word_sets[doc.doc_id] = set(doc.text.split())

        # Only ingest narrative docs — GitHub code is not meaningful for
        # LLM fact extraction and would take days with a local gguf LLM
        narrative_docs = [doc for doc in docs if doc.source == "narrative"]

        # Load checkpoint and skip already-processed docs
        completed = self._load_checkpoint()
        remaining = [doc for doc in narrative_docs if doc.doc_id not in completed]
        print(f"  Checkpoint: {len(completed)} already done, {len(remaining)} remaining (total narrative: {len(narrative_docs)})")

        if not remaining:
            print(f"  Mem0 ingestion complete (all docs already processed)")
            return

        total = len(remaining)
        print(f"  Adding {total} docs to Mem0 with {max_workers} workers...")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        done_count = 0
        done_lock = threading.Lock()
        error_count = 0
        batch_to_save: list[int] = []

        def _add_one(doc: DatabaseEntry) -> bool:
            try:
                self._memory.add(
                    messages=[{"role": "user", "content": doc.text}],
                    user_id=self._user_id,
                    metadata={"doc_id": doc.doc_id},
                    infer=True,
                )
                return True
            except Exception as exc:
                logger.warning("Mem0 add error for doc %d: %s", doc.doc_id, exc)
                return False

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_add_one, doc): doc for doc in remaining}
            for future in as_completed(futures):
                doc = futures[future]
                ok = future.result()
                with done_lock:
                    done_count += 1
                    if not ok:
                        error_count += 1
                    else:
                        batch_to_save.append(doc.doc_id)
                    if len(batch_to_save) >= 10:
                        self._save_checkpoint(batch_to_save)
                        batch_to_save = []
                    if done_count % 50 == 0:
                        print(f"    Progress: {done_count}/{total} (errors: {error_count})")

        if batch_to_save:
            self._save_checkpoint(batch_to_save)

        print(f"  Mem0 ingestion complete: {done_count}/{total} docs, {error_count} errors")

    def find(self, query: str, top_k: int = 10) -> list[int]:
        if self._memory is None:
            return []

        try:
            internal_k = max(20, 6 * top_k)
            result = self._memory.search(
                query=query,
                user_id=self._user_id,
                limit=internal_k,
            )

            raw_items: list[dict[str, Any]] = []
            if isinstance(result, dict) and "results" in result:
                raw_items = [item for item in result["results"] if isinstance(item, dict)]
            elif isinstance(result, list):
                raw_items = [item for item in result if isinstance(item, dict)]

            # Strategy 1: Metadata-based matching (direct doc_id from add metadata)
            doc_scores: dict[int, float] = {}
            metadata_matched = False
            for item in raw_items:
                score = float(item.get("score", 0.0))
                metadata = item.get("metadata") or {}
                doc_id = metadata.get("doc_id")
                if isinstance(doc_id, int) and doc_id in self._docs:
                    doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), score)
                    metadata_matched = True

            # Strategy 2: Word overlap fallback (for data without metadata)
            if not metadata_matched and self._doc_word_sets:
                for item in raw_items:
                    memory_text = (item.get("memory", "") or item.get("text", "")).strip()
                    score = float(item.get("score", 0.0))

                    mem_words = set(memory_text.split())
                    best_doc_id = None
                    best_overlap = 0
                    for doc_id, doc_words in self._doc_word_sets.items():
                        overlap = len(mem_words & doc_words)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_doc_id = doc_id

                    if best_doc_id is not None and best_overlap > 0:
                        doc_scores[best_doc_id] = max(doc_scores.get(best_doc_id, 0.0), score)

            ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            return [doc_id for doc_id, _ in ranked[:top_k]]

        except Exception as exc:
            logger.warning("Mem0 search error: %s", exc)
            return []
