"""Mem0 baseline."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from src.systems.base import MemorySystem, QueryResult

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


class Mem0Baseline(MemorySystem):
    """Mem0 baseline."""

    @staticmethod
    @contextlib.contextmanager
    def _proxy_env_disabled():
        proxy_keys = (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        )
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

    @staticmethod
    def _create_mem0_memory(memory_config):
        import httpx
        import mem0.llms.deepseek as mem0_deepseek_llm_module
        import mem0.embeddings.openai as mem0_openai_embedding_module
        import mem0.llms.openai as mem0_openai_llm_module
        from mem0 import Memory
        from openai import OpenAI

        original_embedding_openai = mem0_openai_embedding_module.OpenAI
        original_llm_openai = mem0_openai_llm_module.OpenAI
        def _openai_factory(*args, **kwargs):
            kwargs.setdefault("http_client", httpx.Client(timeout=60.0, trust_env=False))
            return OpenAI(*args, **kwargs)

        def _deepseek_generate_with_disabled_thinking(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
            params = self._get_supported_params(messages=messages, **kwargs)
            params.update(
                {
                    "model": self.config.model,
                    "messages": messages,
                    "extra_body": {"thinking": {"type": "disabled"}},
                }
            )
            if response_format:
                params["response_format"] = response_format
            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice
            response = self.client.chat.completions.create(**params)
            return self._parse_response(response, tools)

        mem0_openai_embedding_module.OpenAI = _openai_factory
        mem0_openai_llm_module.OpenAI = _openai_factory
        mem0_deepseek_llm_module.DeepSeekLLM.generate_response = _deepseek_generate_with_disabled_thinking
        try:
            with Mem0Baseline._proxy_env_disabled():
                return Memory.from_config(memory_config)
        finally:
            mem0_openai_embedding_module.OpenAI = original_embedding_openai
            mem0_openai_llm_module.OpenAI = original_llm_openai

    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedder_model: Optional[str] = None,
        embedder_base_url: Optional[str] = None,
        embedder_api_key: Optional[str] = None,
        expose_metadata_in_context: bool = False,
        top_k: int = 5,
        use_infer_updates: bool = True,
        chain_ingest_workers: int = 1,
        internal_fact_k: Optional[int] = None,
        run_id: Optional[str] = None,
    ):
        super().__init__(name="Mem0")
        self.expose_metadata_in_context = expose_metadata_in_context
        self.top_k = top_k
        self.use_infer_updates = use_infer_updates
        self.chain_ingest_workers = max(1, int(chain_ingest_workers))
        self.internal_fact_k = max(1, int(internal_fact_k)) if internal_fact_k is not None else None
        self._instance_tag = self._normalize_instance_tag(run_id) if run_id else self._build_instance_tag()
        self.user_id = f"benchmark_user_{self._instance_tag}"
        self.collection_name = f"mem0_state_version_{self._instance_tag}"
        self._chain_counter = 0
        self._chain_counter_lock = threading.Lock()

        runtime_dir = os.path.join(tempfile.gettempdir(), "bitempqa_mem0_runtime")
        os.makedirs(runtime_dir, exist_ok=True)
        history_db_path = os.path.join(runtime_dir, f"history_{self._instance_tag}.db")
        os.environ["MEM0_DIR"] = runtime_dir

        llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        llm_base_url = llm_base_url or "https://open.bigmodel.cn/api/paas/v4"
        llm_model = llm_model or "GLM-4-Flash"
        embedder_api_key = embedder_api_key or llm_api_key
        embedder_base_url = (embedder_base_url or "https://api.siliconflow.cn/v1").rstrip("/")
        if embedder_base_url.endswith("/embeddings"):
            embedder_base_url = embedder_base_url[: -len("/embeddings")]

        os.environ["OPENAI_API_KEY"] = llm_api_key
        os.environ["OPENAI_BASE_URL"] = llm_base_url

        config = {
            "custom_fact_extraction_prompt": STATE_VERSION_FACT_EXTRACTION_PROMPT,
            "custom_update_memory_prompt": STATE_VERSION_UPDATE_MEMORY_PROMPT,
            "llm": {
                "provider": "deepseek" if "deepseek" in (llm_base_url or "").lower() else "openai",
                "config": (
                    {
                        "model": llm_model,
                        "deepseek_base_url": llm_base_url,
                        "api_key": llm_api_key,
                        "temperature": 0.0,
                    }
                    if "deepseek" in (llm_base_url or "").lower()
                    else {
                        "model": llm_model,
                        "openai_base_url": llm_base_url,
                        "api_key": llm_api_key,
                        "temperature": 0.0,
                    }
                ),
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedder_model or "Pro/BAAI/bge-m3",
                    "openai_base_url": embedder_base_url,
                    "api_key": embedder_api_key,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "embedding_model_dims": 1024,
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": self.collection_name,
                },
            },
            "history_db_path": history_db_path,
            "version": "v1.1",
        }

        self._memory_config = config
        self.m = self._create_mem0_memory(self._memory_config)
        logger.info(
            "Mem0 initialized (llm_model=%s, embedder_model=%s, metadata_exposed=%s, infer_updates=%s, user_id=%s, collection_name=%s)",
            llm_model,
            embedder_model or "Pro/BAAI/bge-m3",
            self.expose_metadata_in_context,
            self.use_infer_updates,
            self.user_id,
            self.collection_name,
        )

    @staticmethod
    def _build_instance_tag() -> str:
        raw = f"{os.getpid()}_{time.time_ns()}"
        return re.sub(r"[^A-Za-z0-9_]+", "_", raw)

    @staticmethod
    def _normalize_instance_tag(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value or "").strip("_")
        return normalized or Mem0Baseline._build_instance_tag()

    def _next_source_chain_id(self) -> str:
        with self._chain_counter_lock:
            self._chain_counter += 1
            return f"chain_{self._chain_counter:05d}"

    def _remember_with_metadata(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        source_chain_id = (metadata or {}).get("source_chain_id")
        source_node_id = (metadata or {}).get("source_node_id")
        text_len = len(text or "")
        started_at = time.time()
        logger.info(
            "Mem0 add start chain=%s node=%s text_len=%s infer=%s",
            source_chain_id,
            source_node_id,
            text_len,
            self.use_infer_updates,
        )
        try:
            result = self.m.add(
                messages=[{"role": "user", "content": text}],
                user_id=self.user_id,
                metadata=metadata,
                infer=self.use_infer_updates,
            )
            elapsed_ms = (time.time() - started_at) * 1000
            if isinstance(result, dict) and "results" in result:
                memories = result["results"]
                if memories:
                    memory_id = str(memories[0].get("id", ""))
                    logger.info(
                        "Mem0 add success chain=%s node=%s elapsed_ms=%.1f memory_id=%s result_count=%s",
                        source_chain_id,
                        source_node_id,
                        elapsed_ms,
                        memory_id,
                        len(memories),
                    )
                    return memory_id
            fallback_id = f"mem0_{int(time.time())}"
            logger.info(
                "Mem0 add completed without explicit result list chain=%s node=%s elapsed_ms=%.1f fallback_id=%s",
                source_chain_id,
                source_node_id,
                elapsed_ms,
                fallback_id,
            )
            return fallback_id
        except Exception as exc:
            elapsed_ms = (time.time() - started_at) * 1000
            logger.warning(
                "Mem0 add error chain=%s node=%s elapsed_ms=%.1f error=%s",
                source_chain_id,
                source_node_id,
                elapsed_ms,
                exc,
            )
            return ""

    def remember(self, text: str) -> str:
        return self._remember_with_metadata(text=text, metadata=None)

    def remember_many(self, texts: list[str]) -> list[str]:
        source_chain_id = self._next_source_chain_id()
        memory_ids: list[str] = []
        for index, text in enumerate(texts, start=1):
            memory_ids.append(
                self._remember_with_metadata(
                    text=text,
                    metadata={
                        "source_chain_id": source_chain_id,
                        "source_node_id": f"{source_chain_id}_node_{index:04d}",
                        "source_surface_order": index,
                    },
                )
            )
        return memory_ids

    def remember_chain(self, chain_id: str, node_ids: list[str], texts: list[str]) -> list[str]:
        memory_ids: list[str] = []
        for index, text in enumerate(texts):
            node_id = node_ids[index] if index < len(node_ids) else f"{chain_id}_node_{index + 1:04d}"
            memory_ids.append(
                self._remember_with_metadata(
                    text=text,
                    metadata={
                        "source_chain_id": chain_id,
                        "source_node_id": node_id,
                        "source_surface_order": index + 1,
                    },
                )
            )
        return memory_ids

    @staticmethod
    def _memory_text(item: Dict[str, Any]) -> str:
        return (item.get("memory", "") or item.get("text", "")).strip()

    @staticmethod
    def _metadata_value(item: Dict[str, Any], key: str) -> Any:
        metadata = item.get("metadata")
        if isinstance(metadata, dict) and key in metadata:
            return metadata.get(key)
        return item.get(key)

    @classmethod
    def _format_memory_fact(cls, item: Dict[str, Any]) -> str:
        memory_text = cls._memory_text(item)
        metadata = item.get("metadata") or {}
        parts = [memory_text] if memory_text else []
        source_name = metadata.get("source")
        if source_name:
            parts.append(f"source={source_name}")
        return " | ".join(part for part in parts if part)

    @classmethod
    def _group_raw_results_by_source_node(cls, raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}

        for item in raw_items:
            source_node_id = cls._metadata_value(item, "source_node_id") or item.get("id") or "unknown_node"
            source_chain_id = cls._metadata_value(item, "source_chain_id")
            surface_order = cls._metadata_value(item, "source_surface_order")
            fact_text = cls._memory_text(item)
            score = float(item.get("score") or 0.0)

            group = grouped.setdefault(
                str(source_node_id),
                {
                    "source_node_id": str(source_node_id),
                    "source_chain_id": source_chain_id,
                    "source_surface_order": surface_order,
                    "facts": [],
                    "fact_scores": [],
                    "max_score": score,
                    "score_sum": 0.0,
                },
            )
            if fact_text:
                group["facts"].append(fact_text)
                group["fact_scores"].append({"text": fact_text, "score": score})
            group["max_score"] = max(float(group["max_score"]), score)
            group["score_sum"] = float(group["score_sum"]) + score

        grouped_results = list(grouped.values())
        grouped_results.sort(
            key=lambda item: (
                -float(item["max_score"]),
                -len(item["facts"]),
                -float(item["score_sum"]),
                str(item["source_node_id"]),
            )
        )
        return grouped_results

    @staticmethod
    def _bundle_text(group: Dict[str, Any]) -> str:
        seen: set[str] = set()
        unique_ranked_facts: list[str] = []
        for fact_entry in sorted(group.get("fact_scores", []), key=lambda item: -float(item.get("score") or 0.0)):
            text = (fact_entry.get("text") or "").strip()
            if text and text not in seen:
                seen.add(text)
                unique_ranked_facts.append(text)

        if not unique_ranked_facts:
            return ""

        bullet_lines = [f"- {text}" for text in unique_ranked_facts]
        return "\n".join(bullet_lines)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        start_time = time.time()
        try:
            effective_top_k = top_k if top_k is not None else self.top_k
            internal_fact_k = self.internal_fact_k if self.internal_fact_k is not None else max(20, 6 * effective_top_k)
            result = self.m.search(
                query=question,
                user_id=self.user_id,
                limit=internal_fact_k,
            )

            raw_items: List[Dict[str, Any]] = []
            if isinstance(result, dict) and "results" in result:
                raw_items = [item for item in result["results"] if isinstance(item, dict)]
            elif isinstance(result, list):
                raw_items = [item for item in result if isinstance(item, dict)]

            grouped_results = self._group_raw_results_by_source_node(raw_items)
            top_groups = grouped_results[:effective_top_k]

            facts: List[str] = []
            for group in top_groups:
                bundled_text = self._bundle_text(group)
                if bundled_text:
                    facts.append(bundled_text)

            retrieved_context = "\n".join(facts) if facts else ""
            return QueryResult(
                answer=retrieved_context,
                retrieved_context=retrieved_context,
                retrieved_facts=facts,
                confidence=1.0 if facts else 0.0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={
                    "num_results": len(facts),
                    "internal_fact_k": internal_fact_k,
                    "metadata_exposed": self.expose_metadata_in_context,
                    "raw_results": raw_items,
                    "grouped_results": top_groups,
                },
            )
        except Exception as exc:
            logger.warning("Mem0 search error: %s", exc)
            return QueryResult(
                answer="",
                retrieved_context="",
                retrieved_facts=[],
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(exc)},
            )

    def reset(self) -> None:
        reset_done = False
        try:
            self.m.delete_all(user_id=self.user_id)
            reset_done = True
        except Exception as exc:
            logger.warning("Mem0 reset error: %s", exc)

        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(host="localhost", port=6333)
            existing = {collection.name for collection in client.get_collections().collections}
            if self.collection_name in existing:
                client.delete_collection(self.collection_name)
                reset_done = True
        except Exception as exc:
            logger.warning("Mem0 Qdrant reset fallback error: %s", exc)

        try:
            history_db_path = self._memory_config.get("history_db_path")
            if history_db_path:
                for suffix in ("", "-journal", "-wal", "-shm"):
                    file_path = f"{history_db_path}{suffix}"
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except OSError:
                            logger.warning("Mem0 history file remove skipped: %s", file_path)
            self.m = self._create_mem0_memory(self._memory_config)
        except Exception as exc:
            logger.warning("Mem0 reinitialize error after reset: %s", exc)

        if reset_done:
            logger.info("Mem0 reset complete")
