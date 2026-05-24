"""Graphiti retriever: temporal knowledge graph retrieval.

Uses:
  - Neo4j (bolt://localhost:7687) for graph storage
  - Local xinference LLM for KG extraction
  - Local sentence-transformers for embeddings
  - Local cross-encoder for reranking

Install: pip install graphiti-core neo4j httpx openai pydantic sentence-transformers
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import typing
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .base import DatabaseEntry, Retriever

from graphiti_core.llm_client.client import LLMClient

logger = logging.getLogger(__name__)

# ─── Shared async event loop ───

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_ready = threading.Event()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    if _loop is None or _loop.is_closed() or _loop_thread is None or not _loop_thread.is_alive():
        _loop_ready.clear()
        _loop = asyncio.new_event_loop()

        def _run_loop(loop: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(loop)
            _loop_ready.set()
            loop.run_forever()

        _loop_thread = threading.Thread(target=_run_loop, args=(_loop,), name="graphiti-event-loop", daemon=True)
        _loop_thread.start()
        _loop_ready.wait()
    return _loop


def _run_async(coro):
    future = asyncio.run_coroutine_threadsafe(coro, _get_loop())
    return future.result()


# ─── Local sentence-transformers embedder ───

class LocalEmbedderClient:
    """EmbedderClient backed by local sentence-transformers (no API needed)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    async def create(self, input_data) -> list[float]:
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list):
            texts = input_data
        else:
            texts = [str(input_data)]
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings[0].tolist()


# ─── Local cross-encoder reranker ───

class LocalCrossEncoderClient:
    """CrossEncoderClient backed by local cross-encoder model (no API needed)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"  Loading cross-encoder reranker: {self._model_name}")
            self._model = CrossEncoder(self._model_name)

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        self._load_model()
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(passages, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked


# ─── Robust OpenAI-compatible LLM client ───

class RobustOpenAIGenericClient(LLMClient):
    """LLMClient subclass that bypasses graphiti-core's retry mechanism.

    graphiti-core's _generate_response_with_retry retries on JSONDecodeError,
    but gguf models like gemma produce unpredictable JSON formatting (code fences,
    trailing commas, etc). We override generate_response() directly to:
      1. Inject JSON schema (as base class does)
      2. Call the LLM directly (no retry decorator)
      3. Use robust multi-strategy JSON parsing
      4. Return a valid dict on best-effort basis (never raise JSONDecodeError)
    """

    def __init__(self, config, cache: bool = False):
        super().__init__(config, cache)
        self._openai_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=getattr(config, "base_url", None),
        )

    @staticmethod
    def _parse_json_robust(raw_text: str) -> dict[str, typing.Any] | None:
        import re
        import yaml
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines: lines = lines[1:]
            if lines and lines[-1].strip() == "```": lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
        for candidate in [cleaned, re.sub(r",(\s*[}\]])", r"\1", cleaned)]:
            parsed = None
            try:
                parsed = json.loads(candidate)
            except Exception:
                try:
                    parsed = yaml.safe_load(candidate)
                except Exception:
                    continue
            if not isinstance(parsed, dict):
                continue
            return RobustOpenAIGenericClient._strip_schema_wrapper(parsed)
        return None

    @staticmethod
    def _strip_schema_wrapper(parsed: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Remove JSON schema metadata that gguf models echo back.

        Gemma produces three patterns:
        1. Schema+data at root:
           {"properties": {...}, "title": "...", "extracted_node_names": ["data"]}
        2. Data inside properties:
           {"properties": {"nodes": [...], "edges": [...]}}
        3. Pure schema echo (no data):
           {"properties": {"field": {"type": "array", ...}}, "title": "...", "type": "object"}
        """
        schema_keys = {"properties", "title", "type", "required", "description",
                       "default", "items", "$schema", "additionalProperties"}

        # Pattern 1: Data at root alongside schema metadata
        root_data_keys = {k for k, v in parsed.items()
                         if k not in schema_keys and not isinstance(v, (dict,)) or
                         (k not in schema_keys and isinstance(v, (list,)) and v)}
        if root_data_keys:
            return {k: v for k, v in parsed.items() if k not in schema_keys}

        # Pattern 2: Data inside "properties"
        if "properties" in parsed and isinstance(parsed["properties"], dict):
            inner = parsed["properties"]
            has_data = any(
                (isinstance(v, list) and v) or
                (isinstance(v, dict) and not all(k in schema_keys for k in v))
                for v in inner.values()
            )
            if has_data:
                return inner

        # Pattern 3: Pure schema echo — extract field names and return defaults
        # Build {"field1": default1, "field2": default2} from schema metadata
        if "properties" in parsed and isinstance(parsed["properties"], dict):
            inner = parsed["properties"]
            defaults: dict[str, typing.Any] = {}
            for field_name, field_schema in inner.items():
                if isinstance(field_schema, dict):
                    ftype = field_schema.get("type", "")
                    if ftype == "array":
                        defaults[field_name] = field_schema.get("default", [])
                    elif ftype == "string":
                        defaults[field_name] = field_schema.get("default", "")
                    elif ftype == "boolean":
                        defaults[field_name] = field_schema.get("default", False)
                    elif ftype == "integer":
                        defaults[field_name] = field_schema.get("default", 0)
                    elif "default" in field_schema:
                        defaults[field_name] = field_schema["default"]
                    else:
                        defaults[field_name] = None
            if defaults:
                return defaults

        return parsed

    @staticmethod
    def _build_default(response_model: type[BaseModel]) -> dict[str, typing.Any]:
        """Build a valid default dict for a Pydantic model, even without default constructors."""
        schema = response_model.model_json_schema()
        props = schema.get("properties", {})
        defs = schema.get("$defs", schema.get("definitions", {}))
        result: dict[str, typing.Any] = {}
        for field_name, field_info in props.items():
            if "default" in field_info:
                result[field_name] = field_info["default"]
            elif field_info.get("type") == "array":
                result[field_name] = []
            elif field_info.get("type") == "string":
                result[field_name] = ""
            elif field_info.get("type") == "boolean":
                result[field_name] = False
            elif field_info.get("type") == "integer":
                result[field_name] = 0
            elif field_info.get("type") == "number":
                result[field_name] = 0.0
            elif "$ref" in field_info:
                result[field_name] = {}
            else:
                result[field_name] = None
        try:
            return response_model.model_validate(result).model_dump()
        except Exception:
            return result

    @staticmethod
    def _coerce_to_schema(
        parsed: dict[str, typing.Any], response_model: type[BaseModel],
    ) -> dict[str, typing.Any] | None:
        """Try to coerce gemma's output to match the expected Pydantic schema."""
        import pydantic_core
        schema = response_model.model_json_schema()
        props = schema.get("properties", {})
        result: dict[str, typing.Any] = {}

        for field_name, field_info in props.items():
            if field_name not in parsed:
                # Use default if available
                if "default" in field_info:
                    result[field_name] = field_info["default"]
                else:
                    return None
                continue

            value = parsed[field_name]
            expected_type = field_info.get("type", "")

            if expected_type == "array" and not isinstance(value, list):
                item_type = field_info.get("items", {}).get("type", "string")
                if isinstance(value, str) and item_type == "string":
                    result[field_name] = [value]
                elif isinstance(value, dict):
                    result[field_name] = [value]
                else:
                    return None
            elif expected_type == "string" and isinstance(value, list):
                result[field_name] = ", ".join(str(v) for v in value)
            elif expected_type == "string" and isinstance(value, dict):
                result[field_name] = json.dumps(value)
            elif expected_type == "boolean" and isinstance(value, str):
                result[field_name] = value.lower() in ("true", "yes", "1")
            else:
                result[field_name] = value

        try:
            return response_model.model_validate(result).model_dump()
        except Exception:
            return None

    async def generate_response(
        self,
        messages,
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 16384,
    ) -> dict[str, typing.Any]:
        # Inject JSON schema into prompt (same as base class)
        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[-1].content += (
                f"\n\nRespond with a JSON object in the following format:\n\n{serialized_model}"
            )

        openai_messages: list[ChatCompletionMessageParam] = []
        for message in messages:
            content = self._clean_input(message.content)
            openai_messages.append({"role": message.role, "content": content})

        extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if self.model and "qwen" in self.model.lower()
            else None
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await self._openai_client.chat.completions.create(**kwargs)
        raw_text = response.choices[0].message.content or ""

        if response_model is None:
            return {"content": raw_text, "usage": {}}

        model_name = response_model.__name__
        parsed = self._parse_json_robust(raw_text)
        if parsed is not None:
            try:
                result = response_model.model_validate(parsed).model_dump()
                return result
            except Exception as e:
                # Try to coerce common gemma mistakes
                coerced = self._coerce_to_schema(parsed, response_model)
                if coerced is not None:
                    return coerced

        # Last resort: construct valid default from schema
        logger.warning("Could not parse LLM output as JSON, returning fallback for %s", response_model.__name__)
        return self._build_default(response_model)

    async def _generate_response(
        self, messages, response_model=None, max_tokens=16384,
    ) -> dict[str, typing.Any]:
        # Should never be called since we override generate_response(),
        # but required by ABC. Delegate to generate_response().
        return await self.generate_response(messages, response_model, max_tokens)


# ─── Main retriever ───

class GraphitiRetriever(Retriever):
    """Graphiti temporal knowledge graph retriever.

    Ingests documents as episodes into a Neo4j-backed knowledge graph,
    retrieves via COMBINED_HYBRID_SEARCH_RRF search.
    Only ingests narrative docs — GitHub code is not meaningful for KG extraction.
    """

    @property
    def supported_sources(self) -> list[str] | None:
        return ["narrative"]

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "tmg2024secure",
        llm_model: str = "gemma4-26b-32k",
        llm_base_url: str = "http://localhost:9997/v1",
        llm_api_key: str = "not-ollama",
        embedder_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_coroutines: int = 10,
    ):
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._llm_api_key = llm_api_key
        self._embedder_model = embedder_model
        self._reranker_model = reranker_model
        self._max_coroutines = max_coroutines
        self._graphiti = None
        self._write_counter = 0
        self._write_lock = threading.Lock()
        self._docs: dict[int, str] = {}
        self._text_to_doc_id: dict[str, int] = {}

    def _init_graphiti(self):
        if self._graphiti is not None:
            return

        from graphiti_core import Graphiti
        from graphiti_core.llm_client.config import LLMConfig
        import graphiti_core.helpers as graphiti_helpers

        os.environ["SEMAPHORE_LIMIT"] = str(self._max_coroutines)
        graphiti_helpers.SEMAPHORE_LIMIT = self._max_coroutines

        print(f"  LLM: {self._llm_model} @ {self._llm_base_url}")
        print(f"  Embedder: {self._embedder_model} (local sentence-transformers)")
        print(f"  Reranker: {self._reranker_model} (local cross-encoder)")

        llm_client = RobustOpenAIGenericClient(
            config=LLMConfig(
                api_key=self._llm_api_key,
                model=self._llm_model,
                base_url=self._llm_base_url,
                temperature=0.0,
                max_tokens=16384,
            ),
            cache=False,
        )
        embedder = LocalEmbedderClient(model_name=self._embedder_model)
        cross_encoder = LocalCrossEncoderClient(model_name=self._reranker_model)

        self._graphiti = Graphiti(
            uri=self._neo4j_uri,
            user=self._neo4j_user,
            password=self._neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            store_raw_episode_content=True,
        )
        try:
            _run_async(self._graphiti.build_indices_and_constraints())
        except Exception as exc:
            logger.warning("Graphiti build_indices error: %s", exc)

    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        print("  Initializing Graphiti...")
        self._init_graphiti()

        # Store all docs for lookup
        for doc in docs:
            self._docs[doc.doc_id] = doc.text
            self._text_to_doc_id[doc.text[:200]] = doc.doc_id

        from graphiti_core.nodes import EpisodeType

        # Only ingest narrative docs — GitHub code is not meaningful for KG extraction
        # and would take days with a local gguf LLM
        narrative_docs = [doc for doc in docs if doc.source == "narrative"]
        total = len(narrative_docs)
        print(f"  Adding {total} narrative episodes to Graphiti (skipping {len(docs) - total} GitHub docs)...")

        for i, doc in enumerate(narrative_docs):
            try:
                with self._write_lock:
                    self._write_counter += 1
                    write_index = self._write_counter
                reference_time = datetime.now(timezone.utc)
                _run_async(
                    self._graphiti.add_episode(
                        name=f"ep_{write_index}",
                        episode_body=doc.text,
                        source_description="bitempqa_benchmark",
                        reference_time=reference_time,
                        source=EpisodeType.text,
                    )
                )
            except Exception as exc:
                logger.warning("Graphiti add_episode error for doc %d: %s", doc.doc_id, exc)
            if (i + 1) % 50 == 0:
                print(f"    Progress: {i + 1}/{total}")
        print(f"  Graphiti ingestion complete: {total} episodes")

    def find(self, query: str, top_k: int = 10) -> list[int]:
        if self._graphiti is None:
            return []

        episode_texts: list[str] = []
        seen: set[str] = set()

        # Strategy 1: KG edge search (best quality, needs good LLM extraction)
        try:
            from graphiti_core.search.search_filters import SearchFilters
            edges = _run_async(
                self._graphiti.search(query=query, num_results=top_k * 3, search_filter=SearchFilters())
            )
            for edge in edges if isinstance(edges, list) else []:
                fact = getattr(edge, "fact", "") or ""
                if fact and fact not in seen:
                    seen.add(fact)
                    episode_texts.append(fact)
        except Exception as exc:
            logger.debug("KG search failed: %s", exc)

        # Strategy 2: Embedding-based episode search (fallback when KG has no edges)
        if len(episode_texts) < top_k:
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                model = SentenceTransformer(self._embedder_model)
                query_vec = model.encode([query], convert_to_numpy=True)[0]

                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_password))
                with driver.session() as session:
                    result = session.run("MATCH (e:Episodic) WHERE e.content IS NOT NULL RETURN e.content AS content, e.name AS name")
                    episodes = [(r["name"], r["content"]) for r in result if r["content"]]

                if episodes:
                    names, texts = zip(*episodes)
                    ep_vecs = model.encode(list(texts), convert_to_numpy=True)
                    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
                    ep_norms = ep_vecs / (np.linalg.norm(ep_vecs, axis=1, keepdims=True) + 1e-10)
                    sims = ep_norms @ query_norm
                    top_indices = np.argsort(sims)[::-1][:top_k * 2]
                    for idx in top_indices:
                        text = texts[idx]
                        if text not in seen:
                            seen.add(text)
                            episode_texts.append(text)
                driver.close()
            except Exception as exc:
                logger.debug("Embedding fallback search failed: %s", exc)

        if not episode_texts:
            return []

        # Map episode texts back to doc_ids
        doc_scores: dict[int, float] = {}
        for i, text in enumerate(episode_texts):
            # Fast path: direct prefix match
            prefix_key = text[:200]
            if prefix_key in self._text_to_doc_id:
                doc_id = self._text_to_doc_id[prefix_key]
                doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), 1000.0 - i)
                continue

            # Fallback: word overlap (only against narrative docs, not all)
            best_doc_id = None
            best_overlap = 0
            text_words = set(text.split())
            for key, doc_id in self._text_to_doc_id.items():
                full_text = self._docs[doc_id]
                overlap = len(text_words & set(full_text.split()))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_doc_id = doc_id
            if best_doc_id is not None and best_overlap > 5:
                doc_scores[best_doc_id] = max(doc_scores.get(best_doc_id, 0.0), float(best_overlap))

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_k]]
