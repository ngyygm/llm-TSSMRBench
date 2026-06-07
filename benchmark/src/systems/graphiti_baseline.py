"""Graphiti baseline.

This adapter stores raw node text in Graphiti and queries it for retrieval."""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
import re
import threading
import time
import typing
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from src.systems.base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)

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

        _loop_thread = threading.Thread(
            target=_run_loop,
            args=(_loop,),
            name='graphiti-event-loop',
            daemon=True,
        )
        _loop_thread.start()
        _loop_ready.wait()
    return _loop


def _run_async(coro):
    future = asyncio.run_coroutine_threadsafe(coro, _get_loop())
    return future.result()


def _parse_datetime(time_str: str | None) -> datetime | None:
    """Parse an ISO-like datetime string into a UTC datetime when possible."""
    if not time_str:
        return None

    raw = time_str.strip()
    candidates = [raw, raw.replace('Z', '+00:00')]
    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError:
            continue

    for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _to_utc_iso(dt: datetime | None) -> str | None:
    """Convert a datetime to a UTC ISO-8601 string."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace('+00:00', 'Z')


def _normalize_dt(dt: datetime | None) -> datetime | None:
    """Normalize a datetime into timezone-aware UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class SiliconFlowRerankerClient(CrossEncoderClient):
    """OpenAI-compatible reranker client for SiliconFlow."""

    def __init__(self, api_key: str, base_url: str, model: str = 'BAAI/bge-reranker-v2-m3'):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.endpoint = self.base_url if self.base_url.endswith('/rerank') else f'{self.base_url}/rerank'
        self.model = model

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            response = await client.post(
                self.endpoint,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': self.model,
                    'query': query,
                    'documents': passages,
                    'return_documents': True,
                    'top_n': len(passages),
                },
            )
            response.raise_for_status()
            payload = response.json()
        ranked: list[tuple[str, float]] = []
        for item in payload.get('results', []):
            index = item.get('index')
            if isinstance(index, int) and 0 <= index < len(passages):
                ranked.append((passages[index], float(item.get('relevance_score', 0.0))))
        if len(ranked) != len(passages):
            seen = {text for text, _ in ranked}
            for passage in passages:
                if passage not in seen:
                    ranked.append((passage, 0.0))
        return ranked


class RobustOpenAIGenericClient(OpenAIGenericClient):
    """OpenAI-generic client with more robust JSON handling."""

    @staticmethod
    def _example_from_annotation(annotation: typing.Any) -> typing.Any:
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)

        if origin in (list, list[typing.Any].__origin__ if hasattr(list[typing.Any], "__origin__") else list):
            inner = args[0] if args else str
            return [RobustOpenAIGenericClient._example_from_annotation(inner)]

        if origin in (typing.Union, getattr(typing, "UnionType", object())):
            non_none = [arg for arg in args if arg is not type(None)]
            if non_none:
                return RobustOpenAIGenericClient._example_from_annotation(non_none[0])
            return ""

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return RobustOpenAIGenericClient._example_from_model(annotation)

        if annotation in (str,):
            return ""
        if annotation in (int,):
            return 0
        if annotation in (float,):
            return 0.0
        if annotation in (bool,):
            return False

        return ""

    @staticmethod
    def _example_from_model(model: type[BaseModel]) -> dict[str, typing.Any]:
        example: dict[str, typing.Any] = {}
        for name, field in model.model_fields.items():
            example[name] = RobustOpenAIGenericClient._example_from_annotation(field.annotation)
        return example

    @classmethod
    def _response_shape_hint(cls, response_model: type[BaseModel]) -> str:
        example = cls._example_from_model(response_model)
        return json.dumps(example, ensure_ascii=False)

    @staticmethod
    def _looks_like_json_schema(payload: dict[str, typing.Any]) -> bool:
        schema_keys = {"$defs", "properties", "required", "title", "type", "items"}
        return bool(payload) and len(set(payload.keys()) & schema_keys) >= 3

    @classmethod
    def _normalize_structured_payload(
        cls, payload: dict[str, typing.Any], response_model: type[BaseModel]
    ) -> dict[str, typing.Any] | None:
        if cls._looks_like_json_schema(payload):
            return None

        field_names = set(response_model.model_fields.keys())
        if field_names.issubset(payload.keys()):
            return payload

        wrapper_candidates = [
            response_model.__name__,
            response_model.__name__.lower(),
            response_model.__name__.replace("_", ""),
        ]
        for key in wrapper_candidates:
            nested = payload.get(key)
            if isinstance(nested, dict) and field_names.issubset(nested.keys()):
                return nested

        if len(payload) == 1:
            only_value = next(iter(payload.values()))
            if isinstance(only_value, dict):
                if field_names.issubset(only_value.keys()):
                    return only_value
            elif isinstance(only_value, list) and len(field_names) == 1:
                only_field = next(iter(field_names))
                return {only_field: only_value}

        return payload

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned

    @staticmethod
    def _extract_json_region(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    @staticmethod
    def _remove_trailing_commas(text: str) -> str:
        import re

        return re.sub(r",(\s*[}\]])", r"\1", text)

    @classmethod
    def _parse_structured_output(cls, raw_text: str) -> dict[str, typing.Any]:
        import yaml

        candidates = []
        cleaned = cls._strip_code_fences(raw_text)
        candidates.append(cleaned)
        extracted = cls._extract_json_region(cleaned)
        if extracted != cleaned:
            candidates.append(extracted)
        candidates.append(cls._remove_trailing_commas(extracted))

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as exc:
                last_error = exc

        for candidate in candidates:
            try:
                parsed = yaml.safe_load(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("Empty structured output", raw_text, 0)

    async def _generate_response(
        self,
        messages,
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 16384,
        model_size=None,
    ) -> dict[str, typing.Any]:
        del model_size

        openai_messages: list[ChatCompletionMessageParam] = []
        schema_instruction = None
        if response_model is not None:
            schema_instruction = (
                "Return one valid JSON object instance only. "
                "Do not output markdown, comments, explanations, or the JSON schema itself. "
                "Do not output keys such as $defs, title, type, properties, required, or items. "
                "Return a concrete JSON value that matches this template shape:\n"
                f"{self._response_shape_hint(response_model)}"
            )

        for message in messages:
            content = self._clean_input(message.content)
            if message.role == "system" and schema_instruction:
                content = f"{content}\n\n{schema_instruction}"
                schema_instruction = None
            openai_messages.append({"role": message.role, "content": content})

        if schema_instruction:
            openai_messages.insert(0, {"role": "system", "content": schema_instruction})

        response_format: dict[str, Any] | None = None
        if response_model is not None:
            response_format = {"type": "json_object"}

        extra_body: dict[str, Any] | None = None
        if self.model and "qwen" in self.model.lower():
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        elif self.model and "deepseek-v4-flash" in self.model.lower():
            extra_body = {"thinking": {"type": "disabled"}}

        async def _call(model_messages: list[ChatCompletionMessageParam]) -> str:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=model_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format=response_format,  # type: ignore[arg-type]
                extra_body=extra_body,  # type: ignore[arg-type]
            )
            return response.choices[0].message.content or ""

        raw_text = await _call(openai_messages)
        if response_model is None:
            return {"content": raw_text, "usage": {}}

        parsed = self._parse_structured_output(raw_text)
        normalized = self._normalize_structured_payload(parsed, response_model)
        if normalized is not None:
            try:
                validated = response_model.model_validate(normalized)
                return validated.model_dump()
            except Exception:
                pass

        repair_messages = list(openai_messages)
        repair_messages.append(
            {
                "role": "system",
                "content": (
                    "Your previous answer was invalid. "
                    "Return one concrete JSON object instance only. "
                    "Do not return a JSON schema. "
                    "Do not wrap the object under an extra top-level model name. "
                    f"Return only this shape: {self._response_shape_hint(response_model)}"
                ),
            }
        )
        repaired_raw_text = await _call(repair_messages)
        repaired = self._parse_structured_output(repaired_raw_text)
        normalized_repaired = self._normalize_structured_payload(repaired, response_model)
        if normalized_repaired is not None:
            try:
                validated = response_model.model_validate(normalized_repaired)
                return validated.model_dump()
            except Exception:
                return normalized_repaired
        return repaired


class GraphitiBaseline(MemorySystem):
    """Graphiti-based memory baseline."""

    def __init__(
        self,
        neo4j_uri: str = 'bolt://localhost:7687',
        neo4j_user: str = 'neo4j',
        neo4j_password: str = 'graphiti',
        llm_model: str = 'GLM-4-Flash',
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedder_model: str = 'Pro/BAAI/bge-m3',
        embedder_base_url: Optional[str] = None,
        embedder_api_key: Optional[str] = None,
        reranker_model: str = 'BAAI/bge-reranker-v2-m3',
        reranker_base_url: Optional[str] = None,
        reranker_api_key: Optional[str] = None,
        top_k: int = 5,
        max_coroutines: int = 10,
        chain_ingest_workers: int = 5,
        run_id: Optional[str] = None,
    ):
        super().__init__(name='Graphiti')
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.embedder_model = embedder_model
        self.embedder_base_url = embedder_base_url
        self.embedder_api_key = embedder_api_key
        self.reranker_model = reranker_model
        self.reranker_base_url = reranker_base_url
        self.reranker_api_key = reranker_api_key
        self.top_k = top_k
        self.max_coroutines = max_coroutines
        self.chain_ingest_workers = chain_ingest_workers
        self._write_counter = 0
        self._write_lock = threading.Lock()
        self._instance_tag = self._normalize_instance_tag(run_id) if run_id else self._build_instance_tag()
        self._group_id = f"graphiti_{self._instance_tag}"
        self._init_graphiti()

    @staticmethod
    def _build_instance_tag() -> str:
        raw = f"{os.getpid()}_{time.time_ns()}"
        return re.sub(r"[^A-Za-z0-9_]+", "_", raw)

    @staticmethod
    def _normalize_instance_tag(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value or "").strip("_")
        return normalized or GraphitiBaseline._build_instance_tag()

    def _init_graphiti(self) -> None:
        """Select the most useful fact texts from the current search result."""
        os.environ['SEMAPHORE_LIMIT'] = str(self.max_coroutines)
        from graphiti_core import Graphiti
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        import graphiti_core.graphiti as graphiti_module
        import graphiti_core.helpers as graphiti_helpers

        graphiti_helpers.SEMAPHORE_LIMIT = self.max_coroutines
        graphiti_module.get_default_group_id = lambda _provider: self._group_id

        llm_api_key = self.llm_api_key or os.environ.get('OPENAI_API_KEY', '')
        llm_base_url = self.llm_base_url or os.environ.get(
            'OPENAI_BASE_URL',
            'https://open.bigmodel.cn/api/paas/v4',
        )
        embedder_api_key = self.embedder_api_key or llm_api_key
        embedder_base_url = (self.embedder_base_url or 'https://api.siliconflow.cn/v1').rstrip('/')
        if embedder_base_url.endswith('/embeddings'):
            embedder_base_url = embedder_base_url[:-len('/embeddings')]
        reranker_api_key = self.reranker_api_key or embedder_api_key
        reranker_base_url = (self.reranker_base_url or 'https://api.siliconflow.cn/v1').rstrip('/')

        llm_client = RobustOpenAIGenericClient(
            config=LLMConfig(
                api_key=llm_api_key,
                model=self.llm_model,
                base_url=llm_base_url,
                temperature=0.0,
                max_tokens=16384,
            ),
            cache=False,
            max_tokens=16384,
        )

        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=embedder_api_key,
                embedding_model=self.embedder_model,
                base_url=embedder_base_url,
            )
        )

        cross_encoder = SiliconFlowRerankerClient(
            api_key=reranker_api_key,
            base_url=reranker_base_url,
            model=self.reranker_model,
        )

        self.graphiti = Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            max_coroutines=self.max_coroutines,
        )

        try:
            _run_async(self.graphiti.build_indices_and_constraints())
        except Exception as exc:
            logger.warning('Graphiti build_indices error: %s', exc)

    @staticmethod
    def _format_edge_fact(edge: Any) -> str:
        """Build the Graphiti instance and dependent clients."""
        parts = [getattr(edge, 'fact', '')]
        valid_at = _to_utc_iso(getattr(edge, 'valid_at', None))
        invalid_at = _to_utc_iso(getattr(edge, 'invalid_at', None))
        if valid_at:
            parts.append(f'valid_at={valid_at}')
        if invalid_at:
            parts.append(f'invalid_at={invalid_at}')
        return ' | '.join(part for part in parts if part)

    @staticmethod
    def _format_node_fact(node: Any) -> str:
        """Reset Graphiti storage for a fresh evaluation run."""
        parts = [f"node={getattr(node, 'name', '')}"]
        summary = getattr(node, 'summary', None)
        if summary:
            parts.append(f"summary={summary}")
        return ' | '.join(part for part in parts if part)

    @staticmethod
    def _format_episode_fact(episode: Any) -> str:
        """Return the number of stored episodes when available."""
        parts = [f"Episode: {getattr(episode, 'content', '')}"]
        event_time = _to_utc_iso(getattr(episode, 'valid_at', None))
        if event_time:
            parts.append(f'valid_at={event_time}')
        return ' | '.join(part for part in parts if part)

    def _find_existing_episode_uuid(self, text: str) -> str | None:
        normalized = text.strip()
        if not normalized:
            return None

        async def _query_existing() -> str | None:
            records, _, _ = await self.graphiti.driver.execute_query(
                """
                MATCH (e:Episodic {content: $content, group_id: $group_id})
                RETURN e.uuid AS uuid
                ORDER BY e.created_at ASC
                LIMIT 1
                """,
                content=normalized,
                group_id=self._group_id,
                routing_='r',
            )
            if not records:
                return None
            return records[0].get('uuid')

        try:
            return _run_async(_query_existing())
        except Exception as exc:
            logger.warning('Graphiti existing-episode lookup error: %s', exc)
            return None

    def _find_existing_episode_uuids_bulk(self, texts: list[str]) -> dict[str, str]:
        normalized_texts = [text.strip() for text in texts if text and text.strip()]
        if not normalized_texts:
            return {}

        async def _query_existing_bulk() -> dict[str, str]:
            records, _, _ = await self.graphiti.driver.execute_query(
                """
                MATCH (e:Episodic)
                WHERE e.content IN $contents AND e.group_id = $group_id
                RETURN e.content AS content, e.uuid AS uuid, e.created_at AS created_at
                ORDER BY e.created_at ASC
                """,
                contents=normalized_texts,
                group_id=self._group_id,
                routing_='r',
            )
            existing: dict[str, str] = {}
            for record in records:
                content = record.get('content')
                episode_uuid = record.get('uuid')
                if content and episode_uuid and content not in existing:
                    existing[content] = episode_uuid
            return existing

        try:
            return _run_async(_query_existing_bulk())
        except Exception as exc:
            logger.warning('Graphiti bulk existing-episode lookup error: %s', exc)
            return {}

    @staticmethod
    def _format_community_fact(community: Any) -> str:
        """Expose lightweight diagnostics about the current Graphiti state."""
        parts = [f"community={getattr(community, 'name', '')}"]
        summary = getattr(community, 'summary', None)
        if summary:
            parts.append(f"summary={summary}")
        return ' | '.join(part for part in parts if part)

    def _build_retrieval_units(self, search_results: SearchResults) -> list[dict[str, Any]]:
        """Group Graphiti results by episode so one memory write occupies one retrieval slot."""
        units: list[dict[str, Any]] = []
        edges = [edge for edge in search_results.edges if getattr(edge, 'fact', '')]
        episodes = list(search_results.episodes or [])

        if not episodes:
            return [
                {
                    "unit_type": "edge_only",
                    "episode_uuid": None,
                    "text": self._format_edge_fact(edge),
                    "episode_text": "",
                    "supporting_edge_facts": [self._format_edge_fact(edge)],
                }
                for edge in edges
                if self._format_edge_fact(edge)
            ]

        edges_by_episode: dict[str, list[Any]] = {}
        for edge in edges:
            linked_episode_ids = list(getattr(edge, 'episodes', []) or [])
            if not linked_episode_ids:
                continue
            for episode_uuid in linked_episode_ids:
                if not episode_uuid:
                    continue
                edges_by_episode.setdefault(episode_uuid, []).append(edge)

        seen_episode_ids: set[str] = set()
        seen_episode_texts: dict[str, dict[str, Any]] = {}
        for episode in episodes:
            episode_uuid = getattr(episode, 'uuid', None)
            episode_text = self._format_episode_fact(episode)
            supporting_edges = edges_by_episode.get(episode_uuid or "", [])
            supporting_edge_facts: list[str] = []
            seen_edge_facts: set[str] = set()
            for edge in supporting_edges:
                formatted_edge = self._format_edge_fact(edge)
                if not formatted_edge or formatted_edge in seen_edge_facts:
                    continue
                seen_edge_facts.add(formatted_edge)
                supporting_edge_facts.append(formatted_edge)

            parts: list[str] = []
            if episode_text:
                parts.append(episode_text)
            if supporting_edge_facts:
                support_block = "\n".join(f"- {fact}" for fact in supporting_edge_facts)
                parts.append(f"Supporting graph facts:\n{support_block}")
            text = "\n".join(part for part in parts if part).strip()
            if not text:
                continue
            if episode_text in seen_episode_texts:
                existing = seen_episode_texts[episode_text]
                existing_edges = existing["supporting_edge_facts"]
                for fact in supporting_edge_facts:
                    if fact not in existing_edges:
                        existing_edges.append(fact)
                if existing_edges:
                    support_block = "\n".join(f"- {fact}" for fact in existing_edges)
                    existing["text"] = f"{existing['episode_text']}\nSupporting graph facts:\n{support_block}"
                continue

            unit = {
                "unit_type": "episode",
                "episode_uuid": episode_uuid,
                "text": text,
                "episode_text": episode_text,
                "supporting_edge_facts": supporting_edge_facts,
            }
            if episode_uuid:
                seen_episode_ids.add(episode_uuid)
            seen_episode_texts[episode_text] = unit
            units.append(unit)

        attached_episode_edge_ids = {
            getattr(edge, 'uuid', None)
            for episode_uuid in seen_episode_ids
            for edge in edges_by_episode.get(episode_uuid, [])
            if getattr(edge, 'uuid', None)
        }
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid', None)
            if edge_uuid in attached_episode_edge_ids:
                continue
            formatted_edge = self._format_edge_fact(edge)
            if not formatted_edge:
                continue
            units.append(
                {
                    "unit_type": "edge_only",
                    "episode_uuid": None,
                    "text": formatted_edge,
                    "episode_text": "",
                    "supporting_edge_facts": [formatted_edge],
                }
            )

        return units

    def _build_retrieved_facts(self, search_results: SearchResults) -> list[str]:
        """Return top-k episode texts; keep graph-derived support only in metadata."""
        units = self._build_retrieval_units(search_results)
        episode_facts = [unit["episode_text"] for unit in units if unit["unit_type"] == "episode" and unit["episode_text"]]
        if episode_facts:
            return episode_facts
        return [unit["text"] for unit in units if unit["text"]]

    @staticmethod
    def _earliest_datetime(current: datetime | None, candidate: datetime) -> datetime:
        """Choose a reference time for Graphiti insertion from the text when possible."""
        current = _normalize_dt(current)
        candidate = _normalize_dt(candidate)
        if current is None:
            return candidate
        return candidate if candidate <= current else current

    def _apply_result_timestamps(self, result: Any, event_dt: datetime, record_dt: datetime) -> None:
        """Insert one memory write into Graphiti as a new episode."""
        episode = getattr(result, "episode", None)
        if episode is not None:
            episode.valid_at = event_dt
            episode.created_at = record_dt
            _run_async(episode.save(self.graphiti.driver))

        for edge in getattr(result, "edges", []) or []:
            edge.valid_at = event_dt
            edge.created_at = self._earliest_datetime(getattr(edge, "created_at", None), record_dt)
            _run_async(edge.save(self.graphiti.driver))

        for node in getattr(result, "nodes", []) or []:
            node.created_at = self._earliest_datetime(getattr(node, "created_at", None), record_dt)
            _run_async(node.save(self.graphiti.driver))

        for episodic_edge in getattr(result, "episodic_edges", []) or []:
            episodic_edge.created_at = self._earliest_datetime(
                getattr(episodic_edge, "created_at", None),
                record_dt,
            )
            _run_async(episodic_edge.save(self.graphiti.driver))

        for community in getattr(result, "communities", []) or []:
            community.created_at = self._earliest_datetime(
                getattr(community, "created_at", None),
                record_dt,
            )
            _run_async(community.save(self.graphiti.driver))

        for community_edge in getattr(result, "community_edges", []) or []:
            community_edge.created_at = self._earliest_datetime(
                getattr(community_edge, "created_at", None),
                record_dt,
            )
            _run_async(community_edge.save(self.graphiti.driver))

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, datetime):
            return _to_utc_iso(value)
        if isinstance(value, enum.Enum):
            return value.value
        if isinstance(value, dict):
            return {str(k): GraphitiBaseline._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [GraphitiBaseline._json_safe(item) for item in value]
        return value

    @staticmethod
    def _serialize_graph_object(obj: Any, fields: list[str]) -> dict[str, Any]:
        """Query Graphiti and convert the result into benchmark retrieval text."""
        data: dict[str, Any] = {}
        for field in fields:
            value = getattr(obj, field, None)
            data[field] = GraphitiBaseline._json_safe(value)
        return data

    def _build_graphiti_metadata(self, search_results: SearchResults) -> dict[str, Any]:
        """Transform Graphiti search output into our unified QueryResult schema."""
        unique_episodes: list[Any] = []
        seen_episode_texts: set[str] = set()
        for episode in search_results.episodes:
            content = getattr(episode, 'content', '') or ''
            if content in seen_episode_texts:
                continue
            seen_episode_texts.add(content)
            unique_episodes.append(episode)
        return {
            "retrieval_units": self._build_retrieval_units(search_results),
            "retrieved_edges": [
                self._serialize_graph_object(
                    edge,
                    [
                        "uuid",
                        "name",
                        "fact",
                        "valid_at",
                        "invalid_at",
                        "created_at",
                        "source_node_uuid",
                        "target_node_uuid",
                        "episodes",
                        "group_id",
                    ],
                )
                for edge in search_results.edges
            ],
            "retrieved_nodes": [
                self._serialize_graph_object(
                    node,
                    [
                        "uuid",
                        "name",
                        "summary",
                        "created_at",
                        "group_id",
                        "labels",
                        "attributes",
                    ],
                )
                for node in search_results.nodes
            ],
            "retrieved_episodes": [
                self._serialize_graph_object(
                    episode,
                    [
                        "uuid",
                        "name",
                        "content",
                        "source",
                        "source_description",
                        "valid_at",
                        "created_at",
                        "group_id",
                        "entity_edges",
                    ],
                )
                for episode in unique_episodes
            ],
            "retrieved_communities": [
                self._serialize_graph_object(
                    community,
                    [
                        "uuid",
                        "name",
                        "summary",
                        "created_at",
                        "group_id",
                    ],
                )
                for community in search_results.communities
            ],
        }

    def _search_graphiti(self, question: str, search_filter: Any) -> SearchResults:
        """Rerank candidate facts with the configured cross-encoder when enabled."""
        if hasattr(self.graphiti, "search_"):
            results = _run_async(
                self.graphiti.search_(
                    query=question,
                    config=COMBINED_HYBRID_SEARCH_RRF,
                    group_ids=[self._group_id],
                    search_filter=search_filter,
                )
            )
            if isinstance(results, SearchResults):
                return results

        if hasattr(self.graphiti, "search"):
            edges = _run_async(
                self.graphiti.search(
                    query=question,
                    group_ids=[self._group_id],
                    num_results=5,
                    search_filter=search_filter,
                )
            )
            return SearchResults(edges=edges if isinstance(edges, list) else [])

        return SearchResults()

    def remember(
        self,
        text: str,
    ) -> str:
        """Store one memory text in Graphiti."""
        try:
            existing_uuid = self._find_existing_episode_uuid(text)
            if existing_uuid:
                return existing_uuid
            with self._write_lock:
                self._write_counter += 1
                write_index = self._write_counter
            reference_time = datetime.now(timezone.utc)

            _run_async(
                self.graphiti.add_episode(
                    name=f"ep_{write_index}",
                    episode_body=text,
                    source_description="state_version_benchmark",
                    reference_time=reference_time,
                    source=EpisodeType.text,
                )
            )
            return f"graphiti_ep_{write_index}"
        except Exception as exc:
            logger.warning('Graphiti add_episode error: %s', exc)
            return ''

    def remember_many(self, texts: list[str]) -> list[str]:
        """Store one chain's node texts sequentially so each node becomes an ordered episode."""
        ids: list[str] = []
        for text in texts:
            ids.append(self.remember(text=text))
        return ids

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """Search Graphiti with a plain question string."""
        start_time = time.time()

        try:
            from graphiti_core.search.search_filters import SearchFilters

            search_filter = SearchFilters()
            search_results = self._search_graphiti(question, search_filter)

            effective_top_k = top_k if top_k is not None else self.top_k
            facts = self._build_retrieved_facts(search_results)[:effective_top_k]
            retrieved_context = '\n'.join(facts)
            graphiti_metadata = self._build_graphiti_metadata(search_results)

            return QueryResult(
                answer=retrieved_context,
                retrieved_context=retrieved_context,
                retrieved_facts=facts,
                confidence=1.0 if facts else 0.0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={
                    'num_results': len(facts),
                    'search_mode': 'graphiti.search__edges_plus_episodes',
                    'edge_count': len(search_results.edges),
                    'node_count': len(search_results.nodes),
                    'episode_count': len(search_results.episodes),
                    'community_count': len(search_results.communities),
                    'edge_reranker_scores': list(search_results.edge_reranker_scores),
                    **graphiti_metadata,
                },
            )
        except Exception as exc:
            logger.warning('Graphiti search error: %s', exc)
            return QueryResult(
                answer='',
                retrieved_context='',
                retrieved_facts=[],
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(exc)},
            )

    def reset(self) -> None:
        """Close Neo4j and Graphiti resources safely."""
        try:
            _run_async(self.graphiti.close())
        except Exception as exc:
            logger.warning('Graphiti close error: %s', exc)

        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            with driver.session() as session:
                for _ in range(100):
                    result = session.run(
                        'MATCH (n {group_id: $group_id}) WITH n LIMIT 5000 DETACH DELETE n RETURN count(n) AS cnt',
                        group_id=self._group_id,
                    )
                    if result.single()['cnt'] == 0:
                        break
            driver.close()
        except Exception as exc:
            logger.warning('Graphiti Neo4j clear error: %s', exc)

        self._write_counter = 0
        self._init_graphiti()
