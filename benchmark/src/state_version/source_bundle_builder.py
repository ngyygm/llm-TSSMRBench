"""Utilities for building GitHub source bundles from normalized raw artifacts."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional

from openai import OpenAI

from .schemas import BuildTask, RawGithubArtifactRecord, SourceBundleItem, SourceBundleRecord

SUMMARY_SYSTEM_PROMPT = (
    "You prepare normalized GitHub artifact summaries for a state-version retrieval benchmark. "
    "Return exactly one JSON object and no extra commentary."
)


def strip_code_fences(raw: str) -> str:
    """Remove common markdown fences around JSON output."""

    text = (raw or "").strip()
    if text.startswith("```json"):
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if text.startswith("```"):
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse one JSON object from the model output."""

    cleaned = strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("model output must be a JSON object")
        return data
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(cleaned[start : end + 1])
            if not isinstance(data, dict):
                raise ValueError("model output must be a JSON object")
            return data
        raise


def normalize_text(value: Any) -> str:
    """Normalize arbitrary text-like values into a compact multiline string."""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def resolve_artifact_text(record: RawGithubArtifactRecord) -> str:
    """Choose the best available text to place into a source bundle item."""

    if normalize_text(record.summary):
        return normalize_text(record.summary)
    return normalize_text(record.raw_text)


def artifact_group_key(record: RawGithubArtifactRecord) -> str:
    """Choose a stable grouping key for one raw artifact."""

    if record.state_chain_id:
        return f"state_chain_id::{record.state_chain_id}"
    if record.bundle_key:
        return f"bundle_key::{record.bundle_key}"
    return f"repo_focus::{record.repo}::{record.focus_event}"


def sort_group_records(records: Iterable[RawGithubArtifactRecord]) -> list[RawGithubArtifactRecord]:
    """Sort records within a bundle in a stable audit-friendly order."""

    return sorted(
        records,
        key=lambda record: (
            record.artifact_order if record.artifact_order is not None else 10**9,
            record.time_hint or "",
            record.artifact_type,
            record.artifact_ref,
        ),
    )


def group_raw_artifacts(records: Iterable[RawGithubArtifactRecord]) -> list[list[RawGithubArtifactRecord]]:
    """Group raw artifacts into bundle candidates."""

    groups: "OrderedDict[str, list[RawGithubArtifactRecord]]" = OrderedDict()
    for record in records:
        groups.setdefault(artifact_group_key(record), []).append(record)
    return [sort_group_records(group_records) for group_records in groups.values()]


def default_bundle_summary(repo: str, focus_event: str, item_count: int) -> str:
    """Build a deterministic fallback bundle summary."""

    return (
        f"This bundle contains {item_count} GitHub artifacts from {repo} tracking the evolving state of "
        f"{focus_event}."
    )


class GithubArtifactSummarizer:
    """LLM-backed summarizer for raw GitHub artifacts."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt_path: Path,
        temperature: float = 0.2,
        timeout: int = 180,
        max_tokens: int = 1200,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        use_json_mode: bool = True,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_json_mode = use_json_mode

    def build_prompt(self, record: RawGithubArtifactRecord) -> str:
        payload = {
            "repo": record.repo,
            "focus_event": record.focus_event,
            "artifact_type": record.artifact_type,
            "artifact_ref": record.artifact_ref,
            "title": record.title,
            "time_hint": record.time_hint,
            "source_url": record.source_url,
            "raw_text": normalize_text(record.raw_text),
            "notes": record.notes,
            "metadata": record.metadata,
        }
        return (
            f"{self.prompt_template}\n\n"
            "## Input raw_github_artifact\n"
            f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
        )

    def _call_llm(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                if self.use_json_mode and attempt == 0:
                    self.use_json_mode = False
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise last_error
        raise RuntimeError("Artifact summarization failed without a captured exception")

    def summarize(self, record: RawGithubArtifactRecord) -> tuple[str, Optional[str], str]:
        prompt = self.build_prompt(record)
        raw = self._call_llm(prompt)
        payload = parse_json_object(raw)
        summary = normalize_text(payload.get("summary"))
        time_hint = normalize_text(payload.get("time_hint")) or None
        if not summary:
            raise ValueError("artifact summarizer returned an empty summary")
        return summary, time_hint, raw


def assign_groups_to_tasks(
    grouped_records: list[list[RawGithubArtifactRecord]],
    tasks: list[BuildTask],
    existing_state_chain_ids: set[str],
    allow_explicit_overwrite: bool = False,
    allow_implicit_overwrite: bool = False,
) -> list[tuple[BuildTask, list[RawGithubArtifactRecord]]]:
    """Assign grouped raw artifacts to manifest tasks."""

    task_by_state_chain_id = {task.state_chain_id: task for task in tasks}
    assignments: list[tuple[BuildTask, list[RawGithubArtifactRecord]]] = []
    explicitly_assigned: set[str] = set()
    implicit_groups: list[list[RawGithubArtifactRecord]] = []

    for group in grouped_records:
        explicit_ids = {record.state_chain_id for record in group if record.state_chain_id}
        if explicit_ids:
            if len(explicit_ids) != 1:
                raise ValueError(f"group has inconsistent explicit state_chain_id values: {sorted(explicit_ids)}")
            state_chain_id = next(iter(explicit_ids))
            if state_chain_id not in task_by_state_chain_id:
                raise ValueError(f"explicit state_chain_id {state_chain_id} is not present in the selected manifest")
            if state_chain_id in existing_state_chain_ids and not allow_explicit_overwrite:
                raise ValueError(f"explicit state_chain_id {state_chain_id} already exists in source_bundles.jsonl")
            assignments.append((task_by_state_chain_id[state_chain_id], group))
            explicitly_assigned.add(state_chain_id)
        else:
            implicit_groups.append(group)

    remaining_tasks = [
        task
        for task in tasks
        if (allow_implicit_overwrite or task.state_chain_id not in existing_state_chain_ids)
        and task.state_chain_id not in explicitly_assigned
    ]
    if len(implicit_groups) > len(remaining_tasks):
        raise ValueError(
            f"not enough remaining manifest tasks for implicit groups: "
            f"{len(implicit_groups)} groups vs {len(remaining_tasks)} tasks"
        )

    for group, task in zip(implicit_groups, remaining_tasks):
        assignments.append((task, group))

    return sorted(assignments, key=lambda item: item[0].state_chain_id)


def build_source_bundle_record(
    task: BuildTask,
    records: list[RawGithubArtifactRecord],
    artifact_summaries: dict[str, str],
    inferred_time_hints: Optional[dict[str, Optional[str]]] = None,
) -> SourceBundleRecord:
    """Build one source-bundle record from grouped artifacts and one assigned task."""

    if not records:
        raise ValueError("cannot build a source bundle from an empty record group")

    first = records[0]
    repo = first.repo
    focus_event = first.focus_event
    bundle_summary = next((normalize_text(record.bundle_summary) for record in records if record.bundle_summary), "")
    notes: list[str] = []
    source_bundle_items: list[SourceBundleItem] = []
    inferred_time_hints = inferred_time_hints or {}

    for record in records:
        artifact_key = f"{record.artifact_type}::{record.artifact_ref}"
        summary = artifact_summaries[artifact_key]
        time_hint = record.time_hint or inferred_time_hints.get(artifact_key)
        source_bundle_items.append(
            SourceBundleItem(
                artifact_type=record.artifact_type,
                artifact_ref=record.artifact_ref,
                title=record.title,
                time_hint=time_hint,
                summary=summary,
            )
        )
        notes.extend(record.notes)

    metadata: dict[str, Any] = {
        "repo": repo,
        "artifact_count": len(records),
    }
    for key, value in first.metadata.items():
        metadata[key] = value

    if not bundle_summary:
        bundle_summary = default_bundle_summary(repo, focus_event, len(records))

    return SourceBundleRecord(
        sample_id=task.sample_id,
        state_chain_id=task.state_chain_id,
        domain=task.domain,
        language=task.language,
        focus_event=focus_event,
        source_title=f"{repo}: {focus_event}",
        bundle_summary=bundle_summary,
        source_bundle_items=source_bundle_items,
        notes=list(dict.fromkeys(note for note in notes if note)),
        source_metadata=metadata,
    )
