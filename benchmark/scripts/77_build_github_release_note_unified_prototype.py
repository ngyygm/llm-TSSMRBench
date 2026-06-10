#!/usr/bin/env python3
"""Build unified release-window prototypes from GitHub Releases.

This script does not modify the older release-note builders. It creates a new
single-file prototype format where chunk metadata and QA live in one JSON file.

Design:
1. use the latest N release notes of a repository as one release-evolution window;
2. ask one LLM prompt to produce:
   - memory_unit_text for each release chunk
   - three questions over the same window
3. do not emit a redundant bundle-level nodes list;
4. per-question source_chunk_ids define the gold evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
import yaml

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from src.state_version.github_collection import (  # noqa: E402
    clip_text,
    iso_day,
    normalize_text,
    parse_datetime,
)
from src.utils.config_env import load_yaml_with_env  # noqa: E402


DEFAULT_OUT_DIR = (
    ROOT
    / "benchmark"
    / "data"
    / "prototypes"
    / "github_release_note_v2"
    / "multirepo_10repo_30release_unified_v1"
)
DEFAULT_CONFIG = ROOT / "benchmark" / "configs" / "state_version_build_config.yaml"
DEFAULT_CACHE_DIR = ROOT / "benchmark" / "data" / "cache" / "github_release_note_unified_v1"
DEFAULT_RAW_RELEASE_DIR = ROOT / "benchmark" / "data" / "raw" / "github_release_notes_v1"
SYSTEM_PROMPT = (
    "You are a careful benchmark data writer for GitHub temporal version-memory retrieval. "
    "Return exactly one valid JSON object and no extra commentary."
)


@dataclass(frozen=True)
class ReleaseRepoSpec:
    repo: str
    prototype_id: str
    window_title: str


REPO_SPECS: list[ReleaseRepoSpec] = [
    ReleaseRepoSpec("fastapi/fastapi", "fastapi_release_window", "fastapi/fastapi recent release evolution"),
    ReleaseRepoSpec("pandas-dev/pandas", "pandas_release_window", "pandas-dev/pandas recent release evolution"),
    ReleaseRepoSpec("hashicorp/terraform", "terraform_release_window", "hashicorp/terraform recent release evolution"),
    ReleaseRepoSpec("redis/redis", "redis_release_window", "redis/redis recent release evolution"),
    ReleaseRepoSpec("prometheus/prometheus", "prometheus_release_window", "prometheus/prometheus recent release evolution"),
    ReleaseRepoSpec("grafana/grafana", "grafana_release_window", "grafana/grafana recent release evolution"),
    ReleaseRepoSpec("apache/airflow", "airflow_release_window", "apache/airflow recent release evolution"),
    ReleaseRepoSpec("celery/celery", "celery_release_window", "celery/celery recent release evolution"),
    ReleaseRepoSpec("vercel/next.js", "nextjs_release_window", "vercel/next.js recent release evolution"),
    ReleaseRepoSpec("pytest-dev/pytest", "pytest_release_window", "pytest-dev/pytest recent release evolution"),
]


class JsonLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        extra_body: dict[str, Any] | None = None,
        max_retries: int = 3,
        retry_delay: float = 4.0,
    ) -> None:
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            http_client=httpx.Client(timeout=timeout, trust_env=False),
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def call(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        use_json_mode = True
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                response = self.client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or ""
                return parse_json_object(raw)
            except Exception as exc:
                last_error = exc
                if use_json_mode and attempt == 0:
                    use_json_mode = False
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise last_error
        raise RuntimeError("LLM call failed without a captured exception")


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"response does not contain a JSON object: {text[:200]}")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("expected top-level JSON object")
    return payload


def normalize_ws(text: str) -> str:
    return " ".join((text or "").replace("\r", "\n").split())


def slugify(text: str) -> str:
    lowered = normalize_ws(text).lower()
    lowered = __import__("re").sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_") or "release_note"


def create_llm(config_path: Path) -> JsonLLM:
    cfg = load_yaml_with_env(config_path)
    llm_cfg = cfg.get("llm", {}) or {}
    return JsonLLM(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        model=llm_cfg["generation_model"],
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 12000),
        timeout=llm_cfg.get("timeout", 180),
        extra_body=llm_cfg.get("extra_body") or {},
        max_retries=3,
        retry_delay=4.0,
    )


def make_cache_key(kind: str, payload: dict[str, Any], *, model_name: str) -> str:
    packed = json.dumps({"kind": kind, "model": model_name, "payload": payload}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def llm_cached_json(llm: JsonLLM, prompt: str, *, cache_dir: Path, key_payload: dict[str, Any], kind: str) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = make_cache_key(
        kind,
        {
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            **key_payload,
        },
        model_name=llm.model,
    )
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = llm.call(prompt)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def release_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        parse_datetime(row.get("published_at") or row.get("created_at") or row.get("time_hint")),
        str(row.get("tag_name") or ""),
        str(row.get("memory_node_id") or ""),
    )


def make_memory_node_id(row: dict[str, Any]) -> str:
    repo_slug = str(row["repo"]).replace("/", "__").replace("-", "_").replace(".", "_")
    tag_slug = slugify(str(row.get("tag_name") or row.get("artifact_ref") or "release"))
    return f"{repo_slug}__release__{tag_slug}"


def build_unified_release_window_prompt(spec: ReleaseRepoSpec, rows: list[dict[str, Any]]) -> str:
    compact_rows = [
        {
            "memory_node_id": row["memory_node_id"],
            "repo": row["repo"],
            "artifact_type": row["artifact_type"],
            "artifact_ref": row["artifact_ref"],
            "tag_name": row.get("tag_name"),
            "title": row.get("title"),
            "published_at": row.get("published_at"),
            "time_hint": row.get("time_hint"),
            "source_url": row.get("source_url"),
            "raw_text": row["raw_text"],
        }
        for row in rows
    ]

    schema_hint = {
        "prototype_id": spec.prototype_id,
        "domain": "github_release_evolution",
        "repo": spec.repo,
        "source_type": "release_note",
        "window_title": spec.window_title,
        "window_summary": "2-3 sentence summary of the recent release-evolution window for this repository",
        "chunks": [
            {
                "memory_node_id": "must match one provided release node id",
                "memory_unit_text": "detailed English factual summary of only this release note; explicitly name the repository and the release version/tag",
            }
        ],
        "questions": [
            {
                "question_id": "string",
                "task_type": "single_state_lookup / cross_version_comparison / temporal_version_ordering",
                "difficulty": "low or high",
                "answer_format": "multiple_choice",
                "query_text": "question text that explicitly names the repository",
                "options": [
                    {"option_id": "A", "text": "option text"},
                    {"option_id": "B", "text": "option text"},
                    {"option_id": "C", "text": "option text"},
                    {"option_id": "D", "text": "option text"},
                ],
                "correct_option_id": "A",
                "expected_answer": "gold answer",
                "source_chunk_ids": ["memory_node_id_1"],
                "answer_support": [{"memory_node_id": "memory_node_id_1", "answer_text": "support text"}],
            }
        ],
    }

    airflow_style_example = {
        "prototype_id": "airflow_release_window",
        "domain": "github_release_evolution",
        "repo": "apache/airflow",
        "source_type": "release_note",
        "window_title": "apache/airflow recent release evolution",
        "window_summary": "This window tracks the recent apache/airflow release stream across core releases and related official release artifacts.",
        "chunks": [
            {
                "memory_node_id": "apache__airflow__release__3_0_0",
                "memory_unit_text": "In the apache/airflow release 3.0.0, the project announces the general availability of Airflow 3.0 with a service-oriented architecture, a React-based UI, enhanced security, DAG versioning, improved backfills, event-driven scheduling, and remote execution support.",
            }
        ],
        "questions": [
            {
                "question_id": "q1_single_state_lookup",
                "task_type": "single_state_lookup",
                "difficulty": "low",
                "answer_format": "multiple_choice",
                "query_text": "In the apache/airflow releases, which release introduces a service-oriented architecture together with DAG versioning and event-driven scheduling?",
                "options": [
                    {"option_id": "A", "text": "Airflow 2.10.5"},
                    {"option_id": "B", "text": "Airflow 3.0.0"},
                    {"option_id": "C", "text": "Airflow 3.0.2"},
                    {"option_id": "D", "text": "Airflow 2.11.0"},
                ],
                "correct_option_id": "B",
                "expected_answer": "Airflow 3.0.0",
                "source_chunk_ids": ["apache__airflow__release__3_0_0"],
                "answer_support": [
                    {
                        "memory_node_id": "apache__airflow__release__3_0_0",
                        "answer_text": "introduces a new service-oriented architecture ... DAG versioning ... event-driven scheduling",
                    }
                ],
            },
            {
                "question_id": "q2_cross_version_comparison",
                "task_type": "cross_version_comparison",
                "difficulty": "high",
                "answer_format": "multiple_choice",
                "query_text": "Across the apache/airflow releases, how does the scheduling-related change described in Airflow 2.11.0 differ from the platform-level change described in Airflow 3.0.0?",
                "options": [
                    {"option_id": "A", "text": "2.11.0 introduces DeltaTriggerTimetable migration support, while 3.0.0 announces the broader Airflow 3 platform shift with service-oriented architecture and event-driven scheduling."},
                    {"option_id": "B", "text": "2.11.0 removes DAG versioning, while 3.0.0 restores execution_date."},
                    {"option_id": "C", "text": "Both releases only update Helm chart images without scheduler changes."},
                    {"option_id": "D", "text": "2.11.0 deprecates Python 3.12, while 3.0.0 removes all trigger-based timetables."},
                ],
                "correct_option_id": "A",
                "expected_answer": "2.11.0 introduces DeltaTriggerTimetable migration support, while 3.0.0 announces the broader Airflow 3 platform shift with service-oriented architecture and event-driven scheduling.",
                "source_chunk_ids": [
                    "apache__airflow__release__2_11_0",
                    "apache__airflow__release__3_0_0",
                ],
                "answer_support": [
                    {"memory_node_id": "apache__airflow__release__2_11_0", "answer_text": "introduces DeltaTriggerTimetable ... to help users begin transitioning before upgrading to Airflow 3.0"},
                    {"memory_node_id": "apache__airflow__release__3_0_0", "answer_text": "introduces a new service-oriented architecture ... event-driven scheduling"},
                ],
            },
            {
                "question_id": "q3_temporal_version_ordering",
                "task_type": "temporal_version_ordering",
                "difficulty": "high",
                "answer_format": "multiple_choice",
                "query_text": "In the apache/airflow release window, order the following release-content states from earliest to latest: (1) a release that adds DeltaTriggerTimetable migration support, (2) a release that announces the general availability of Airflow 3.0, (3) a Helm Chart release that updates the default Airflow image to 3.0.2.",
                "options": [
                    {"option_id": "A", "text": "1 -> 2 -> 3"},
                    {"option_id": "B", "text": "2 -> 1 -> 3"},
                    {"option_id": "C", "text": "1 -> 3 -> 2"},
                    {"option_id": "D", "text": "3 -> 2 -> 1"},
                ],
                "correct_option_id": "A",
                "expected_answer": "1 -> 2 -> 3",
                "source_chunk_ids": [
                    "apache__airflow__release__2_11_0",
                    "apache__airflow__release__3_0_0",
                    "apache__airflow__release__helm_chart_1_17_0",
                ],
                "answer_support": [
                    {"memory_node_id": "apache__airflow__release__2_11_0", "answer_text": "introduces DeltaTriggerTimetable"},
                    {"memory_node_id": "apache__airflow__release__3_0_0", "answer_text": "announces the General Availability of Apache Airflow 3.0"},
                    {"memory_node_id": "apache__airflow__release__helm_chart_1_17_0", "answer_text": "default Airflow image is updated to 3.0.2"},
                ],
            },
        ],
    }

    instructions = """
Build one unified repository release-evolution prototype from the provided official GitHub release-note nodes.

Requirements:
- Use only the provided release-note nodes.
- Produce one single JSON object matching the output schema.
- Only generate chunk-level memory_unit_text values keyed by memory_node_id. The program will attach the original metadata and raw_text from the provided input.
- For every chunk, write memory_unit_text in English.
- Every memory_unit_text must explicitly name both the repository and the current release version or tag.
- Every memory_unit_text must describe only the current release note itself.
- Do not compare to earlier or later releases inside memory_unit_text.
- Do not mention benchmark setup, memory pools, retrieval, neighboring releases, hidden context, or any chain-construction process.
- Generate exactly 3 multiple-choice questions, one per task type:
  - single_state_lookup
  - cross_version_comparison
  - temporal_version_ordering
- Every query_text must explicitly name the repository.
- Use source_chunk_ids as the question-level gold evidence field.
- Do not output a separate nodes list.
- Do not output a focus_event field.
- For single_state_lookup, ask either:
  - which release contains a specific release-note content item
  - what content appears in a specific release
- For cross_version_comparison, compare two releases using concrete release-content differences.
- For temporal_version_ordering, use 3 to 5 release-content states.
- For temporal_version_ordering, you may mention version numbers, tag names, or dates when they are naturally part of the release-content descriptions.
- For temporal_version_ordering, do not reveal the answer in the options.
- The question text should describe the candidate release-content states and number them, for example (1), (2), (3), (4).
- The answer options for temporal_version_ordering should only be orderings of those indices, such as "1 -> 3 -> 2".
- More generally, do not make the correct answer obvious in query_text or options. The question should still require the system to retrieve the corresponding chunk-level memory_unit_text content rather than answering purely from surface chronology in the question text or options.
- All questions must be answerable from the provided memory_unit_text content.
"""

    return (
        f"{textwrap.dedent(instructions).strip()}\n\n"
        "## Output schema shape\n"
        f"```json\n{json.dumps(schema_hint, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Reference style example\n"
        f"```json\n{json.dumps(airflow_style_example, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Release-note nodes\n"
        f"```json\n{json.dumps({'repo': spec.repo, 'window_title': spec.window_title, 'release_nodes': compact_rows}, ensure_ascii=False, indent=2)}\n```"
    )


def normalize_unified_output(payload: dict[str, Any], spec: ReleaseRepoSpec, raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["memory_node_id"]: row for row in raw_rows}
    chunks_out: list[dict[str, Any]] = []
    for item in payload.get("chunks") or []:
        node_id = item.get("memory_node_id")
        if node_id not in by_id:
            continue
        source = by_id[node_id]
        chunks_out.append(
            {
                "memory_node_id": source["memory_node_id"],
                "repo": source["repo"],
                "artifact_type": source["artifact_type"],
                "artifact_ref": source["artifact_ref"],
                "tag_name": source.get("tag_name"),
                "title": source.get("title"),
                "time_hint": source.get("time_hint"),
                "published_at": source.get("published_at"),
                "source_url": source.get("source_url"),
                "raw_text": source["raw_text"],
                "memory_unit_text": normalize_text(item.get("memory_unit_text") or ""),
            }
        )
    seen = {row["memory_node_id"] for row in chunks_out}
    for row in raw_rows:
        if row["memory_node_id"] in seen:
            continue
        chunks_out.append(
            {
                "memory_node_id": row["memory_node_id"],
                "repo": row["repo"],
                "artifact_type": row["artifact_type"],
                "artifact_ref": row["artifact_ref"],
                "tag_name": row.get("tag_name"),
                "title": row.get("title"),
                "time_hint": row.get("time_hint"),
                "published_at": row.get("published_at"),
                "source_url": row.get("source_url"),
                "raw_text": row["raw_text"],
                "memory_unit_text": "",
            }
        )

    allowed_ids = {row["memory_node_id"] for row in chunks_out}
    questions = []
    for index, question in enumerate(payload.get("questions") or [], start=1):
        task_type = normalize_ws(str(question.get("task_type") or ""))
        if task_type not in {"single_state_lookup", "cross_version_comparison", "temporal_version_ordering"}:
            continue
        options = []
        for opt in (question.get("options") or [])[:4]:
            option_id = normalize_ws(str(opt.get("option_id") or ""))
            text = normalize_ws(str(opt.get("text") or ""))
            if option_id and text:
                options.append({"option_id": option_id, "text": text})
        if len(options) != 4:
            continue
        questions.append(
            {
                "question_id": normalize_ws(str(question.get("question_id") or f"q{index}_{task_type}")),
                "task_type": task_type,
                "difficulty": normalize_ws(str(question.get("difficulty") or "high")),
                "answer_format": "multiple_choice",
                "query_text": normalize_ws(str(question.get("query_text") or "")),
                "options": options,
                "correct_option_id": normalize_ws(str(question.get("correct_option_id") or "")),
                "expected_answer": normalize_ws(str(question.get("expected_answer") or "")),
                "source_chunk_ids": [
                    node_id
                    for node_id in (question.get("source_chunk_ids") or question.get("gold_node_ids") or [])
                    if node_id in allowed_ids
                ],
                "answer_support": [
                    {
                        "memory_node_id": item["memory_node_id"],
                        "answer_text": normalize_ws(str(item.get("answer_text") or "")),
                    }
                    for item in (question.get("answer_support") or [])
                    if item.get("memory_node_id") in allowed_ids
                ],
            }
        )

    return {
        "prototype_id": spec.prototype_id,
        "domain": "github_release_evolution",
        "repo": spec.repo,
        "source_type": "release_note",
        "window_title": normalize_ws(str(payload.get("window_title") or spec.window_title)),
        "window_summary": normalize_ws(str(payload.get("window_summary") or spec.window_title)),
        "chunks": chunks_out,
        "questions": questions,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified release-note GitHub prototypes from GitHub Releases.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--chunks-per-window", type=int, default=30)
    parser.add_argument("--repo-id", default=None, help="Optional single prototype_id to regenerate.")
    parser.add_argument("--raw-release-root", type=Path, default=DEFAULT_RAW_RELEASE_DIR)
    return parser.parse_args()


def load_local_release_rows(spec: ReleaseRepoSpec, raw_root: Path) -> list[dict[str, Any]]:
    local_json = raw_root / spec.repo.replace("/", "__") / "releases.json"
    if not local_json.exists():
        raise FileNotFoundError(f"Missing local release file: {local_json}")
    release_rows = json.loads(local_json.read_text(encoding="utf-8"))
    rows = [
        {
            "repo": spec.repo,
            "artifact_type": "release_note",
            "artifact_ref": f"release:{release.get('tag_name') or release.get('id')}",
            "tag_name": release.get("tag_name"),
            "title": str(release.get("name") or release.get("tag_name") or "").strip() or None,
            "published_at": release.get("published_at"),
            "created_at": release.get("created_at"),
            "time_hint": iso_day(release.get("published_at") or release.get("created_at")),
            "source_url": release.get("html_url"),
            "raw_text": clip_text(
                "\n".join(
                    [
                        f"Repository: {spec.repo}",
                        f"Release: {release.get('name') or release.get('tag_name')}",
                        f"Tag: {release.get('tag_name')}; draft: {release.get('draft')}; prerelease: {release.get('prerelease')}; published_at: {release.get('published_at')}; created_at: {release.get('created_at')}",
                        f"Release notes:\n{normalize_text(release.get('body') or '')}",
                    ]
                )
            ),
        }
        for release in release_rows
        if normalize_text(release.get("body") or "") and not bool(release.get("draft"))
    ]
    for row in rows:
        row["memory_node_id"] = make_memory_node_id(row)
    rows.sort(key=release_sort_key)
    return rows


def main() -> None:
    args = parse_args()
    llm = create_llm(args.config)

    selected_specs = [spec for spec in REPO_SPECS if not args.repo_id or spec.prototype_id == args.repo_id]
    if not selected_specs:
        raise ValueError("No repository specs selected.")

    index_rows = []
    for spec in selected_specs:
        rows = load_local_release_rows(spec, args.raw_release_root)
        if len(rows) < args.chunks_per_window:
            raise ValueError(f"{spec.repo} has only {len(rows)} usable release notes.")
        window_rows = rows[-args.chunks_per_window :]
        prompt = build_unified_release_window_prompt(spec, window_rows)
        payload = llm_cached_json(
            llm,
            prompt,
            cache_dir=args.cache_dir / "unified_window",
            key_payload={"repo": spec.repo, "window_rows": window_rows},
            kind="unified_release_window",
        )
        normalized = normalize_unified_output(payload, spec, window_rows)

        out_dir = args.out_dir / spec.prototype_id
        out_dir.mkdir(parents=True, exist_ok=True)
        write_text(out_dir / "prototype.json", json.dumps(normalized, ensure_ascii=False, indent=2) + "\n")
        write_text(
            out_dir / "README.md",
            (
                f"# {normalized['window_title']}\n\n"
                f"- repo: `{normalized['repo']}`\n"
                f"- source_type: `{normalized['source_type']}`\n"
                f"- total_chunks: `{len(normalized['chunks'])}`\n"
                f"- question_count: `{len(normalized['questions'])}`\n\n"
                "This prototype stores official GitHub release notes in one unified JSON file.\n"
            ),
        )
        index_rows.append(
            {
                "prototype_id": normalized["prototype_id"],
                "repo": normalized["repo"],
                "source_type": normalized["source_type"],
                "window_title": normalized["window_title"],
                "total_chunks": len(normalized["chunks"]),
                "question_count": len(normalized["questions"]),
            }
        )

    write_text(args.out_dir / "prototype_index.jsonl", "\n\n".join(json.dumps(r, ensure_ascii=False, indent=2) for r in index_rows) + "\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "prototype_count": len(index_rows),
                "chunks_per_window": args.chunks_per_window,
                "out_dir": str(args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
