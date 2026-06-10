#!/usr/bin/env python3
"""Run baseline evaluation on one narrative prototype built from chapter memory nodes.

This script adapts the direct-bundle narrative prototype format:
- chapter_chunks.jsonl
- chain_qa_bundles.json

into a minimal in-memory dataset that reuses the existing state-version
evaluation runner, systems, answer generator, and judge.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "benchmark") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "benchmark"))

from src.evaluation.answer_generator import AnswerGenerator
from src.evaluation.judge import LLMJudge
from src.state_version.evaluation import (
    StateVersionDataset,
    StateVersionEvaluationRunner,
    summarize_state_version_results,
)
from src.state_version.schemas import (
    ChainProfile,
    ChoiceOption,
    SourcePointer,
    StateChainNode,
    StateChainSample,
    StateQuestion,
)
from src.utils.config_env import load_yaml_with_env

DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "configs" / "state_version_experiment_config.yaml"
DEFAULT_PROTO_DIR = (
    REPO_ROOT
    / "benchmark"
    / "data"
    / "prototypes"
    / "narrative_state_summary_v2_llm"
    / "pride_and_prejudice_auto_multi_full_direct_bundle"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmark"
    / "data"
    / "prototype_eval_results"
    / "pride_and_prejudice_auto_multi_full_direct_bundle"
)
GLOBAL_CHAIN_ID = "prototype_global_memory_pool"
GITHUB_RAW_CODE_CHAR_LIMIT = 12000
UNIFIED_TOP_K = 10


def display_system_name(system_name: str) -> str:
    if system_name == "Full Context":
        return "Oracle Gold Context"
    return system_name


def parse_version_key(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version or "")
    return tuple(int(part) for part in parts) if parts else (0,)


def slugify(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_") or "prototype"


def compact_github_raw_code(text: str, max_chars: int = GITHUB_RAW_CODE_CHAR_LIMIT) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_chars:
        return raw
    head_budget = max_chars - 160
    if head_budget < 1000:
        head_budget = max_chars
    head = raw[:head_budget].rstrip()
    return f"{head}\n\n# [truncated for embedding budget]"


def load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml_with_env(path)


def load_pretty_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        payload, next_index = decoder.raw_decode(text, index)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} contains a non-object JSON value")
        rows.append(payload)
        index = next_index
    return rows


def _load_create_systems():
    script_path = REPO_ROOT / "benchmark" / "scripts" / "68_run_state_version_evaluation.py"
    spec = importlib.util.spec_from_file_location("state_version_eval_cli", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load evaluation CLI module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_systems


def build_dataset(
    *,
    prototype_dir: Path,
    normalize_top_k: bool,
) -> tuple[StateVersionDataset, dict[str, Any]]:
    unified_payload_path = prototype_dir / "prototype.json"
    if unified_payload_path.exists():
        bundle_payload = json.loads(unified_payload_path.read_text(encoding="utf-8"))
        memory_node_file = "prototype.json"
        chapter_rows = list(bundle_payload.get("chunks") or [])
        bundles = [
            {
                "chain_id": bundle_payload.get("prototype_id") or prototype_dir.name,
                "chain_title": bundle_payload.get("window_title") or prototype_dir.name,
                "chain_summary": bundle_payload.get("window_summary") or "",
                "source_chunk_ids": [row["memory_node_id"] for row in chapter_rows if row.get("memory_node_id")],
                "questions": bundle_payload.get("questions") or [],
            }
        ]
        if not chapter_rows:
            raise ValueError("No chunks found in unified prototype.json")
        if not bundles[0]["questions"]:
            raise ValueError("No questions found in unified prototype.json")
    else:
        bundle_payload = json.loads((prototype_dir / "chain_qa_bundles.json").read_text(encoding="utf-8"))
        memory_node_file = bundle_payload.get("memory_node_file") or "chapter_chunks.jsonl"
        chapter_rows = load_pretty_jsonl(prototype_dir / memory_node_file)
        bundles = bundle_payload.get("bundles") or []
        if not chapter_rows:
            raise ValueError(f"No memory nodes found in prototype directory: {memory_node_file}")
        if not bundles:
            raise ValueError("No chain bundles found in prototype directory")

    is_github_code_proto = bool(
        chapter_rows
        and "repo" in chapter_rows[0]
        and "file_path" in chapter_rows[0]
        and "version" in chapter_rows[0]
    )
    is_github_artifact_proto = bool(
        chapter_rows
        and "repo" in chapter_rows[0]
        and "artifact_type" in chapter_rows[0]
        and "artifact_ref" in chapter_rows[0]
    )
    if is_github_code_proto:
        chapter_rows = sorted(
            chapter_rows,
            key=lambda row: (
                parse_version_key(str(row.get("version") or "")),
                str(row.get("anchor_id") or ""),
                str(row.get("memory_node_id") or ""),
            ),
        )
    elif is_github_artifact_proto:
        chapter_rows = sorted(
            chapter_rows,
            key=lambda row: (
                str(row.get("repo") or ""),
                str(row.get("bundle_key") or ""),
                str(row.get("time_hint") or ""),
                int(row.get("artifact_order") or 0),
                str(row.get("artifact_ref") or ""),
                str(row.get("memory_node_id") or ""),
            ),
        )
    else:
        chapter_rows = sorted(chapter_rows, key=lambda row: int(row["chapter_index"]))
    chapter_by_id = {row["memory_node_id"]: row for row in chapter_rows}

    chain_nodes: list[StateChainNode] = []
    for idx, row in enumerate(chapter_rows, start=1):
        if is_github_code_proto:
            span_hint = f"{row.get('file_path')}@{row.get('version')}#{row.get('anchor_id')}"
            raw_code = compact_github_raw_code(row.get("raw_text") or "")
            memory_text = (
                f"{row['memory_unit_text']}\n\n[Raw Code Block]\n{raw_code}"
            ).strip()
            artifact_ref = row["memory_node_id"]
            artifact_type = "version_code_block"
            surface_order = idx
        elif is_github_artifact_proto:
            span_hint = f"{row.get('bundle_key')}::{row.get('artifact_type')}::{row.get('artifact_ref')}"
            memory_text = row["memory_unit_text"]
            artifact_ref = row["memory_node_id"]
            artifact_type = str(row.get("artifact_type") or "github_artifact")
            surface_order = idx
        else:
            span_hint = row.get("chapter_heading") or f"chapter_{row['chapter_index']}"
            memory_text = row["memory_unit_text"]
            artifact_ref = row["chunk_id"]
            artifact_type = "chapter_chunk"
            surface_order = int(row["chapter_index"])
        chain_nodes.append(
            StateChainNode(
                node_id=row["memory_node_id"],
                surface_order=surface_order,
                text=memory_text,
                progress_label="active",
                perspective_label="retrospective",
                relation_label="continues",
                salience_label="core",
                supersedes=[],
                depends_on=[],
                source_pointer=SourcePointer(
                    artifact_type=artifact_type,
                    artifact_ref=artifact_ref,
                    span_hint=span_hint,
                ),
            )
        )

    if unified_payload_path.exists():
        source_title = bundle_payload.get("window_title") or f"{bundle_payload.get('repo') or 'github'}::release_window"
        focus_event = "multiple temporal semantic state evolution questions over one repository release window"
        raw_domain = str(bundle_payload.get("domain") or "github_evolution")
        domain = "github_evolution" if raw_domain == "github_release_evolution" else raw_domain
        chain_summary = bundle_payload.get("window_summary") or "All release-note chunks in one repository window are stored into a single noisy memory pool."
    elif is_github_code_proto:
        if bundle_payload.get("file_path"):
            source_title = f"{bundle_payload['repo']}::{bundle_payload['file_path']}"
        else:
            source_title = f"{bundle_payload['repo']}::github_version_memory_pool"
        focus_event = "multiple temporal semantic state evolution events in one code-version memory pool"
        domain = "github_evolution"
        chain_summary = "All code-version memory nodes in one repository window are stored into a single noisy memory pool."
    elif is_github_artifact_proto:
        if bundle_payload.get("bundle_key"):
            source_title = f"{bundle_payload['repo']}::{bundle_payload['bundle_key']}"
        else:
            source_title = f"{bundle_payload.get('repo') or 'github'}::github_artifact_memory_pool"
        focus_event = "multiple temporal semantic state evolution events in one GitHub artifact-text memory pool"
        domain = "github_evolution"
        chain_summary = "All GitHub artifact memory nodes in one repository window are stored into a single noisy memory pool."
    else:
        source_title = bundle_payload["novel_title"]
        focus_event = "multiple temporal semantic state evolution events in one novel"
        domain = "narrative_evolution"
        chain_summary = "All chapter memory nodes from one full novel are stored into a single noisy memory pool."

    chain = StateChainSample(
        sample_id=f"{(bundle_payload.get('novel_id') or slugify(source_title))}_prototype_global",
        state_chain_id=GLOBAL_CHAIN_ID,
        domain=domain,
        language="en",
        focus_event=focus_event,
        chain_summary=chain_summary,
        source_kind=domain,
        source_title=source_title,
        chain_profile=ChainProfile(
            node_count=len(chain_nodes),
            competition_strength="high",
            lexical_overlap_band="high",
        ),
        chain_nodes=chain_nodes,
    )

    questions: list[StateQuestion] = []
    meta_by_question_id: dict[str, Any] = {}
    top_k_rewrites: list[dict[str, Any]] = []
    for bundle in bundles:
        bundle_id = bundle["chain_id"]
        focus_event = bundle.get("focus_event") or bundle.get("chain_title") or f"release evolution in {bundle_payload.get('repo') or 'repository'}"
        if bundle.get("nodes"):
            bundle_node_ids = [
                item["memory_node_id"]
                for item in (bundle.get("nodes") or [])
                if item.get("memory_node_id") in chapter_by_id
            ]
        else:
            bundle_node_ids = [node_id for node_id in (bundle.get("source_chunk_ids") or []) if node_id in chapter_by_id]
        for q in bundle.get("questions") or []:
            unique_qid = f"{bundle_id}__{q['question_id']}"
            gold_ids = list(q.get("gold_node_ids") or q.get("source_chunk_ids") or [])
            derived_question_family = "multi_version" if len(gold_ids) > 1 else "single_version"
            if normalize_top_k:
                dynamic_top_k = UNIFIED_TOP_K
            else:
                dynamic_top_k = int(q.get("dynamic_top_k") or UNIFIED_TOP_K)
            original_top_k = q.get("dynamic_top_k")
            if original_top_k != dynamic_top_k:
                top_k_rewrites.append(
                    {
                        "question_id": unique_qid,
                        "original_dynamic_top_k": original_top_k,
                        "normalized_dynamic_top_k": dynamic_top_k,
                        "gold_node_count": len(gold_ids),
                    }
                )

            options = None
            if q.get("options") is not None:
                options = [ChoiceOption(**item) for item in q["options"]]

            question = StateQuestion(
                question_id=unique_qid,
                state_chain_id=GLOBAL_CHAIN_ID,
                difficulty_level=q["difficulty"],
                question_family=derived_question_family,
                answerability="answerable",
                answer_format=q["answer_format"],
                query_text=q["query_text"],
                options=options,
                correct_option_id=q.get("correct_option_id"),
                expected_answer=q["expected_answer"],
                gold_node_ids=gold_ids,
                adversarial_node_ids=[],
                oracle_context_node_ids=gold_ids,
                dynamic_top_k=dynamic_top_k,
                reasoning_chain=list(q.get("reasoning_sketch") or []),
            )
            questions.append(question)
            meta_by_question_id[unique_qid] = {
                "bundle_id": bundle_id,
                "focus_event": focus_event,
                "task_type": q.get("task_type"),
                "original_question_id": q["question_id"],
                "gold_evidence": q.get("answer_support") or [],
                "adversarial_evidence": [],
            }

            for node_id in gold_ids:
                if node_id not in chapter_by_id:
                    raise ValueError(f"Question {unique_qid} references unknown memory_node_id: {node_id}")

    dataset = StateVersionDataset(
        chains={GLOBAL_CHAIN_ID: chain},
        questions_by_chain={GLOBAL_CHAIN_ID: questions},
        chain_split={GLOBAL_CHAIN_ID: "prototype"},
        chain_domain={GLOBAL_CHAIN_ID: domain},
    )
    audit = {
        "prototype_id": bundle_payload.get("novel_id") or slugify(source_title),
        "source_title": source_title,
        "chapter_count": len(chapter_rows),
        "bundle_count": len(bundles),
        "question_count": len(questions),
        "internal_budget_policy": f"fixed_top_k={UNIFIED_TOP_K}" if normalize_top_k else "question provided or derived fallback",
        "budget_rewrites": top_k_rewrites,
        "question_meta": meta_by_question_id,
        "memory_node_file": memory_node_file,
        "domain": domain,
    }
    return dataset, audit


def attach_question_metadata(run: dict[str, Any], question_meta: dict[str, Any]) -> dict[str, Any]:
    for row in run.get("question_results") or []:
        meta = question_meta.get(row["question_id"]) or {}
        row["bundle_id"] = meta.get("bundle_id")
        row["bundle_focus_event"] = meta.get("focus_event")
        row["task_type"] = meta.get("task_type")
        row["prototype_gold_evidence"] = meta.get("gold_evidence") or []
        row["prototype_adversarial_evidence"] = meta.get("adversarial_evidence") or []
        row["original_question_id"] = meta.get("original_question_id")
        row.pop("question_family", None)
        row.pop("dynamic_top_k", None)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline evaluation on one narrative prototype")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prototype-dir", type=Path, default=DEFAULT_PROTO_DIR)
    parser.add_argument("--systems", nargs="*", default=["bm25", "faiss", "mem0", "graphiti"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--normalize-top-k", action="store_true", default=True)
    parser.add_argument("--no-normalize-top-k", dest="normalize_top_k", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    eval_cfg = load_yaml(args.config)
    dataset, audit = build_dataset(prototype_dir=args.prototype_dir, normalize_top_k=args.normalize_top_k)

    create_systems = _load_create_systems()
    systems = create_systems(eval_cfg, eval_cfg, only_systems=args.systems)
    if not systems:
        raise ValueError("No systems were enabled for prototype evaluation")

    answer_cfg = eval_cfg["answer_generator"]
    answer_generator = AnswerGenerator(
        base_url=answer_cfg["base_url"],
        api_key=answer_cfg["api_key"],
        model=answer_cfg["model"],
        temperature=answer_cfg.get("temperature", 0.0),
        extra_body=answer_cfg.get("extra_body") or {},
    )

    judge = None
    judge_cfg = eval_cfg.get("llm_judge") or {}
    if judge_cfg.get("enabled", True):
        judge = LLMJudge(
            base_url=judge_cfg["base_url"],
            api_key=judge_cfg["api_key"],
            model=judge_cfg["model"],
            temperature=judge_cfg.get("temperature", 0.0),
            max_workers=judge_cfg.get("max_workers", 3),
            timeout=judge_cfg.get("timeout", 60),
            extra_body=judge_cfg.get("extra_body") or {},
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prototype_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    runner = StateVersionEvaluationRunner(
        dataset=dataset,
        systems=systems,
        answer_generator=answer_generator,
        judge=judge,
        random_context_pool_nodes=[],
    )

    for system in systems:
        run = runner.run_system(
            system,
            output_dir=output_dir,
            resume=False,
            max_workers=args.max_workers,
            save_every=args.save_every,
        )
        run["system_name"] = display_system_name(run["system_name"])
        for row in run.get("question_results") or []:
            row["system_name"] = display_system_name(row["system_name"])
        run = attach_question_metadata(run, audit["question_meta"])
        run["summary"] = summarize_state_version_results(run["question_results"])
        # overwrite saved artifacts with enriched metadata
        from src.state_version.evaluation import save_state_version_run

        save_state_version_run(output_dir=output_dir, run=run, save_per_question_jsonl=True)
        logging.info(
            "Prototype evaluation finished for %s: QA=%.3f Cov=%.3f CSR=%.3f",
            run["system_name"],
            run["summary"]["overall"].get("qa_accuracy") or 0.0,
            run["summary"]["overall"].get("gold_coverage_at_k") or 0.0,
            run["summary"]["overall"].get("complete_support_rate") or 0.0,
        )


if __name__ == "__main__":
    main()
