#!/usr/bin/env python3
"""Evaluate a merged unified GitHub release-note dataset with incremental persistence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "benchmark") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmark"))

from src.evaluation.answer_generator import AnswerGenerator
from src.evaluation.judge import LLMJudge
from src.state_version.evaluation import (
    StateVersionDataset,
    StateVersionEvaluationRunner,
    save_state_version_run,
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

DEFAULT_CONFIG = ROOT / "benchmark" / "configs" / "state_version_experiment_config.yaml"
DEFAULT_MERGED_JSON = (
    ROOT
    / "benchmark"
    / "data"
    / "prototypes"
    / "github_release_note_v2"
    / "formal_300repo_unified_v1"
    / "official_300_merged.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "benchmark"
    / "data"
    / "prototype_eval_results"
    / "official_300repo_release_unified_v1_simple_baselines"
)
GLOBAL_CHAIN_ID = "prototype_global_memory_pool"
UNIFIED_TOP_K = 10


def load_eval_module() -> Any:
    script_path = ROOT / "benchmark" / "scripts" / "71_run_narrative_prototype_evaluation.py"
    spec = importlib.util.spec_from_file_location("prototype_eval_v71", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluation module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def build_dataset_from_unified_payload(payload: dict[str, Any]) -> tuple[StateVersionDataset, dict[str, Any]]:
    chapter_rows = list(payload.get("chunks") or [])
    if not chapter_rows:
        raise ValueError(f"{payload.get('prototype_id')}: no chunks found")
    bundle = {
        "chain_id": payload.get("prototype_id") or "prototype",
        "chain_title": payload.get("window_title") or "release window",
        "chain_summary": payload.get("window_summary") or "",
        "source_chunk_ids": [row["memory_node_id"] for row in chapter_rows if row.get("memory_node_id")],
        "questions": payload.get("questions") or [],
    }
    if not bundle["questions"]:
        raise ValueError(f"{payload.get('prototype_id')}: no questions found")

    chapter_rows = sorted(
        chapter_rows,
        key=lambda row: (
            str(row.get("repo") or ""),
            str(row.get("time_hint") or ""),
            str(row.get("published_at") or ""),
            str(row.get("artifact_ref") or ""),
            str(row.get("memory_node_id") or ""),
        ),
    )
    chapter_by_id = {row["memory_node_id"]: row for row in chapter_rows}

    chain_nodes: list[StateChainNode] = []
    for idx, row in enumerate(chapter_rows, start=1):
        span_hint = f"{row.get('artifact_type')}::{row.get('artifact_ref')}"
        memory_text = str(row.get("memory_unit_text") or "").strip()
        if not memory_text:
            raise ValueError(
                f"{payload.get('prototype_id')} chunk {row.get('memory_node_id')} has blank memory_unit_text"
            )
        chain_nodes.append(
            StateChainNode(
                node_id=row["memory_node_id"],
                surface_order=idx,
                text=memory_text,
                progress_label="active",
                perspective_label="retrospective",
                relation_label="continues",
                salience_label="core",
                supersedes=[],
                depends_on=[],
                source_pointer=SourcePointer(
                    artifact_type=str(row.get("artifact_type") or "release_note"),
                    artifact_ref=row["memory_node_id"],
                    span_hint=span_hint,
                ),
            )
        )

    raw_domain = str(payload.get("domain") or "github_evolution")
    domain = "github_evolution" if raw_domain == "github_release_evolution" else raw_domain
    source_title = payload.get("window_title") or f"{payload.get('repo') or 'github'}::release_window"
    chain = StateChainSample(
        sample_id=f"{payload.get('prototype_id') or 'prototype'}_global",
        state_chain_id=GLOBAL_CHAIN_ID,
        domain=domain,
        language="en",
        focus_event="multiple temporal semantic state evolution questions over one repository release window",
        chain_summary=payload.get("window_summary")
        or "All release-note chunks in one repository window are stored into a single noisy memory pool.",
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
    for q in bundle["questions"]:
        unique_qid = f"{bundle['chain_id']}__{q['question_id']}"
        gold_ids = list(q.get("gold_node_ids") or q.get("source_chunk_ids") or [])
        derived_question_family = "multi_version" if len(gold_ids) > 1 else "single_version"
        options = [ChoiceOption(**item) for item in (q.get("options") or [])] if q.get("options") else None

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
            dynamic_top_k=UNIFIED_TOP_K,
            reasoning_chain=list(q.get("reasoning_sketch") or []),
        )
        questions.append(question)
        meta_by_question_id[unique_qid] = {
            "bundle_id": bundle["chain_id"],
            "focus_event": payload.get("window_title"),
            "task_type": q.get("task_type"),
            "original_question_id": q["question_id"],
            "gold_evidence": q.get("answer_support") or [],
            "adversarial_evidence": [],
        }
        for node_id in gold_ids:
            if node_id not in chapter_by_id:
                raise ValueError(f"{payload.get('prototype_id')} question {unique_qid} references unknown node {node_id}")

    dataset = StateVersionDataset(
        chains={GLOBAL_CHAIN_ID: chain},
        questions_by_chain={GLOBAL_CHAIN_ID: questions},
        chain_split={GLOBAL_CHAIN_ID: "prototype"},
        chain_domain={GLOBAL_CHAIN_ID: domain},
    )
    audit = {
        "prototype_id": payload.get("prototype_id"),
        "repo": payload.get("repo"),
        "source_title": source_title,
        "chapter_count": len(chapter_rows),
        "bundle_count": 1,
        "question_count": len(questions),
        "internal_budget_policy": f"fixed_top_k={UNIFIED_TOP_K}",
        "budget_rewrites": [],
        "question_meta": meta_by_question_id,
        "memory_node_file": "prototype.json",
        "domain": domain,
    }
    return dataset, audit


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baselines on a merged unified GitHub release-note dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--merged-json", type=Path, default=DEFAULT_MERGED_JSON)
    parser.add_argument("--systems", nargs="*", default=["full_context", "bm25", "faiss"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    eval_module = load_eval_module()
    display_system_name = eval_module.display_system_name
    load_yaml = eval_module.load_yaml
    create_systems = eval_module._load_create_systems()
    attach_question_metadata = eval_module.attach_question_metadata

    eval_cfg = load_yaml(args.config)
    merged = load_json(args.merged_json)
    prototypes = list(merged.get("prototypes") or [])
    if args.limit > 0:
        prototypes = prototypes[: args.limit]
    if not prototypes:
        raise ValueError("No prototypes found in merged JSON")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "merged_dataset_meta.json",
        {
            "merged_json": str(args.merged_json),
            "dataset_id": merged.get("dataset_id"),
            "prototype_count": len(prototypes),
            "systems": args.systems,
        },
    )

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

    progress_path = output_dir / "progress.jsonl"
    progress_rows: list[dict[str, Any]] = read_jsonl(progress_path) if args.resume else []
    completed_ids = {str(row.get("prototype_id")) for row in progress_rows}

    for index, payload in enumerate(prototypes, start=1):
        prototype_id = str(payload["prototype_id"])
        repo = str(payload.get("repo") or "")
        if args.resume and prototype_id in completed_ids:
            logging.info("Skipping already completed prototype: %s", prototype_id)
            continue
        logging.info("Evaluating %s/%s: %s", index, len(prototypes), prototype_id)
        dataset, audit = build_dataset_from_unified_payload(payload)
        per_proto_output = output_dir / prototype_id
        write_json(per_proto_output / "prototype_audit.json", audit)

        systems = create_systems(eval_cfg, eval_cfg, only_systems=args.systems)
        runner = StateVersionEvaluationRunner(
            dataset=dataset,
            systems=systems,
            answer_generator=answer_generator,
            judge=judge,
            random_context_pool_nodes=[],
            question_task_type_map={
                qid: str(meta.get("task_type") or "")
                for qid, meta in audit["question_meta"].items()
                if meta.get("task_type")
            },
        )

        progress_entry: dict[str, Any] = {
            "prototype_id": prototype_id,
            "repo": repo,
            "question_count": audit["question_count"],
            "systems": {},
        }

        for system in systems:
            run = runner.run_system(
                system,
                output_dir=per_proto_output,
                resume=False,
                max_workers=args.max_workers,
                save_every=args.save_every,
            )
            run["system_name"] = display_system_name(run["system_name"])
            for row in run.get("question_results") or []:
                row["system_name"] = display_system_name(row["system_name"])
            run = attach_question_metadata(run, audit["question_meta"])
            run["summary"] = summarize_state_version_results(run["question_results"])
            save_state_version_run(output_dir=per_proto_output, run=run, save_per_question_jsonl=True)

            system_slug = str(getattr(system, "name", run["system_name"]))
            progress_entry["systems"][system_slug] = run["summary"]["overall"]
            logging.info(
                "%s %s: QA=%.3f Cov=%.3f CSR=%.3f",
                prototype_id,
                run["system_name"],
                run["summary"]["overall"].get("qa_accuracy") or 0.0,
                run["summary"]["overall"].get("gold_coverage_at_k") or 0.0,
                run["summary"]["overall"].get("complete_support_rate") or 0.0,
            )

        progress_rows.append(progress_entry)
        write_jsonl(progress_path, progress_rows)

    aggregate_by_system: dict[str, list[dict[str, Any]]] = {}
    for row in progress_rows:
        prototype_id = str(row.get("prototype_id"))
        per_proto_output = output_dir / prototype_id
        for filename, system_key in [
            ("oracle_gold_context.questions.jsonl", "Full Context"),
            ("bm25.questions.jsonl", "BM25"),
            ("faiss_vector_store.questions.jsonl", "FAISS Vector Store"),
        ]:
            path = per_proto_output / filename
            if not path.exists():
                continue
            aggregate_by_system.setdefault(system_key, []).extend(read_jsonl(path))

    summary_index: list[dict[str, Any]] = []
    for system_key, rows in aggregate_by_system.items():
        if not rows:
            continue
        summary = summarize_state_version_results(rows)
        system_name = display_system_name(system_key)
        aggregate_run = {
            "system_name": system_name,
            "question_results": rows,
            "summary": summary,
        }
        save_state_version_run(output_dir=output_dir, run=aggregate_run, save_per_question_jsonl=True)
        summary_index.append(
            {
                "system_name": system_name,
                "summary": summary["overall"],
                "question_count": len(rows),
            }
        )

    write_json(output_dir / "aggregate_summary.json", {"systems": summary_index})


if __name__ == "__main__":
    main()
