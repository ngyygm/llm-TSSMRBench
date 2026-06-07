#!/usr/bin/env python3
"""Evaluate the merged official GitHub release-note dataset in one global mixed memory pool."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import threading
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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

DEFAULT_CONFIG = ROOT / "benchmark" / "configs" / "state_version_experiment_config_deepseek_flash_memory.yaml"
DEFAULT_MERGED_JSON = (
    ROOT
    / "benchmark"
    / "data"
    / "prototypes"
    / "github_release_note_v2"
    / "formal_300repo_unified_v1"
    / "official_300_merged.json"
)


def load_eval71_module() -> Any:
    script_path = ROOT / "benchmark" / "scripts" / "71_run_narrative_prototype_evaluation.py"
    spec = importlib.util.spec_from_file_location("prototype_eval_v71_globalpool", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluation module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sanitize_run_id(value: str) -> str:
    import re

    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value or "").strip("_")
    return normalized or "globalpool"


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_system_run_id(eval_cfg: dict[str, Any], system_name: str, output_dir: Path) -> tuple[str, dict[str, Any]]:
    run_meta_path = output_dir / f"{system_name}_memory_run.json"
    existing = _load_json_if_exists(run_meta_path)
    if existing and existing.get("run_id"):
        run_id = str(existing["run_id"])
    else:
        run_id = _sanitize_run_id(f"{output_dir.name}_{system_name}")
        write_json(
            run_meta_path,
            {
                "system": system_name,
                "run_id": run_id,
                "output_dir": str(output_dir),
            },
        )

    cfg = copy.deepcopy(eval_cfg)
    cfg.setdefault("systems", {}).setdefault(system_name, {})
    cfg["systems"][system_name]["run_id"] = run_id
    return run_id, cfg


def _load_ingest_checkpoint(path: Path) -> dict[str, Any]:
    payload = _load_json_if_exists(path)
    if payload is None:
        return {
            "ingestion_complete": False,
            "completed_nodes_by_chain": {},
            "completed_chain_count": 0,
            "completed_node_count": 0,
        }
    payload.setdefault("ingestion_complete", False)
    payload.setdefault("completed_nodes_by_chain", {})
    payload.setdefault("completed_chain_count", 0)
    payload.setdefault("completed_node_count", 0)
    return payload


def _save_ingest_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def ingest_global_pool_with_resume(
    *,
    runner: StateVersionEvaluationRunner,
    system: Any,
    output_dir: Path,
    resume: bool,
) -> None:
    if system.name == "Full Context":
        return

    system_slug = runner._system_slug(system.name)
    checkpoint_path = output_dir / f"{system_slug}.ingest_checkpoint.json"
    progress_path = output_dir / f"{system_slug}.ingest_progress.jsonl"
    checkpoint = _load_ingest_checkpoint(checkpoint_path) if resume else {
        "ingestion_complete": False,
        "completed_nodes_by_chain": {},
        "completed_chain_count": 0,
        "completed_node_count": 0,
    }

    if checkpoint.get("ingestion_complete"):
        logging.info("%s ingestion checkpoint already complete; skipping ingest", system.name)
        return

    completed_nodes_by_chain: dict[str, set[str]] = {
        chain_id: set(node_ids or [])
        for chain_id, node_ids in (checkpoint.get("completed_nodes_by_chain") or {}).items()
    }

    if resume and checkpoint.get("completed_node_count", 0) > 0:
        logging.info(
            "Resuming %s ingest from checkpoint: %s nodes across %s chains already completed",
            system.name,
            checkpoint.get("completed_node_count", 0),
            checkpoint.get("completed_chain_count", 0),
        )
    else:
        system.reset()
        completed_nodes_by_chain = {}
        checkpoint = {
            "ingestion_complete": False,
            "completed_nodes_by_chain": {},
            "completed_chain_count": 0,
            "completed_node_count": 0,
        }
        _save_ingest_checkpoint(checkpoint_path, checkpoint)

    state_lock = threading.Lock()
    chain_ingest_workers = max(1, int(getattr(system, "chain_ingest_workers", 1)))

    chain_payloads: list[tuple[str, list[tuple[str, str]]]] = []
    for chain_id in sorted(runner.dataset.chains):
        chain = runner.dataset.chains[chain_id]
        ordered_nodes = sorted(chain.chain_nodes, key=lambda node: node.surface_order)
        node_pairs = [(node.node_id, node.text) for node in ordered_nodes]
        chain_payloads.append((chain.state_chain_id, node_pairs))

    def persist_progress(chain_id: str, node_id: str) -> None:
        with state_lock:
            chain_done = completed_nodes_by_chain.setdefault(chain_id, set())
            chain_done.add(node_id)
            checkpoint["completed_nodes_by_chain"] = {
                key: sorted(value)
                for key, value in completed_nodes_by_chain.items()
            }
            checkpoint["completed_chain_count"] = sum(
                1 for _, pairs in chain_payloads if len(completed_nodes_by_chain.get(_, set())) >= len(pairs)
            )
            checkpoint["completed_node_count"] = sum(len(value) for value in completed_nodes_by_chain.values())
            with progress_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps({"chain_id": chain_id, "node_id": node_id}, ensure_ascii=False) + "\n")
            _save_ingest_checkpoint(checkpoint_path, checkpoint)

    def ingest_chain_nodes(chain_id: str, node_pairs: list[tuple[str, str]]) -> None:
        done = completed_nodes_by_chain.get(chain_id, set())
        total = len(node_pairs)
        for index, (node_id, text) in enumerate(node_pairs, start=1):
            if node_id in done:
                continue
            logging.info("Ingesting %s node %s/%s: %s", chain_id, index, total, node_id)
            system.remember_chain(chain_id, [node_id], [text])
            persist_progress(chain_id, node_id)

    with ThreadPoolExecutor(max_workers=chain_ingest_workers) as executor:
        futures = [
            executor.submit(ingest_chain_nodes, chain_id, node_pairs)
            for chain_id, node_pairs in chain_payloads
            if len(completed_nodes_by_chain.get(chain_id, set())) < len(node_pairs)
        ]
        for future in as_completed(futures):
            future.result()

    checkpoint["ingestion_complete"] = True
    checkpoint["completed_nodes_by_chain"] = {
        key: sorted(value)
        for key, value in completed_nodes_by_chain.items()
    }
    checkpoint["completed_chain_count"] = len(chain_payloads)
    checkpoint["completed_node_count"] = sum(len(pairs) for _, pairs in chain_payloads)
    _save_ingest_checkpoint(checkpoint_path, checkpoint)


def build_global_dataset_from_merged_payload(merged: dict[str, Any]) -> tuple[StateVersionDataset, dict[str, Any]]:
    chains: dict[str, StateChainSample] = {}
    questions_by_chain: dict[str, list[StateQuestion]] = defaultdict(list)
    chain_split: dict[str, str] = {}
    chain_domain: dict[str, str] = {}
    audit_meta: dict[str, Any] = {}

    prototypes = list(merged.get("prototypes") or [])
    for payload in prototypes:
        prototype_id = str(payload["prototype_id"])
        repo = str(payload.get("repo") or "")
        chapter_rows = list(payload.get("chunks") or [])
        if not chapter_rows:
            raise ValueError(f"{prototype_id}: no chunks found")
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
            memory_text = str(row.get("memory_unit_text") or "").strip()
            if not memory_text:
                raise ValueError(f"{prototype_id} chunk {row.get('memory_node_id')} has blank memory_unit_text")
            span_hint = f"{row.get('artifact_type')}::{row.get('artifact_ref')}"
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
        source_title = payload.get("window_title") or f"{repo}::release_window"
        chain = StateChainSample(
            sample_id=f"{prototype_id}_global",
            state_chain_id=prototype_id,
            domain=domain,
            language="en",
            focus_event=payload.get("window_title") or "repository release evolution window",
            chain_summary=payload.get("window_summary") or "",
            source_kind=domain,
            source_title=source_title,
            chain_profile=ChainProfile(
                node_count=len(chain_nodes),
                competition_strength="high",
                lexical_overlap_band="high",
            ),
            chain_nodes=chain_nodes,
        )
        chains[prototype_id] = chain
        chain_split[prototype_id] = "prototype"
        chain_domain[prototype_id] = domain

        for q in payload.get("questions") or []:
            unique_qid = f"{prototype_id}__{q['question_id']}"
            gold_ids = list(q.get("gold_node_ids") or q.get("source_chunk_ids") or [])
            derived_question_family = "multi_version" if len(gold_ids) > 1 else "single_version"
            options = [ChoiceOption(**item) for item in (q.get("options") or [])] if q.get("options") else None
            question = StateQuestion(
                question_id=unique_qid,
                state_chain_id=prototype_id,
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
                dynamic_top_k=10,
                reasoning_chain=list(q.get("reasoning_sketch") or []),
            )
            questions_by_chain[prototype_id].append(question)
            audit_meta[unique_qid] = {
                "bundle_id": prototype_id,
                "focus_event": payload.get("window_title"),
                "task_type": q.get("task_type"),
                "original_question_id": q["question_id"],
                "repo": repo,
                "gold_evidence": q.get("answer_support") or [],
            }
            for node_id in gold_ids:
                if node_id not in chapter_by_id:
                    raise ValueError(f"{prototype_id} question {unique_qid} references unknown node {node_id}")

    dataset = StateVersionDataset(
        chains=chains,
        questions_by_chain=dict(questions_by_chain),
        chain_split=chain_split,
        chain_domain=chain_domain,
    )
    audit = {
        "dataset_id": merged.get("dataset_id"),
        "prototype_count": len(prototypes),
        "chain_count": len(chains),
        "question_count": sum(len(v) for v in questions_by_chain.values()),
        "question_meta": audit_meta,
        "memory_node_file": "official_300_merged.json",
    }
    return dataset, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-system evaluation over the official merged GitHub release dataset in one mixed memory pool.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--merged-json", type=Path, default=DEFAULT_MERGED_JSON)
    parser.add_argument("--system", choices=["mem0", "graphiti", "bm25", "faiss", "full_context"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    eval71 = load_eval71_module()
    load_yaml = eval71.load_yaml
    create_systems = eval71._load_create_systems()
    attach_question_metadata = eval71.attach_question_metadata
    display_system_name = eval71.display_system_name

    merged = load_json(args.merged_json)
    dataset, audit = build_global_dataset_from_merged_payload(merged)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "global_pool_audit.json", audit)

    eval_cfg = load_yaml(args.config)
    _, eval_cfg = _ensure_system_run_id(eval_cfg, args.system, args.output_dir)
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
            max_workers=judge_cfg.get("max_workers", 50),
            timeout=judge_cfg.get("timeout", 60),
            extra_body=judge_cfg.get("extra_body") or {},
        )

    systems = create_systems(eval_cfg, eval_cfg, only_systems=[args.system])
    if len(systems) != 1:
        raise RuntimeError(f"Expected one system for {args.system}, got {len(systems)}")
    system = systems[0]

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

    ingest_global_pool_with_resume(
        runner=runner,
        system=system,
        output_dir=args.output_dir,
        resume=args.resume,
    )

    run = runner.run_system(
        system,
        output_dir=args.output_dir,
        resume=args.resume,
        max_workers=args.max_workers,
        save_every=args.save_every,
        pre_ingested=(system.name != "Full Context"),
    )
    run["system_name"] = display_system_name(run["system_name"])
    for row in run.get("question_results") or []:
        row["system_name"] = display_system_name(row["system_name"])
    run = attach_question_metadata(run, audit["question_meta"])
    run["summary"] = summarize_state_version_results(run["question_results"])
    save_state_version_run(output_dir=args.output_dir, run=run, save_per_question_jsonl=True)
    write_json(
        args.output_dir / "aggregate_summary.json",
        {
            "systems": [
                {
                    "system_name": run["system_name"],
                    "summary": run["summary"]["overall"],
                    "question_count": len(run["question_results"]),
                }
            ]
        },
    )


if __name__ == "__main__":
    main()
