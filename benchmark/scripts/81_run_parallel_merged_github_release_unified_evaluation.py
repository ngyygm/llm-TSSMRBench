#!/usr/bin/env python3
"""Run merged unified GitHub release-note evaluation in parallel, one prototype per subprocess."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "benchmark") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmark"))

from src.evaluation.answer_generator import AnswerGenerator
from src.evaluation.judge import LLMJudge
from src.state_version.evaluation import StateVersionEvaluationRunner, save_state_version_run, summarize_state_version_results

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


def load_eval80_module() -> Any:
    script_path = ROOT / "benchmark" / "scripts" / "80_run_merged_github_release_unified_evaluation.py"
    spec = importlib.util.spec_from_file_location("merged_eval_v80_parallel", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluation module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_eval71_module() -> Any:
    script_path = ROOT / "benchmark" / "scripts" / "71_run_narrative_prototype_evaluation.py"
    spec = importlib.util.spec_from_file_location("prototype_eval_v71_parallel", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluation module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if raw:
            rows.append(json.loads(raw))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_worker_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _evaluate_one_prototype(
    *,
    payload: dict[str, Any],
    config_path: str,
    output_dir: str,
    system_name: str,
    question_workers: int,
    save_every: int,
) -> dict[str, Any]:
    eval80 = load_eval80_module()
    eval71 = load_eval71_module()

    load_yaml = eval71.load_yaml
    create_systems = eval71._load_create_systems()
    attach_question_metadata = eval71.attach_question_metadata
    display_system_name = eval71.display_system_name

    output_root = Path(output_dir)
    prototype_id = str(payload["prototype_id"])
    repo = str(payload.get("repo") or "")
    per_proto_output = output_root / prototype_id

    eval_cfg = load_yaml(Path(config_path))
    dataset, audit = eval80.build_dataset_from_unified_payload(payload)
    write_json(per_proto_output / "prototype_audit.json", audit)

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

    systems = create_systems(eval_cfg, eval_cfg, only_systems=[system_name])
    if len(systems) != 1:
        raise RuntimeError(f"{prototype_id}: expected one system for {system_name}, got {len(systems)}")
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

    run = runner.run_system(
        system,
        output_dir=per_proto_output,
        resume=False,
        max_workers=question_workers,
        save_every=save_every,
    )
    run["system_name"] = display_system_name(run["system_name"])
    for row in run.get("question_results") or []:
        row["system_name"] = display_system_name(row["system_name"])
    run = attach_question_metadata(run, audit["question_meta"])
    run["summary"] = summarize_state_version_results(run["question_results"])
    save_state_version_run(output_dir=per_proto_output, run=run, save_per_question_jsonl=True)

    return {
        "prototype_id": prototype_id,
        "repo": repo,
        "question_count": audit["question_count"],
        "system_name": run["system_name"],
        "summary": run["summary"]["overall"],
        "status": "success",
    }


def _worker_entry(kwargs: dict[str, Any]) -> dict[str, Any]:
    payload = kwargs["payload"]
    prototype_id = str(payload["prototype_id"])
    repo = str(payload.get("repo") or "")
    try:
        return _evaluate_one_prototype(**kwargs)
    except Exception as exc:
        return {
            "prototype_id": prototype_id,
            "repo": repo,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _aggregate_system_results(output_dir: Path, display_name: str, system_slug: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for prototype_dir in sorted(output_dir.iterdir()):
        if not prototype_dir.is_dir():
            continue
        path = prototype_dir / f"{system_slug}.questions.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path))

    summary = summarize_state_version_results(rows) if rows else {"overall": {"question_count": 0}, "breakdowns": {}, "notes": {}}
    run = {
        "system_name": display_name,
        "question_results": rows,
        "summary": summary,
    }
    save_state_version_run(output_dir=output_dir, run=run, save_per_question_jsonl=True)
    return {
        "system_name": display_name,
        "summary": summary.get("overall") or {},
        "question_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel formal evaluation over merged unified GitHub release-note prototypes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--merged-json", type=Path, default=DEFAULT_MERGED_JSON)
    parser.add_argument("--system", choices=["mem0", "graphiti"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prototype-workers", type=int, default=50)
    parser.add_argument("--question-workers", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--worker-payload", type=Path, default=None)
    parser.add_argument("--worker-result", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.worker_payload is not None and args.worker_result is not None:
        payload = read_json(args.worker_payload)
        result = _worker_entry(
            {
                "payload": payload,
                "config_path": str(args.config),
                "output_dir": str(args.output_dir),
                "system_name": args.system,
                "question_workers": args.question_workers,
                "save_every": args.save_every,
            }
        )
        write_worker_json(args.worker_result, result)
        return

    merged = read_json(args.merged_json)
    prototypes = list(merged.get("prototypes") or [])
    if args.limit > 0:
        prototypes = prototypes[: args.limit]
    if not prototypes:
        raise ValueError("No prototypes found in merged JSON")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "controller_meta.json",
        {
            "merged_json": str(args.merged_json),
            "dataset_id": merged.get("dataset_id"),
            "prototype_count": len(prototypes),
            "system": args.system,
            "prototype_workers": args.prototype_workers,
            "question_workers": args.question_workers,
            "save_every": args.save_every,
        },
    )

    progress_path = args.output_dir / "progress.jsonl"
    progress_rows = read_jsonl(progress_path) if args.resume else []
    completed_ids = {
        str(row.get("prototype_id"))
        for row in progress_rows
        if row.get("status") == "success"
    }

    remaining = [payload for payload in prototypes if str(payload["prototype_id"]) not in completed_ids]
    logging.info("System=%s total=%s remaining=%s completed=%s", args.system, len(prototypes), len(remaining), len(completed_ids))

    if remaining:
        payload_dir = args.output_dir / "_worker_payloads"
        result_dir = args.output_dir / "_worker_results"
        payload_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        pending = list(remaining)
        active: dict[str, dict[str, Any]] = {}
        max_active = max(1, args.prototype_workers)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        def launch_one(payload: dict[str, Any]) -> None:
            prototype_id = str(payload["prototype_id"])
            payload_path = payload_dir / f"{prototype_id}.json"
            result_path = result_dir / f"{prototype_id}.json"
            write_worker_json(payload_path, payload)
            cmd = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--config",
                str(args.config),
                "--merged-json",
                str(args.merged_json),
                "--system",
                args.system,
                "--output-dir",
                str(args.output_dir),
                "--question-workers",
                str(args.question_workers),
                "--save-every",
                str(args.save_every),
                "--worker-payload",
                str(payload_path),
                "--worker-result",
                str(result_path),
            ]
            if args.verbose:
                cmd.append("--verbose")
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                creationflags=creationflags,
            )
            active[prototype_id] = {
                "process": proc,
                "result_path": result_path,
                "payload_path": payload_path,
            }

        while pending or active:
            while pending and len(active) < max_active:
                launch_one(pending.pop(0))

            finished: list[str] = []
            for prototype_id, info in list(active.items()):
                proc: subprocess.Popen = info["process"]
                if proc.poll() is None:
                    continue
                result_path: Path = info["result_path"]
                if result_path.exists():
                    result = read_json(result_path)
                else:
                    result = {
                        "prototype_id": prototype_id,
                        "status": "error",
                        "error_type": "WorkerExitWithoutResult",
                        "error": f"Worker exited with code {proc.returncode} without result file",
                    }
                append_jsonl(progress_path, result)
                status = result.get("status")
                if status == "success":
                    summary = result.get("summary") or {}
                    logging.info(
                        "%s %s: ACC=%.3f Cov=%.3f CSR=%.3f",
                        result.get("prototype_id"),
                        result.get("system_name"),
                        summary.get("qa_accuracy") or 0.0,
                        summary.get("gold_coverage_at_k") or 0.0,
                        summary.get("complete_support_rate") or 0.0,
                    )
                else:
                    logging.error("%s failed: %s", result.get("prototype_id"), result.get("error"))
                finished.append(prototype_id)

            for prototype_id in finished:
                active.pop(prototype_id, None)

            if active:
                time.sleep(2.0)

    display_name = "Mem0" if args.system == "mem0" else "Graphiti"
    aggregate = _aggregate_system_results(args.output_dir, display_name, args.system)
    write_json(args.output_dir / "aggregate_summary.json", {"systems": [aggregate]})


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
