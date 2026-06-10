#!/usr/bin/env python3
"""Repair Mem0 node-level alignment using grouped source-node identifiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from src.state_version.evaluation import (  # noqa: E402
    _compute_formal_distractor_to_gold_ratio,
    _compute_gold_rank_positions,
    _explicit_retrieved_pairs_from_metadata,
    summarize_state_version_results,
)


DEFAULT_RESULT_DIR = (
    ROOT
    / "benchmark"
    / "data"
    / "prototype_eval_results"
    / "official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10"
)
DEFAULT_QUESTIONS = DEFAULT_RESULT_DIR / "mem0.questions.jsonl"
DEFAULT_SUMMARY = DEFAULT_RESULT_DIR / "mem0.summary.json"

TASK_PRIMARY_K = {
    "single_state_lookup": 3,
    "cross_version_comparison": 8,
    "temporal_version_ordering": 10,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_task(task_type: str | None) -> str:
    value = str(task_type or "").strip()
    if value == "temporal_ordering":
        return "temporal_version_ordering"
    return value


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def gold_node_ids_from_row(row: dict[str, Any]) -> list[str]:
    if row.get("gold_node_ids"):
        return ordered_unique([str(value) for value in row["gold_node_ids"]])
    return ordered_unique(
        [
            str(item.get("memory_node_id"))
            for item in (row.get("prototype_gold_evidence") or [])
            if item.get("memory_node_id")
        ]
    )


def adversarial_node_ids_from_row(row: dict[str, Any]) -> list[str]:
    if row.get("adversarial_node_ids"):
        return ordered_unique([str(value) for value in row["adversarial_node_ids"]])
    return ordered_unique(
        [
            str(item.get("memory_node_id"))
            for item in (row.get("prototype_adversarial_evidence") or [])
            if item.get("memory_node_id")
        ]
    )


def matched_node_ids_from_metadata(
    *,
    state_chain_id: str,
    metadata: dict[str, Any] | None,
    fallback: list[str],
) -> list[str]:
    explicit_pairs = _explicit_retrieved_pairs_from_metadata(metadata)
    if not explicit_pairs:
        return ordered_unique([str(value) for value in fallback if value])

    matched: list[str] = []
    seen: set[str] = set()
    for source_chain_id, node_id in explicit_pairs:
        if source_chain_id and source_chain_id != state_chain_id:
            continue
        if node_id and node_id not in seen:
            seen.add(node_id)
            matched.append(node_id)
    return matched


def refresh_result_fields(
    *,
    state_chain_id: str,
    answerability: str,
    gold_node_ids: list[str],
    adversarial_node_ids: list[str],
    result: dict[str, Any],
) -> dict[str, Any]:
    matched_node_ids = matched_node_ids_from_metadata(
        state_chain_id=state_chain_id,
        metadata=result.get("query_metadata") or {},
        fallback=list(result.get("matched_node_ids") or []),
    )
    gold_set = set(gold_node_ids)
    adversarial_set = set(adversarial_node_ids)
    matched_gold = [node_id for node_id in matched_node_ids if node_id in gold_set]
    matched_adv = [node_id for node_id in matched_node_ids if node_id in adversarial_set]
    matched_non_gold = [node_id for node_id in matched_node_ids if node_id not in gold_set]

    if answerability == "answerable":
        support_coverage = len(matched_gold) / max(len(gold_node_ids), 1)
        complete_support = len(matched_gold) == len(gold_node_ids)
        support_precision = len(matched_gold) / max(len(matched_node_ids), 1) if matched_node_ids else 0.0
        distractor_to_gold_ratio = _compute_formal_distractor_to_gold_ratio(matched_node_ids, gold_node_ids)
    else:
        support_coverage = None
        complete_support = False
        support_precision = None
        distractor_to_gold_ratio = None

    result["matched_node_ids"] = matched_node_ids
    result["matched_gold_node_ids"] = matched_gold
    result["matched_adversarial_node_ids"] = matched_adv
    result["matched_non_gold_node_ids"] = matched_non_gold
    result["support_coverage"] = support_coverage
    result["complete_support"] = complete_support
    result["support_precision"] = support_precision
    result["distractor_to_gold_ratio"] = distractor_to_gold_ratio
    result["is_correct_without_gold_support"] = bool(
        result.get("is_correct") and (support_coverage or 0.0) == 0.0
    )
    result.update(_compute_gold_rank_positions(matched_node_ids, gold_node_ids))
    return result


def repair_row(row: dict[str, Any]) -> dict[str, Any]:
    row["task_type"] = canonical_task(row.get("task_type"))
    gold_node_ids = gold_node_ids_from_row(row)
    adversarial_node_ids = adversarial_node_ids_from_row(row)
    answerability = str(row.get("answerability") or "")
    state_chain_id = str(row.get("state_chain_id") or "")

    per_k_results = row.get("per_k_results") or {}
    repaired_per_k: dict[str, Any] = {}
    for key, payload in per_k_results.items():
        repaired_per_k[str(key)] = refresh_result_fields(
            state_chain_id=state_chain_id,
            answerability=answerability,
            gold_node_ids=gold_node_ids,
            adversarial_node_ids=adversarial_node_ids,
            result=dict(payload),
        )
    row["per_k_results"] = repaired_per_k

    analysis_top_ks = [int(value) for value in (row.get("analysis_top_ks") or [])]
    primary_k = max(analysis_top_ks) if analysis_top_ks else TASK_PRIMARY_K.get(row["task_type"], 10)
    primary = repaired_per_k[str(primary_k)]

    for field in (
        "matched_node_ids",
        "matched_gold_node_ids",
        "matched_adversarial_node_ids",
        "support_coverage",
        "complete_support",
        "support_precision",
        "distractor_to_gold_ratio",
        "is_correct_without_gold_support",
        "gold_rank_positions",
        "first_gold_rank",
        "any_gold_within",
        "all_gold_within",
    ):
        row[field] = primary.get(field)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair Mem0 node-level alignment using grouped source ids.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = load_jsonl(args.questions)
    repaired_rows = [repair_row(dict(row)) for row in rows]
    dump_jsonl(args.questions, repaired_rows)

    summary_payload = {
        "system_name": "Mem0",
        "summary": summarize_state_version_results(repaired_rows),
    }
    args.summary.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    zero_correct = sum(1 for row in repaired_rows if row.get("is_correct_without_gold_support"))
    print(
        json.dumps(
            {
                "questions_path": str(args.questions),
                "summary_path": str(args.summary),
                "question_count": len(repaired_rows),
                "zero_correct_count": zero_correct,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
