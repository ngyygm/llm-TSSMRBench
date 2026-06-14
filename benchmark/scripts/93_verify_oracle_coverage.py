#!/usr/bin/env python3
"""Deterministically verify Oracle Gold Context coverage on the current dataset.

The Oracle baseline injects the annotated gold nodes directly into the answer
model's context. By construction its support coverage and complete-support rate
must therefore be exactly 1.0 -- provided every annotated gold id actually
exists in the question's chain. A prior Oracle run reported Cov/CSR slightly
below 1.0 because it was computed against an earlier dataset revision in which
a few gold ids were missing (and ``_build_full_context_query_result`` silently
dropped them); those ids have since been repaired.

This script re-derives Oracle coverage from the CURRENT released dataset without
calling any answer model (coverage does not depend on the answer LLM). It proves
Cov = CSR = 1.0 for all questions, so the ``1.0 by construction`` claim in the
paper is correct and auditable on the committed data.

Oracle ACC still requires the answer model and must be regenerated with:

    python benchmark/scripts/82_run_merged_github_release_unified_global_pool_evaluation.py \\
      --system full_context \\
      --merged-json <official_300_merged.json> \\
      --output-dir <...>/official_300repo_release_unified_v1_simple_baselines_conda_taskk
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERGED_JSON = (
    ROOT
    / "benchmark"
    / "data"
    / "prototypes"
    / "github_release_note_v2"
    / "formal_300repo_unified_v1"
    / "official_300_merged.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark"
    / "data"
    / "prototype_eval_results"
    / "official_300repo_release_unified_v1_paper_artifacts"
    / "oracle_coverage_proof.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-json", type=Path, default=DEFAULT_MERGED_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    merged = json.loads(args.merged_json.read_text(encoding="utf-8"))
    total = 0
    missing_total = 0
    all_complete = True
    missing_examples: list[str] = []
    for proto in merged.get("prototypes") or []:
        chunk_ids = {str(c.get("memory_node_id") or "") for c in (proto.get("chunks") or [])}
        for q in proto.get("questions") or []:
            gold = list(q.get("source_chunk_ids") or q.get("gold_node_ids") or [])
            missing = [g for g in gold if g not in chunk_ids]
            total += 1
            missing_total += len(missing)
            if missing or len(gold) == 0:
                all_complete = False
                if len(missing_examples) < 10:
                    missing_examples.append(f"{proto.get('repo')}/{q.get('question_id')}: missing {missing}")
            # coverage = len(present gold)/len(gold); complete iff all present

    result = {
        "dataset": str(args.merged_json),
        "questions_checked": total,
        "gold_ids_missing_across_dataset": missing_total,
        "oracle_coverage_equals_1_for_all": all_complete and missing_total == 0,
        "oracle_csr_equals_1_for_all": all_complete and missing_total == 0,
        "note": "Oracle injects gold nodes directly; since every gold id exists in its chain on the current dataset, Cov=CSR=1.0 for all questions. Oracle ACC still requires an answer-model rerun.",
        "rerun_command": "python benchmark/scripts/82_run_merged_github_release_unified_global_pool_evaluation.py --system full_context --merged-json <official_300_merged.json> --output-dir <...>/official_300repo_release_unified_v1_simple_baselines_conda_taskk",
        "missing_examples": missing_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["oracle_coverage_equals_1_for_all"]:
        raise SystemExit("ERROR: Oracle coverage is NOT 1.0 on the current dataset; investigate missing gold ids.")


if __name__ == "__main__":
    main()
