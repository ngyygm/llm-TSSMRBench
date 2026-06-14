#!/usr/bin/env python3
"""Validate the integrity of the released TSSMRBench merged dataset.

Checks (hard errors fail the build; warnings are reported):
* every prototype has exactly 30 memory nodes;
* every question's gold ``source_chunk_ids`` exists in its prototype's chunks;
* task-specific support-size invariants:
    - single_state_lookup      |G| == 1
    - cross_version_comparison |G| == 2
    - temporal_version_ordering|G| in {3,4}
* every question has exactly 4 real options A-D and correct_option_id in A-D;
* correct_option_id distribution per task (position-skew report);
* memory_node_id uniqueness within a prototype;
* no blank memory_unit_text.

Exits 1 if any hard error is found, so this can gate a release.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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
REAL_OPTIONS = {"A", "B", "C", "D"}
EXPECTED_SUPPORT = {
    "single_state_lookup": {1},
    "cross_version_comparison": {2},
    "temporal_version_ordering": {3, 4},
}


def validate(merged_path: Path) -> int:
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    prototypes = list(merged.get("prototypes") or [])
    errors: list[str] = []
    warnings: list[str] = []

    per_task_correct: dict[str, Counter] = {}
    per_task_gold: dict[str, Counter] = {}

    for proto in prototypes:
        pid = proto.get("prototype_id", "?")
        repo = proto.get("repo", "?")
        chunks = proto.get("chunks") or []
        if len(chunks) != 30:
            warnings.append(f"{repo}: {len(chunks)} chunks (expected 30)")
        chunk_ids = [str(c.get("memory_node_id") or "") for c in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            errors.append(f"{repo}: duplicate memory_node_id within prototype")
        chunk_id_set = set(chunk_ids)
        for c in chunks:
            if not str(c.get("memory_unit_text") or "").strip():
                errors.append(f"{repo}: blank memory_unit_text for {c.get('memory_node_id')}")

        for q in proto.get("questions") or []:
            qid = q.get("question_id", "?")
            tt = q.get("task_type")
            gold = list(q.get("source_chunk_ids") or q.get("gold_node_ids") or [])
            correct = str(q.get("correct_option_id") or "").upper()
            options = q.get("options") or []
            opt_ids = [str(o.get("option_id") or "").upper() for o in options]

            # gold existence
            missing = [g for g in gold if g not in chunk_id_set]
            if missing:
                errors.append(f"{repo}/{qid}: gold ids missing from chunks: {missing}")
            # support-size invariant
            if tt in EXPECTED_SUPPORT and len(gold) not in EXPECTED_SUPPORT[tt]:
                errors.append(
                    f"{repo}/{qid}: task={tt} has |G|={len(gold)} (expected {sorted(EXPECTED_SUPPORT[tt])})"
                )
            # options
            if opt_ids[:4] != ["A", "B", "C", "D"]:
                errors.append(f"{repo}/{qid}: option ids not A/B/C/D: {opt_ids}")
            if len([o for o in opt_ids if o in REAL_OPTIONS]) != 4:
                errors.append(f"{repo}/{qid}: expected exactly 4 real options, got {opt_ids}")
            if correct not in REAL_OPTIONS:
                errors.append(f"{repo}/{qid}: correct_option_id={correct!r} not in A-D")
            # distribution bookkeeping
            per_task_correct.setdefault(tt, Counter())[correct] += 1
            per_task_gold.setdefault(tt, Counter())[len(gold)] += 1

    print("=== Per-task correct_option_id distribution (position skew) ===")
    for tt in ["single_state_lookup", "cross_version_comparison", "temporal_version_ordering"]:
        dist = per_task_correct.get(tt, Counter())
        n = sum(dist.values()) or 1
        skew = " ".join(f"{L}={dist.get(L,0)}({dist.get(L,0)/n*100:.1f}%)" for L in ["A", "B", "C", "D", "E"])
        print(f"  {tt:32} n={n}  {skew}")
        if tt in ("cross_version_comparison", "temporal_version_ordering") and dist.get("A", 0) / n > 0.6:
            warnings.append(
                f"{tt}: correct option is A for {dist.get('A',0)/n*100:.1f}% of questions (>60%); "
                "option position is not balanced -- always-A baseline is inflated. "
                "Run 91_shuffle_balance_options.py before re-evaluation."
            )

    print("\n=== Per-task gold support-size distribution ===")
    for tt in ["single_state_lookup", "cross_version_comparison", "temporal_version_ordering"]:
        dist = per_task_gold.get(tt, Counter())
        print(f"  {tt:32} " + " ".join(f"|G|={k}:{v}" for k, v in sorted(dist.items())))

    print(f"\nprototypes={len(prototypes)}  errors={len(errors)}  warnings={len(warnings)}")
    for e in errors:
        print(f"  ERROR: {e}")
    for w in warnings:
        print(f"  WARN : {w}")

    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the released merged dataset.")
    parser.add_argument("--merged-json", type=Path, default=DEFAULT_MERGED_JSON)
    args = parser.parse_args()
    sys.exit(validate(args.merged_json))


if __name__ == "__main__":
    main()
