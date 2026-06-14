#!/usr/bin/env python3
"""Compute deterministic, retrieval-free answer-position baselines for TSSMRBench.

Motivation
----------
The released multiple-choice questions are heavily position-skewed: the correct
option is option ``A`` for ~97.7% of cross-version questions and ~93% of
temporal-ordering questions (see the research-integrity audit). A model that
*always* picks ``A`` -- with zero retrieval and zero reasoning -- would therefore
score above every reported memory system on those two task families.

This script makes that confound explicit and reproducible. It computes three
deterministic baselines that depend ONLY on the gold option labels (no answer
model, no retrieval, no API):

* ``always_A``    -- always pick option A.
* ``majority``    -- always pick the single most-frequent gold letter within each
                     task family (the strongest position-only baseline).
* ``uniform``     -- pick uniformly at random over the four real options A-D
                     (expected accuracy = 0.25; reported for reference).

These baselines are intended to be reported alongside system ACC so that readers
can see how much of each system's score is recoverable from option position
alone. They are also the recommended acceptance bar: a memory system should beat
``majority`` by a clear margin before its ACC is attributed to retrieval quality.
"""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "benchmark"
    / "data"
    / "prototype_eval_results"
    / "official_300repo_release_unified_v1_paper_artifacts"
)
DEFAULT_PAPER_TABLE_DIR = ROOT / "paper" / "tables" / "generated"

REAL_OPTIONS = ["A", "B", "C", "D"]
TASK_ORDER = [
    "single_state_lookup",
    "cross_version_comparison",
    "temporal_version_ordering",
]


def load_questions(merged_path: Path) -> list[dict[str, Any]]:
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for payload in merged.get("prototypes") or []:
        repo = str(payload.get("repo") or "")
        for q in payload.get("questions") or []:
            correct = str(q.get("correct_option_id") or "").strip().upper()
            rows.append(
                {
                    "repo": repo,
                    "task_type": q.get("task_type"),
                    "question_id": q.get("question_id"),
                    "correct_option_id": correct,
                    "num_options": len([o for o in (q.get("options") or []) if str(o.get("option_id") or "").upper() in REAL_OPTIONS]),
                }
            )
    return rows


def accuracy(rows: list[dict[str, Any]], pick: str) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if row["correct_option_id"] == pick)
    return hits / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute retrieval-free answer-position baselines.")
    parser.add_argument("--merged-json", type=Path, default=DEFAULT_MERGED_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-table-dir", type=Path, default=DEFAULT_PAPER_TABLE_DIR)
    args = parser.parse_args()

    rows = load_questions(args.merged_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_table_dir.mkdir(parents=True, exist_ok=True)

    # Per-task distribution + baselines
    csv_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"tasks": {}, "overall": {}}
    for task in TASK_ORDER:
        task_rows = [row for row in rows if row["task_type"] == task]
        dist = Counter(row["correct_option_id"] for row in task_rows)
        n = len(task_rows) or 1
        majority_letter, _ = dist.most_common(1)[0] if dist else ("A", 0)
        record = {
            "task_type": task,
            "n": len(task_rows),
            "dist_A": dist.get("A", 0),
            "dist_B": dist.get("B", 0),
            "dist_C": dist.get("C", 0),
            "dist_D": dist.get("D", 0),
            "dist_E": dist.get("E", 0),
            "always_A_acc": accuracy(task_rows, "A"),
            "majority_letter": majority_letter,
            "majority_acc": accuracy(task_rows, majority_letter),
            "uniform_acc": 0.25,
        }
        summary["tasks"][task] = record
        csv_rows.append(record)

    overall_dist = Counter(row["correct_option_id"] for row in rows)
    overall_majority, _ = overall_dist.most_common(1)[0]
    summary["overall"] = {
        "n": len(rows),
        "dist": dict(overall_dist),
        "always_A_acc": accuracy(rows, "A"),
        "majority_letter": overall_majority,
        "majority_acc": accuracy(rows, overall_majority),
        "uniform_acc": 0.25,
    }

    (args.output_dir / "option_position_baselines.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "option_position_baselines.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # Paper table
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append(
        "\\caption{Retrieval-free answer-position baselines (depend only on gold option labels, "
        "no answer model). \\texttt{always-A} always picks option~A; \\texttt{majority} always picks "
        "the most frequent gold letter within each task; \\texttt{uniform} is random over A--D (0.25). "
        "Because option position is never shuffled at construction time, these baselines expose how much "
        "of each system's ACC is recoverable from position alone. A memory system should clear "
        "\\texttt{majority} by a wide margin before its ACC is read as retrieval quality.}"
    )
    lines.append("\\label{tab:option-position-baselines}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{lccccrrr}")
    lines.append("\\toprule")
    lines.append("Task & Gold A\\% & Gold B\\% & Gold C\\% & Gold D\\% & always-A & majority & uniform \\\\")
    lines.append("\\midrule")
    label = {
        "single_state_lookup": "Single",
        "cross_version_comparison": "Cross",
        "temporal_version_ordering": "Temporal",
    }
    for task in TASK_ORDER:
        rec = summary["tasks"][task]
        n = rec["n"] or 1
        lines.append(
            f"{label[task]} & {rec['dist_A']/n*100:.1f} & {rec['dist_B']/n*100:.1f} & "
            f"{rec['dist_C']/n*100:.1f} & {rec['dist_D']/n*100:.1f} & {rec['always_A_acc']:.4f} & "
            f"{rec['majority_acc']:.4f} & {rec['uniform_acc']:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    table_tex = "\n".join(lines) + "\n"
    (args.output_dir / "option_position_baselines.tex").write_text(table_tex, encoding="utf-8")
    (args.paper_table_dir / "option_position_baselines.tex").write_text(table_tex, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
