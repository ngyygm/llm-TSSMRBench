#!/usr/bin/env python3
"""Produce a position-balanced, option-shuffled copy of the TSSMRBench dataset.

Why this exists
---------------
The released questions are position-skewed: the correct option is ``A`` for
~97.7% of cross-version and ~93% of temporal-ordering questions. A model that
always picks ``A`` then beats every reported memory system on those tasks (see
``90_option_position_baselines.py``). The correct content of each option is
fine; only the *letter* assigned to the correct content is skewed.

This script re-emits the dataset with the option order of every question
permuted so that, within each task family, the correct option lands on A/B/C/D
in a balanced (~25% each) and randomized fashion. The option *content* and the
gold support are unchanged; only labels move. A permutation record is written
for auditability and so old per-question results can be re-keyed if needed.

Re-evaluating the systems on this balanced dataset removes the positional
confound from ACC. Run:

    python benchmark/scripts/82_run_merged_github_release_unified_global_pool_evaluation.py \\
      --merged-json <balanced path>/official_300_merged_balanced.json ...

for each system with your API key.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
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
DEFAULT_OUTPUT_PATH = (
    ROOT
    / "benchmark"
    / "data"
    / "prototypes"
    / "github_release_note_v2"
    / "formal_300repo_unified_v1"
    / "official_300_merged_balanced.json"
)
REAL_OPTIONS = ["A", "B", "C", "D"]
TASK_ORDER = [
    "single_state_lookup",
    "cross_version_comparison",
    "temporal_version_ordering",
]
SEED = 20240611


def _stable_question_key(payload: dict[str, Any], qid: str) -> int:
    digest = hashlib.sha1(qid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _balanced_letter_plan(task_questions: list[tuple[str, dict[str, Any]]], seed: int) -> dict[str, str]:
    """Return qid -> target correct letter, balanced ~25% per letter, randomized."""
    n = len(task_questions)
    base_seed = seed
    letters: list[str] = []
    for letter in REAL_OPTIONS:
        letters.extend([letter] * (n // len(REAL_OPTIONS)))
    # distribute the remainder round-robin so counts differ by at most 1
    for i in range(n - len(letters)):
        letters.append(REAL_OPTIONS[i % len(REAL_OPTIONS)])
    rng = random.Random(base_seed ^ (n * 2654435761 & 0xFFFFFFFF))
    rng.shuffle(letters)
    return {qid: letter for (qid, _q), letter in zip(task_questions, letters)}


def shuffle_prototype(prototype: dict[str, Any], target_letters: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    new_proto = copy.deepcopy(prototype)
    audit: list[dict[str, Any]] = []
    prototype_id = str(prototype.get("prototype_id") or "")
    for q in new_proto.get("questions") or []:
        qid = q["question_id"]
        uid = f"{prototype_id}__{qid}"
        options = q.get("options") or []
        real = [o for o in options if str(o.get("option_id") or "").upper() in REAL_OPTIONS]
        if len(real) != len(REAL_OPTIONS):
            audit.append({"question_id": uid, "skipped": f"expected 4 real options, got {len(real)}"})
            continue
        correct_old = str(q.get("correct_option_id") or "").upper()
        try:
            correct_idx = [str(o.get("option_id") or "").upper() for o in real].index(correct_old)
        except ValueError:
            audit.append({"question_id": uid, "skipped": f"correct_option_id {correct_old} not in options"})
            continue
        correct_text = real[correct_idx]["text"]
        other_texts = [real[i]["text"] for i in range(len(real)) if i != correct_idx]

        target = target_letters.get(uid, "A")
        target_idx = REAL_OPTIONS.index(target)

        rng = random.Random(_stable_question_key(prototype, qid))
        shuffled_others = other_texts[:]
        rng.shuffle(shuffled_others)

        new_order: list[str] = [""] * len(REAL_OPTIONS)
        new_order[target_idx] = correct_text
        oi = 0
        for slot in range(len(REAL_OPTIONS)):
            if new_order[slot] == "":
                new_order[slot] = shuffled_others[oi]
                oi += 1

        new_options = [
            {"option_id": REAL_OPTIONS[i], "text": new_order[i]}
            for i in range(len(REAL_OPTIONS))
        ]
        q["options"] = new_options
        q["correct_option_id"] = target
        audit.append(
            {
                "question_id": uid,
                "task_type": q.get("task_type"),
                "old_correct": correct_old,
                "new_correct": target,
                "permutation": [REAL_OPTIONS[new_order.index(real[i]["text"])] for i in range(len(real))],
            }
        )
    return new_proto, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffle options and balance correct-option position per task.")
    parser.add_argument("--merged-json", type=Path, default=DEFAULT_MERGED_JSON)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    seed = args.seed
    merged = json.loads(args.merged_json.read_text(encoding="utf-8"))
    prototypes = list(merged.get("prototypes") or [])

    # Collect (uid, question) per task across all prototypes. question_id is only
    # unique within a prototype, so key by the eval-style unique id prototype_id__question_id.
    by_task: dict[str, list[tuple[str, dict[str, Any]]]] = {t: [] for t in TASK_ORDER}
    for proto in prototypes:
        prototype_id = str(proto.get("prototype_id") or "")
        for q in proto.get("questions") or []:
            uid = f"{prototype_id}__{q['question_id']}"
            tt = q.get("task_type")
            if tt in by_task:
                by_task[tt].append((uid, q))

    target_letters: dict[str, str] = {}
    for task, items in by_task.items():
        items.sort(key=lambda pair: pair[0])
        target_letters.update(_balanced_letter_plan(items, seed))

    new_prototypes: list[dict[str, Any]] = []
    audit_log: list[dict[str, Any]] = []
    for proto in prototypes:
        prototype_id = str(proto.get("prototype_id") or "")
        # restrict target map to this prototype's questions
        proto_uids = [f"{prototype_id}__{q['question_id']}" for q in proto.get("questions") or []]
        proto_targets = {uid: target_letters[uid] for uid in proto_uids if uid in target_letters}
        new_proto, audit = shuffle_prototype(proto, proto_targets)
        new_prototypes.append(new_proto)
        audit_log.extend(audit)

    out = copy.deepcopy(merged)
    out["prototypes"] = new_prototypes
    out["dataset_id"] = str(merged.get("dataset_id", "")) + "_balanced"
    out["balance_meta"] = {
        "source": str(args.merged_json),
        "seed": seed,
        "description": "Option order permuted per question so correct_option_id is balanced (~25% each of A-D) within every task family. Option content and gold support are unchanged.",
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_path.parent / "official_300_merged_balanced_permutation_audit.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in audit_log) + "\n", encoding="utf-8"
    )

    # Verify balance
    new_rows = []
    for proto in new_prototypes:
        for q in proto.get("questions") or []:
            new_rows.append((q.get("task_type"), str(q.get("correct_option_id") or "").upper()))
    for task in TASK_ORDER:
        dist = Counter(letter for tt, letter in new_rows if tt == task)
        n = sum(dist.values()) or 1
        print(f"{task}: " + " ".join(f"{L}={dist[L]}({dist[L]/n*100:.1f}%)" for L in REAL_OPTIONS))
    print(f"\nWrote balanced dataset -> {args.output_path}")
    print(f"Wrote permutation audit -> {args.output_path.parent / 'official_300_merged_balanced_permutation_audit.jsonl'}")


if __name__ == "__main__":
    main()
