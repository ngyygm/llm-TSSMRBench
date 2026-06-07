#!/usr/bin/env python3
"""Merge official unified GitHub release-window prototypes into one JSON file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTO_ROOT = (
    ROOT
    / "benchmark"
    / "data"
    / "prototypes"
    / "github_release_note_v2"
    / "formal_300repo_unified_v1"
)
DEFAULT_INDEX = DEFAULT_PROTO_ROOT / "prototype_index_official_300.jsonl"
DEFAULT_OUTPUT = DEFAULT_PROTO_ROOT / "official_300_merged.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge official 300 unified prototype.json files into one JSON file.")
    parser.add_argument("--prototype-root", type=Path, default=DEFAULT_PROTO_ROOT)
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    index_rows = read_jsonl(args.index_file)
    prototypes: list[dict[str, Any]] = []

    for row in index_rows:
        prototype_id = str(row["prototype_id"])
        prototype_path = args.prototype_root / prototype_id / "prototype.json"
        payload = json.loads(prototype_path.read_text(encoding="utf-8"))
        payload["_merge_meta"] = {
            "prototype_dir": str(prototype_path.parent),
            "candidate_labels": row.get("candidate_labels") or [],
            "stargazers_count": row.get("stargazers_count"),
        }
        prototypes.append(payload)

    merged = {
        "dataset_id": "official_300repo_release_unified_v1",
        "dataset_title": "Official 300-repository GitHub release-note unified benchmark",
        "source_type": "github_release_note_unified",
        "generated_at": utc_now(),
        "prototype_count": len(prototypes),
        "official_index_file": str(args.index_file),
        "prototypes": prototypes,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "merged_file": str(args.output_file),
                "prototype_count": len(prototypes),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
