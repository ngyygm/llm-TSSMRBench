"""Shared progress tracking for resumable state-version build scripts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utcnow_iso() -> str:
    """Return a stable UTC timestamp for progress snapshots."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ProgressTracker:
    """Persist per-item outcomes so long-running build jobs can resume safely."""

    def __init__(
        self,
        path: Path,
        *,
        script_name: str,
        total_items: Optional[int],
        run_context: Optional[dict[str, Any]] = None,
        resume: bool = False,
    ) -> None:
        self.path = path
        self.script_name = script_name
        self.total_items = total_items
        self.run_context = run_context or {}
        self.started_at = utcnow_iso()
        self.items: dict[str, dict[str, Any]] = {}

        if resume and self.path.exists():
            self._load_existing()
        else:
            self.save()

    def _load_existing(self) -> None:
        raw_text = self.path.read_text(encoding="utf-8-sig")
        if not raw_text.strip():
            self.save()
            return
        payload = json.loads(raw_text)
        self.started_at = str(payload.get("started_at") or utcnow_iso())
        self.total_items = payload.get("total_items", self.total_items)
        self.run_context = payload.get("run_context") or self.run_context
        self.items = payload.get("items") or {}

    def has_outcome(self, item_id: str, outcomes: set[str]) -> bool:
        record = self.items.get(item_id)
        return bool(record and record.get("outcome") in outcomes)

    def record(
        self,
        item_id: str,
        *,
        outcome: str,
        message: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        increment_attempt: bool = True,
    ) -> None:
        previous = self.items.get(item_id, {})
        attempts = int(previous.get("attempts", 0))
        if increment_attempt:
            attempts += 1

        self.items[item_id] = {
            "item_id": item_id,
            "outcome": outcome,
            "attempts": attempts,
            "message": message,
            "metadata": metadata or {},
            "updated_at": utcnow_iso(),
        }
        self.save()

    def outcome_counts(self) -> dict[str, int]:
        counter = Counter(str(record.get("outcome")) for record in self.items.values())
        return dict(sorted(counter.items()))

    def compact_counts(self) -> str:
        counts = self.outcome_counts()
        if not counts:
            return "no-progress"
        return ", ".join(f"{key}={value}" for key, value in counts.items())

    def to_dict(self) -> dict[str, Any]:
        return {
            "script_name": self.script_name,
            "started_at": self.started_at,
            "updated_at": utcnow_iso(),
            "total_items": self.total_items,
            "processed_items": len(self.items),
            "outcome_counts": self.outcome_counts(),
            "run_context": self.run_context,
            "items": self.items,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        self.path.write_text(payload, encoding="utf-8")
