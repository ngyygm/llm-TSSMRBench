"""Utilities for building narrative source bundles from ordered summary sources."""

from __future__ import annotations

from typing import Any

from .schemas import SourceBundleItem, SourceBundleRecord


def build_narrative_summary_source_bundle(
    *,
    sample_id: str,
    state_chain_id: str,
    language: str,
    focus_event: str,
    source_title: str,
    title: str,
    author: str,
    source_url: str,
    source_provider: str,
    bundle_summary: str,
    notes: list[str],
    item_specs: list[dict[str, Any]],
    source_kind: str = "plot_summary",
) -> SourceBundleRecord:
    """Build one narrative source bundle from ordered summary blocks."""

    source_bundle_items = [
        SourceBundleItem(
            artifact_type=str(item["artifact_type"]),
            artifact_ref=str(item["artifact_ref"]),
            title=str(item.get("title", "")).strip() or None,
            time_hint=str(item.get("time_hint", "")).strip() or None,
            summary=str(item["summary"]).strip(),
        )
        for item in item_specs
    ]

    return SourceBundleRecord(
        sample_id=sample_id,
        state_chain_id=state_chain_id,
        domain="narrative_evolution",
        language=language,
        focus_event=focus_event,
        source_title=source_title,
        bundle_summary=bundle_summary,
        source_bundle_items=source_bundle_items,
        notes=notes,
        source_metadata={
            "work": title,
            "author": author,
            "source_url": source_url,
            "source_provider": source_provider,
            "source_kind": source_kind,
        },
    )
