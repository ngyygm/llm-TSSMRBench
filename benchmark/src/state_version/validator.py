"""Validation helpers for the independent state-version benchmark."""

from __future__ import annotations

import json
import math
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml
from pydantic import ValidationError

from .schemas import DatasetBuildConfig, RawGithubArtifactRecord, SourceBundleRecord, StateChainSample, StateQuestion

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
MC_OPTION_IDS = ["A", "B", "C", "D", "E"]
BOOLEAN_OPTION_IDS = ["A", "B", "C"]
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
NODE_REFERENCE_RE = re.compile(r"\b(?:node|nodes)\b|\bn\d{3,}\b", re.IGNORECASE)
MC_BAD_OPTION_RE = re.compile(r"\b(?:all of the above|none of the above|both a and b|both b and c|both c and d)\b", re.IGNORECASE)
COMPOUND_QUERY_RE = re.compile(
    r"\b(?:and\s+(?:what|why|how|when|whether|did|does|is|are|was|were|who|which))\b",
    re.IGNORECASE,
)
GITHUB_VISIBLE_ID_PATTERNS = [
    re.compile(r"\bissue\s*#\d+\b", re.IGNORECASE),
    re.compile(r"\bpull request\s*#\d+\b", re.IGNORECASE),
    re.compile(r"\bpr#\d+\b", re.IGNORECASE),
    re.compile(r"\bissue#\d+\b", re.IGNORECASE),
    re.compile(r"\bcomment#\d+\b", re.IGNORECASE),
    re.compile(r"\bevent#\d+\b", re.IGNORECASE),
    re.compile(r"\bcommit\s+[0-9a-f]{7,40}\b", re.IGNORECASE),
]
NARRATIVE_BELIEF_FOCUS_RE = re.compile(
    r"\b(?:belief|beliefs|thought|thoughts|confusion|doubt|interpretation|understanding|opinion|feelings)\b",
    re.IGNORECASE,
)


class ValidationIssue(dict):
    """One validation issue."""

    def __init__(self, severity: str, scope: str, item_id: str, message: str) -> None:
        super().__init__(
            severity=severity,
            scope=scope,
            item_id=item_id,
            message=message,
        )


class ValidationReport:
    """Mutable validation report with hard errors and softer warnings."""

    def __init__(self, scope: str, item_id: str) -> None:
        self.scope = scope
        self.item_id = item_id
        self.errors: list[ValidationIssue] = []
        self.warnings: list[ValidationIssue] = []
        self.metrics: dict[str, Any] = {}

    @property
    def passed(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(ValidationIssue("error", self.scope, self.item_id, message))

    def add_warning(self, message: str) -> None:
        self.warnings.append(ValidationIssue("warning", self.scope, self.item_id, message))

    def extend(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "item_id": self.item_id,
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": self.metrics,
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load one JSONL-like file.

    Supports both classic one-object-per-line JSONL and a human-auditable
    multiline JSON-object sequence separated by whitespace.
    """

    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []

    records: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            records = []
            break
    if records:
        return records

    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            payload, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON sequence at {path}:{exc.lineno}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
        records.append(payload)
        index = next_index
    return records


def load_build_config(config_path: Path) -> DatasetBuildConfig:
    """Load the YAML build configuration."""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return DatasetBuildConfig(**payload)


def count_text_units(text: str, language: str) -> int:
    """Count approximate text units for minimum-length filtering."""

    if language == "en":
        return len(WORD_RE.findall(text))
    return len([char for char in text if not char.isspace()])


def _normalize_text_signature(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = re.sub(r"[`\"'“”‘’.,;:!?()\[\]{}]+", "", normalized)
    return normalized


def _ensure_unique(items: Iterable[str]) -> bool:
    materialized = list(items)
    return len(materialized) == len(set(materialized))


def validate_state_chain_payload(payload: dict[str, Any], config: DatasetBuildConfig) -> ValidationReport:
    """Validate one state-chain sample against the new spec."""

    sample_id = str(payload.get("sample_id", "unknown_sample"))
    report = ValidationReport(scope="state_chain", item_id=sample_id)

    try:
        sample = StateChainSample(**payload)
    except ValidationError as exc:
        report.add_error(str(exc))
        return report

    node_ids = [node.node_id for node in sample.chain_nodes]
    surface_orders = [node.surface_order for node in sample.chain_nodes]
    if not _ensure_unique(node_ids):
        report.add_error("node_id values must be unique within a state chain")
    if sorted(surface_orders) != list(range(1, len(surface_orders) + 1)):
        report.add_error("surface_order must be contiguous and start from 1")

    min_nodes = config.hard_limits.min_nodes_by_domain[sample.domain]
    if len(sample.chain_nodes) < min_nodes:
        report.add_error(f"{sample.domain} requires at least {min_nodes} nodes")

    max_nodes = config.hard_limits.max_nodes_by_domain[sample.domain]
    if len(sample.chain_nodes) > max_nodes:
        report.add_error(f"{sample.domain} allows at most {max_nodes} nodes")

    min_units = config.hard_limits.min_text_units_by_language[sample.language]
    short_nodes: list[str] = []
    for node in sample.chain_nodes:
        unit_count = count_text_units(node.text, sample.language)
        if unit_count < min_units:
            short_nodes.append(f"{node.node_id}={unit_count}")
    if short_nodes:
        report.add_error(
            f"node text below hard minimum {min_units} {sample.language} units: {', '.join(short_nodes)}"
        )

    leaked_identifier_nodes: list[str] = []
    for node in sample.chain_nodes:
        if URL_RE.search(node.text):
            leaked_identifier_nodes.append(f"{node.node_id}=url")
            continue
        if sample.domain == "github_evolution":
            for pattern in GITHUB_VISIBLE_ID_PATTERNS:
                matched = pattern.search(node.text)
                if matched:
                    leaked_identifier_nodes.append(f"{node.node_id}={matched.group(0)}")
                    break
    if leaked_identifier_nodes:
        report.add_error(
            "visible text leaks raw source identifiers that should stay only in source_pointer: "
            + ", ".join(leaked_identifier_nodes)
        )

    distractor_count = sum(1 for node in sample.chain_nodes if node.salience_label == "distractor")
    core_count = sum(1 for node in sample.chain_nodes if node.salience_label == "core")
    progress_counts = Counter(node.progress_label for node in sample.chain_nodes)
    core_artifact_refs = {
        f"{node.source_pointer.artifact_type}::{node.source_pointer.artifact_ref}"
        for node in sample.chain_nodes
        if node.salience_label == "core"
    }
    distractor_artifact_refs = [
        f"{node.source_pointer.artifact_type}::{node.source_pointer.artifact_ref}"
        for node in sample.chain_nodes
        if node.salience_label == "distractor"
    ]
    distractor_relations = [
        node.relation_label
        for node in sample.chain_nodes
        if node.salience_label == "distractor"
    ]
    text_signature_counts = Counter(_normalize_text_signature(node.text) for node in sample.chain_nodes if node.text.strip())
    duplicate_signatures = [sig for sig, count in text_signature_counts.items() if sig and count >= 2]
    distractor_source_counts = Counter(distractor_artifact_refs)
    if core_count == 0:
        report.add_warning("chain has no core nodes")

    distractor_ratio = distractor_count / len(sample.chain_nodes)
    min_required_distractors = max(2, math.ceil(0.2 * len(sample.chain_nodes)))
    if distractor_count < min_required_distractors:
        report.add_error(
            f"chain must contain at least {min_required_distractors} distractor nodes for its length; do not return an overly clean chain"
        )

    if sample.chain_profile.competition_strength in {"medium", "high"} and distractor_count < 2:
        report.add_warning("competition_strength is marked medium/high but distractor count is still low")

    distractor_only_refs = set(distractor_artifact_refs) - core_artifact_refs
    if distractor_count >= 2 and distractor_artifact_refs and not distractor_only_refs:
        message = "distractor nodes reuse only the same source artifacts as core nodes; add adjacent but distinct related evidence or reduce distractors"
        if sample.domain == "github_evolution":
            report.add_error(message)
        else:
            report.add_warning(message)
    if (
        distractor_count >= 2
        and distractor_artifact_refs
        and not distractor_only_refs
        and distractor_relations
        and all(relation == "explains" for relation in distractor_relations)
    ):
        report.add_error(
            "distractor nodes are only explanation-style restatements of main-arc artifacts; use adjacent but distinct related subthreads instead"
        )

    if duplicate_signatures:
        report.add_error(
            "chain contains duplicate or near-duplicate node texts; merge repeated nodes instead of copying the same state wording"
        )

    repeated_distractor_sources = [key for key, count in distractor_source_counts.items() if key and count >= 2]
    if repeated_distractor_sources:
        message = (
            "distractor nodes repeatedly reuse the same source artifact; use distinct nearby branches or merge the duplicates: "
            + ", ".join(sorted(repeated_distractor_sources))
        )
        if sample.domain == "github_evolution":
            report.add_error(message)
        else:
            report.add_warning(message)

    if not any(node.progress_label == "resolved" for node in sample.chain_nodes):
        report.add_warning("chain has no resolved node; answerable final-state questions may be limited")
    if len(progress_counts) < 2:
        report.add_warning(
            "chain uses fewer than two progress_label values; it may not express enough temporal-semantic state change"
        )
    elif len(progress_counts) < 3:
        report.add_warning(
            "chain uses only two progress_label values; check whether richer planned/active/resolved/invalidated distinctions are available in the source"
        )

    if sample.domain == "narrative_evolution" and NARRATIVE_BELIEF_FOCUS_RE.search(sample.focus_event):
        report.add_warning(
            "narrative focus_event looks belief-centric or purely interpretive; prefer an externally anchored event arc and keep belief shifts as supporting nodes instead of the whole focal thread"
        )

    report.metrics = {
        "node_count": len(sample.chain_nodes),
        "core_count": core_count,
        "distractor_count": distractor_count,
        "distractor_ratio": round(distractor_ratio, 4),
        "competition_strength": sample.chain_profile.competition_strength,
        "lexical_overlap_band": sample.chain_profile.lexical_overlap_band,
        "progress_label_counts": dict(sorted(progress_counts.items())),
    }
    return report


def validate_state_chain_payload_with_bundle(
    payload: dict[str, Any],
    bundle_payload: dict[str, Any],
    config: DatasetBuildConfig,
) -> ValidationReport:
    """Validate one state chain plus its grounding back to the assigned source bundle."""

    report = validate_state_chain_payload(payload, config)
    if not report.passed:
        return report

    try:
        sample = StateChainSample(**payload)
    except ValidationError:
        return report

    try:
        bundle = SourceBundleRecord(**bundle_payload)
    except ValidationError as exc:
        report.add_error(f"backing source bundle is invalid: {exc}")
        return report

    if sample.state_chain_id != bundle.state_chain_id:
        report.add_error("state_chain_id does not match the assigned source bundle")
    if sample.sample_id != bundle.sample_id:
        report.add_error("sample_id does not match the assigned source bundle")
    if sample.domain != bundle.domain:
        report.add_error("domain does not match the assigned source bundle")
    if sample.language != bundle.language:
        report.add_error("language does not match the assigned source bundle")
    if sample.focus_event != bundle.focus_event:
        report.add_error("focus_event does not match the assigned source bundle")

    allowed_sources = {
        (item.artifact_type, item.artifact_ref)
        for item in bundle.source_bundle_items
    }
    invalid_source_pointers: list[str] = []
    for node in sample.chain_nodes:
        key = (node.source_pointer.artifact_type, node.source_pointer.artifact_ref)
        if key not in allowed_sources:
            invalid_source_pointers.append(
                f"{node.node_id}={node.source_pointer.artifact_type}::{node.source_pointer.artifact_ref}"
            )
    if invalid_source_pointers:
        report.add_error(
            "source_pointer references artifacts that are not present in the assigned source_bundle: "
            + ", ".join(invalid_source_pointers)
        )

    return report


def validate_source_bundle_payload(payload: dict[str, Any], config: DatasetBuildConfig) -> ValidationReport:
    """Validate one source-bundle record before it is sent to the model."""

    sample_id = str(payload.get("sample_id", "unknown_source_bundle"))
    report = ValidationReport(scope="source_bundle", item_id=sample_id)

    try:
        bundle = SourceBundleRecord(**payload)
    except ValidationError as exc:
        report.add_error(str(exc))
        return report

    if bundle.language != config.language:
        report.add_error(f"source bundle language must be {config.language}, got {bundle.language}")

    if bundle.bundle_summary is not None and not bundle.bundle_summary.strip():
        report.add_error("bundle_summary cannot be blank when provided")

    report.metrics = {
        "domain": bundle.domain,
        "item_count": len(bundle.source_bundle_items),
        "has_bundle_summary": bundle.bundle_summary is not None,
    }
    return report


def validate_raw_github_artifact_payload(payload: dict[str, Any]) -> ValidationReport:
    """Validate one normalized raw GitHub artifact record."""

    item_id = f"{payload.get('artifact_type', 'artifact')}::{payload.get('artifact_ref', 'unknown')}"
    report = ValidationReport(scope="raw_github_artifact", item_id=item_id)

    try:
        artifact = RawGithubArtifactRecord(**payload)
    except ValidationError as exc:
        report.add_error(str(exc))
        return report

    if artifact.split is not None and not artifact.split.strip():
        report.add_error("split cannot be blank when provided")

    report.metrics = {
        "repo": artifact.repo,
        "focus_event": artifact.focus_event,
        "has_summary": bool((artifact.summary or "").strip()),
        "has_raw_text": bool((artifact.raw_text or "").strip()),
    }
    return report


def validate_question_payload(
    payload: dict[str, Any],
    chain_payload: dict[str, Any],
    config: DatasetBuildConfig,
) -> ValidationReport:
    """Validate one question against the new spec and its backing chain."""

    question_id = str(payload.get("question_id", "unknown_question"))
    report = ValidationReport(scope="question", item_id=question_id)

    try:
        question = StateQuestion(**payload)
    except ValidationError as exc:
        report.add_error(str(exc))
        return report

    try:
        chain = StateChainSample(**chain_payload)
    except ValidationError as exc:
        report.add_error(f"backing chain is invalid: {exc}")
        return report

    if question.state_chain_id != chain.state_chain_id:
        report.add_error("question.state_chain_id does not match chain.state_chain_id")

    node_ids = {node.node_id for node in chain.chain_nodes}
    node_salience = {node.node_id: node.salience_label for node in chain.chain_nodes}
    missing_gold = sorted(set(question.gold_node_ids) - node_ids)
    if missing_gold:
        report.add_error(f"gold_node_ids reference missing node ids: {missing_gold}")

    missing_adv = sorted(set(question.adversarial_node_ids) - node_ids)
    if missing_adv:
        report.add_error(f"adversarial_node_ids reference missing node ids: {missing_adv}")

    overlap = sorted(set(question.gold_node_ids) & set(question.adversarial_node_ids))
    if overlap:
        report.add_error(f"gold_node_ids and adversarial_node_ids overlap: {overlap}")

    non_core_gold = sorted(
        node_id
        for node_id in question.gold_node_ids
        if node_salience.get(node_id) != "core"
    )
    if non_core_gold:
        report.add_error(
            f"answerable questions must use only core gold_node_ids, found non-core gold nodes: {non_core_gold}"
        )

    expected_dynamic_top_k = math.ceil(1.5 * len(question.gold_node_ids))
    if question.dynamic_top_k != expected_dynamic_top_k:
        report.add_error(
            f"dynamic_top_k must equal ceil(1.5 * len(gold_node_ids))={expected_dynamic_top_k}"
        )
    if question.dynamic_top_k is not None and question.dynamic_top_k >= len(chain.chain_nodes):
        report.add_error(
            "dynamic_top_k must stay strictly smaller than total chain node count for answerable questions"
        )

    if NODE_REFERENCE_RE.search(question.query_text):
        report.add_error("query_text must not mention node numbers, node ranges, or raw node ids")
    if COMPOUND_QUERY_RE.search(question.query_text):
        report.add_error("query_text must be a single question, not a compound question with two asks")

    if question.answer_format == "multiple_choice":
        option_ids = [option.option_id for option in question.options or []]
        if option_ids != MC_OPTION_IDS:
            report.add_error("multiple_choice questions must use option ids A/B/C/D/E in order")
        else:
            insufficiency_text = (question.options or [])[4].text
            if insufficiency_text != config.canonical_abstentions.multiple_choice:
                report.add_error("multiple_choice option E must use the canonical insufficiency text")
        if question.correct_option_id == "E":
            report.add_error("answerable multiple_choice questions cannot use E as the correct option")
        bad_options = [
            option.option_id
            for option in question.options or []
            if MC_BAD_OPTION_RE.search(option.text)
        ]
        if bad_options:
            report.add_error(
                f"multiple_choice questions must not use 'all of the above' or similar meta-options: {bad_options}"
            )

    if question.answer_format == "boolean":
        option_ids = [option.option_id for option in question.options or []]
        if option_ids != BOOLEAN_OPTION_IDS:
            report.add_error("boolean questions must use option ids A/B/C in order")
        else:
            insufficiency_text = (question.options or [])[2].text
            if insufficiency_text != config.canonical_abstentions.boolean:
                report.add_error("boolean option C must use the canonical insufficiency text")
        if question.correct_option_id == "C":
            report.add_error("answerable boolean questions cannot use C as the correct option")

    if question.answer_format == "abstractive":
        if question.expected_answer == config.canonical_abstentions.abstractive:
            report.add_error("answerable abstractive questions cannot use the canonical abstention answer")

    report.metrics = {
        "difficulty_level": question.difficulty_level,
        "question_family": question.question_family,
        "answerability": question.answerability,
        "answer_format": question.answer_format,
        "gold_node_count": len(question.gold_node_ids),
        "adversarial_node_count": len(question.adversarial_node_ids),
        "dynamic_top_k": question.dynamic_top_k,
    }
    return report


def summarize_question_distribution(question_payloads: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Compute simple distribution counts for reporting."""

    buckets = {
        "difficulty_level": Counter(),
        "question_family": Counter(),
        "answerability": Counter(),
        "answer_format": Counter(),
    }
    for payload in question_payloads:
        for key, counter in buckets.items():
            value = payload.get(key)
            if value is not None:
                counter[value] += 1
    return {key: dict(sorted(counter.items())) for key, counter in buckets.items()}


def group_questions_by_chain(question_payloads: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group question payloads by state_chain_id."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in question_payloads:
        groups[str(payload.get("state_chain_id"))].append(payload)
    return groups
