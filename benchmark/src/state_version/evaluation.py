"""Evaluation pipeline for the independent state-version benchmark."""

from __future__ import annotations

import json
import hashlib
import logging
import random
import re
import statistics
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.evaluation.answer_generator import AnswerGenerator
from src.evaluation.judge import LLMJudge
from src.state_version.schemas import StateChainSample, StateQuestion
from src.state_version.validator import load_jsonl
from src.systems.base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
DEFAULT_ANALYSIS_TOP_KS = (6, 8, 10)
DEFAULT_ANALYSIS_TOP_KS_BY_TASK = {
    "single_state_lookup": (1, 2, 3),
    "cross_version_comparison": (2, 5, 8),
    "temporal_version_ordering": (5, 8, 10),
}


@dataclass
class StateVersionDataset:
    """Materialized state-version evaluation dataset."""

    chains: Dict[str, StateChainSample]
    questions_by_chain: Dict[str, list[StateQuestion]]
    chain_split: Dict[str, str]
    chain_domain: Dict[str, str]


def build_memory_view_dataset(
    dataset: StateVersionDataset,
    *,
    memory_view: str = "full",
) -> StateVersionDataset:
    """Return a dataset view used for memory ingestion."""

    if memory_view == "full":
        return dataset
    if memory_view != "core_only":
        raise ValueError(f"Unsupported memory_view: {memory_view}")

    transformed_chains: dict[str, StateChainSample] = {}
    for chain_id, chain in dataset.chains.items():
        core_nodes = [
            node
            for node in sorted(chain.chain_nodes, key=lambda item: item.surface_order)
            if node.salience_label == "core"
        ]
        transformed_chains[chain_id] = chain.model_copy(
            update={
                "chain_nodes": core_nodes,
                "chain_profile": chain.chain_profile.model_copy(update={"node_count": len(core_nodes)}),
            }
        )

    return StateVersionDataset(
        chains=transformed_chains,
        questions_by_chain=dataset.questions_by_chain,
        chain_split=dataset.chain_split,
        chain_domain=dataset.chain_domain,
    )


def _normalize_text(text: str) -> str:
    cleaned = text or ""
    cleaned = cleaned.replace("Episode:", " ")
    cleaned = cleaned.replace("KG:", " ")
    cleaned = re.sub(r"\|\s*(?:event_time|record_time|source|valid_at|invalid_at|summary)=.*", "", cleaned)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return cleaned.strip()


def _tokenize(text: str) -> set[str]:
    normalized = _normalize_text(text)
    tokens = set(WORD_RE.findall(normalized))
    if tokens:
        return tokens
    return {normalized} if normalized else set()


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(WORD_RE.findall(text))


def _overlap_score(left: str, right: str) -> float:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        shorter = min(len(left_norm), len(right_norm))
        longer = max(len(left_norm), len(right_norm))
        if shorter >= 24:
            return 0.95 * (shorter / longer)

    left_tokens = _tokenize(left_norm)
    right_tokens = _tokenize(right_norm)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    if union == 0:
        return 0.0
    return intersection / union


def _match_retrieved_nodes(
    chain: StateChainSample,
    retrieved_facts: Iterable[str],
    *,
    metadata: Optional[dict[str, Any]] = None,
    threshold: float = 0.35,
) -> list[str]:
    explicit_pairs = _explicit_retrieved_pairs_from_metadata(metadata)
    if explicit_pairs:
        valid_node_ids = {node.node_id for node in chain.chain_nodes}
        ordered = []
        seen: set[str] = set()
        for source_chain_id, node_id in explicit_pairs:
            if source_chain_id and source_chain_id != chain.state_chain_id:
                continue
            if node_id in valid_node_ids and node_id not in seen:
                ordered.append(node_id)
                seen.add(node_id)
        return ordered

    matched_node_ids: list[str] = []
    seen: set[str] = set()
    for fact in retrieved_facts:
        best_node_id: Optional[str] = None
        best_score = 0.0
        for node in chain.chain_nodes:
            score = _overlap_score(fact, node.text)
            if score > best_score:
                best_score = score
                best_node_id = node.node_id
        if best_node_id is not None and best_score >= threshold and best_node_id not in seen:
            seen.add(best_node_id)
            matched_node_ids.append(best_node_id)
    return matched_node_ids


def _explicit_retrieved_pairs_from_metadata(
    metadata: Optional[dict[str, Any]],
) -> list[tuple[str, str]]:
    payload = metadata or {}
    explicit_node_ids = list(payload.get("retrieved_source_node_ids") or [])
    explicit_chain_ids = list(payload.get("retrieved_source_chain_ids") or [])
    if explicit_node_ids:
        if explicit_chain_ids and len(explicit_chain_ids) == len(explicit_node_ids):
            return [
                (str(chain_id or ""), str(node_id))
                for chain_id, node_id in zip(explicit_chain_ids, explicit_node_ids)
                if node_id is not None
            ]
        return [("", str(node_id)) for node_id in explicit_node_ids if node_id is not None]

    grouped_results = payload.get("grouped_results")
    if isinstance(grouped_results, list):
        pairs: list[tuple[str, str]] = []
        for item in grouped_results:
            if not isinstance(item, dict):
                continue
            node_id = item.get("source_node_id")
            if node_id is None:
                continue
            pairs.append((str(item.get("source_chain_id") or ""), str(node_id)))
        if pairs:
            return pairs

    return []


def _mean(values: list[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _compute_formal_distractor_to_gold_ratio(
    matched_node_ids: list[str],
    gold_node_ids: list[str],
) -> Optional[float]:
    gold_set = set(gold_node_ids)
    if not gold_node_ids:
        return None

    matched_gold_count = sum(1 for node_id in matched_node_ids if node_id in gold_set)
    matched_non_gold_count = sum(1 for node_id in matched_node_ids if node_id not in gold_set)
    total_gold = len(gold_node_ids)

    if matched_gold_count == total_gold:
        last_gold_index = max(
            idx for idx, node_id in enumerate(matched_node_ids) if node_id in gold_set
        )
        distractors_before_last_gold = sum(
            1 for node_id in matched_node_ids[:last_gold_index] if node_id not in gold_set
        )
        return distractors_before_last_gold / total_gold

    if matched_gold_count > 0:
        return matched_non_gold_count / matched_gold_count

    return float(matched_non_gold_count)


def _synthetic_time(order: int) -> str:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(days=order - 1)).isoformat().replace("+00:00", "Z")


def _build_full_context_query_result(chain: StateChainSample, question: StateQuestion) -> QueryResult:
    node_by_id = {node.node_id: node for node in chain.chain_nodes}
    context_node_ids = list(question.oracle_context_node_ids or question.gold_node_ids)
    selected_nodes = [
        node_by_id[node_id]
        for node_id in context_node_ids
        if node_id in node_by_id
    ]
    baseline_name = "oracle_gold_context"

    facts = [node.text for node in selected_nodes]
    context = "\n".join(facts)
    return QueryResult(
        answer=context,
        retrieved_context=context,
        retrieved_facts=facts,
        confidence=1.0 if facts else 0.0,
        latency_ms=0.0,
        metadata={
            "baseline": baseline_name,
            "num_results": len(facts),
            "retrieved_source_node_ids": [node.node_id for node in selected_nodes],
            "retrieved_source_chain_ids": [chain.state_chain_id for _ in selected_nodes],
        },
    )


def _build_random_context_query_result(
    random_pool_nodes: list[tuple[str, str, str]],
    chain: StateChainSample,
    question: StateQuestion,
) -> QueryResult:
    start = time.time()
    seed = int(hashlib.md5(question.question_id.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    candidates = [
        (chain_id, node_id, text)
        for chain_id, node_id, text in random_pool_nodes
        if chain_id != chain.state_chain_id
    ]
    rng.shuffle(candidates)
    selected = candidates[: max(int(question.dynamic_top_k or 0), 0)]
    facts = [text for _, _, text in selected]
    context = "\n".join(facts)
    return QueryResult(
        answer=context,
        retrieved_context=context,
        retrieved_facts=facts,
        confidence=1.0 if facts else 0.0,
        latency_ms=(time.time() - start) * 1000,
        metadata={
            "baseline": "random_context",
            "num_results": len(facts),
            "pool_size": len(candidates),
            "sampling_seed": seed,
            "retrieved_source_chain_ids": [chain_id for chain_id, _, _ in selected],
            "retrieved_source_node_ids": [node_id for _, node_id, _ in selected],
        },
    )


def _slice_query_result(query_result: QueryResult, top_k: int) -> QueryResult:
    facts = list(query_result.retrieved_facts or [])[:top_k]
    metadata = dict(query_result.metadata or {})

    if "retrieved_source_node_ids" in metadata:
        metadata["retrieved_source_node_ids"] = list(metadata.get("retrieved_source_node_ids") or [])[:top_k]
    if "retrieved_source_chain_ids" in metadata:
        metadata["retrieved_source_chain_ids"] = list(metadata.get("retrieved_source_chain_ids") or [])[:top_k]
    if "grouped_results" in metadata and isinstance(metadata.get("grouped_results"), list):
        metadata["grouped_results"] = list(metadata["grouped_results"])[:top_k]
    if "raw_results" in metadata and isinstance(metadata.get("raw_results"), list):
        metadata["raw_results"] = list(metadata["raw_results"])
    if not list(metadata.get("retrieved_source_node_ids") or []):
        explicit_pairs = _explicit_retrieved_pairs_from_metadata(metadata)
        if explicit_pairs:
            metadata["retrieved_source_chain_ids"] = [chain_id for chain_id, _ in explicit_pairs]
            metadata["retrieved_source_node_ids"] = [node_id for _, node_id in explicit_pairs]
    metadata["sliced_top_k"] = top_k

    return QueryResult(
        answer="\n".join(facts),
        retrieved_context="\n".join(facts),
        retrieved_facts=facts,
        confidence=query_result.confidence,
        latency_ms=query_result.latency_ms,
        metadata=metadata,
    )


def _compute_gold_rank_positions(
    matched_node_ids: list[str],
    gold_node_ids: list[str],
) -> dict[str, Any]:
    gold_set = set(gold_node_ids)
    positions = [idx + 1 for idx, node_id in enumerate(matched_node_ids) if node_id in gold_set]
    first_gold_rank = min(positions) if positions else None
    return {
        "gold_rank_positions": positions,
        "first_gold_rank": first_gold_rank,
        "any_gold_within": {
            str(k): bool(positions and first_gold_rank is not None and first_gold_rank <= k)
            for k in (1, 3, 5, 6, 8, 10)
        },
        "all_gold_within": {
            str(k): len(gold_set) > 0 and len({node_id for node_id in matched_node_ids[:k] if node_id in gold_set}) == len(gold_set)
            for k in (1, 3, 5, 6, 8, 10)
        },
    }


def load_state_version_dataset(
    dataset_root: Path,
    language: str,
    phase: str,
    domains: list[str],
    splits: list[str],
) -> StateVersionDataset:
    """Load state chains and questions from the formal dataset layout."""

    chains: dict[str, StateChainSample] = {}
    questions_by_chain: dict[str, list[StateQuestion]] = defaultdict(list)
    chain_split: dict[str, str] = {}
    chain_domain: dict[str, str] = {}

    for domain in domains:
        for split in splits:
            split_dir = dataset_root / language / phase / domain / split
            chain_records = load_jsonl(split_dir / "state_chains.jsonl")
            question_records = load_jsonl(split_dir / "questions.jsonl")

            for payload in chain_records:
                sample = StateChainSample(**payload)
                chains[sample.state_chain_id] = sample
                chain_split[sample.state_chain_id] = split
                chain_domain[sample.state_chain_id] = domain

            for payload in question_records:
                question = StateQuestion(**payload)
                questions_by_chain[question.state_chain_id].append(question)

    for chain_id, question_list in questions_by_chain.items():
        question_list.sort(key=lambda item: item.question_id)

    logger.info(
        "Loaded state-version dataset: %s chains, %s questions",
        len(chains),
        sum(len(items) for items in questions_by_chain.values()),
    )
    return StateVersionDataset(
        chains=chains,
        questions_by_chain=dict(questions_by_chain),
        chain_split=chain_split,
        chain_domain=chain_domain,
    )


def summarize_state_version_results(question_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question results into report-friendly summary stats."""

    def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
        answerable = rows
        correct_answerable = [row for row in answerable if row["is_correct"]]
        zero_gold_answerable = [row for row in answerable if (row.get("support_coverage") or 0.0) == 0.0]
        unsupported_correct_answerable = [
            row for row in correct_answerable if not row.get("complete_support", False)
        ]

        return {
            "question_count": len(rows),
            "qa_accuracy": _rate(sum(1 for row in rows if row["is_correct"]), len(rows)),
            "answerable_accuracy": _rate(sum(1 for row in answerable if row["is_correct"]), len(answerable)),
            "gold_coverage_at_k": _mean([row["support_coverage"] for row in answerable if row["support_coverage"] is not None]),
            "complete_support_rate": _rate(sum(1 for row in answerable if row["complete_support"]), len(answerable)),
            "answerable_zero_gold_rate": _rate(len(zero_gold_answerable), len(answerable)),
            "answerable_zero_gold_count": len(zero_gold_answerable),
            "correct_without_complete_support_rate": _rate(
                len(unsupported_correct_answerable),
                len(correct_answerable),
            ),
            "correct_avg_matched_node_count": _mean(
                [len(row.get("matched_node_ids") or []) for row in correct_answerable]
            ),
            "correct_avg_gold_coverage": _mean(
                [row["support_coverage"] for row in correct_answerable if row["support_coverage"] is not None]
            ),
            "distractor_to_gold_ratio": _mean(
                [row["distractor_to_gold_ratio"] for row in answerable if row["distractor_to_gold_ratio"] is not None]
            ),
            "retrieval_latency_ms": _mean([row["retrieval_latency_ms"] for row in rows]),
            "retrieved_context_token_count": _mean([row["retrieved_context_token_count"] for row in rows]),
        }

    summary = {
        "overall": summarize_bucket(question_results),
        "breakdowns": {},
    }

    grouping_fields: list[str] = ["domain", "split", "difficulty_level", "answer_format"]
    if any(row.get("task_type") for row in question_results):
        grouping_fields.insert(2, "task_type")
    elif any(row.get("question_family") for row in question_results):
        grouping_fields.insert(2, "question_family")

    for field in grouping_fields:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in question_results:
            grouped[str(row.get(field))].append(row)
        summary["breakdowns"][field] = {
            key: summarize_bucket(rows)
            for key, rows in sorted(grouped.items(), key=lambda item: item[0])
        }

    summary["notes"] = {
        "task_type": "recommended as the main task-structure breakdown when available because it directly reflects the benchmark's target question types.",
        "difficulty_level": "retained as a default breakdown because low/high still provides a compact view of evidence complexity without requiring gold-node-count grouping in the main paper.",
        "gold_node_count": "retained per question for optional appendix analysis because it indicates how many version-memory nodes are required to support an answer.",
        "retrieval_failure_metric": "answerable_zero_gold_rate measures how often an answerable question retrieved no gold support at all; correct_without_complete_support_rate measures how often a correct answer was produced without fully retrieving the required gold evidence.",
        "correct_answer_retrieval_metric": "correct_avg_matched_node_count and correct_avg_gold_coverage characterize how much chain evidence the retriever actually returned on questions that were answered correctly.",
        "non_gold_pollution_metric": "distractor_to_gold_ratio is interpreted as the ratio of retrieved non-gold nodes to retrieved gold support, even when the dataset does not explicitly label adversarial nodes.",
        "removed_metric": "support_precision is omitted from aggregate reports because it is largely redundant with distractor_to_gold_ratio; end_to_end_latency is omitted in favor of retrieved_context_token_count.",
    }

    return summary


class StateVersionEvaluationRunner:
    """Evaluate one or more memory systems on the state-version benchmark."""

    def __init__(
        self,
        dataset: StateVersionDataset,
        systems: list[MemorySystem],
        answer_generator: AnswerGenerator,
        judge: Optional[LLMJudge] = None,
        random_context_pool_nodes: Optional[list[tuple[str, str, str]]] = None,
        analysis_top_ks: Optional[list[int]] = None,
        analysis_top_ks_by_task: Optional[dict[str, list[int] | tuple[int, ...]]] = None,
        question_task_type_map: Optional[dict[str, str]] = None,
    ) -> None:
        self.dataset = dataset
        self.systems = systems
        self.answer_generator = answer_generator
        self.judge = judge
        self.random_context_pool_nodes = random_context_pool_nodes or []
        self.analysis_top_ks = sorted({int(k) for k in (analysis_top_ks or list(DEFAULT_ANALYSIS_TOP_KS)) if int(k) > 0})
        self.analysis_top_ks_by_task = {
            task_type: sorted({int(k) for k in values if int(k) > 0})
            for task_type, values in (analysis_top_ks_by_task or DEFAULT_ANALYSIS_TOP_KS_BY_TASK).items()
        }
        self.question_task_type_map = question_task_type_map or {}

    def _resolve_task_type(self, question: StateQuestion) -> Optional[str]:
        task_type = self.question_task_type_map.get(question.question_id)
        if task_type:
            return task_type
        qid = question.question_id
        for candidate in DEFAULT_ANALYSIS_TOP_KS_BY_TASK:
            if candidate in qid:
                return candidate
        return None

    def _resolve_analysis_top_ks(self, question: StateQuestion) -> list[int]:
        task_type = self._resolve_task_type(question)
        if task_type and task_type in self.analysis_top_ks_by_task:
            values = self.analysis_top_ks_by_task[task_type]
            if values:
                return values
        return self.analysis_top_ks

    def _ingest_chain(self, system: MemorySystem, chain: StateChainSample) -> None:
        ordered_nodes = sorted(chain.chain_nodes, key=lambda node: node.surface_order)
        node_ids = [node.node_id for node in ordered_nodes]
        texts = [node.text for node in ordered_nodes]
        system.remember_chain(chain.state_chain_id, node_ids, texts)

    def _ingest_global_pool(self, system: MemorySystem) -> None:
        chain_payloads: list[tuple[str, list[str], list[str]]] = []
        for chain_id in sorted(self.dataset.chains):
            chain = self.dataset.chains[chain_id]
            ordered_nodes = sorted(chain.chain_nodes, key=lambda node: node.surface_order)
            texts = [node.text for node in ordered_nodes]
            node_ids = [node.node_id for node in ordered_nodes]
            chain_payloads.append((chain.state_chain_id, node_ids, texts))

        chain_ingest_workers = max(1, int(getattr(system, "chain_ingest_workers", 1)))
        if chain_ingest_workers <= 1:
            for chain_id, node_ids, texts in chain_payloads:
                system.remember_chain(chain_id, node_ids, texts)
            return

        with ThreadPoolExecutor(max_workers=chain_ingest_workers) as executor:
            futures = [
                executor.submit(system.remember_chain, chain_id, node_ids, texts)
                for chain_id, node_ids, texts in chain_payloads
            ]
            for future in futures:
                future.result()

    def _score_structured_question(self, question: StateQuestion, generated_answer: str) -> tuple[bool, Optional[str]]:
        normalized = (generated_answer or "").strip().upper()
        if question.answer_format == "multiple_choice":
            return normalized == (question.correct_option_id or "").upper(), normalized or None
        if question.answer_format == "boolean":
            label_map = {
                "A": "A",
                "B": "B",
                "C": "C",
                "YES": "A",
                "NO": "B",
                "INSUFFICIENT": "C",
            }
            option_id = label_map.get(normalized)
            return option_id == question.correct_option_id, option_id
        return False, None

    @staticmethod
    def _canonical_abstention(question: StateQuestion) -> str:
        if question.answer_format == "multiple_choice":
            return "E"
        if question.answer_format == "boolean":
            return "C"
        return "Insufficient information to support reasoning."

    def _score_abstractive_question(
        self,
        system_name: str,
        question: StateQuestion,
        generated_answer: str,
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if self.judge is None:
            normalized_pred = (generated_answer or "").strip().lower()
            normalized_gold = question.expected_answer.strip().lower()
            return normalized_pred == normalized_gold, None

        judge_result = self.judge.judge_answer(
            question=question.query_text,
            gold_answer=question.expected_answer,
            generated_answer=generated_answer,
            qa_id=question.question_id,
            system_name=system_name,
        )
        return bool(judge_result.get("is_correct", False)), judge_result

    def _evaluate_question(
        self,
        system: MemorySystem,
        chain: StateChainSample,
        split: str,
        question: StateQuestion,
    ) -> dict[str, Any]:
        def evaluate_from_query_result(query_result_slice: QueryResult, *, retrieval_latency_ms: float) -> dict[str, Any]:
            generation_error: Optional[str] = None
            judge_error: Optional[str] = None
            evaluation_status = "ok"
            generation_start = time.time()
            answer_generation_result = None
            if not (query_result_slice.retrieved_context or "").strip():
                generated_answer = self._canonical_abstention(question)
                answer_prompt_payload = {
                    "system_prompt": "",
                    "user_prompt": "",
                    "prompt_mode": question.answer_format,
                    "short_circuit": "empty_retrieved_context",
                }
            else:
                choices = None
                if question.answer_format == "multiple_choice" and question.options:
                    choices = [option.text for option in question.options[:4]]
                try:
                    answer_generation_result = self.answer_generator.generate_detailed(
                        question=question.query_text,
                        retrieved_context=query_result_slice.retrieved_context,
                        choices=choices,
                        answer_type=question.answer_format,
                    )
                    generated_answer = answer_generation_result.answer
                    generation_error = answer_generation_result.error
                    answer_prompt_payload = {
                        "system_prompt": answer_generation_result.system_prompt,
                        "user_prompt": answer_generation_result.user_prompt,
                        "prompt_mode": answer_generation_result.prompt_mode,
                    }
                except Exception as exc:
                    generated_answer = self._canonical_abstention(question)
                    generation_error = str(exc)
                    answer_prompt_payload = {
                        "system_prompt": "",
                        "user_prompt": "",
                        "prompt_mode": question.answer_format,
                        "exception": str(exc),
                    }
                if generation_error and evaluation_status == "ok":
                    evaluation_status = "generation_error"
            generation_latency_ms = (time.time() - generation_start) * 1000

            selected_option_id: Optional[str] = None
            judge_result: Optional[dict[str, Any]] = None
            if question.answer_format in {"multiple_choice", "boolean"}:
                is_correct, selected_option_id = self._score_structured_question(question, generated_answer)
            else:
                try:
                    is_correct, judge_result = self._score_abstractive_question(system.name, question, generated_answer)
                except Exception as exc:
                    is_correct = False
                    judge_error = str(exc)
                    judge_result = {
                        "is_correct": False,
                        "confidence": "low",
                        "reason": judge_error,
                        "raw_response": "",
                        "mode": "answer_judge",
                    }
                if judge_result and judge_result.get("reason") == "API error after 3 retries":
                    judge_error = judge_result["reason"]
                if judge_error and evaluation_status == "ok":
                    evaluation_status = "judge_error"

            matched_node_ids = _match_retrieved_nodes(
                chain,
                query_result_slice.retrieved_facts,
                metadata=query_result_slice.metadata,
            )
            gold_set = set(question.gold_node_ids)
            adv_set = set(question.adversarial_node_ids)
            matched_gold = sorted(gold_set & set(matched_node_ids))
            matched_adv = sorted(adv_set & set(matched_node_ids))
            matched_non_gold = sorted(set(matched_node_ids) - gold_set)

            support_coverage: Optional[float] = None
            complete_support = False
            support_precision: Optional[float] = None
            distractor_to_gold_ratio: Optional[float] = None
            if question.answerability == "answerable":
                support_coverage = len(matched_gold) / max(len(question.gold_node_ids), 1)
                complete_support = len(matched_gold) == len(question.gold_node_ids)
                support_precision = len(matched_gold) / max(len(matched_node_ids), 1) if matched_node_ids else 0.0
                distractor_to_gold_ratio = _compute_formal_distractor_to_gold_ratio(
                    matched_node_ids,
                    question.gold_node_ids,
                )

            return {
                "generated_answer": generated_answer,
                "answer_generation_raw": answer_generation_result.raw_response if answer_generation_result else "",
                "answer_prompt": answer_prompt_payload,
                "selected_option_id": selected_option_id,
                "is_correct": is_correct,
                "judge_result": judge_result,
                "generation_error": generation_error,
                "judge_error": judge_error,
                "evaluation_status": evaluation_status,
                "retrieved_context": query_result_slice.retrieved_context,
                "retrieved_facts": query_result_slice.retrieved_facts,
                "retrieved_fact_count": len(query_result_slice.retrieved_facts),
                "matched_node_ids": matched_node_ids,
                "matched_gold_node_ids": matched_gold,
                "matched_adversarial_node_ids": matched_adv,
                "matched_non_gold_node_ids": matched_non_gold,
                "support_coverage": support_coverage,
                "complete_support": complete_support,
                "support_precision": support_precision,
                "distractor_to_gold_ratio": distractor_to_gold_ratio,
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "retrieved_context_token_count": _count_tokens(query_result_slice.retrieved_context),
                "query_metadata": query_result_slice.metadata,
                "is_correct_without_gold_support": bool(is_correct and (support_coverage or 0.0) == 0.0),
                **_compute_gold_rank_positions(matched_node_ids, question.gold_node_ids),
            }

        start_time = time.time()
        retrieval_error: Optional[str] = None
        evaluation_status = "ok"
        question_analysis_top_ks = self._resolve_analysis_top_ks(question)
        max_requested_k = max(question_analysis_top_ks + [int(question.dynamic_top_k or 0), 10])

        if system.name == "Full Context":
            query_result = _build_full_context_query_result(chain, question)
        elif system.name == "Random Context":
            query_result = _build_random_context_query_result(self.random_context_pool_nodes, chain, question)
        else:
            try:
                query_result = system.query(
                    question=question.query_text,
                    top_k=max_requested_k,
                )
            except Exception as exc:
                retrieval_error = str(exc)
                evaluation_status = "retrieval_error"
                query_result = QueryResult(
                    answer="",
                    retrieved_context="",
                    retrieved_facts=[],
                    confidence=0.0,
                    latency_ms=0.0,
                    metadata={"error": retrieval_error},
                )

        retrieval_latency_ms = query_result.latency_ms
        per_k_results: dict[str, Any] = {}
        for top_k in question_analysis_top_ks:
            sliced = _slice_query_result(query_result, top_k)
            per_k_results[str(top_k)] = evaluate_from_query_result(sliced, retrieval_latency_ms=retrieval_latency_ms)

        primary_k = str(max(question_analysis_top_ks))
        primary = per_k_results[primary_k]
        matched_node_ids = primary["matched_node_ids"]
        matched_gold = primary["matched_gold_node_ids"]
        matched_adv = primary["matched_adversarial_node_ids"]

        return {
            "question_id": question.question_id,
            "state_chain_id": question.state_chain_id,
            "system_name": system.name,
            "domain": chain.domain,
            "split": split,
            "source_title": chain.source_title,
            "focus_event": chain.focus_event,
            "difficulty_level": question.difficulty_level,
            "answerability": question.answerability,
            "answer_format": question.answer_format,
            "gold_node_count": len(question.gold_node_ids),
            "adversarial_node_count": len(question.adversarial_node_ids),
            "chain_node_count": len(chain.chain_nodes),
            "query_text": question.query_text,
            "expected_answer": question.expected_answer,
            "generated_answer": primary["generated_answer"],
            "answer_generation_raw": primary["answer_generation_raw"],
            "answer_prompt": primary["answer_prompt"],
            "selected_option_id": primary["selected_option_id"],
            "is_correct": primary["is_correct"],
            "judge_result": primary["judge_result"],
            "retrieval_error": retrieval_error,
            "generation_error": primary["generation_error"],
            "judge_error": primary["judge_error"],
            "evaluation_status": evaluation_status,
            "retrieved_context": primary["retrieved_context"],
            "retrieved_facts": primary["retrieved_facts"],
            "retrieved_fact_count": primary["retrieved_fact_count"],
            "matched_node_ids": matched_node_ids,
            "matched_gold_node_ids": matched_gold,
            "matched_adversarial_node_ids": matched_adv,
            "support_coverage": primary["support_coverage"],
            "complete_support": primary["complete_support"],
            "support_precision": primary["support_precision"],
            "distractor_to_gold_ratio": primary["distractor_to_gold_ratio"],
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": primary["generation_latency_ms"],
            "end_to_end_latency_ms": (time.time() - start_time) * 1000,
            "retrieved_context_token_count": primary["retrieved_context_token_count"],
            "query_metadata": query_result.metadata,
            "retrieval_top_k_max": max_requested_k,
            "analysis_top_ks": question_analysis_top_ks,
            "per_k_results": per_k_results,
            "is_correct_without_gold_support": primary["is_correct_without_gold_support"],
            "gold_rank_positions": primary["gold_rank_positions"],
            "first_gold_rank": primary["first_gold_rank"],
            "any_gold_within": primary["any_gold_within"],
            "all_gold_within": primary["all_gold_within"],
        }

    @staticmethod
    def _system_slug(system_name: str) -> str:
        return system_name.lower().replace(" ", "_").replace("/", "_")

    def _save_incremental_run(
        self,
        output_dir: Path,
        system_name: str,
        question_results: list[dict[str, Any]],
    ) -> None:
        run = {
            "system_name": system_name,
            "question_results": question_results,
            "summary": summarize_state_version_results(question_results),
        }
        save_state_version_run(output_dir=output_dir, run=run, save_per_question_jsonl=True)

    def run_system(
        self,
        system: MemorySystem,
        output_dir: Optional[Path] = None,
        resume: bool = True,
        max_workers: int = 1,
        save_every: int = 10,
        pre_ingested: bool = False,
    ) -> dict[str, Any]:
        logger.info("Starting state-version evaluation for %s", system.name)
        question_results: list[dict[str, Any]] = []
        saved_question_ids: set[str] = set()

        if output_dir is not None and resume:
            system_slug = self._system_slug(system.name)
            existing_path = output_dir / f"{system_slug}.questions.jsonl"
            existing_rows = load_saved_question_results(existing_path)
            if existing_rows:
                question_results.extend(existing_rows)
                saved_question_ids = {row["question_id"] for row in existing_rows}
                logger.info(
                    "Resuming %s with %s previously saved question results",
                    system.name,
                    len(existing_rows),
                )

        if pre_ingested:
            pass
        elif system.name == "Random Context":
            system.reset()
        elif system.name != "Full Context":
            system.reset()
            self._ingest_global_pool(system)

        tasks: list[tuple[StateChainSample, str, StateQuestion]] = []
        for chain_id in sorted(self.dataset.chains):
            questions = self.dataset.questions_by_chain.get(chain_id, [])
            if not questions:
                continue
            chain = self.dataset.chains[chain_id]
            split = self.dataset.chain_split[chain_id]
            for question in questions:
                if question.question_id in saved_question_ids:
                    continue
                tasks.append((chain, split, question))

        if not tasks:
            logger.info("No remaining questions to evaluate for %s", system.name)
        elif max_workers <= 1:
            for chain, split, question in tasks:
                question_results.append(self._evaluate_question(system, chain, split, question))
                if output_dir is not None and (len(question_results) % max(save_every, 1) == 0):
                    self._save_incremental_run(output_dir, system.name, question_results)
        else:
            max_pending = max(max_workers, 1)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                pending = {}
                task_iter = iter(tasks)

                def submit_next(batch_count: int) -> int:
                    while batch_count < max_pending:
                        try:
                            chain, split, question = next(task_iter)
                        except StopIteration:
                            break
                        future = executor.submit(self._evaluate_question, system, chain, split, question)
                        pending[future] = question.question_id
                        batch_count += 1
                    return batch_count

                submitted = submit_next(0)
                while pending:
                    done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        question_id = pending.pop(future)
                        try:
                            question_results.append(future.result())
                        except Exception as exc:
                            logger.exception("Question evaluation failed for %s on %s: %s", question_id, system.name, exc)
                        submitted -= 1
                    question_results.sort(key=lambda row: row["question_id"])
                    if output_dir is not None:
                        self._save_incremental_run(output_dir, system.name, question_results)
                    submitted = submit_next(submitted)

        question_results.sort(key=lambda row: row["question_id"])

        return {
            "system_name": system.name,
            "question_results": question_results,
            "summary": summarize_state_version_results(question_results),
        }

    def run_all(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for system in self.systems:
            runs.append(self.run_system(system))
        return runs


def save_state_version_run(
    output_dir: Path,
    run: dict[str, Any],
    save_per_question_jsonl: bool = True,
) -> None:
    """Persist one system run to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    system_slug = run["system_name"].lower().replace(" ", "_").replace("/", "_")

    summary_path = output_dir / f"{system_slug}.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "system_name": run["system_name"],
                "summary": run["summary"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if save_per_question_jsonl:
        per_question_path = output_dir / f"{system_slug}.questions.jsonl"
        with per_question_path.open("w", encoding="utf-8") as fp:
            for record in run["question_results"]:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_saved_question_results(per_question_path: Path) -> list[dict[str, Any]]:
    """Load saved per-question results for resume."""
    if not per_question_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with per_question_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows
