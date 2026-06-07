"""LLM-assisted review helpers for the independent state-version benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from .chain_generator import parse_json_object
from .question_generator import StateQuestionGenerator
from .schemas import BuildTask, DatasetBuildConfig, SourceBundleRecord, StateChainSample
from .validator import ValidationReport

SYSTEM_PROMPT = (
    "You are a careful benchmark reviewer. "
    "Return exactly one valid JSON object and no extra commentary."
)


class StateVersionReviewer:
    """Review chain and QA artifacts, then provide structured repair guidance."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8000,
        timeout: int = 180,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        use_json_mode: bool = True,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_json_mode = use_json_mode

    @staticmethod
    def load_prompt(prompt_path: Path) -> str:
        return prompt_path.read_text(encoding="utf-8")

    def _call_llm(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                if self.use_json_mode and attempt == 0:
                    self.use_json_mode = False
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise last_error
        raise RuntimeError("LLM review call failed without a captured exception")

    @staticmethod
    def _validation_summary(report: ValidationReport) -> dict[str, Any]:
        return {
            "passed": report.passed,
            "errors": [issue["message"] for issue in report.errors],
            "warnings": [issue["message"] for issue in report.warnings],
            "metrics": report.metrics,
        }

    @staticmethod
    def build_chain_review_prompt(
        template: str,
        task: BuildTask,
        bundle: SourceBundleRecord,
        chain: StateChainSample,
        config: DatasetBuildConfig,
        validation_report: ValidationReport,
    ) -> str:
        task_block = {
            "sample_id": task.sample_id,
            "state_chain_id": task.state_chain_id,
            "domain": task.domain,
            "language": task.language,
            "focus_event": chain.focus_event,
            "source_title": chain.source_title,
            "hard_limits": {
                "min_nodes": config.hard_limits.min_nodes_by_domain[task.domain],
                "max_nodes": config.hard_limits.max_nodes_by_domain[task.domain],
                "min_text_units": config.hard_limits.min_text_units_by_language[task.language],
            },
            "validator_summary": StateVersionReviewer._validation_summary(validation_report),
        }
        return (
            f"{template}\n\n"
            "## Task block\n"
            f"```json\n{json.dumps(task_block, ensure_ascii=False, indent=2)}\n```\n\n"
            "## Source bundle for groundedness checks\n"
            f"```json\n{json.dumps(bundle.model_dump(), ensure_ascii=False, indent=2)}\n```\n\n"
            "## Candidate state_chain to review\n"
            f"```json\n{json.dumps(chain.model_dump(), ensure_ascii=False, indent=2)}\n```"
        )

    @staticmethod
    def build_question_review_prompt(
        template: str,
        task: BuildTask,
        chain: StateChainSample,
        questions: list[dict[str, Any]],
        config: DatasetBuildConfig,
        validation_report: ValidationReport,
    ) -> str:
        task_block = {
            "sample_id": task.sample_id,
            "state_chain_id": task.state_chain_id,
            "domain": task.domain,
            "language": task.language,
            "focus_event": chain.focus_event,
            "source_title": chain.source_title,
            "required_generation_plan": StateQuestionGenerator.build_suggested_plan(task, config),
            "canonical_abstentions": config.canonical_abstentions.model_dump(),
            "validator_summary": StateVersionReviewer._validation_summary(validation_report),
        }
        payload = {"state_chain_id": task.state_chain_id, "questions": questions}
        return (
            f"{template}\n\n"
            "## Task block\n"
            f"```json\n{json.dumps(task_block, ensure_ascii=False, indent=2)}\n```\n\n"
            "## Frozen state_chain\n"
            f"```json\n{json.dumps(chain.model_dump(), ensure_ascii=False, indent=2)}\n```\n\n"
            "## Candidate question set to review\n"
            f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
        )

    @staticmethod
    def normalize_chain_review(payload: dict[str, Any], state_chain_id: str) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["state_chain_id"] = state_chain_id
        normalized["pass"] = bool(normalized.get("pass", False))
        normalized["major_issues"] = [str(item) for item in normalized.get("major_issues", []) if str(item).strip()]
        normalized["minor_issues"] = [str(item) for item in normalized.get("minor_issues", []) if str(item).strip()]
        normalized["repair_suggestions"] = [
            str(item) for item in normalized.get("repair_suggestions", []) if str(item).strip()
        ]
        scores = normalized.get("quality_scores") or {}
        if not isinstance(scores, dict):
            scores = {}
        normalized["quality_scores"] = {
            "groundedness": int(scores.get("groundedness", 0) or 0),
            "state_separability": int(scores.get("state_separability", 0) or 0),
            "competition_strength": int(scores.get("competition_strength", 0) or 0),
            "text_richness": int(scores.get("text_richness", 0) or 0),
            "answerability_support": int(scores.get("answerability_support", 0) or 0),
        }
        return normalized

    @staticmethod
    def normalize_question_review(
        payload: dict[str, Any],
        state_chain_id: str,
        question_ids: list[str],
    ) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["state_chain_id"] = state_chain_id
        normalized["pass"] = bool(normalized.get("pass", False))
        normalized["major_issues"] = [str(item) for item in normalized.get("major_issues", []) if str(item).strip()]
        normalized["minor_issues"] = [str(item) for item in normalized.get("minor_issues", []) if str(item).strip()]
        reports = normalized.get("question_reports") or []
        if not isinstance(reports, list):
            reports = []
        cleaned_reports: list[dict[str, Any]] = []
        report_by_id: dict[str, dict[str, Any]] = {}
        for item in reports:
            if not isinstance(item, dict):
                continue
            question_id = str(item.get("question_id") or "").strip()
            if not question_id:
                continue
            report_by_id[question_id] = {
                "question_id": question_id,
                "pass": bool(item.get("pass", False)),
                "issue_types": [str(v) for v in item.get("issue_types", []) if str(v).strip()],
                "suggestion": str(item.get("suggestion") or "").strip(),
            }
        for question_id in question_ids:
            cleaned_reports.append(
                report_by_id.get(
                    question_id,
                    {
                        "question_id": question_id,
                        "pass": False,
                        "issue_types": ["missing_review_report"],
                        "suggestion": "The reviewer did not provide a report for this question.",
                    },
                )
            )
        normalized["question_reports"] = cleaned_reports
        return normalized

    def review_chain(
        self,
        task: BuildTask,
        bundle: SourceBundleRecord,
        chain: StateChainSample,
        prompt_path: Path,
        config: DatasetBuildConfig,
        validation_report: ValidationReport,
    ) -> tuple[dict[str, Any], str, str]:
        template = self.load_prompt(prompt_path)
        prompt = self.build_chain_review_prompt(template, task, bundle, chain, config, validation_report)
        raw = self._call_llm(prompt)
        payload = self.normalize_chain_review(parse_json_object(raw), task.state_chain_id)
        return payload, raw, prompt

    def review_questions(
        self,
        task: BuildTask,
        chain: StateChainSample,
        questions: list[dict[str, Any]],
        prompt_path: Path,
        config: DatasetBuildConfig,
        validation_report: ValidationReport,
    ) -> tuple[dict[str, Any], str, str]:
        template = self.load_prompt(prompt_path)
        prompt = self.build_question_review_prompt(template, task, chain, questions, config, validation_report)
        raw = self._call_llm(prompt)
        question_ids = [str(item.get("question_id")) for item in questions]
        payload = self.normalize_question_review(parse_json_object(raw), task.state_chain_id, question_ids)
        return payload, raw, prompt

    @staticmethod
    def build_chain_repair_prompt(
        base_prompt: str,
        previous_chain: dict[str, Any],
        review_payload: dict[str, Any],
        validation_errors: list[str],
    ) -> str:
        issues = review_payload.get("major_issues", []) + review_payload.get("minor_issues", [])
        suggestions = review_payload.get("repair_suggestions", [])
        guidance = [
            "## Content repair request",
            "The previous state_chain passed through generation but still needs review-driven correction.",
            "Regenerate the entire state_chain JSON from scratch while preserving the same fixed identifiers and focal event.",
            "Review issues:",
        ]
        guidance.extend(f"- {issue}" for issue in issues)
        if validation_errors:
            guidance.append("Validator findings:")
            guidance.extend(f"- {issue}" for issue in validation_errors)
        if suggestions:
            guidance.append("Repair suggestions:")
            guidance.extend(f"- {item}" for item in suggestions)
        guidance.extend(
            [
                "",
                "Additional rules:",
                "1. Keep the main chain intact but repair mislabeled or weakly separated nodes.",
                "2. Preserve realistic distractor nodes when they are useful, but do not let distractors replace core evidence.",
                "3. Keep node count within the configured limits.",
                "4. Do not leak raw GitHub identifiers into visible text.",
            ]
        )
        return (
            f"{base_prompt}\n\n"
            + "\n".join(guidance)
            + "\n\n## Previous state_chain to repair\n"
            + f"```json\n{json.dumps(previous_chain, ensure_ascii=False, indent=2)}\n```"
        )

    @staticmethod
    def build_question_repair_prompt(
        base_prompt: str,
        previous_questions: list[dict[str, Any]],
        review_payload: dict[str, Any],
        validation_errors: list[str],
    ) -> str:
        question_reports = review_payload.get("question_reports", [])
        failing_reports = [item for item in question_reports if not item.get("pass", False)]
        guidance = [
            "## Content repair request",
            "The previous question set needs review-driven correction.",
            "Regenerate the entire question-set JSON from scratch while preserving the same state_chain_id.",
            "Review issues:",
        ]
        guidance.extend(f"- {issue}" for issue in review_payload.get("major_issues", []))
        guidance.extend(f"- {issue}" for issue in review_payload.get("minor_issues", []))
        if validation_errors:
            guidance.append("Validator findings:")
            guidance.extend(f"- {issue}" for issue in validation_errors)
        if failing_reports:
            guidance.append("Question-level repair targets:")
            for item in failing_reports:
                suggestion = str(item.get("suggestion") or "").strip()
                issue_types = ", ".join(item.get("issue_types", []))
                guidance.append(f"- {item.get('question_id')}: {issue_types or 'unspecified_issue'}")
                if suggestion:
                    guidance.append(f"  Repair suggestion: {suggestion}")
        guidance.extend(
            [
                "",
                "Additional rules:",
                "1. Keep the per-chain count and distribution targets exact whenever possible.",
                "2. High-difficulty answerable questions must remain multi-version and cite at least two gold nodes.",
                "3. Multiple-choice and boolean questions must include the full fixed option sets.",
                "4. Keep questions single-question and single-answer.",
                "5. Use the canonical abstention texts exactly where required.",
            ]
        )
        return (
            f"{base_prompt}\n\n"
            + "\n".join(guidance)
            + "\n\n## Previous question set to repair\n"
            + f"```json\n{json.dumps({'questions': previous_questions}, ensure_ascii=False, indent=2)}\n```"
        )
