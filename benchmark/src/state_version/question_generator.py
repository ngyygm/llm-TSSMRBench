"""Independent question-generation utilities for the state-version benchmark."""

from __future__ import annotations

from collections import Counter
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from openai import OpenAI

from .schemas import BuildTask, DatasetBuildConfig, StateChainSample
from .validator import ValidationReport, summarize_question_distribution, validate_question_payload

SYSTEM_PROMPT = (
    "You are a careful benchmark QA writer. "
    "Return exactly one valid JSON object and no extra commentary."
)
REASONING_SPLIT_RE = re.compile(r"[;\n]+|(?<=[.!?])\s+")
GENERIC_QUERY_RE = re.compile(
    r"\b(?:what are the main issues|what are the key steps(?: taken)?|what is the latest status|what is the current status)\b",
    re.IGNORECASE,
)
META_QUERY_RE = re.compile(
    r"\b(?:according to (?:the )?(?:chain|narrative summary)|as described in|based on (?:the )?node|chain nodes?|without mentioning node numbers|without mentioning node ids|provided in the chain nodes?)\b",
    re.IGNORECASE,
)
GENERIC_OPTION_RE = re.compile(
    r"\b(?:planned|in progress|resolved|invalidated|deprecated|fully implemented|not implemented|unknown|unclear)\b",
    re.IGNORECASE,
)
META_OPTION_RE = re.compile(
    r"\b(?:all of the above|none of the above|both [a-d] and [a-d]|neither [a-d] nor [a-d])\b",
    re.IGNORECASE,
)


class QuestionGenerationValidationError(ValueError):
    """Raised when question generation keeps failing validation."""

    def __init__(self, message: str, *, raw_response: str, payload: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.payload = payload


def strip_code_fences(raw: str) -> str:
    """Remove common markdown code fences around JSON output."""

    text = (raw or "").strip()
    if text.startswith("```json"):
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if text.startswith("```"):
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse one JSON object from model output."""

    cleaned = strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("model output must be a JSON object")
        return data
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("model output must be a JSON object")
            return data
        raise


class StateQuestionGenerator:
    """Generate questions for one frozen state chain."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 12000,
        timeout: int = 180,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        use_json_mode: bool = True,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> None:
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            http_client=httpx.Client(timeout=timeout, trust_env=False),
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_json_mode = use_json_mode
        self.extra_body = extra_body or {}
        self.last_usage: dict[str, Any] | None = None

    @staticmethod
    def load_prompt(prompt_path: Path) -> str:
        return prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def build_question_blueprint(plan: dict[str, int], format_counts: dict[str, int]) -> list[dict[str, str]]:
        """Build one deterministic per-chain question blueprint."""

        blueprint: list[dict[str, str]] = []

        def append_slots(count: int, *, difficulty: str, answerability: str, answer_format: str) -> None:
            family = "multi_version" if difficulty == "high" else "single_version"
            for _ in range(count):
                blueprint.append(
                    {
                        "difficulty_level": difficulty,
                        "answerability": answerability,
                        "question_family": family,
                        "answer_format": answer_format,
                    }
                )

        append_slots(min(3, plan["low_answerable"]), difficulty="low", answerability="answerable", answer_format="multiple_choice")
        remaining_low_answerable = plan["low_answerable"] - min(3, plan["low_answerable"])
        append_slots(remaining_low_answerable, difficulty="low", answerability="answerable", answer_format="boolean")
        append_slots(min(2, plan["high_answerable"]), difficulty="high", answerability="answerable", answer_format="multiple_choice")
        remaining_high_answerable = plan["high_answerable"] - min(2, plan["high_answerable"])
        append_slots(remaining_high_answerable, difficulty="high", answerability="answerable", answer_format="abstractive")
        if len(blueprint) != sum(plan.values()):
            raise ValueError("question blueprint size does not match the required plan")

        actual_formats = {
            "multiple_choice": sum(1 for slot in blueprint if slot["answer_format"] == "multiple_choice"),
            "boolean": sum(1 for slot in blueprint if slot["answer_format"] == "boolean"),
            "abstractive": sum(1 for slot in blueprint if slot["answer_format"] == "abstractive"),
        }
        if actual_formats != format_counts:
            raise ValueError(f"question blueprint format mismatch: expected {format_counts}, got {actual_formats}")

        return blueprint

    @classmethod
    def build_suggested_plan(cls, task: BuildTask, config: DatasetBuildConfig) -> dict[str, Any]:
        """Construct a helpful per-chain suggestion without making it a hard quota."""

        target_count = task.recommended_question_count
        explicit_plan = dict(task.recommended_question_plan)
        if explicit_plan:
            format_counts = {
                "multiple_choice": max(1, round(target_count * config.distribution_targets.answer_format["multiple_choice"])),
                "boolean": max(1, round(target_count * config.distribution_targets.answer_format["boolean"])),
                "abstractive": max(1, target_count - round(target_count * config.distribution_targets.answer_format["multiple_choice"]) - round(target_count * config.distribution_targets.answer_format["boolean"])),
            }
            return {
                "target_question_count": target_count,
                "suggested_difficulty_answerability_plan": explicit_plan,
                "suggested_answer_format_counts": format_counts,
                "question_blueprint": cls.build_question_blueprint(explicit_plan, format_counts),
            }

        low_count = round(target_count * config.distribution_targets.difficulty["low"])
        high_count = target_count - low_count
        answerable_count = round(target_count * config.distribution_targets.answerability["answerable"])
        low_ratio = low_count / target_count if target_count else 0.0
        low_answerable = min(low_count, math.floor(answerable_count * low_ratio))
        high_answerable = answerable_count - low_answerable
        fallback_plan = {
            "low_answerable": max(0, low_answerable),
            "high_answerable": max(0, high_answerable),
            "low_unanswerable": 0,
            "high_unanswerable": 0,
        }
        total = sum(fallback_plan.values())
        if total != target_count:
            fallback_plan["high_answerable"] += target_count - total

        mc_count = max(1, round(target_count * config.distribution_targets.answer_format["multiple_choice"]))
        boolean_count = max(1, round(target_count * config.distribution_targets.answer_format["boolean"]))
        abstractive_count = max(1, target_count - mc_count - boolean_count)
        while mc_count + boolean_count + abstractive_count > target_count:
            if mc_count >= max(boolean_count, abstractive_count) and mc_count > 1:
                mc_count -= 1
            elif boolean_count >= abstractive_count and boolean_count > 1:
                boolean_count -= 1
            else:
                abstractive_count -= 1
        while mc_count + boolean_count + abstractive_count < target_count:
            mc_count += 1

        format_counts = {
            "multiple_choice": mc_count,
            "boolean": boolean_count,
            "abstractive": abstractive_count,
        }
        return {
            "target_question_count": target_count,
            "suggested_difficulty_answerability_plan": fallback_plan,
            "suggested_answer_format_counts": format_counts,
            "question_blueprint": cls.build_question_blueprint(fallback_plan, format_counts),
        }

    @classmethod
    def build_user_prompt(
        cls,
        template: str,
        task: BuildTask,
        chain: StateChainSample,
        config: DatasetBuildConfig,
    ) -> str:
        suggested_plan = cls.build_suggested_plan(task, config)
        blueprint = suggested_plan.get("question_blueprint", [])
        slot_lines = [
            (
                f"{index + 1}. difficulty={slot['difficulty_level']}, "
                f"answerability={slot['answerability']}, "
                f"question_family={slot['question_family']}, "
                f"answer_format={slot['answer_format']}"
            )
            for index, slot in enumerate(blueprint)
        ]
        task_block = {
            "sample_id": task.sample_id,
            "state_chain_id": task.state_chain_id,
            "domain": task.domain,
            "language": task.language,
            "focus_event": chain.focus_event,
            "source_title": chain.source_title,
            "target_question_count_range": task.target_question_count_range,
            "recommended_question_count": task.recommended_question_count,
            "canonical_abstentions": config.canonical_abstentions.model_dump(),
            "suggested_generation_plan": suggested_plan,
        }
        return (
            f"{template}\n\n"
            "## Task-specific fixed identifiers and targets\n"
            "Use exactly these values in the output where applicable.\n"
            f"```json\n{json.dumps(task_block, ensure_ascii=False, indent=2)}\n```\n\n"
            f"Generate a complete question set, not a single example. "
            f"Return exactly {task.recommended_question_count} questions if possible, and never return fewer than "
            f"{task.target_question_count_range['min']} or more than {task.target_question_count_range['max']}.\n\n"
            "## Why node metadata must stay out of the question text\n"
            "At evaluation time, the tested memory system only sees the memory texts that were written into the system. "
            "It does not see node ids, chain position numbers, annotation labels, or benchmark-internal metadata. "
            "Therefore, the question text must sound like a natural user query over remembered content, not like a question about dataset structure. "
            "Never mention node ids, node order, chain nodes, hidden labels, or any wording copied from annotation instructions.\n\n"
            "## Required slot order\n"
            "You must fill the blueprint slots in this exact order. "
            "Question 1 must satisfy slot 1, question 2 must satisfy slot 2, and so on. "
            "Do not reorder slots. Do not collapse structured slots into abstractive questions.\n"
            f"{chr(10).join(slot_lines)}\n\n"
            "## Input frozen state_chain\n"
            "Treat the following chain as the only evidence source. "
            "Do not invent outside facts, and do not invent missing chain nodes.\n"
            f"```json\n{json.dumps(chain.model_dump(), ensure_ascii=False, indent=2)}\n```"
        )

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
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                response = self.client.chat.completions.create(**kwargs)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    if hasattr(usage, "model_dump"):
                        self.last_usage = usage.model_dump()
                    elif hasattr(usage, "dict"):
                        self.last_usage = usage.dict()
                    else:
                        self.last_usage = dict(usage)
                else:
                    self.last_usage = None
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                if self.use_json_mode and attempt == 0:
                    self.use_json_mode = False
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise last_error
        raise RuntimeError("LLM call failed without a captured exception")

    @staticmethod
    def _normalize_nullable_scalar(value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a"}:
            return None
        return value

    @staticmethod
    def _normalize_reasoning_chain(value: Any) -> list[str]:
        if value in (None, "", "null", "None"):
            return []

        steps: list[str] = []
        if isinstance(value, str):
            steps = [segment.strip() for segment in REASONING_SPLIT_RE.split(value) if segment.strip()]
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned:
                        steps.append(cleaned)
        else:
            return []

        if len(steps) == 1:
            expanded = [segment.strip() for segment in REASONING_SPLIT_RE.split(steps[0]) if segment.strip()]
            if len(expanded) > 1:
                steps = expanded

        if len(steps) > 4:
            steps = steps[:4]
        return steps

    @classmethod
    def _normalize_question_fields(
        cls,
        question: dict[str, Any],
        config: DatasetBuildConfig,
        *,
        chain_node_count: Optional[int] = None,
    ) -> dict[str, Any]:
        normalized = dict(question)
        for key in ["difficulty_level", "question_family", "answerability", "answer_format"]:
            value = normalized.get(key)
            if isinstance(value, str):
                normalized[key] = value.strip().lower()

        normalized["correct_option_id"] = cls._normalize_nullable_scalar(normalized.get("correct_option_id"))
        if isinstance(normalized.get("correct_option_id"), str):
            normalized["correct_option_id"] = normalized["correct_option_id"].strip().upper()

        normalized["dynamic_top_k"] = cls._normalize_nullable_scalar(normalized.get("dynamic_top_k"))
        gold_node_ids = normalized.get("gold_node_ids") or []
        if isinstance(gold_node_ids, list):
            normalized["dynamic_top_k"] = math.ceil(1.5 * len(gold_node_ids))

        options = normalized.get("options")
        if normalized.get("answer_format") == "abstractive":
            normalized["options"] = None
            normalized["correct_option_id"] = None
        elif options in ("", "null", "None"):
            normalized["options"] = None

        if isinstance(normalized.get("options"), list):
            for option in normalized["options"]:
                if isinstance(option, dict) and isinstance(option.get("option_id"), str):
                    option["option_id"] = option["option_id"].strip().upper()

        normalized["reasoning_chain"] = cls._normalize_reasoning_chain(normalized.get("reasoning_chain"))
        if len(normalized["reasoning_chain"]) == 1:
            normalized["reasoning_chain"].append(
                "Nearby competing nodes are relevant but do not settle the requested state."
            )

        if normalized.get("answer_format") == "multiple_choice" and isinstance(normalized.get("options"), list):
            option_map = {
                option["option_id"]: option
                for option in normalized["options"]
                if isinstance(option, dict) and isinstance(option.get("option_id"), str)
            }
            if set(option_map).issubset({"A", "B", "C", "D", "E"}):
                option_map.setdefault(
                    "E",
                    {"option_id": "E", "text": config.canonical_abstentions.multiple_choice},
                )
                if set(option_map) == {"A", "B", "C", "D", "E"}:
                    normalized["options"] = [option_map[key] for key in ["A", "B", "C", "D", "E"]]
                else:
                    normalized["answer_format"] = "abstractive"
                    normalized["options"] = None
                    normalized["correct_option_id"] = None

        if normalized.get("answer_format") == "boolean" and isinstance(normalized.get("options"), list):
            option_map = {
                option["option_id"]: option
                for option in normalized["options"]
                if isinstance(option, dict) and isinstance(option.get("option_id"), str)
            }
            if set(option_map).issubset({"A", "B", "C"}):
                option_map.setdefault(
                    "C",
                    {"option_id": "C", "text": config.canonical_abstentions.boolean},
                )
                if set(option_map) == {"A", "B", "C"}:
                    normalized["options"] = [option_map[key] for key in ["A", "B", "C"]]
                else:
                    normalized["answer_format"] = "abstractive"
                    normalized["options"] = None
                    normalized["correct_option_id"] = None

        return normalized

    @staticmethod
    def normalize_payload(
        payload: dict[str, Any],
        task: BuildTask,
        config: DatasetBuildConfig,
        chain: StateChainSample,
    ) -> dict[str, Any]:
        payload["state_chain_id"] = task.state_chain_id
        raw_questions = payload.get("questions")
        if not isinstance(raw_questions, list):
            raise ValueError("question generator must return a JSON object with a questions list")

        normalized_questions: list[dict[str, Any]] = []
        for index, question in enumerate(raw_questions, start=1):
            if not isinstance(question, dict):
                raise ValueError("each generated question must be a JSON object")
            normalized = StateQuestionGenerator._normalize_question_fields(
                question,
                config,
                chain_node_count=len(chain.chain_nodes),
            )
            normalized["state_chain_id"] = task.state_chain_id
            normalized["question_id"] = f"{task.sample_id}_q{index:03d}"
            normalized_questions.append(normalized)
        payload["questions"] = normalized_questions
        return payload

    @staticmethod
    def build_semantic_retry_guidance(previous_payload: Optional[dict[str, Any]]) -> list[str]:
        """Derive extra regeneration hints from a bad previous draft."""

        if previous_payload is None:
            return []

        questions = previous_payload.get("questions")
        if not isinstance(questions, list):
            return []

        generic_query_ids: list[str] = []
        meta_query_ids: list[str] = []
        generic_option_ids: list[str] = []
        meta_option_ids: list[str] = []
        repeated_stem_counter: Counter[str] = Counter()

        for question in questions:
            if not isinstance(question, dict):
                continue
            question_id = str(question.get("question_id", "unknown_question"))
            query_text = str(question.get("query_text", "")).strip()
            if GENERIC_QUERY_RE.search(query_text):
                generic_query_ids.append(question_id)
            if META_QUERY_RE.search(query_text):
                meta_query_ids.append(question_id)

            stem_tokens = re.findall(r"[A-Za-z0-9]+", query_text.lower())[:5]
            if stem_tokens:
                repeated_stem_counter[" ".join(stem_tokens)] += 1

            if str(question.get("answer_format")) == "multiple_choice":
                options = question.get("options")
                if isinstance(options, list):
                    generic_option_count = 0
                    has_meta_option = False
                    for option in options:
                        if not isinstance(option, dict):
                            continue
                        option_id = str(option.get("option_id", "")).upper()
                        option_text = str(option.get("text", "")).strip()
                        if option_id in {"A", "B", "C", "D"} and GENERIC_OPTION_RE.fullmatch(option_text.lower()):
                            generic_option_count += 1
                        if option_id in {"A", "B", "C", "D"} and META_OPTION_RE.search(option_text):
                            has_meta_option = True
                    if generic_option_count >= 2:
                        generic_option_ids.append(question_id)
                    if has_meta_option:
                        meta_option_ids.append(question_id)

        hints: list[str] = []
        if generic_query_ids:
            hints.append(
                "Rewrite these query_text items into chain-specific state-disambiguation questions instead of broad recap prompts: "
                + ", ".join(generic_query_ids[:8])
            )
        if meta_query_ids:
            hints.append(
                "Rewrite these query_text items as natural user-facing questions. Never mention the chain, node numbers, or authoring process: "
                + ", ".join(meta_query_ids[:8])
            )
        if generic_option_ids:
            hints.append(
                "Replace generic label-like multiple-choice options in these questions with chain-specific competing states grounded in adversarial nodes: "
                + ", ".join(generic_option_ids[:8])
            )
        if meta_option_ids:
            hints.append(
                "Replace meta-options such as 'both A and B', 'neither A nor B', or 'all of the above' with standalone concrete answer choices in these questions: "
                + ", ".join(meta_option_ids[:8])
            )

        repeated_stems = [stem for stem, count in repeated_stem_counter.items() if count >= 3 and stem]
        if repeated_stems:
            hints.append(
                "Diversify repeated template-like question openings. Do not keep reusing the same broad stem across many questions."
            )
        return hints

    @staticmethod
    def build_validation_retry_prompt(
        base_prompt: str,
        task: BuildTask,
        error_messages: list[str],
        previous_payload: Optional[dict[str, Any]] = None,
    ) -> str:
        guidance = [
            "## Regeneration request",
            "Your previous draft failed schema or hard validation.",
            "Regenerate the entire JSON object from scratch and fix every error below.",
            "Validation errors:",
        ]
        guidance.extend(f"- {message}" for message in error_messages)
        guidance.extend(
            [
                "",
                "Important fixes:",
                f"1. Return between {task.target_question_count_range['min']} and {task.target_question_count_range['max']} questions.",
                f"2. Prefer exactly {task.recommended_question_count} questions unless the chain strongly resists that exact count.",
                "3. Every question must be answerable only from the chain.",
                "4. Every multiple-choice question must use A/B/C/D/E and the canonical E insufficiency option.",
                "5. Every boolean question must use A/B/C and the canonical C insufficiency option.",
                "6. High-difficulty answerable questions must use at least two gold nodes.",
                "7. Every gold_node_ids and adversarial_node_ids entry must reference real node ids from the chain.",
                "8. Every gold_node_id must come from a core node. Never make a distractor node the direct answer target.",
                "9. Keep the questions single-question and single-answer; do not bundle two questions into one.",
                "10. Do not output one illustrative sample question; output the full question set.",
                "11. Every high-difficulty answerable slot must remain multi-version and must cite at least two gold_node_ids.",
                "12. If a validation error mentions a specific question_id, repair that exact question instead of leaving the slot malformed.",
                "13. Never mention node numbers, node ids, or phrases like 'as described in node 3' inside query_text.",
                "14. Never use 'all of the above', 'none of the above', or multi-select style options.",
                "15. For answerable questions, reasoning_chain must contain 2 to 4 short bullet-like steps after normalization.",
                "16. Follow the question_blueprint slot-by-slot in order; do not swap slot types or answer formats.",
                "17. Do not convert multiple_choice or boolean blueprint slots into abstractive questions for convenience.",
                "18. If the blueprint requires a structured answer format, return the full structured options with the canonical insufficiency option.",
                "19. Every query_text must be one single question only. Do not ask two things joined by 'and'. A single sentence that asks for two fields is still invalid, such as asking for exact words and also who spoke them.",
                "20. Never write query_text that refers to node numbering, chain construction, or any internal annotation artifact.",
                "21. Never copy instruction phrases such as 'according to the narrative summary provided in the chain nodes' or 'without mentioning node numbers' into query_text.",
                "22. Never use composite answer options such as 'both A and B', 'neither A nor B', or any meta-option that refers to other option letters.",
                "23. Remember the evaluation setting: the tested memory system only sees memory texts, not node ids, chain order, or hidden labels, so every query_text must be answerable from visible memory content alone.",
                "24. If the chain text contains explicit dates, release periods, chapter stages, or relative time markers, you may ask when a state held or what state held at that time, but still target core-node evidence only.",
            ]
        )
        semantic_hints = StateQuestionGenerator.build_semantic_retry_guidance(previous_payload)
        if semantic_hints:
            guidance.extend(["", "Quality fixes for generic or templated drafts:"])
            guidance.extend(f"- {hint}" for hint in semantic_hints)
        prompt = f"{base_prompt}\n\n" + "\n".join(guidance)
        if previous_payload is not None:
            prompt += (
                "\n\n## Previous draft to repair\n"
                "Use this draft as a repair target, but return a fully corrected replacement.\n"
                f"```json\n{json.dumps(previous_payload, ensure_ascii=False, indent=2)}\n```"
            )
        return prompt

    @staticmethod
    def validate_question_set_payload(
        payload: dict[str, Any],
        chain_payload: dict[str, Any],
        task: BuildTask,
        config: DatasetBuildConfig,
        *,
        enforce_exact_count: bool = True,
        enforce_distribution: bool = True,
    ) -> ValidationReport:
        report = ValidationReport(scope="question_set", item_id=task.sample_id)
        questions = payload.get("questions")
        if not isinstance(questions, list):
            report.add_error("questions must be a list")
            return report

        question_count = len(questions)
        min_count = task.target_question_count_range["min"]
        max_count = task.target_question_count_range["max"]
        if question_count < min_count or question_count > max_count:
            issue_text = f"question count must be within [{min_count}, {max_count}], got {question_count}"
            if enforce_exact_count:
                report.add_error(issue_text)
            else:
                report.add_warning(issue_text)
        if question_count != task.recommended_question_count:
            issue_text = (
                f"question count must match the required target {task.recommended_question_count}, got {question_count}"
            )
            if enforce_exact_count:
                report.add_error(issue_text)
            else:
                report.add_warning(issue_text)

        question_ids = [str(question.get("question_id", "")) for question in questions]
        duplicate_ids = sorted(question_id for question_id in set(question_ids) if question_ids.count(question_id) > 1)
        if duplicate_ids:
            report.add_error(f"duplicate question_id values detected: {duplicate_ids}")

        for question_payload in questions:
            question_id = str(question_payload.get("question_id", "unknown_question"))
            difficulty = str(question_payload.get("difficulty_level"))
            answerability = str(question_payload.get("answerability"))
            gold_node_ids = question_payload.get("gold_node_ids") or []
            if (
                difficulty == "high"
                and answerability == "answerable"
                and isinstance(gold_node_ids, list)
                and len(gold_node_ids) < 2
            ):
                report.add_error(
                    f"{question_id}: high-difficulty answerable questions must cite at least two gold_node_ids"
                )
            question_report = validate_question_payload(question_payload, chain_payload, config)
            report.extend(question_report)

        distribution = summarize_question_distribution(questions)
        required_plan = StateQuestionGenerator.build_suggested_plan(task, config)
        required_combo = required_plan["suggested_difficulty_answerability_plan"]
        actual_combo = {
            "low_answerable": 0,
            "high_answerable": 0,
            "low_unanswerable": 0,
            "high_unanswerable": 0,
        }
        for question in questions:
            difficulty = str(question.get("difficulty_level"))
            answerability = str(question.get("answerability"))
            key = f"{difficulty}_{answerability}"
            if key in actual_combo:
                actual_combo[key] += 1
        if actual_combo != required_combo:
            issue_text = f"difficulty/answerability distribution must match {required_combo}, got {actual_combo}"
            if enforce_distribution:
                report.add_error(issue_text)
            else:
                report.add_warning(issue_text)

        required_formats = required_plan["suggested_answer_format_counts"]
        actual_formats = {
            "multiple_choice": distribution["answer_format"].get("multiple_choice", 0),
            "boolean": distribution["answer_format"].get("boolean", 0),
            "abstractive": distribution["answer_format"].get("abstractive", 0),
        }
        if actual_formats != required_formats:
            issue_text = f"answer_format distribution must match {required_formats}, got {actual_formats}"
            if enforce_distribution:
                report.add_error(issue_text)
            else:
                report.add_warning(issue_text)

        report.metrics = {
            "question_count": question_count,
            "distribution": distribution,
            "required_plan": required_plan,
            "enforce_exact_count": enforce_exact_count,
            "enforce_distribution": enforce_distribution,
        }
        return report

    def generate(
        self,
        task: BuildTask,
        chain: StateChainSample,
        prompt_path: Path,
        config: DatasetBuildConfig,
        raw_override: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], str]:
        template = self.load_prompt(prompt_path)
        base_prompt = self.build_user_prompt(template, task, chain, config)
        raw = raw_override if raw_override is not None else self._call_llm(base_prompt)
        last_errors: list[str] = []
        last_payload: Optional[dict[str, Any]] = None

        for validation_attempt in range(3):
            payload = self.normalize_payload(parse_json_object(raw), task, config, chain)
            last_payload = payload
            report = self.validate_question_set_payload(payload, chain.model_dump(), task, config)
            if report.passed:
                return payload["questions"], raw

            last_errors = [issue["message"] for issue in report.errors]
            if raw_override is not None or validation_attempt == 2:
                message = "; ".join(last_errors)
                raise QuestionGenerationValidationError(
                    f"generated questions failed validation: {message}",
                    raw_response=raw,
                    payload=last_payload,
                )

            retry_prompt = self.build_validation_retry_prompt(base_prompt, task, last_errors, previous_payload=payload)
            raw = self._call_llm(retry_prompt)

        message = "; ".join(last_errors)
        raise QuestionGenerationValidationError(
            f"generated questions failed validation: {message}",
            raw_response=raw,
            payload=last_payload,
        )
