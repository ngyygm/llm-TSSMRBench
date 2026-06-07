"""LLM-based answer generation from retrieved context."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

MC_SYSTEM_PROMPT = """You are a strict evaluation assistant for a memory benchmark.

You must answer using only the retrieved memory.
You must not use outside knowledge, background knowledge, guessing, or prior world knowledge.
Your job is to choose the best-supported answer from the retrieved memory.

Output rule:
- Return exactly one uppercase option letter: A, B, C, D, or E.
- Do not output any explanation, punctuation, JSON, markdown, or extra text.
- If the retrieved memory contains relevant evidence for one option, choose that option actively.
- Return E only when the retrieved memory is completely irrelevant or contains no answer-bearing evidence for any option."""

MC_USER_PROMPT = """You may answer only from the retrieved memory below.
If the retrieved memory contains relevant evidence for one option, choose the best-supported option.
Return E only if the retrieved memory is completely irrelevant or contains no answer-bearing evidence.

Retrieved memory:
{retrieved_context}

Question:
{question}

Options:
{choices_text}

Return exactly one uppercase letter: A, B, C, D, or E."""

BOOLEAN_SYSTEM_PROMPT = """You are a strict evaluation assistant for a memory benchmark.

You must answer using only the retrieved memory.
You must not use outside knowledge, background knowledge, guessing, or prior world knowledge.
You must decide whether the retrieved memory alone is sufficient.

Output rule:
- Return exactly one uppercase option letter: A, B, or C.
- A = Yes
- B = No
- C = The retrieved memory is insufficient to make a judgment
- Do not output any explanation, punctuation, JSON, markdown, or extra text."""

BOOLEAN_USER_PROMPT = """You may answer only from the retrieved memory below.
If the retrieved memory does not contain enough support to answer Yes or No, you must return C.

Retrieved memory:
{retrieved_context}

Question:
{question}

Return exactly one uppercase letter: A, B, or C."""

ABSTRACTIVE_SYSTEM_PROMPT = """You are a strict evaluation assistant for a memory benchmark.

You must answer using only the retrieved memory.
You must not use outside knowledge, background knowledge, guessing, or prior world knowledge.
You must answer positively whenever the retrieved memory contains relevant answer-bearing evidence.

Output rule:
- If the retrieved memory is insufficient, return exactly:
Insufficient information to support reasoning.
- Otherwise answer in plain English using only retrieved-memory evidence.
- The answer must be concise and between 10 and 100 words.
- Do not be over-conservative: if the retrieved memory is relevant and supports an answer, answer directly.
- Return the insufficiency sentence only when the retrieved memory is completely irrelevant or lacks any answer-bearing evidence.
- Do not mention the benchmark, hidden nodes, retrieval process, or your reasoning process."""

ABSTRACTIVE_USER_PROMPT = """You may answer only from the retrieved memory below.
If the retrieved memory is relevant and contains answer-bearing evidence, answer directly from it.
Return exactly the sentence below only if the retrieved memory is completely irrelevant or lacks any answer-bearing evidence:
Insufficient information to support reasoning.

Retrieved memory:
{retrieved_context}

Question:
{question}"""

STRICT_MC_FALLBACK = "Return only one uppercase letter from {letters}. No explanation."
STRICT_BOOLEAN_FALLBACK = "Return only one uppercase letter: A, B, or C. No explanation."
STRICT_ABSTRACTIVE_FALLBACK = (
    "Use only the retrieved memory. If insufficient, return exactly 'Insufficient information to support reasoning.' "
    "Otherwise answer in 10 to 100 words with no extra commentary."
)


@dataclass
class AnswerGenerationResult:
    answer: str
    raw_response: str
    error: Optional[str]
    system_prompt: str
    user_prompt: str
    prompt_mode: str


def _format_choices(choices: List[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [f"{labels[i]}. {choice}" for i, choice in enumerate(choices) if i < len(labels)]
    lines.append(f"{labels[len(choices)]}. Insufficient information")
    return "\n".join(lines)


def parse_mc_answer(response: str, num_choices: int, allow_abstain: bool = False) -> Optional[int]:
    if not response:
        return None
    label = response.strip().upper()
    if len(label) == 1 and "A" <= label <= "Z":
        idx = ord(label) - ord("A")
    else:
        match = re.search(r"\b([A-Z])\b", label)
        idx = ord(match.group(1)) - ord("A") if match else -1
    max_choices = num_choices + 1 if allow_abstain else num_choices
    if 0 <= idx < max_choices:
        return idx
    logger.warning("Could not parse strict MC answer from: %s", response[:100])
    return None


def _normalize_choice_text(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", (text or "").strip().lower())


def parse_mc_answer_with_choices(
    response: str,
    choices: List[str],
    allow_abstain: bool = False,
) -> Optional[int]:
    idx = parse_mc_answer(response, len(choices), allow_abstain=allow_abstain)
    if idx is not None:
        return idx
    normalized = _normalize_choice_text(response)
    if not normalized:
        return None
    matches: list[int] = []
    for i, choice in enumerate(choices):
        normalized_choice = _normalize_choice_text(choice)
        if not normalized_choice:
            continue
        if normalized == normalized_choice:
            return i
        if len(normalized) >= 4 and (
            normalized in normalized_choice or normalized_choice in normalized
        ):
            matches.append(i)
    if len(matches) == 1:
        return matches[0]
    if allow_abstain and normalized in {
        "e",
        "insufficient",
        "insufficientinformation",
        "insufficientinformationtosupportreasoning",
    }:
        return len(choices)
    logger.warning("Could not parse MC answer against choices from: %s", response[:100])
    return None


class AnswerGenerator:
    """Generates answers from retrieved context using one shared LLM."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "deepseek-v4-pro",
        temperature: float = 0.0,
        timeout: int = 60,
        extra_body: Optional[dict[str, Any]] = None,
    ):
        http_client = httpx.Client(timeout=timeout, trust_env=False)
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, http_client=http_client)
        self.model = model
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.last_error: Optional[str] = None
        self.last_raw_response: str = ""
        self.last_prompt: dict[str, str] = {}

    def generate(
        self,
        question: str,
        retrieved_context: str,
        choices: Optional[List[str]] = None,
        answer_type: Optional[str] = None,
    ) -> str:
        return self.generate_detailed(
            question=question,
            retrieved_context=retrieved_context,
            choices=choices,
            answer_type=answer_type,
        ).answer

    def generate_detailed(
        self,
        question: str,
        retrieved_context: str,
        choices: Optional[List[str]] = None,
        answer_type: Optional[str] = None,
    ) -> AnswerGenerationResult:
        self.last_error = None
        self.last_raw_response = ""
        self.last_prompt = {}

        is_mc = bool(choices)
        is_boolean = answer_type == "boolean"

        if is_mc:
            system_prompt = MC_SYSTEM_PROMPT
            user_prompt = MC_USER_PROMPT.format(
                retrieved_context=retrieved_context,
                question=question,
                choices_text=_format_choices(choices or []),
            )
            raw, used_prompt = self._generate_with_retry(
                system_prompt,
                user_prompt,
                max_tokens=16,
                parse_validator=lambda text: parse_mc_answer_with_choices(text, choices or [], allow_abstain=True) is not None,
                fallback_user_prompt=(
                    f"{user_prompt}\n\n"
                    + STRICT_MC_FALLBACK.format(
                        letters="/".join(chr(ord("A") + i) for i in range(len(choices or []) + 1))
                    )
                ),
            )
            answer = self._extract_mc_letter(raw, choices or [], allow_abstain=True)
            result = AnswerGenerationResult(
                answer=answer,
                raw_response=raw,
                error=self.last_error,
                system_prompt=system_prompt,
                user_prompt=used_prompt,
                prompt_mode="multiple_choice",
            )
            self.last_raw_response = raw
            self.last_prompt = {
                "system_prompt": system_prompt,
                "user_prompt": used_prompt,
                "prompt_mode": "multiple_choice",
            }
            return result

        if is_boolean:
            system_prompt = BOOLEAN_SYSTEM_PROMPT
            user_prompt = BOOLEAN_USER_PROMPT.format(
                retrieved_context=retrieved_context,
                question=question,
            )
            raw, used_prompt = self._generate_with_retry(
                system_prompt,
                user_prompt,
                max_tokens=16,
                parse_validator=lambda text: self._extract_boolean_label(text) in {"A", "B", "C"},
                fallback_user_prompt=f"{user_prompt}\n\n{STRICT_BOOLEAN_FALLBACK}",
            )
            answer = self._extract_boolean_label(raw)
            result = AnswerGenerationResult(
                answer=answer,
                raw_response=raw,
                error=self.last_error,
                system_prompt=system_prompt,
                user_prompt=used_prompt,
                prompt_mode="boolean",
            )
            self.last_raw_response = raw
            self.last_prompt = {
                "system_prompt": system_prompt,
                "user_prompt": used_prompt,
                "prompt_mode": "boolean",
            }
            return result

        system_prompt = ABSTRACTIVE_SYSTEM_PROMPT
        user_prompt = ABSTRACTIVE_USER_PROMPT.format(
            retrieved_context=retrieved_context,
            question=question,
        )
        raw, used_prompt = self._generate_with_retry(
            system_prompt,
            user_prompt,
            max_tokens=180,
            parse_validator=self._validate_abstractive_answer,
            fallback_user_prompt=f"{user_prompt}\n\n{STRICT_ABSTRACTIVE_FALLBACK}",
        )
        answer = raw.strip()
        result = AnswerGenerationResult(
            answer=answer,
            raw_response=raw,
            error=self.last_error,
            system_prompt=system_prompt,
            user_prompt=used_prompt,
            prompt_mode="abstractive",
        )
        self.last_raw_response = raw
        self.last_prompt = {
            "system_prompt": system_prompt,
            "user_prompt": used_prompt,
            "prompt_mode": "abstractive",
        }
        return result

    def _generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
        parse_validator=None,
        fallback_user_prompt: Optional[str] = None,
    ) -> tuple[str, str]:
        for attempt in range(3):
            try:
                prompt = fallback_user_prompt if (attempt >= 1 and fallback_user_prompt) else user_prompt
                raw = self._call_llm(system_prompt, prompt, max_tokens=max_tokens).strip()
                if not raw:
                    self.last_error = "empty_response"
                    logger.warning("Answer generation returned empty content (attempt %s/3)", attempt + 1)
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    return "", prompt
                if parse_validator is not None and not parse_validator(raw):
                    self.last_error = f"unparseable_response: {raw[:120]}"
                    logger.warning(
                        "Answer generation returned unparseable content (attempt %s/3): %s",
                        attempt + 1,
                        raw[:120],
                    )
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                self.last_error = None
                return raw, prompt
            except Exception as exc:
                self.last_error = str(exc)
                logger.warning("Answer generation error (attempt %s/3): %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return "", fallback_user_prompt or user_prompt

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
        max_tokens: int = 256,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _extract_mc_letter(self, raw: str, choices: List[str], allow_abstain: bool = False) -> str:
        idx = parse_mc_answer_with_choices(raw, choices, allow_abstain=allow_abstain)
        return chr(ord("A") + idx) if idx is not None else ""

    @staticmethod
    def _extract_boolean_label(raw: str) -> str:
        lowered = (raw or "").strip().lower()
        if len(lowered) == 1 and lowered in {"a", "b", "c"}:
            return lowered.upper()
        match = re.search(r"\b([abc])\b", lowered)
        if match:
            return match.group(1).upper()
        if lowered in {"a", "yes"}:
            return "A"
        if lowered in {"b", "no"}:
            return "B"
        if lowered in {
            "c",
            "insufficient",
            "the retrieved memory is insufficient to make a judgment",
            "the retrieved memory is insufficient to make a judgment.",
        }:
            return "C"
        return ""

    @staticmethod
    def _validate_abstractive_answer(raw: str) -> bool:
        text = (raw or "").strip()
        if not text:
            return False
        if text == "Insufficient information to support reasoning.":
            return True
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)
        return 10 <= len(words) <= 100

    def generate_mc(
        self,
        question: str,
        retrieved_context: str,
        choices: List[str],
    ) -> Optional[int]:
        raw = self.generate(question, retrieved_context, choices)
        return parse_mc_answer_with_choices(raw, choices, allow_abstain=True)
