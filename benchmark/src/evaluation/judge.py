"""LLM-as-a-judge utilities."""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

CONTEXT_JUDGE_PROMPT = """You are a strict evaluator.
Decide whether the retrieved context supports the gold answer.

Question:
{question}

Gold answer:
{ground_truth}

Retrieved context:
{retrieved_context}

Return exactly three lines:
Judgment: Correct or Wrong
Confidence: High, Medium, or Low
Reason: one short sentence"""

ANSWER_JUDGE_SYSTEM = """You are a strict answer judge for a memory-grounded benchmark.

Judge only whether the system answer is correct with respect to the question and gold answer.
Do not reward outside knowledge.
If the system answer goes beyond the gold answer but stays consistent with it, that can still be correct.
If the system answer is unsupported, empty, off-topic, or contradicts the gold answer, mark it wrong.

Return JSON only."""

ANSWER_JUDGE_PROMPT = """Decide whether the system answer is correct.

Question:
{question}

Gold answer:
{gold_answer}

System answer:
{generated_answer}

Judging rules:
- Use the gold answer as the reference.
- Do not infer correctness from outside knowledge.
- For time-related questions, equivalent dates or equivalent time phases count as correct.
- A longer answer is still correct if it preserves the same core state and does not contradict the gold answer.
- If the answer is empty, irrelevant, unsupported by the gold answer, or contradictory, mark it wrong.

Return JSON only with this schema:
{{"label":"CORRECT" or "WRONG","reason":"one short sentence"}}"""


def parse_context_judge_response(response: str) -> Tuple[bool, str, str]:
    if not response:
        return False, "low", "empty response"
    is_correct = False
    judge_match = re.search(r"Judgment[:\s]*(Correct|Wrong)", response, re.IGNORECASE)
    if judge_match:
        is_correct = judge_match.group(1).lower() == "correct"
    else:
        head = response[:50].lower()
        if "correct" in head and "wrong" not in head:
            is_correct = True
        elif "wrong" in head:
            is_correct = False
    confidence = "low"
    conf_match = re.search(r"Confidence[:\s]*(High|Medium|Low)", response, re.IGNORECASE)
    if conf_match:
        confidence = conf_match.group(1).lower()
    reason = ""
    reason_match = re.search(r"Reason[:\s]*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
    return is_correct, confidence, reason


def parse_answer_judge_response(response: str) -> Tuple[bool, str]:
    if not response:
        return False, "empty response"
    try:
        json_match = re.search(r'\{[^}]*"label"\s*:\s*"(CORRECT|WRONG)"[^}]*\}', response, re.IGNORECASE)
        if json_match:
            data = json.loads(json_match.group())
            label = str(data.get("label", "WRONG")).upper()
            return label == "CORRECT", str(data.get("reason", response[:200]))
    except json.JSONDecodeError:
        pass
    upper = response.upper()
    if "CORRECT" in upper and "WRONG" not in upper[:50]:
        return True, response[:200]
    if "WRONG" in upper:
        return False, response[:200]
    return False, response[:200]


class LLMJudge:
    """LLM-as-a-judge for QA evaluation."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "deepseek-v4-pro",
        temperature: float = 0.0,
        max_workers: int = 3,
        timeout: int = 60,
        cache_path: Optional[Path] = None,
        mode: str = "answer_judge",
        extra_body: Optional[dict[str, Any]] = None,
    ):
        http_client = httpx.Client(timeout=timeout, trust_env=False)
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, http_client=http_client)
        self.model = model
        self.temperature = temperature
        self.max_workers = max_workers
        self.mode = mode
        self.cache: Dict[str, dict] = {}
        self.cache_path = cache_path
        self.extra_body = extra_body or {}
        self._load_cache()

    def _cache_key(self, qa_id: str, system_name: str) -> str:
        return f"{qa_id}|||{system_name}"

    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                logger.info("Loaded %s cached judge results", len(self.cache))
            except Exception as exc:
                logger.warning("Failed to load judge cache: %s", exc)

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def judge_single(
        self,
        question: str,
        ground_truth: str,
        retrieved_context: str,
        qa_id: str,
        system_name: str,
    ) -> dict:
        key = self._cache_key(qa_id, system_name)
        if key in self.cache:
            return self.cache[key]
        prompt = CONTEXT_JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            retrieved_context=retrieved_context[:1000],
        )
        for attempt in range(3):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a strict and fair evaluator. Follow the required output format exactly."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 256,
                }
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                resp = self.client.chat.completions.create(**kwargs)
                response_text = resp.choices[0].message.content or ""
                is_correct, confidence, reason = parse_context_judge_response(response_text)
                result = {
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "reason": reason,
                    "raw_response": response_text,
                    "mode": "context_judge",
                    "system_prompt": "You are a strict and fair evaluator. Follow the required output format exactly.",
                    "user_prompt": prompt,
                }
                self.cache[key] = result
                return result
            except Exception as exc:
                logger.warning("Judge API error (attempt %s/3): %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        result = {
            "is_correct": False,
            "confidence": "low",
            "reason": "API error after 3 retries",
            "raw_response": "",
            "mode": "context_judge",
            "system_prompt": "You are a strict and fair evaluator. Follow the required output format exactly.",
            "user_prompt": prompt,
        }
        self.cache[key] = result
        return result

    def judge_answer(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
        qa_id: str,
        system_name: str,
    ) -> dict:
        key = self._cache_key(qa_id, system_name)
        if key in self.cache:
            return self.cache[key]
        prompt = ANSWER_JUDGE_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )
        for attempt in range(3):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": ANSWER_JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 256,
                }
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                resp = self.client.chat.completions.create(**kwargs)
                response_text = resp.choices[0].message.content or ""
                is_correct, reason = parse_answer_judge_response(response_text)
                result = {
                    "is_correct": is_correct,
                    "confidence": "high" if is_correct else "low",
                    "reason": reason,
                    "raw_response": response_text,
                    "mode": "answer_judge",
                    "system_prompt": ANSWER_JUDGE_SYSTEM,
                    "user_prompt": prompt,
                }
                self.cache[key] = result
                return result
            except Exception as exc:
                logger.warning("Answer judge API error (attempt %s/3): %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        result = {
            "is_correct": False,
            "confidence": "low",
            "reason": "API error after 3 retries",
            "raw_response": "",
            "mode": "answer_judge",
            "system_prompt": ANSWER_JUDGE_SYSTEM,
            "user_prompt": prompt,
        }
        self.cache[key] = result
        return result

    def judge_batch(self, items: list, progress_callback=None) -> Dict[str, dict]:
        to_judge = []
        for item in items:
            key = self._cache_key(item["qa_id"], item["system_name"])
            if key not in self.cache:
                to_judge.append(item)
        if not to_judge:
            logger.info("All items already cached")
            return self.cache

        completed = 0
        total = len(to_judge)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for item in to_judge:
                if "generated_answer" in item:
                    future = executor.submit(
                        self.judge_answer,
                        item["question"],
                        item["gold_answer"],
                        item["generated_answer"],
                        item["qa_id"],
                        item["system_name"],
                    )
                else:
                    future = executor.submit(
                        self.judge_single,
                        item["question"],
                        item["ground_truth"],
                        item["retrieved_context"],
                        item["qa_id"],
                        item["system_name"],
                    )
                futures[future] = item

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    item = futures[future]
                    key = self._cache_key(item["qa_id"], item["system_name"])
                    self.cache[key] = {
                        "is_correct": False,
                        "confidence": "low",
                        "reason": f"Thread error: {exc}",
                        "raw_response": "",
                        "mode": "unknown",
                        "system_prompt": "",
                        "user_prompt": "",
                    }
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        self._save_cache()
        return self.cache
