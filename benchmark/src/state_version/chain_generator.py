"""Independent state-chain generation utilities."""

from __future__ import annotations

from collections import Counter
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from openai import OpenAI

from .schemas import BuildTask, DatasetBuildConfig, SourceBundleRecord, StateChainSample
from .validator import validate_state_chain_payload_with_bundle

SYSTEM_PROMPT = (
    "You are a careful benchmark data builder. "
    "Return exactly one valid JSON object and no extra commentary."
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def strip_code_fences(raw: str) -> str:
    """Remove common markdown code fences around JSON output."""

    text = (raw or "").strip()
    if text.startswith("```json"):
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if text.startswith("```"):
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse one JSON object from the model output."""

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


class StateChainGenerator:
    """Generate one state-chain sample from one source bundle."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
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
    def build_user_prompt(
        template: str,
        task: BuildTask,
        bundle: SourceBundleRecord,
    ) -> str:
        task_block = {
            "sample_id": task.sample_id,
            "state_chain_id": task.state_chain_id,
            "domain": task.domain,
            "language": task.language,
            "source_kind": task.domain,
            "source_title": bundle.source_title,
            "focus_event": bundle.focus_event,
        }
        return (
            f"{template}\n\n"
            "## Task-specific fixed identifiers\n"
            "Use exactly these values in the output JSON for the corresponding fields:\n"
            f"```json\n{json.dumps(task_block, ensure_ascii=False, indent=2)}\n```\n\n"
            "## Input source_bundle\n"
            "Treat the following bundle as the only source of evidence. "
            "You may rewrite it into natural benchmark node texts, but do not invent unsupported facts.\n"
            "If a source_bundle_item contains time_hint, treat it as an optional cue for natural time wording in node text. "
            "Use it only when it fits naturally, such as a date, month, release period, chapter stage, or relative temporal phrase. "
            "Do not copy the literal field name time_hint into text or force time wording into every node.\n"
            "Every node.source_pointer.artifact_type and node.source_pointer.artifact_ref must exactly match one input "
            "source_bundle_item. Never invent synthetic discussions, pull requests, review comments, or artifact ids.\n"
            f"```json\n{json.dumps(bundle.model_dump(), ensure_ascii=False, indent=2)}\n```"
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
                    # Fallback for providers that reject response_format=json_object.
                    self.use_json_mode = False
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise last_error
        raise RuntimeError("LLM call failed without a captured exception")

    @staticmethod
    def build_validation_retry_prompt(
        base_prompt: str,
        error_messages: list[str],
        previous_payload: Optional[dict[str, Any]] = None,
    ) -> str:
        """Append structured correction instructions after a failed validation pass."""

        guidance = [
            "## Regeneration request",
            "Your previous draft failed schema or hard-limit validation.",
            "Regenerate the entire JSON object from scratch and fix every error below.",
            "Validation errors:",
        ]
        guidance.extend(f"- {message}" for message in error_messages)
        guidance.extend(
            [
                "",
                "Important fixes:",
                "1. Every English node text must meet the hard minimum length requirement.",
                "2. Rewrite every short English node to safely exceed the minimum; aim for roughly 50 to 120 words per node unless a node genuinely needs more.",
                "3. Expand short nodes into grounded, self-contained natural language with enough context.",
                "4. If many nodes are too short, merge nearby artifacts into fewer but richer nodes instead of keeping many shallow nodes.",
                "5. Preserve the same focal event and keep the fixed identifiers exactly valid.",
                "6. Keep distractor competition, but do not waste nodes on tiny discussion fragments.",
                "7. Every source_pointer must reuse an artifact_type/artifact_ref pair that already exists in the input source_bundle.",
                "8. Do not invent synthetic source artifacts such as made-up discussion ids, review comments, or pull requests.",
                "9. Visible text must not contain raw source identifiers such as issue numbers, pull request numbers, comment ids, commit hashes, or URLs. Keep those only in source_pointer.",
                "10. Distractor nodes should usually come from adjacent but distinct related subthreads, not from near-duplicate rewrites of the same main-event state.",
                "11. Do not create distractors by merely rephrasing a core node from the same source artifact with an explains-style gloss.",
                "12. At least max(2, ceil(0.2 * node_count)) chain_nodes must literally use salience_label='distractor'. Do not return an all-core or nearly-all-core chain.",
                "13. Distractor nodes should be different event lines that reuse the same entities, components, or setting words, not hidden extra steps of the same main event.",
                "14. Keep node text simple: one main fact plus one short qualifier is enough.",
                "15. For GitHub bundles, preferred distractor sources include workaround comments, build or install failure reports, review-status signals, issue-closure signals, rejected alternative paths, and neighboring compatibility or packaging branches.",
                "16. If the current focal event cannot support at least two real distractors from nearby but distinct event lines, switch to a different focal event inside the same bundle rather than returning a clean chain.",
                "17. If the input bundle does not contain enough adjacent-but-distinct distractor-ready material to satisfy that quota, treat the event as unsuitable rather than fabricating paraphrastic distractors.",
                "18. Do not choose a vague belief-only or confusion-only focal event when the source supports a more concrete externally anchored event arc.",
                "19. If reliable dates, release periods, chapter stages, or relative time hints are present in the source bundle, you may preserve some of them naturally in node text, but only where they fit cleanly.",
                "20. Do not end the chain with a summary-style distractor by default. The final node should usually be the last core state.",
                "21. Before answering, privately count distractor nodes and make sure there are at least two from distinct nearby branches or artifacts.",
            ]
        )
        semantic_hints = StateChainGenerator.build_semantic_retry_guidance(previous_payload)
        if semantic_hints:
            guidance.extend(["", "Quality fixes for generic or weak chains:"])
            guidance.extend(f"- {hint}" for hint in semantic_hints)
        prompt = f"{base_prompt}\n\n" + "\n".join(guidance)
        if previous_payload is not None:
            prompt += (
                "\n\n## Previous draft to repair\n"
                "Use this draft as a repair target. You may keep or merge nodes, but the replacement must satisfy the hard minimum length requirement.\n"
                f"```json\n{json.dumps(previous_payload, ensure_ascii=False, indent=2)}\n```"
            )
        return prompt

    @staticmethod
    def build_semantic_retry_guidance(previous_payload: Optional[dict[str, Any]]) -> list[str]:
        """Derive extra regeneration hints from a weak previous chain draft."""

        if previous_payload is None:
            return []

        nodes = previous_payload.get("chain_nodes")
        if not isinstance(nodes, list):
            return []

        hints: list[str] = []
        distractor_count = 0
        source_counter: Counter[str] = Counter()
        signature_counter: Counter[str] = Counter()

        for node in nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("salience_label")) == "distractor":
                distractor_count += 1
            pointer = node.get("source_pointer")
            if isinstance(pointer, dict):
                source_key = f"{pointer.get('artifact_type')}::{pointer.get('artifact_ref')}"
                source_counter[source_key] += 1

            text = str(node.get("text", "")).strip().lower()
            words = WORD_RE.findall(text)
            if words:
                signature_counter[" ".join(words[:10])] += 1

        if len(nodes) >= 8 and distractor_count < 2:
            hints.append(
                "Preserve the complete main chain, but explicitly keep at least max(2, ceil(0.2 * node_count)) realistic distractor nodes tied to the same focal event."
            )

        heavily_reused_sources = [key for key, count in source_counter.items() if count >= 3 and key]
        if heavily_reused_sources:
            hints.append(
                "Do not keep reusing the same source artifact for many near-duplicate nodes. Merge shallow restatements and retain only clearly separable state slices."
            )

        repeated_signatures = [sig for sig, count in signature_counter.items() if count >= 2 and sig]
        if repeated_signatures:
            hints.append(
                "Several nodes look like paraphrases of the same broad state. Rewrite or merge them so each remaining node marks a retrieval-relevant state change."
            )

        if distractor_count > 0:
            hints.append(
                "If distractors are currently too entangled with the main arc, replace some of them with different event lines that reuse the same entities or vocabulary but are not themselves main-event states."
            )
        else:
            hints.append(
                "Your previous draft had no usable distractor nodes. Explicitly convert at least two adjacent-but-distinct source artifacts into distractors instead of labeling every node as core."
            )
        if distractor_count > 0 and all(
            isinstance(node, dict) and str(node.get("relation_label")) == "explains"
            for node in nodes
            if isinstance(node, dict) and str(node.get("salience_label")) == "distractor"
        ):
            hints.append(
                "Do not keep all distractors as explanation-style paraphrases. Use distinct neighboring subthreads, or reduce distractors instead of fabricating them."
            )

        domain = str(previous_payload.get("domain", "")).strip()
        if domain == "github_evolution":
            hints.append(
                "For GitHub chains, good distractors often come from workaround comments, failed reproduction notes, build or packaging friction, review-commented states, or closure signals that sound final before the event is truly settled."
            )
        else:
            hints.append(
                "For narrative chains, good distractors should be separate concrete event lines involving the same people, place, or object words, such as a different battle action, search action, travel action, letter action, household action, negotiation, or rescue branch near the same timeline."
            )
        hints.append(
            "When reliable time hints exist in the source bundle, you may carry some of them into node text naturally so the temporal state progression is easier to recover, but do not force them into every node."
        )
        hints.append(
            "If a node is carrying too many names or details, split or simplify it so one node mainly expresses one event fact and one short qualifier."
        )
        hints.append(
            "For narrative chains, avoid focal events whose versions all feel too similar. If most core nodes currently read like general process narration, switch to a richer focal event or rewrite each core node so the current event status, control, or meaning is more clearly different from the previous one."
        )
        hints.append(
            "For narrative chains, most core nodes should contain some natural temporal anchor when the source supports it, such as a repeated-round number, chapter stage, relative time phrase, or before/after cue."
        )

        return hints

    @staticmethod
    def normalize_sample(
        payload: dict[str, Any],
        task: BuildTask,
        bundle: SourceBundleRecord,
    ) -> dict[str, Any]:
        payload["sample_id"] = task.sample_id
        payload["state_chain_id"] = task.state_chain_id
        payload["domain"] = task.domain
        payload["language"] = task.language
        payload["source_kind"] = task.domain
        payload["source_title"] = bundle.source_title
        if not payload.get("focus_event"):
            payload["focus_event"] = bundle.focus_event
        if payload.get("chain_profile", {}).get("node_count") is None and "chain_nodes" in payload:
            payload.setdefault("chain_profile", {})["node_count"] = len(payload.get("chain_nodes", []))
        return payload

    def generate(
        self,
        task: BuildTask,
        bundle: SourceBundleRecord,
        prompt_path: Path,
        config: DatasetBuildConfig,
        raw_override: Optional[str] = None,
    ) -> tuple[StateChainSample, str]:
        template = self.load_prompt(prompt_path)
        base_prompt = self.build_user_prompt(template, task, bundle)
        raw = raw_override if raw_override is not None else self._call_llm(base_prompt)
        last_errors: list[str] = []

        for validation_attempt in range(3):
            payload = self.normalize_sample(parse_json_object(raw), task, bundle)
            report = validate_state_chain_payload_with_bundle(payload, bundle.model_dump(), config)
            if report.passed:
                return StateChainSample(**payload), raw

            last_errors = [issue["message"] for issue in report.errors]
            if raw_override is not None or validation_attempt == 2:
                message = "; ".join(last_errors)
                raise ValueError(f"generated state chain failed validation: {message}")

            retry_prompt = self.build_validation_retry_prompt(base_prompt, last_errors, previous_payload=payload)
            raw = self._call_llm(retry_prompt)

        message = "; ".join(last_errors)
        raise ValueError(f"generated state chain failed validation: {message}")
