"""Helpers for building narrative source bundles from catalog-level summary pages."""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from .narrative_summary import build_narrative_summary_source_bundle
from .schemas import SourceBundleRecord


SYSTEM_PROMPT = (
    "You are a careful benchmark narrative source-bundle builder. "
    "Return exactly one valid JSON object and no extra commentary."
)

WIKIPEDIA_TITLE_OVERRIDES: dict[str, str] = {
    "The Adventures of Huckleberry Finn": "Adventures_of_Huckleberry_Finn",
    "Emma": "Emma_(novel)",
    "Persuasion": "Persuasion_(novel)",
    "North and South": "North_and_South_(Gaskell_novel)",
    "Vanity Fair": "Vanity_Fair_(novel)",
    "Kim": "Kim_(novel)",
    "The Portrait of a Lady": "The_Portrait_of_a_Lady",
    "The Last of the Mohicans": "The_Last_of_the_Mohicans",
    "Villette": "Villette",
    "Great Expectations": "Great_Expectations",
    "Jane Eyre": "Jane_Eyre",
    "Bleak House": "Bleak_House",
    "Middlemarch": "Middlemarch",
    "Tess of the d'Urbervilles": "Tess_of_the_d%27Urbervilles",
    "Les Miserables": "Les_Mis%C3%A9rables",
    "The Last Man": "The_Last_Man_(novel)",
}

SPARKNOTES_SLUG_OVERRIDES: dict[str, str] = {
    "Pride and Prejudice": "pride",
    "Great Expectations": "greatex",
    "Jane Eyre": "janeeyre",
    "Wuthering Heights": "wuthering",
    "Sense and Sensibility": "sensibility",
    "Emma": "emma",
    "David Copperfield": "davidcopperfield",
    "The Moonstone": "moonstone",
    "Far from the Madding Crowd": "maddingcrowd",
    "The Return of the Native": "returnofnative",
    "The Mill on the Floss": "millonthefloss",
    "Frankenstein": "frankenstein",
    "Strange Case of Dr Jekyll and Mr Hyde": "jekyllhyde",
    "A Tale of Two Cities": "twocities",
    "The Scarlet Letter": "scarletletter",
    "Treasure Island": "treasureisland",
    "Silas Marner": "silasmarner",
    "Dracula": "dracula",
    "Middlemarch": "middlemarch",
    "Tess of the d'Urbervilles": "tess",
    "North and South": "northandsouth",
    "Persuasion": "persuasion",
    "The Mayor of Casterbridge": "mayorofcasterbridge",
    "Bleak House": "bleakhouse",
    "Villette": "villette",
    "Vanity Fair": "vanityfair",
    "Kim": "kim",
    "The Portrait of a Lady": "portraitofalady",
    "The Last of the Mohicans": "mohicans",
}

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style|noscript)\b.*?>.*?</\1>")
COMMENT_RE = re.compile(r"(?is)<!--.*?-->")
WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
REF_TAG_RE = re.compile(r"(?is)<ref\b[^>/]*/>|<ref\b.*?>.*?</ref>")
WIKIPEDIA_HEADING_RE = re.compile(r"(?m)^==+\s*(.+?)\s*==+\s*$")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
BELIEF_CENTRIC_FOCUS_RE = re.compile(
    r"\b(?:evolving|changing|mixed|growing)?\s*(?:belief|beliefs|feelings|thoughts|confusion|doubt|interpretation|understanding|opinion)\b"
    r"|(?:\bevolving\s+relationship\s+with\b|\brelationship\s+with\b)",
    re.IGNORECASE,
)
NEAR_DUPLICATE_ITEM_RE = re.compile(
    r"\b(?:still|again|further|continued|remained)\b.*\b(?:shocked|confused|uncertain|suspicious|doubtful)\b",
    re.IGNORECASE,
)


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
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("model output must be a JSON object")
    return data


def slugify_title_for_litcharts(title: str) -> str:
    """Create one best-effort LitCharts slug from a book title."""

    lowered = title.lower().replace("&", " and ")
    lowered = lowered.replace("'", "").replace("’", "")
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered).strip("-")
    return lowered


def wikipedia_title_slug(title: str) -> str:
    """Create a raw-wikitext Wikipedia page slug for one title."""

    slug = WIKIPEDIA_TITLE_OVERRIDES.get(title, title.replace(" ", "_"))
    if "%" in slug or "_" in slug:
        return slug
    return urllib.parse.quote(slug)


def candidate_summary_sources(title: str, preferred_provider: Optional[str] = None) -> list[dict[str, str]]:
    """Return ordered summary-source candidates for one title."""

    candidates: list[dict[str, str]] = []
    all_candidates: list[dict[str, str]] = []
    all_candidates.append(
        {
            "provider": "wikipedia_plot_section",
            "url": f"https://en.wikipedia.org/w/index.php?title={wikipedia_title_slug(title)}&action=raw",
        }
    )
    sparknotes_slug = SPARKNOTES_SLUG_OVERRIDES.get(title)
    if sparknotes_slug:
        all_candidates.append(
            {
                "provider": "sparknotes_full_book_summary",
                "url": f"https://www.sparknotes.com/lit/{sparknotes_slug}/summary/",
            }
        )
    litcharts_slug = slugify_title_for_litcharts(title)
    if litcharts_slug:
        all_candidates.append(
            {
                "provider": "litcharts_plot_summary",
                "url": f"https://www.litcharts.com/lit/{litcharts_slug}/summary",
            }
        )
    if preferred_provider:
        preferred = [candidate for candidate in all_candidates if candidate["provider"] == preferred_provider]
        remaining = [candidate for candidate in all_candidates if candidate["provider"] != preferred_provider]
        candidates.extend(preferred)
        candidates.extend(remaining)
    else:
        candidates.extend(all_candidates)
    return candidates


def fetch_url_text(url: str, timeout: int = 60, max_retries: int = 3, retry_delay: float = 2.0) -> str:
    """Fetch one public HTML page."""

    last_error: Exception | None = None
    for attempt in range(max_retries):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "BiTempQA-state-version-builder/0.1",
                "Accept-Language": "en-US,en;q=0.9",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f"unable to fetch {url}: {last_error}")


def html_to_readable_text(raw_html: str) -> str:
    """Convert one HTML page into a coarse readable text block."""

    text = COMMENT_RE.sub("\n", raw_html)
    text = SCRIPT_STYLE_RE.sub("\n", text)
    text = re.sub(r"(?i)</(p|div|section|article|main|li|h1|h2|h3|h4|h5|h6|br)>", "\n", text)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = "\n".join(line for line in text.split("\n") if line)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def trim_summary_text(provider: str, readable_text: str, max_chars: int = 28000) -> str:
    """Trim noisy page text into a more model-friendly summary source."""

    text = readable_text
    if provider == "sparknotes_full_book_summary":
        marker = "Full Book Summary"
        if marker in text:
            text = text.split(marker, 1)[1]
        tail_markers = ["Previous section", "Full Book Analysis", "Did you know you can highlight text"]
        for tail_marker in tail_markers:
            if tail_marker in text:
                text = text.split(tail_marker, 1)[0]
    elif provider == "litcharts_plot_summary":
        marker = "Summary"
        if marker in text:
            text = text.split(marker, 1)[1]
        tail_markers = ["Themes", "Characters", "Symbols"]
        for tail_marker in tail_markers:
            if tail_marker in text:
                text = text.split(tail_marker, 1)[0]

    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def _strip_simple_templates(text: str) -> str:
    """Best-effort removal of non-nested MediaWiki templates."""

    previous = None
    current = text
    while previous != current:
        previous = current
        current = re.sub(r"\{\{[^{}]*\}\}", "", current)
    return current


def _clean_wikipedia_wikitext(section: str, max_chars: int = 28000) -> str:
    """Convert a raw Wikipedia wikitext slice into readable prose."""

    section = REF_TAG_RE.sub(" ", section)
    section = COMMENT_RE.sub(" ", section)
    section = _strip_simple_templates(section)
    section = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", section)
    section = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", section)
    section = re.sub(r"<[^>]+>", " ", section)
    section = section.replace("'''", "").replace("''", "")
    section = re.sub(r"(?m)^\*+\s*", "", section)
    section = re.sub(r"(?m)^:+\s*", "", section)
    section = re.sub(r"\{\|.*?\|\}", " ", section, flags=re.S)
    section = html.unescape(section)
    section = "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in section.split("\n"))
    section = "\n".join(line for line in section.split("\n") if line)
    section = MULTI_NEWLINE_RE.sub("\n\n", section)
    section = section.strip()
    if len(section) > max_chars:
        section = section[:max_chars].rstrip() + "..."
    return section


def extract_wikipedia_plot_section(raw_wikitext: str, max_chars: int = 28000, min_chars: int = 800) -> str:
    """Extract a readable plot-style section from Wikipedia raw wikitext."""

    text = raw_wikitext.replace("\r\n", "\n").replace("\r", "\n")
    headings = list(WIKIPEDIA_HEADING_RE.finditer(text))
    start = None
    end = None
    accepted_keywords = ("plot", "synopsis", "summary")
    for index, match in enumerate(headings):
        heading_text = match.group(1).strip().lower()
        if any(keyword in heading_text for keyword in accepted_keywords):
            start = match.end()
            if index + 1 < len(headings):
                end = headings[index + 1].start()
            break
    if start is None:
        raise ValueError("Wikipedia raw page does not contain a recognizable plot section")

    section = _clean_wikipedia_wikitext(text[start:end].strip(), max_chars=max_chars)
    if len(section) < min_chars:
        raise ValueError("Wikipedia plot section is unexpectedly short after extraction")
    return section


def extract_wikipedia_fallback_summary(raw_wikitext: str, max_chars: int = 28000, min_chars: int = 800) -> str:
    """Fallback to a cleaned lead-plus-synopsis block when no strong plot section is available."""

    text = raw_wikitext.replace("\r\n", "\n").replace("\r", "\n")
    headings = list(WIKIPEDIA_HEADING_RE.finditer(text))
    stop_keywords = (
        "characters",
        "background",
        "themes",
        "style",
        "publication",
        "reception",
        "adaptation",
        "analysis",
        "legacy",
        "references",
        "external links",
        "see also",
    )
    end = None
    for match in headings:
        heading_text = match.group(1).strip().lower()
        if any(keyword in heading_text for keyword in stop_keywords):
            end = match.start()
            break
    section = _clean_wikipedia_wikitext(text[:end].strip(), max_chars=max_chars)
    if len(section) < min_chars:
        raise ValueError("Wikipedia fallback summary is unexpectedly short after extraction")
    return section


def resolve_summary_source_text(
    title: str,
    timeout: int = 60,
    preferred_provider: Optional[str] = None,
) -> dict[str, str]:
    """Fetch the first working summary source for one title."""

    errors: list[str] = []
    for candidate in candidate_summary_sources(title, preferred_provider=preferred_provider):
        try:
            raw_text = fetch_url_text(candidate["url"], timeout=timeout)
            if candidate["provider"] == "wikipedia_plot_section":
                try:
                    trimmed = extract_wikipedia_plot_section(raw_text)
                except Exception:
                    trimmed = extract_wikipedia_fallback_summary(raw_text)
            else:
                readable = html_to_readable_text(raw_text)
                trimmed = trim_summary_text(candidate["provider"], readable)
            if len(trimmed) < 900:
                raise ValueError("summary page text is unexpectedly short after trimming")
            return {
                "provider": candidate["provider"],
                "url": candidate["url"],
                "text": trimmed,
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate['provider']}::{candidate['url']}::{exc}")
    raise RuntimeError(f"unable to fetch a working summary source for {title}: {' | '.join(errors)}")


def validate_generated_chain_specs(payload: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    """Validate the model output for one novel-level chain set."""

    chains = payload.get("chains")
    if not isinstance(chains, list):
        raise ValueError("generated payload must contain a chains list")
    if len(chains) != expected_count:
        raise ValueError(f"generated chains count must equal {expected_count}, got {len(chains)}")

    normalized: list[dict[str, Any]] = []
    normalized_focus_events: list[str] = []
    for index, chain in enumerate(chains, start=1):
        if not isinstance(chain, dict):
            raise ValueError("each generated chain spec must be an object")
        focus_event = str(chain.get("focus_event", "")).strip()
        source_title = str(chain.get("source_title", "")).strip()
        bundle_summary = str(chain.get("bundle_summary", "")).strip()
        items = chain.get("items")
        if not focus_event or not source_title or not bundle_summary:
            raise ValueError("focus_event, source_title, and bundle_summary must not be blank")
        if BELIEF_CENTRIC_FOCUS_RE.search(focus_event):
            raise ValueError(
                f"focus_event looks belief-centric or too internal-state-focused: {focus_event!r}. "
                "Choose one externally anchored plot event instead."
            )
        if not isinstance(items, list) or not 6 <= len(items) <= 10:
            raise ValueError("each generated chain spec must contain 6 to 10 items")

        chain_notes = [str(note).strip() for note in chain.get("notes", []) if str(note).strip()]
        normalized_items: list[dict[str, str]] = []
        item_signatures: set[str] = set()
        for item_index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise ValueError("each source-bundle item must be an object")
            title_value = str(item.get("title", "")).strip()
            summary = str(item.get("summary", "")).strip()
            time_hint = str(item.get("time_hint", "")).strip()
            artifact_ref = str(item.get("artifact_ref", "")).strip() or f"summary_item_{index:02d}_{item_index:02d}"
            if not title_value or not summary:
                raise ValueError("every source-bundle item must have non-blank title and summary")
            signature = " ".join(WORD_RE.findall(summary.lower())[:12])
            if signature in item_signatures and signature:
                raise ValueError("items within the same chain are too similar; avoid near-duplicate summary slices")
            item_signatures.add(signature)
            if NEAR_DUPLICATE_ITEM_RE.search(summary):
                raise ValueError(
                    "items look like repeated inner-state paraphrases; use adjacent but distinct subplot material instead"
                )
            normalized_items.append(
                {
                    "artifact_type": "summary",
                    "artifact_ref": artifact_ref,
                    "title": title_value,
                    "time_hint": time_hint or None,
                    "summary": summary,
                }
            )

        normalized.append(
            {
                "focus_event": focus_event,
                "source_title": source_title,
                "bundle_summary": bundle_summary,
                "notes": chain_notes,
                "items": normalized_items,
            }
        )
        normalized_focus_events.append(focus_event.lower())

    if len(set(normalized_focus_events)) != len(normalized_focus_events):
        raise ValueError("different chains from the same novel must use distinct focus_event values")
    return normalized


class NarrativeCatalogBundleGenerator:
    """Generate one set of narrative source bundles from a trusted summary page."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        prompt_path: Path,
        temperature: float = 0.0,
        max_tokens: int = 12000,
        timeout: int = 180,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        use_json_mode: bool = True,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.prompt_path = prompt_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_json_mode = use_json_mode

    def load_prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")

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
        raise RuntimeError("LLM call failed without a captured exception")

    def build_prompt(
        self,
        *,
        title: str,
        author: str,
        target_chain_count: int,
        notes: str,
        source_provider: str,
        source_url: str,
        summary_text: str,
        existing_chain_specs: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        payload = {
            "title": title,
            "author": author,
            "target_chain_count": target_chain_count,
            "builder_notes": notes,
            "source_provider": source_provider,
            "source_url": source_url,
        }
        prompt = (
            f"{self.load_prompt()}\n\n"
            "## Fixed task block\n"
            f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n\n"
            "## Trusted summary source text\n"
            "Treat the following as the only evidence source for constructing source-bundle items. "
            "Do not invent events that are not supported here.\n"
            f"```text\n{summary_text}\n```"
        )
        if existing_chain_specs:
            prompt += (
                "\n\n## Already selected chains for this novel\n"
                "Do not repeat these focal events, source_title phrasings, or near-duplicate event families. "
                "Choose a materially different externally anchored focal event.\n"
                f"```json\n{json.dumps(existing_chain_specs, ensure_ascii=False, indent=2)}\n```"
            )
        return prompt

    def build_retry_prompt(
        self,
        *,
        base_prompt: str,
        previous_raw: str,
        validation_error: str,
        target_chain_count: int,
    ) -> str:
        """Create a targeted retry prompt after one semantic validation failure."""

        return (
            f"{base_prompt}\n\n"
            "## Repair instructions\n"
            f"Your previous output failed validation for this reason: {validation_error}\n"
            f"You must return exactly {target_chain_count} chain objects inside the top-level `chains` list.\n"
            "Do not collapse multiple focal events into one chain.\n"
            "Do not omit chains.\n"
            "Every returned chain must be materially distinct from the others.\n"
            "Every returned chain must contain 6 to 10 items.\n"
            "Avoid belief-centric focal events such as 'X's evolving belief about Y' when the summary supports a more concrete external plot event.\n"
            "If an earlier draft used distractor-ready material, make it adjacent but distinct from the main arc rather than repeating the same shock, suspicion, or realization in several items.\n"
            "If the trusted summary does not support distinct neighboring subplot material, do not fabricate distractors through paraphrases of the main arc. Preserve the main arc with all-core items instead.\n"
            "Return strict JSON only.\n\n"
            "## Previous invalid output\n"
            f"```json\n{previous_raw}\n```"
        )

    def generate_chain_specs(
        self,
        *,
        title: str,
        author: str,
        target_chain_count: int,
        notes: str,
        source_provider: str,
        source_url: str,
        summary_text: str,
    ) -> tuple[list[dict[str, Any]], str]:
        collected_specs: list[dict[str, Any]] = []
        raw_outputs: list[str] = []

        for _ in range(target_chain_count):
            base_prompt = self.build_prompt(
                title=title,
                author=author,
                target_chain_count=1,
                notes=notes,
                source_provider=source_provider,
                source_url=source_url,
                summary_text=summary_text,
                existing_chain_specs=collected_specs,
            )
            prompt = base_prompt
            last_error: Optional[Exception] = None
            last_raw = ""
            for attempt in range(self.max_retries):
                raw = self._call_llm(prompt)
                last_raw = raw
                raw_outputs.append(raw)
                try:
                    payload = parse_json_object(raw)
                    generated = validate_generated_chain_specs(payload, 1)
                    candidate = generated[0]
                    combined = validate_generated_chain_specs({"chains": [*collected_specs, candidate]}, len(collected_specs) + 1)
                    collected_specs = combined
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.max_retries - 1:
                        raise
                    prompt = self.build_retry_prompt(
                        base_prompt=base_prompt,
                        previous_raw=raw,
                        validation_error=str(exc),
                        target_chain_count=1,
                    )
                    time.sleep(self.retry_delay * (attempt + 1))
            else:
                raise RuntimeError(f"failed to generate valid chain specs: {last_error}; last_raw={last_raw[:800]}")

        return collected_specs, "\n\n".join(raw_outputs)


def chain_specs_to_source_bundles(
    *,
    sample_task_pairs: list[tuple[str, str]],
    language: str,
    title: str,
    author: str,
    source_provider: str,
    source_url: str,
    chain_specs: list[dict[str, Any]],
) -> list[SourceBundleRecord]:
    """Convert validated chain specs into SourceBundleRecord objects."""

    if len(sample_task_pairs) != len(chain_specs):
        raise ValueError("sample_task_pairs and chain_specs must have the same length")

    bundles: list[SourceBundleRecord] = []
    for (sample_id, state_chain_id), spec in zip(sample_task_pairs, chain_specs):
        bundles.append(
            build_narrative_summary_source_bundle(
                sample_id=sample_id,
                state_chain_id=state_chain_id,
                language=language,
                focus_event=spec["focus_event"],
                source_title=spec["source_title"],
                title=title,
                author=author,
                source_url=source_url,
                source_provider=source_provider,
                bundle_summary=spec["bundle_summary"],
                notes=list(spec.get("notes", [])),
                item_specs=list(spec["items"]),
                source_kind="plot_summary",
            )
        )
    return bundles
