"""Helpers for generating narrative full-text specs from novel catalogs."""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from .narrative_fulltext import ChapterBlock


SYSTEM_PROMPT = (
    "You are a careful benchmark narrative full-text spec builder. "
    "Return exactly one valid JSON object and no extra commentary."
)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def normalize_title(text: str) -> str:
    lowered = (text or "").lower()
    lowered = lowered.replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def strip_code_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```json"):
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if text.startswith("```"):
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = strip_code_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("model output must be a JSON object")
    return data


def pick_plain_text_url(formats: dict[str, str]) -> Optional[str]:
    preferences = [
        "text/plain; charset=utf-8",
        "text/plain; charset=us-ascii",
        "text/plain",
    ]
    for key in preferences:
        value = formats.get(key)
        if value:
            return value
    for key, value in formats.items():
        if key.startswith("text/plain"):
            return value
    return None


def resolve_gutendex_metadata(title: str, author: str, timeout: int = 60) -> dict[str, Any]:
    def _search(query_text: str) -> list[dict[str, Any]]:
        query = urllib.parse.quote(query_text)
        url = f"https://gutendex.com/books?languages=en&copyright=false&mime_type=text/plain&search={query}"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "BiTempQA-state-version-builder/0.1",
                "Accept-Language": "en-US,en;q=0.9",
            },
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("results", []) or []

    last_error: Exception | None = None
    results: list[dict[str, Any]] = []
    for query_text in [f"{title} {author}", title]:
        try:
            results = _search(query_text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        if results:
            break
    if not results:
        if last_error is not None:
            raise last_error
        raise ValueError(f"no Gutendex match for {title} by {author}")

    normalized_title = normalize_title(title)
    normalized_author = normalize_title(author)

    def score(result: dict[str, Any]) -> tuple[int, int, int]:
        result_title = normalize_title(str(result.get("title", "")))
        author_names = [normalize_title(str(item.get("name", ""))) for item in result.get("authors", [])]
        title_exact = int(result_title == normalized_title)
        title_contains = int(normalized_title in result_title or result_title in normalized_title)
        author_match = int(any(normalized_author in name or name in normalized_author for name in author_names))
        return (title_exact, title_contains, author_match)

    ranked = sorted(results, key=lambda item: (*score(item), int(item.get("download_count", 0))), reverse=True)
    best = ranked[0]
    download_url = pick_plain_text_url(best.get("formats", {}) or {})
    if not download_url:
        raise ValueError(f"no plain-text download URL available for {title}")
    return {
        "ebook_id": int(best["id"]),
        "title": str(best["title"]),
        "author": ", ".join(str(item.get("name", "")) for item in best.get("authors", [])) or author,
        "source_page_url": f"https://www.gutenberg.org/ebooks/{best['id']}",
        "download_url": download_url,
        "download_count": int(best.get("download_count", 0)),
    }


def slugify_filename(title: str, ebook_id: int) -> str:
    lowered = title.lower().replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    lowered = re.sub(r"_+", "_", lowered)
    return f"{lowered}_pg{ebook_id}.txt"


def build_chapter_previews(chapters: list[ChapterBlock], max_words: int = 70, max_chapters: int = 80) -> str:
    selected = chapters[:max_chapters]
    lines: list[str] = []
    for chapter in selected:
        words = WORD_RE.findall(chapter.text)
        preview = " ".join(words[:max_words])
        lines.append(f"[Chapter {chapter.index}] {chapter.heading}\n{preview}")
    return "\n\n".join(lines).strip()


def validate_generated_fulltext_spec(payload: dict[str, Any], chapter_count: int) -> dict[str, Any]:
    required_top = ["focus_event", "source_title", "bundle_summary", "windows"]
    for key in required_top:
        if key not in payload:
            raise ValueError(f"missing required field: {key}")
    if not isinstance(payload["windows"], list) or not (6 <= len(payload["windows"]) <= 8):
        raise ValueError("windows must be a list with 6 to 8 entries")

    if "notes" not in payload or payload["notes"] is None:
        payload["notes"] = []
    if not isinstance(payload["notes"], list):
        raise ValueError("notes must be a list")

    for index, window in enumerate(payload["windows"], start=1):
        for key in ["artifact_ref", "title", "chapter_start", "chapter_end", "time_hint", "anchor_terms"]:
            if key not in window:
                raise ValueError(f"window {index} missing required field: {key}")
        if window.get("artifact_type") in {None, ""}:
            window["artifact_type"] = "chapter_window"
        if int(window["chapter_start"]) < 1 or int(window["chapter_end"]) < int(window["chapter_start"]):
            raise ValueError(f"window {index} has invalid chapter range")
        if int(window["chapter_end"]) > chapter_count:
            raise ValueError(f"window {index} exceeds chapter count {chapter_count}")
        anchor_terms = window.get("anchor_terms")
        if not isinstance(anchor_terms, list) or len(anchor_terms) < 3:
            raise ValueError(f"window {index} must have at least 3 anchor_terms")
    return payload


class NarrativeFulltextSpecGenerator:
    """Generate one full-text spec from chapter previews."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        prompt_path: Path,
        temperature: float = 0.0,
        timeout: int = 180,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.prompt_template = prompt_path.read_text(encoding="utf-8").strip()
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_body = extra_body or {}

    def _call_llm(self, prompt: str) -> tuple[str, dict[str, Any]]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            extra_body=self.extra_body or None,
        )
        content = response.choices[0].message.content or ""
        return content, response.model_dump()

    def build_prompt(
        self,
        *,
        title: str,
        author: str,
        notes: str,
        source_page_url: str,
        chapter_previews: str,
    ) -> str:
        return (
            f"{self.prompt_template}\n\n"
            "## Novel Metadata\n"
            f"Title: {title}\n"
            f"Author: {author}\n"
            f"Source page: {source_page_url}\n"
            f"Builder notes: {notes or 'No extra notes provided.'}\n\n"
            "## Chapter Previews\n"
            f"{chapter_previews}\n"
        )

    def build_retry_prompt(self, *, base_prompt: str, previous_raw: str, validation_error: str) -> str:
        return (
            f"{base_prompt}\n\n"
            "## Repair instructions\n"
            f"Your previous output failed validation for this reason: {validation_error}\n"
            "Return exactly one spec.\n"
            "Choose one focal event only.\n"
            "Use 6 to 8 windows only.\n"
            "Prefer richer time-state evolution over flat process narration.\n"
            "Use branch windows that are clearly different event lines, not paraphrases of the main arc.\n"
            "Return strict JSON only.\n\n"
            "## Previous invalid output\n"
            f"```json\n{previous_raw}\n```"
        )

    def generate_spec(
        self,
        *,
        title: str,
        author: str,
        notes: str,
        source_page_url: str,
        chapter_previews: str,
        chapter_count: int,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        base_prompt = self.build_prompt(
            title=title,
            author=author,
            notes=notes,
            source_page_url=source_page_url,
            chapter_previews=chapter_previews,
        )
        prompt = base_prompt
        last_error: Optional[Exception] = None
        last_raw = ""
        last_meta: dict[str, Any] = {}
        for attempt in range(self.max_retries):
            raw, meta = self._call_llm(prompt)
            last_raw = raw
            last_meta = meta
            try:
                payload = parse_json_object(raw)
                validated = validate_generated_fulltext_spec(payload, chapter_count)
                return validated, raw, meta
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries - 1:
                    raise
                prompt = self.build_retry_prompt(
                    base_prompt=base_prompt,
                    previous_raw=raw,
                    validation_error=str(exc),
                )
                time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"failed to generate valid full-text spec: {last_error}; last_raw={last_raw[:800]}")
