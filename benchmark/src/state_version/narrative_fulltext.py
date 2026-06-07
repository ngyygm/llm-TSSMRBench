"""Utilities for building narrative source bundles from full-text novels."""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .schemas import SourceBundleItem, SourceBundleRecord


GUTENBERG_START_PATTERNS = (
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
    "***START OF THE PROJECT GUTENBERG EBOOK",
)

GUTENBERG_END_PATTERNS = (
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
    "***END OF THE PROJECT GUTENBERG EBOOK",
)

CHAPTER_HEADING_RE = re.compile(
    r"(?im)^\s*chapter\s+(?:[IVXLCDM]+|\d+)\b.*$"
)
ROMAN_ONLY_HEADING_RE = re.compile(r"^\s*[IVXLCDM]+(?:\.)?\s*$", re.IGNORECASE)
ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9'’\"?;:,\- ]{4,}$")
ILLUSTRATION_ONLY_RE = re.compile(r"^\[\s*illustration(?:[^\]]*)\]$", re.IGNORECASE)
ILLUSTRATION_PREFIX_RE = re.compile(r"^\[\s*illustration(?:[^\]]*)\]\s*", re.IGNORECASE)


@dataclass
class ChapterBlock:
    """One ordered chapter-like block extracted from a full text."""

    index: int
    heading: str
    text: str


def normalize_text(text: str) -> str:
    """Normalize line endings and collapse trailing whitespace."""

    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove common Project Gutenberg header/footer boilerplate."""

    normalized = normalize_text(text)
    lines = normalized.split("\n")

    start_index = 0
    for index, line in enumerate(lines):
        if any(marker in line for marker in GUTENBERG_START_PATTERNS):
            start_index = index + 1
            break

    end_index = len(lines)
    for index, line in enumerate(lines):
        if any(marker in line for marker in GUTENBERG_END_PATTERNS):
            end_index = index
            break

    cleaned = "\n".join(lines[start_index:end_index]).strip()
    return cleaned or normalized


def split_into_chapters(full_text: str) -> list[ChapterBlock]:
    """Split a normalized full text into chapter-like blocks."""

    cleaned = strip_gutenberg_boilerplate(full_text)
    matches = list(CHAPTER_HEADING_RE.finditer(cleaned))
    if not matches:
        fallback = split_into_line_headings(cleaned)
        return fallback or [ChapterBlock(index=1, heading="FULL_TEXT", text=cleaned)]

    chapters: list[ChapterBlock] = []
    chapter_index = 0
    for match_index, match in enumerate(matches):
        start = match.start()
        end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(cleaned)
        block = cleaned[start:end].strip()
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        heading = lines[0] if lines else f"CHAPTER_{match_index + 1}"
        body = "\n".join(lines[1:]).strip()
        if body:
            if looks_like_contents_block(lines[1:]):
                continue
            chapter_index += 1
            chapters.append(ChapterBlock(index=chapter_index, heading=heading, text=body))
    if not chapters:
        return [ChapterBlock(index=1, heading="FULL_TEXT", text=cleaned)]

    chapters = drop_leading_preface_chapters(chapters)
    return chapters or [ChapterBlock(index=1, heading="FULL_TEXT", text=cleaned)]


def looks_like_prose_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 40:
        return False
    lowered = stripped.lower()
    return any(ch.islower() for ch in stripped) and not lowered.startswith(("title:", "author:", "release date:", "language:"))


def is_heading_candidate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered in {"contents", "preface", "prologue", "epilogue"}:
        return False
    if lowered.startswith(("title:", "author:", "release date:", "language:", "credits:", "project gutenberg")):
        return False
    if ROMAN_ONLY_HEADING_RE.fullmatch(stripped):
        return True
    if ALL_CAPS_HEADING_RE.fullmatch(stripped):
        return True
    if re.fullmatch(r"[IVXLCDM]+\.\s+.+", stripped, flags=re.IGNORECASE):
        return True
    return False


def split_into_line_headings(cleaned: str) -> list[ChapterBlock]:
    """Fallback splitter for novels whose headings are Roman numerals or all-caps titles."""

    lines = cleaned.split("\n")
    heading_line_indices: list[int] = []
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not is_heading_candidate(line):
            continue

        next_nonempty_index = None
        next_nonempty_line = ""
        prose_within_window = False
        for j in range(idx + 1, min(len(lines), idx + 8)):
            probe = lines[j].strip()
            if not probe:
                continue
            if next_nonempty_index is None:
                next_nonempty_index = j
                next_nonempty_line = probe
            if looks_like_prose_line(probe):
                prose_within_window = True
                break

        if not prose_within_window:
            continue
        if next_nonempty_index is None:
            continue
        heading_line_indices.append(idx)

    if not heading_line_indices:
        return []

    chapters: list[ChapterBlock] = []
    for i, line_idx in enumerate(heading_line_indices):
        start = line_idx
        end = heading_line_indices[i + 1] if i + 1 < len(heading_line_indices) else len(lines)
        block_lines = [line.rstrip() for line in lines[start:end]]
        nonempty = [line.strip() for line in block_lines if line.strip()]
        if len(nonempty) < 2:
            continue

        heading_parts = [nonempty[0]]
        body_start = 1
        if ROMAN_ONLY_HEADING_RE.fullmatch(nonempty[0]) and len(nonempty) >= 2 and len(nonempty[1].split()) <= 10:
            heading_parts.append(nonempty[1])
            body_start = 2

        heading = " ".join(heading_parts).strip()
        body = "\n".join(nonempty[body_start:]).strip()
        if not body:
            continue
        if looks_like_contents_block(nonempty[body_start:]):
            continue
        chapters.append(ChapterBlock(index=len(chapters) + 1, heading=heading, text=body))

    return chapters


def split_into_paragraphs(text: str) -> list[str]:
    """Split a chapter range into compact paragraphs."""

    normalized = normalize_text(text)
    raw_parts = re.split(r"\n\s*\n", normalized)
    paragraphs: list[str] = []
    for part in raw_parts:
        compact = " ".join(part.split())
        if not compact:
            continue
        if ILLUSTRATION_ONLY_RE.fullmatch(compact):
            continue
        compact = ILLUSTRATION_PREFIX_RE.sub("", compact)
        if not compact:
            continue
        paragraphs.append(compact)
    return paragraphs


def looks_like_contents_block(lines: list[str]) -> bool:
    """Heuristically detect a chapter block that is actually a contents list."""

    if len(lines) < 3:
        return False

    content_like = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"\s\d+\s*$", stripped):
            content_like += 1
            continue
        if stripped.lower().startswith(("heading to chapter", "tailpiece to chapter")):
            content_like += 1
            continue
        if stripped.startswith("[Illustration"):
            content_like += 1

    return content_like >= max(3, len(lines) // 2)


def is_chapter_one_heading(heading: str) -> bool:
    """Return whether one heading clearly marks chapter one."""

    compact = heading.strip().lower()
    return compact.startswith("chapter i") or compact.startswith("chapter 1")


def drop_leading_preface_chapters(chapters: list[ChapterBlock]) -> list[ChapterBlock]:
    """Drop leading preface-like chapter blocks before the first true Chapter I."""

    if not chapters or is_chapter_one_heading(chapters[0].heading):
        return chapters

    first_true_chapter_index = next(
        (index for index, chapter in enumerate(chapters) if is_chapter_one_heading(chapter.heading)),
        None,
    )
    if first_true_chapter_index is None:
        return chapters

    trimmed = chapters[first_true_chapter_index:]
    return [
        ChapterBlock(index=index, heading=chapter.heading, text=chapter.text)
        for index, chapter in enumerate(trimmed, start=1)
    ]


def score_paragraph(paragraph: str, anchor_terms: Iterable[str]) -> int:
    """Score one paragraph by anchor-term overlap."""

    lowered = paragraph.lower()
    return sum(lowered.count(term.lower()) for term in anchor_terms if term.strip())


def select_anchor_excerpt(
    paragraphs: list[str],
    anchor_terms: list[str],
    max_words: int,
    neighbor_radius: int,
) -> str:
    """Extract an audit-friendly excerpt centered on the best anchor paragraph."""

    if not paragraphs:
        return ""

    if anchor_terms:
        scores = [score_paragraph(paragraph, anchor_terms) for paragraph in paragraphs]
        best_index = max(range(len(paragraphs)), key=lambda index: (scores[index], -index))
        if scores[best_index] == 0:
            best_index = 0
    else:
        best_index = 0

    selected_indices = set(
        range(
            max(0, best_index - max(0, neighbor_radius)),
            min(len(paragraphs), best_index + max(0, neighbor_radius) + 1),
        )
    )
    ordered = [paragraphs[index] for index in sorted(selected_indices)]
    excerpt_words: list[str] = []
    collected_parts: list[str] = []
    for part in ordered:
        candidate_words = part.split()
        if not candidate_words:
            continue
        if len(excerpt_words) >= max_words:
            break
        remaining = max_words - len(excerpt_words)
        collected = " ".join(candidate_words[:remaining])
        collected_parts.append(collected)
        excerpt_words.extend(candidate_words[:remaining])

    if not collected_parts:
        fallback_words = paragraphs[0].split()[:max_words]
        return " ".join(fallback_words)
    return "\n\n".join(collected_parts).strip()


def resolve_chapter_range(
    chapters: list[ChapterBlock],
    chapter_start: int,
    chapter_end: int,
) -> list[ChapterBlock]:
    """Resolve an inclusive 1-based chapter range."""

    if chapter_start < 1 or chapter_end < chapter_start:
        raise ValueError(f"invalid chapter range: {chapter_start}-{chapter_end}")
    selected = [chapter for chapter in chapters if chapter_start <= chapter.index <= chapter_end]
    if not selected:
        raise ValueError(f"chapter range {chapter_start}-{chapter_end} is empty for this text")
    return selected


def build_window_text(
    chapters: list[ChapterBlock],
    chapter_start: int,
    chapter_end: int,
    anchor_terms: Optional[list[str]] = None,
    max_words: int = 260,
    neighbor_radius: int = 1,
) -> tuple[str, str]:
    """Build one source-bundle window text and a span hint."""

    selected = resolve_chapter_range(chapters, chapter_start, chapter_end)
    combined_text = "\n\n".join(chapter.text for chapter in selected)
    paragraphs = split_into_paragraphs(combined_text)
    excerpt = select_anchor_excerpt(
        paragraphs=paragraphs,
        anchor_terms=anchor_terms or [],
        max_words=max_words,
        neighbor_radius=neighbor_radius,
    )
    span_hint = selected[0].heading if chapter_start == chapter_end else f"{selected[0].heading} -> {selected[-1].heading}"
    return excerpt, span_hint


def maybe_download_text(download_url: str, destination: Path, timeout_seconds: int) -> None:
    """Download one full text to the local workspace if it is missing."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with urllib.request.urlopen(download_url, timeout=timeout_seconds) as response:
        raw = response.read()
    destination.write_bytes(raw)


def read_full_text(path: Path) -> str:
    """Read one narrative full-text file using UTF-8 with BOM fallback."""

    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def build_narrative_source_bundle(
    *,
    sample_id: str,
    state_chain_id: str,
    language: str,
    focus_event: str,
    source_title: str,
    title: str,
    author: str,
    source_provider: str,
    source_page_url: str,
    download_url: str,
    local_path: Path,
    chapter_blocks: list[ChapterBlock],
    bundle_summary: str,
    notes: list[str],
    window_specs: list[dict],
    default_max_words: int,
    default_neighbor_radius: int,
) -> SourceBundleRecord:
    """Build one narrative source bundle from a local full-text novel."""

    source_bundle_items: list[SourceBundleItem] = []
    extracted_windows: list[dict[str, str]] = []
    for window in window_specs:
        chapter_start = int(window["chapter_start"])
        chapter_end = int(window.get("chapter_end", chapter_start))
        excerpt, span_hint = build_window_text(
            chapters=chapter_blocks,
            chapter_start=chapter_start,
            chapter_end=chapter_end,
            anchor_terms=list(window.get("anchor_terms", []) or []),
            max_words=int(window.get("max_words", default_max_words)),
            neighbor_radius=int(window.get("neighbor_radius", default_neighbor_radius)),
        )
        source_bundle_items.append(
            SourceBundleItem(
                artifact_type=str(window.get("artifact_type", "chapter_window")),
                artifact_ref=str(window["artifact_ref"]),
                title=str(window.get("title", "")) or None,
                time_hint=str(window.get("time_hint", "")).strip() or None,
                summary=excerpt,
            )
        )
        extracted_windows.append(
            {
                "artifact_ref": str(window["artifact_ref"]),
                "span_hint": span_hint,
                "chapter_start": str(chapter_start),
                "chapter_end": str(chapter_end),
            }
        )

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
            "source_provider": source_provider,
            "source_page_url": source_page_url,
            "download_url": download_url,
            "source_kind": "full_text",
            "local_path": str(local_path),
            "chapter_count": len(chapter_blocks),
            "extracted_windows": extracted_windows,
        },
    )
