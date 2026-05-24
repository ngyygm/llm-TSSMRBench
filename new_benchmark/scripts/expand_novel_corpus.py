"""Expand novel corpus with additional books from Project Gutenberg.

Downloads and chunks 5 classic novels, then merges them with existing
P&P chunks and mutations to create the full narrative database.
"""

import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
NOVELS_DIR = DATA_DIR / "benchmark_narrative" / "novels"

NOVEL_SOURCES = [
    {"file": "great_expectations.txt", "title": "Great Expectations", "author": "Charles Dickens"},
    {"file": "jane_eyre.txt", "title": "Jane Eyre", "author": "Charlotte Brontë"},
    {"file": "wuthering_heights.txt", "title": "Wuthering Heights", "author": "Emily Brontë"},
    {"file": "sense_and_sensibility.txt", "title": "Sense and Sensibility", "author": "Jane Austen"},
    {"file": "emma.txt", "title": "Emma", "author": "Jane Austen"},
]

MIN_WORDS = 20
TARGET_PER_NOVEL = 350  # ~350 paragraphs per novel → ~1750 from 5 novels


def chunk_novel(text: str, min_words: int = MIN_WORDS) -> list[str]:
    """Split novel into paragraphs, filtering short ones."""
    start_markers = ["*** START OF", "***START OF"]
    end_markers = ["*** END OF", "***END OF"]

    start_pos = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos >= 0:
            start_pos = text.find("\n", pos) + 1
            break

    end_pos = len(text)
    for marker in end_markers:
        pos = text.rfind(marker)
        if pos >= 0:
            end_pos = pos
            break

    body = text[start_pos:end_pos].strip()
    paragraphs = re.split(r'\n\s*\n', body)

    return [p.strip() for p in paragraphs if len(p.strip().split()) >= min_words]


def load_existing_chunks() -> list[dict]:
    """Load existing novel_chunks.jsonl."""
    path = DATA_DIR / "benchmark_narrative" / "novel_chunks.jsonl"
    chunks = []
    with open(path) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    print("=== Expand Novel Corpus ===\n")

    # Load existing chunks
    existing = load_existing_chunks()
    pp_originals = [c for c in existing if c.get("type") == "original"]
    pp_mutated = [c for c in existing if c.get("type") == "mutated"]
    print(f"Existing P&P originals: {len(pp_originals)}")
    print(f"Existing P&P mutated:   {len(pp_mutated)}")

    # Chunk each new novel
    new_chunks = []
    next_doc_id = max(c.get("doc_id", c.get("chunk_id", 0)) for c in existing) + 1

    for novel_info in NOVEL_SOURCES:
        fpath = NOVELS_DIR / novel_info["file"]
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue

        text = fpath.read_text(encoding="utf-8")
        paragraphs = chunk_novel(text)
        print(f"\n  {novel_info['title']}: {len(paragraphs)} paragraphs total")

        # Sample evenly if too many
        if len(paragraphs) > TARGET_PER_NOVEL:
            step = len(paragraphs) / TARGET_PER_NOVEL
            indices = [int(i * step) for i in range(TARGET_PER_NOVEL)]
            paragraphs = [paragraphs[i] for i in indices]
            print(f"    Sampled down to {len(paragraphs)} paragraphs")

        for para in paragraphs:
            chunk = {
                "doc_id": next_doc_id,
                "type": "original",
                "source": "gutenberg",
                "title": novel_info["title"],
                "author": novel_info["author"],
                "text": para,
                "word_count": len(para.split()),
            }
            new_chunks.append(chunk)
            next_doc_id += 1

    print(f"\nNew chunks from 5 novels: {len(new_chunks)}")

    # Rebuild novel_chunks.jsonl: P&P originals + new novels + P&P mutated
    output = []

    # P&P originals (keep existing)
    output.extend(pp_originals)
    print(f"P&P originals: {len(pp_originals)}")

    # New novel chunks
    output.extend(new_chunks)
    print(f"New novels: {len(new_chunks)}")

    # P&P mutated (keep existing)
    output.extend(pp_mutated)
    print(f"P&P mutated: {len(pp_mutated)}")

    total = len(output)
    print(f"\nTotal chunks: {total}")

    # Save
    out_path = DATA_DIR / "benchmark_narrative" / "novel_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in output:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved to {out_path}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  P&P originals:     {len(pp_originals)}")
    print(f"  Other novels:      {len(new_chunks)}")
    print(f"  P&P mutated:       {len(pp_mutated)}")
    print(f"  Total:             {total}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
