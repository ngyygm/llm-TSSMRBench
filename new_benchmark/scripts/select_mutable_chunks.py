"""Select P&P paragraphs suitable for mutation-based benchmark.

Filters paragraphs by length, mutability score (dialogue richness,
sentence count, adjective density), and selects top candidates for
Tier A (sentence-level mutation) and Tier B (word-level mutation).
"""

import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Gutenberg header/footer markers
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***"

TIER_A_COUNT = 30
TIER_B_COUNT = 20
TOTAL_COUNT = TIER_A_COUNT + TIER_B_COUNT


def load_novel_body() -> str:
    text = (DATA_DIR / "benchmark_narrative" / "pride_and_prejudice.txt").read_text(encoding="utf-8")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1:
        print("WARNING: Gutenberg markers not found, using full text")
        return text
    return text[start + len(START_MARKER):end].strip()


def split_paragraphs(text: str) -> list[str]:
    paras = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paras if p.strip()]


def count_sentences(text: str) -> int:
    return len(re.findall(r'[.!?]["\']?\s', text)) + 1


def has_dialogue(text: str) -> bool:
    return bool(re.search(r'["“]', text))


def count_adjectives(text: str) -> int:
    patterns = [
        r'\bvery\b', r'\bquite\b', r'\bsuch\b', r'\bso\b',
        r'\brather\b', r'\bextremely\b', r'\bexceedingly\b',
        r'\bparticularly\b', r'\bremarkably\b', r'\bmost\b',
    ]
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)


def is_skip_paragraph(text: str) -> bool:
    lower = text.lower()
    skip_phrases = [
        "chapter ", "project gutenberg", "illustration",
        "produced by", "transcriber", "proofreading",
    ]
    return any(p in lower for p in skip_phrases)


def score_mutability(text: str) -> float:
    words = text.split()
    wc = len(words)
    if wc < 50 or wc > 300:
        return 0.0

    score = 0.0

    # Sentence count (more sentences = more reorder targets)
    sents = count_sentences(text)
    if sents >= 5:
        score += 0.3
    elif sents >= 3:
        score += 0.2
    else:
        score += 0.1

    # Dialogue (rich dialogue = more distinct content)
    if has_dialogue(text):
        score += 0.25

    # Adjectives (more adjectives = more word-level mutation targets)
    adj_count = count_adjectives(text)
    score += min(0.25, adj_count * 0.05)

    # Word count sweet spot (80-180 is ideal)
    if 80 <= wc <= 180:
        score += 0.2
    elif 50 <= wc <= 300:
        score += 0.1

    return score


def main():
    print("=== Select Mutable Chunks ===\n")

    novel_body = load_novel_body()
    paragraphs = split_paragraphs(novel_body)
    print(f"Total paragraphs: {len(paragraphs)}")

    candidates = []
    for i, para in enumerate(paragraphs):
        if is_skip_paragraph(para):
            continue
        score = score_mutability(para)
        if score > 0:
            words = para.split()
            sents = count_sentences(para)
            candidates.append({
                "chunk_id": f"pp_{i:04d}",
                "para_index": i,
                "text": para,
                "word_count": len(words),
                "sentence_count": sents,
                "has_dialogue": has_dialogue(para),
                "mutability_score": round(score, 3),
            })

    candidates.sort(key=lambda c: c["mutability_score"], reverse=True)
    print(f"Scorable candidates: {len(candidates)}")

    # Select Tier A (sentence-level, need >= 3 sentences)
    tier_a_pool = [c for c in candidates if c["sentence_count"] >= 3]
    tier_b_pool = [c for c in candidates if c["mutability_score"] > 0]

    # Pick top, avoiding overlap (same paragraph can't be both tiers)
    selected = []
    used_indices = set()

    for c in tier_a_pool[:TIER_A_COUNT * 3]:
        if len([s for s in selected if s["tier"] == "A"]) >= TIER_A_COUNT:
            break
        if c["para_index"] in used_indices:
            continue
        c["tier"] = "A"
        selected.append(c)
        used_indices.add(c["para_index"])

    for c in tier_b_pool:
        if len([s for s in selected if s["tier"] == "B"]) >= TIER_B_COUNT:
            break
        if c["para_index"] in used_indices:
            continue
        c["tier"] = "B"
        selected.append(c)
        used_indices.add(c["para_index"])

    print(f"\nSelected: {len(selected)} chunks")
    print(f"  Tier A (sentence-level): {sum(1 for s in selected if s['tier'] == 'A')}")
    print(f"  Tier B (word-level): {sum(1 for s in selected if s['tier'] == 'B')}")

    # Stats
    wcs = [s["word_count"] for s in selected]
    print(f"\nWord count range: {min(wcs)}-{max(wcs)}, avg: {sum(wcs)/len(wcs):.0f}")
    print(f"Dialogue chunks: {sum(1 for s in selected if s['has_dialogue'])}")

    # Save
    out_path = DATA_DIR / "benchmark_narrative" / "selected_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in selected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSaved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
