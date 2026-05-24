"""Generate mutated versions of P&P paragraphs with controlled similarity.

For each selected chunk, produces 2 mutated versions:
  Tier A (92-96% word overlap): sentence-level edits
    - reorder 2 adjacent sentences
    - replace a sentence with a paraphrase
    - add/remove a qualifying clause
  Tier B (98-99% word overlap): word-level edits
    - swap adjectives with synonyms
    - change verb tense
    - substitute character descriptors

Also rebuilds novel_chunks.jsonl with original + mutated chunks.
"""

import json
import random
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Reproducibility
SEED = 42
random.seed(SEED)

# Overlap targets (generous ranges; the key is Tier A < Tier B)
TIER_A_MIN, TIER_A_MAX = 0.88, 0.97
TIER_B_MIN, TIER_B_MAX = 0.96, 0.995


def word_jaccard(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


# --- Tier A mutations (sentence-level, must change word sets) ---

def mutation_drop_sentence(sentences: list[str]) -> list[str]:
    """Remove one sentence entirely — reduces word count and changes content."""
    if len(sentences) < 3:
        return sentences[:]
    result = sentences[:]
    idx = random.randint(0, len(result) - 1)
    result.pop(idx)
    return result


def mutation_swap_sentence_pair(sentences: list[str]) -> list[str]:
    """Replace one sentence with another from a pool of Austen-style alternatives."""
    if len(sentences) < 2:
        return sentences[:]

    replacements = [
        "She could not but smile at the thought.",
        "This was a circumstance which deserved consideration.",
        "He seemed scarcely to hear her, and was silent for some moments.",
        "The observation was certainly just, and she acknowledged it as such.",
        "Nothing could be more distressing than such a discovery.",
        "It was a subject, in short, on which reflection would be long indulged.",
        "She felt it to be so, though she had not foreseen it.",
        "The whole of this affair was soon spread abroad.",
        "He spoke with great animation on the subject.",
        "She knew not what to think, nor how to account for it.",
        "This was a fortunate circumstance indeed.",
        "It was some time, however, before she could speak again.",
        "He had certainly shown no disposition to avoid her.",
        "Her spirits were certainly affected by it.",
        "The distinction was not lost upon her.",
        "All this was acknowledged with a very civil smile.",
        "There was some mismanagement in the business.",
        "She had never been more surprised in her life.",
        "He looked at her with an air of quiet satisfaction.",
        "The event proved her judgement to be right.",
    ]

    result = sentences[:]
    idx = random.randint(0, len(result) - 1)
    replacement = random.choice(replacements)
    result[idx] = replacement
    return result


def mutation_reorder_and_modify(sentences: list[str]) -> list[str]:
    """Swap adjacent sentences AND modify one word in one of them."""
    if len(sentences) < 3:
        return sentences[:]
    result = sentences[:]
    # Swap adjacent pair
    idx = random.randint(0, len(result) - 2)
    result[idx], result[idx + 1] = result[idx + 1], result[idx]
    # Replace a content word in one of them
    swap_idx = random.choice([idx, idx + 1])
    sent = result[swap_idx]
    words = sent.split()
    if len(words) > 5:
        # Pick a mid-sentence word and replace with synonym
        mid = random.randint(2, len(words) - 3)
        clean = re.sub(r'[.,;!?"\']', '', words[mid].lower())
        if clean in ADJECTIVE_SWAPS:
            repl = random.choice(ADJECTIVE_SWAPS[clean])
            if words[mid][0].isupper():
                repl = repl[0].upper() + repl[1:]
            words[mid] = repl
        elif clean in CHARACTER_DESCRIPTORS:
            repl = CHARACTER_DESCRIPTORS[clean]
            if isinstance(repl, list):
                repl = random.choice(repl)
            if words[mid][0].isupper():
                repl = repl[0].upper() + repl[1:]
            words[mid] = repl
        else:
            # Drop the word
            words.pop(mid)
        result[swap_idx] = " ".join(words)
    return result


def mutation_replace_phrase(sentences: list[str]) -> list[str]:
    """Replace a phrase in one sentence with a different phrase of similar meaning."""
    if not sentences:
        return sentences[:]
    result = sentences[:]
    idx = random.randint(0, len(result) - 1)
    sent = result[idx]
    words = sent.split()

    phrase_swaps = [
        ("very much", "exceedingly"),
        ("at length", "after some reflection"),
        ("in a manner", "in such a way"),
        ("with great", "with much"),
        ("could not help", "found herself unable to resist"),
        ("was obliged to", "felt compelled to"),
        ("took place", "occurred"),
        ("came into", "entered"),
        ("made no answer", "said nothing"),
        ("looked at her", "regarded her"),
        ("turned away", "averted her gaze"),
        ("went on", "proceeded"),
        ("soon after", "not long thereafter"),
        ("in the course of", "during"),
        ("by no means", "not at all"),
        ("at first", "initially"),
    ]

    for old, new in random.sample(phrase_swaps, min(len(phrase_swaps), 3)):
        if old in sent.lower():
            # Case-insensitive replace
            pattern = re.compile(re.escape(old), re.IGNORECASE)
            result[idx] = pattern.sub(new, sent, count=1)
            break
    return result


TIER_A_MUTATIONS = [
    mutation_drop_sentence,
    mutation_swap_sentence_pair,
    mutation_reorder_and_modify,
    mutation_replace_phrase,
]


# --- Tier B mutations (word-level) ---

ADJECTIVE_SWAPS = {
    "handsome": ["agreeable", "pleasing", "gentlemanlike"],
    "beautiful": ["lovely", "charming", "elegant"],
    "clever": ["sensible", "acute", "discerning"],
    "amiable": ["agreeable", "cordial", "kind"],
    "elegant": ["graceful", "refined", "polished"],
    "proud": ["haughty", "reserved", "disdainful"],
    "silly": ["foolish", "absurd", "ridiculous"],
    "pretty": ["handsome", "pleasant", "attractive"],
    "gentle": ["mild", "tender", "soft"],
    "rich": ["wealthy", "prosperous", "affluent"],
    "happy": ["glad", "cheerful", "delighted"],
    "angry": ["displeased", "offended", "indignant"],
    "anxious": ["uneasy", "concerned", "troubled"],
    "fine": ["excellent", "capital", "superior"],
    "good": ["worthy", "excellent", "admirable"],
    "great": ["considerable", "remarkable", "notable"],
    "young": ["early", "juvenile"],
    "old": ["elderly", "aged"],
    "small": ["little", "slight"],
    "large": ["sizeable", "considerable"],
    "warm": ["cordial", "hearty"],
    "cold": ["distant", "reserved"],
    "quiet": ["still", "peaceful"],
    "loud": ["noisy", "boisterous"],
    "kind": ["benevolent", "generous"],
    "sharp": ["keen", "quick"],
    "soft": ["gentle", "mild"],
    "bright": ["brilliant", "radiant"],
    "dark": ["dim", "shadowed"],
}

VERB_TENSE_SWAPS = {
    "was": "had been",
    "were": "had been",
    "is": "was",
    "are": "were",
    "has": "had",
    "have": "had",
    "does": "did",
    "goes": "went",
    "comes": "came",
    "says": "said",
    "knows": "knew",
    "thinks": "thought",
    "sees": "saw",
    "feels": "felt",
    "gives": "gave",
    "takes": "took",
    "makes": "made",
    "finds": "found",
    "tells": "told",
    "seemed": "had seemed",
    "appeared": "had appeared",
    "looked": "had looked",
}

CHARACTER_DESCRIPTORS = {
    "gentleman": ["man of property", "person of consequence"],
    "lady": ["gentlewoman", "person of fashion"],
    "girl": ["young person", "miss"],
    "man": ["gentleman", "fellow"],
    "woman": ["gentlewoman", "person"],
    "sister": ["relation", "family member"],
    "brother": ["relation", "family member"],
    "friend": ["companion", "associate"],
    "fortune": ["estate", "property"],
    "house": ["residence", "dwelling"],
    "room": ["apartment", "chamber"],
    "party": ["assembly", "gathering"],
    "walk": "stroll",
    "visit": "call",
    "dinner": "supper",
    "carriage": "coach",
}


def mutation_swap_adjectives(text: str) -> str:
    words = text.split()
    changed = 0
    for i, w in enumerate(words):
        clean = re.sub(r'[.,;!?"\']', '', w.lower())
        if clean in ADJECTIVE_SWAPS:
            replacement = random.choice(ADJECTIVE_SWAPS[clean])
            # Preserve case
            if w[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            # Preserve trailing punctuation
            trailing = ""
            for c in reversed(w):
                if c in '.,;!?"\'':
                    trailing = c + trailing
                else:
                    break
            if trailing:
                replacement = replacement + trailing
            words[i] = replacement
            changed += 1
            if changed >= 3:
                break
    return " ".join(words)


def mutation_swap_verbs(text: str) -> str:
    words = text.split()
    changed = 0
    for i, w in enumerate(words):
        clean = re.sub(r'[.,;!?"\']', '', w.lower())
        if clean in VERB_TENSE_SWAPS:
            replacement = VERB_TENSE_SWAPS[clean]
            if w[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            trailing = ""
            for c in reversed(w):
                if c in '.,;!?"\'':
                    trailing = c + trailing
                else:
                    break
            if trailing:
                replacement = replacement + trailing
            words[i] = replacement
            changed += 1
            if changed >= 2:
                break
    return " ".join(words)


def mutation_swap_descriptors(text: str) -> str:
    words = text.split()
    changed = 0
    for i, w in enumerate(words):
        clean = re.sub(r'[.,;!?"\']', '', w.lower())
        if clean in CHARACTER_DESCRIPTORS:
            replacement = CHARACTER_DESCRIPTORS[clean]
            if isinstance(replacement, list):
                replacement = random.choice(replacement)
            if w[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            trailing = ""
            for c in reversed(w):
                if c in '.,;!?"\'':
                    trailing = c + trailing
                else:
                    break
            if trailing:
                replacement = replacement + trailing
            words[i] = replacement
            changed += 1
            if changed >= 2:
                break
    return " ".join(words)


TIER_B_MUTATIONS = [
    mutation_swap_adjectives,
    mutation_swap_verbs,
    mutation_swap_descriptors,
]


def mutation_force_change(text: str) -> str:
    """Fallback: if other mutations had no effect, make a guaranteed change."""
    words = text.split()
    if len(words) > 10:
        # Drop 2-3 non-essential words (articles, prepositions)
        skip = {"the", "a", "an", "of", "to", "in", "and", "was", "is", "had", "been"}
        droppable = [i for i, w in enumerate(words) if re.sub(r'[.,;!?"\']', '', w.lower()) in skip]
        if len(droppable) >= 2:
            for idx in sorted(random.sample(droppable, min(2, len(droppable))), reverse=True):
                words.pop(idx)
        else:
            # Swap two words
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
    return " ".join(words)


def generate_tier_a_version(original: str) -> tuple[str, list[dict]]:
    sentences = split_sentences(original)
    if len(sentences) < 2:
        return original, []

    # Apply 1-2 sentence-level mutations
    mutations_applied = []
    n_mutations = random.randint(1, min(2, len(TIER_A_MUTATIONS)))
    chosen = random.sample(TIER_A_MUTATIONS, n_mutations)

    for mut_fn in chosen:
        old_sents = sentences[:]
        sentences = mut_fn(sentences)
        if sentences != old_sents:
            mutations_applied.append({"type": mut_fn.__name__})

    result = " ".join(sentences)

    # Ensure something changed
    if result == original:
        result = mutation_force_change(original)
        if result != original:
            mutations_applied.append({"type": "force_change"})

    return result, mutations_applied


def generate_tier_b_version(original: str) -> tuple[str, list[dict]]:
    # Apply 1-2 word-level mutations
    text = original
    mutations_applied = []
    n_mutations = random.randint(1, 2)
    chosen = random.sample(TIER_B_MUTATIONS, n_mutations)

    for mut_fn in chosen:
        old_text = text
        text = mut_fn(text)
        if text != old_text:
            mutations_applied.append({"type": mut_fn.__name__})

    # Ensure something changed
    if text == original:
        text = mutation_force_change(original)
        if text != original:
            mutations_applied.append({"type": "force_change"})

    return text, mutations_applied


def generate_mutations_for_chunk(chunk: dict) -> list[dict]:
    """Generate 2 mutated versions for a chunk."""
    original = chunk["text"]
    tier = chunk["tier"]
    versions = []

    for v_idx in range(2):
        version_id = f"v{v_idx + 2}"
        overlap_min = TIER_A_MIN if tier == "A" else TIER_B_MIN
        overlap_max = TIER_A_MAX if tier == "A" else TIER_B_MAX

        # Try up to 10 times to get overlap in range
        best = None
        best_overlap = 0
        for attempt in range(10):
            if tier == "A":
                mutated, edits = generate_tier_a_version(original)
            else:
                mutated, edits = generate_tier_b_version(original)

            overlap = word_jaccard(original, mutated)
            if overlap_min <= overlap <= overlap_max:
                best = (mutated, edits, overlap)
                break
            if abs(overlap - (overlap_min + overlap_max) / 2) < abs(best_overlap - (overlap_min + overlap_max) / 2):
                best = (mutated, edits, overlap)
                best_overlap = overlap

        if best is None:
            # Fallback: just use original with tiny tweak
            words = original.split()
            if len(words) > 10:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
            mutated = " ".join(words)
            edits = [{"type": "fallback_word_swap"}]
            overlap = word_jaccard(original, mutated)
            best = (mutated, edits, overlap)

        mutated_text, edits, overlap = best
        versions.append({
            "chunk_id": chunk["chunk_id"],
            "original": original,
            "version": version_id,
            "text": mutated_text,
            "tier": tier,
            "overlap_with_original": round(overlap, 4),
            "edits": edits,
        })

    return versions


def chunk_novel_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Split novel text into overlapping word-level chunks (same as original pipeline)."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 50:
            break
        chunks.append({
            "position": i,
            "word_count": len(chunk_words),
            "text": " ".join(chunk_words),
        })
        if i + chunk_size >= len(words):
            break
    return chunks


def main():
    print("=== Generate Mutations ===\n")

    # Load selected chunks
    selected_path = DATA_DIR / "benchmark_narrative" / "selected_chunks.jsonl"
    selected = []
    with open(selected_path) as f:
        for line in f:
            selected.append(json.loads(line))
    print(f"Loaded {len(selected)} selected chunks")

    # Generate mutations
    all_mutations = []
    for chunk in selected:
        versions = generate_mutations_for_chunk(chunk)
        all_mutations.extend(versions)

    # Report overlap stats
    tier_a = [m for m in all_mutations if m["tier"] == "A"]
    tier_b = [m for m in all_mutations if m["tier"] == "B"]

    if tier_a:
        overlaps_a = [m["overlap_with_original"] for m in tier_a]
        print(f"\nTier A ({len(tier_a)} mutations):")
        print(f"  Overlap: min={min(overlaps_a):.3f}, max={max(overlaps_a):.3f}, "
              f"avg={sum(overlaps_a)/len(overlaps_a):.3f}")
        # Flag out-of-range
        oor = [m for m in tier_a if not (TIER_A_MIN <= m["overlap_with_original"] <= TIER_A_MAX)]
        if oor:
            print(f"  WARNING: {len(oor)} outside target range [{TIER_A_MIN}, {TIER_A_MAX}]")

    if tier_b:
        overlaps_b = [m["overlap_with_original"] for m in tier_b]
        print(f"\nTier B ({len(tier_b)} mutations):")
        print(f"  Overlap: min={min(overlaps_b):.3f}, max={max(overlaps_b):.3f}, "
              f"avg={sum(overlaps_b)/len(overlaps_b):.3f}")
        oor = [m for m in tier_b if not (TIER_B_MIN <= m["overlap_with_original"] <= TIER_B_MAX)]
        if oor:
            print(f"  WARNING: {len(oor)} outside target range [{TIER_B_MIN}, {TIER_B_MAX}]")

    # Save mutations
    mutations_path = DATA_DIR / "benchmark_narrative" / "mutated_chunks.jsonl"
    with open(mutations_path, "w", encoding="utf-8") as f:
        for m in all_mutations:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_mutations)} mutations to {mutations_path}")

    # --- Rebuild novel_chunks.jsonl ---
    print("\nRebuilding novel_chunks.jsonl...")

    novel_path = DATA_DIR / "benchmark_narrative" / "pride_and_prejudice.txt"
    novel_text = novel_path.read_text(encoding="utf-8")
    original_chunks = chunk_novel_text(novel_text)
    print(f"Original chunks: {len(original_chunks)}")

    # Build mutation lookup: chunk_id -> list of mutations
    mut_by_chunk: dict[str, list[dict]] = {}
    for m in all_mutations:
        mut_by_chunk.setdefault(m["chunk_id"], []).append(m)

    # Write all chunks: originals + mutated
    out_path = DATA_DIR / "benchmark_narrative" / "novel_chunks.jsonl"
    chunk_counter = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for oc in original_chunks:
            f.write(json.dumps({
                "chunk_id": chunk_counter,
                "type": "original",
                "text": oc["text"],
                "position": oc["position"],
                "word_count": oc["word_count"],
            }, ensure_ascii=False) + "\n")
            chunk_counter += 1

        # Add baseline (original paragraph) + mutated chunks at the end
        # Build lookup: chunk_id -> original text
        selected_lookup = {s["chunk_id"]: s for s in selected}

        n_baseline = 0
        for chunk_id, versions in mut_by_chunk.items():
            # Add the original paragraph as a baseline chunk
            orig_text = selected_lookup[chunk_id]["text"]
            f.write(json.dumps({
                "chunk_id": chunk_counter,
                "type": "mutated",
                "scenario_id": chunk_id,
                "state_id": "original",
                "text": orig_text,
                "tier": versions[0]["tier"],
                "overlap_with_original": 1.0,
                "edits": [],
            }, ensure_ascii=False) + "\n")
            chunk_counter += 1
            n_baseline += 1

            # Add mutated versions
            for v in versions:
                f.write(json.dumps({
                    "chunk_id": chunk_counter,
                    "type": "mutated",
                    "scenario_id": chunk_id,
                    "state_id": v["version"],
                    "text": v["text"],
                    "tier": v["tier"],
                    "overlap_with_original": v["overlap_with_original"],
                    "edits": v["edits"],
                }, ensure_ascii=False) + "\n")
                chunk_counter += 1

    print(f"Wrote {chunk_counter} total chunks to {out_path}")
    print(f"  Original: {len(original_chunks)}, Baseline: {n_baseline}, Mutated: {len(all_mutations)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
