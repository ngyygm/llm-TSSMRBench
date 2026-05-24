"""Generate questions for mutation-based narrative benchmark.

Design:
- Queries contain distinctive words shared by ALL versions of a scenario
  (so they localize to the right passage but don't leak version info)
- Difficulty comes entirely from version similarity, not query vagueness
- Mirrors GitHub queries: file+function name localizes to the file,
  then version differentiation is the challenge
"""

import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

SEED = 42
random.seed(SEED)

STOPWORDS = {
    "not", "but", "and", "or", "yet", "so", "for", "nor", "as",
    "when", "where", "while", "after", "before", "though", "although",
    "there", "here", "now", "then", "thus", "indeed", "however",
    "nothing", "something", "everything", "anything", "nobody",
    "some", "such", "what", "which", "who", "whom", "whose",
    "how", "why", "much", "many", "more", "most", "every",
    "upon", "into", "over", "under", "between", "among",
    "yes", "no", "oh", "ah", "well", "only", "never",
    "always", "still", "even", "quite", "rather", "already",
    "if", "do", "did", "done", "am", "is", "was", "were", "be",
    "been", "being", "have", "has", "had", "having",
    "a", "an", "the", "to", "of", "in", "on", "at", "by",
    "with", "from", "up", "out", "off", "down", "all",
    "that", "this", "these", "those", "it",
    "i", "he", "she", "we", "they", "me", "him", "us", "them",
    "my", "his", "her", "our", "your", "its",
    "can", "will", "may", "might", "must", "shall", "does",
    "said", "just", "also", "really", "about", "through",
    "other", "each", "any", "such", "than", "because", "since",
}


def load_mutations() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    path = DATA_DIR / "benchmark_narrative" / "mutated_chunks.jsonl"
    with open(path) as f:
        for line in f:
            m = json.loads(line)
            groups.setdefault(m["chunk_id"], []).append(m)
    return groups


def load_selected_chunks() -> dict[str, dict]:
    chunks = {}
    path = DATA_DIR / "benchmark_narrative" / "selected_chunks.jsonl"
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c["text"]
    return chunks


def load_all_chunks() -> list[dict]:
    chunks = []
    path = DATA_DIR / "benchmark_narrative" / "novel_chunks.jsonl"
    with open(path) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def get_shared_rare_words(
    texts: list[str],
    word_counts: dict[str, int],
    total_chunks: int,
    rarity_pct: float = 0.05,
) -> list[str]:
    """Find content words shared by all texts but rare across the corpus."""
    word_sets = [set(re.findall(r'\b[a-z]{3,}\b', t.lower())) for t in texts]
    shared = word_sets[0]
    for ws in word_sets[1:]:
        shared = shared & ws

    rare = sorted(
        w for w in shared - STOPWORDS
        if word_counts.get(w, 0) <= total_chunks * rarity_pct
    )
    return rare


def build_query(rare_words: list[str], original_text: str) -> str:
    """Build a query using distinctive shared words to localize to the scenario."""
    n_words = min(len(rare_words), random.choice([3, 4, 4, 5]))
    chosen = random.sample(rare_words, n_words)

    # Build a phrase-like query from the chosen words
    templates = [
        "In the passage about " + ", ".join(chosen[:-1]) + " and " + chosen[-1] + ", what is described?",
        "What details are given concerning " + ", ".join(chosen[:3]) + " in this text?",
        "In the text mentioning " + " and ".join(chosen[:2]) + ", what is conveyed?",
        "What is described in the passage about " + " and ".join(chosen[:3]) + "?",
        "Regarding " + ", ".join(chosen[:3]) + ", what does this passage say?",
    ]
    return random.choice(templates)


def build_bm25(corpus_texts: list[str]):
    """Build a lightweight BM25 for localization testing."""
    def tokenize(text):
        cleaned = re.sub(r'[\[\]@(){}<>,;:!?"\'\\]', ' ', text.lower())
        return [t for t in re.split(r'[/\s_\.]+', cleaned) if len(t) > 1]

    corpus = [tokenize(t) for t in corpus_texts]
    doc_freqs = [Counter(doc) for doc in corpus]
    doc_lens = [len(doc) for doc in corpus]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0
    k1, b = 1.5, 0.75

    n_q = Counter()
    for doc in corpus:
        for token in set(doc):
            n_q[token] += 1
    idf = {token: math.log(1 + (len(corpus) - freq + 0.5) / (freq + 0.5))
           for token, freq in n_q.items()}

    def query(query_tokens, top_k=10):
        scores = []
        for i, (doc_tf, doc_len) in enumerate(zip(doc_freqs, doc_lens)):
            score = 0.0
            for token in query_tokens:
                tf = doc_tf.get(token, 0)
                if tf == 0:
                    continue
                idf_val = idf.get(token, 0.0)
                denom = tf + k1 * (1 - b + b * (doc_len / avgdl if avgdl else 0))
                if denom == 0:
                    continue
                score += idf_val * (tf * (k1 + 1)) / denom
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    return query


def main():
    print("=== Generate Narrative Questions (V2: Localizing Queries) ===\n")

    mutation_groups = load_mutations()
    selected_chunks = load_selected_chunks()
    all_chunks = load_all_chunks()
    print(f"Mutation groups: {len(mutation_groups)}")
    print(f"Total database chunks: {len(all_chunks)}")

    # Build word frequency across entire corpus
    word_counts: dict[str, int] = defaultdict(int)
    for c in all_chunks:
        for w in set(re.findall(r'\b[a-z]{3,}\b', c['text'].lower())):
            word_counts[w] += 1
    total_chunks = len(all_chunks)

    # Build BM25 for localization testing
    corpus_texts = [c['text'] for c in all_chunks]
    bm25_query = build_bm25(corpus_texts)

    # Build scenario -> doc_id mapping
    scenario_doc_ids: dict[str, set[int]] = defaultdict(set)
    for i, c in enumerate(all_chunks):
        if c.get('type') == 'mutated':
            scenario_doc_ids[c['scenario_id']].add(i)

    all_questions = []
    skipped_rare = 0
    skipped_dup = 0
    localization_ok = 0
    localization_fail = 0

    for chunk_id in sorted(mutation_groups.keys()):
        versions = mutation_groups[chunk_id]
        tier = versions[0]["tier"]
        original = versions[0]["original"]

        all_texts = [original] + [v["text"] for v in versions]
        all_version_ids = ["original"] + [v["version"] for v in versions]

        # Find shared rare words across all versions
        rare_words = get_shared_rare_words(all_texts, word_counts, total_chunks)

        if len(rare_words) < 3:
            # Fallback: relax rarity threshold
            rare_words = get_shared_rare_words(all_texts, word_counts, total_chunks, rarity_pct=0.15)

        if len(rare_words) < 2:
            skipped_rare += 1
            continue

        # Build query
        query_text = build_query(rare_words, original)

        # Test localization: does BM25 find the scenario in top-3?
        query_tokens = re.findall(r'\b[a-z]{3,}\b', query_text.lower())
        query_tokens = [t for t in query_tokens if t not in STOPWORDS]
        results = bm25_query(query_tokens, top_k=10)
        top3_ids = {idx for idx, _ in results[:3]}

        s_docs = scenario_doc_ids.get(chunk_id, set())
        if not (s_docs & top3_ids):
            localization_fail += 1
            # Try with more rare words
            n_words = min(len(rare_words), 6)
            chosen = random.sample(rare_words, n_words)
            query_text = "In the passage about " + " and ".join(chosen) + ", what is described?"
            query_tokens = [w for w in re.findall(r'\b[a-z]{3,}\b', query_text.lower()) if w not in STOPWORDS]
            results = bm25_query(query_tokens, top_k=10)
            top3_ids = {idx for idx, _ in results[:3]}
            if not (s_docs & top3_ids):
                skipped_rare += 1
                continue
        localization_ok += 1

        for version_id in all_version_ids:
            distractor_states = [vid for vid in all_version_ids if vid != version_id]

            if version_id == "original":
                gold_text = original
                style = "generic"
            else:
                v = [v for v in versions if v["version"] == version_id][0]
                gold_text = v["text"]
                style = "specific"

            # Skip if gold text is identical to any distractor text
            is_dup = False
            for dst in distractor_states:
                if dst == "original":
                    dst_text = original
                else:
                    dv = [v for v in versions if v["version"] == dst]
                    dst_text = dv[0]["text"] if dv else ""
                if gold_text == dst_text:
                    is_dup = True
                    break
            if is_dup:
                skipped_dup += 1
                continue

            question = {
                "id": f"q_mut_{chunk_id}_{version_id}_{style}",
                "type": "single_version",
                "difficulty": "hard" if tier == "B" else "medium",
                "question_style": style,
                "query_text": query_text,
                "gold_scenario": chunk_id,
                "gold_state": version_id,
                "distractor_states": distractor_states,
                "gold_state_summary": gold_text[:100],
                "dynamic_top_k": 1,
                "tier": tier,
            }
            all_questions.append(question)

    print(f"\nGenerated: {len(all_questions)} questions")
    specific = [q for q in all_questions if q["question_style"] == "specific"]
    generic = [q for q in all_questions if q["question_style"] == "generic"]
    print(f"  Specific (mutation as gold): {len(specific)}")
    print(f"  Generic (original as gold): {len(generic)}")
    print(f"  Tier A: {sum(1 for q in all_questions if q['tier'] == 'A')}")
    print(f"  Tier B: {sum(1 for q in all_questions if q['tier'] == 'B')}")
    print(f"\n  Localization: {localization_ok} ok, {localization_fail} needed retry, {skipped_rare} skipped (too few rare words)")
    print(f"  Duplicates skipped: {skipped_dup}")

    out_path = DATA_DIR / "benchmark_narrative" / "questions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_questions)} questions to {out_path}")

    print("\nSample questions:")
    for q in all_questions[:10]:
        print(f"  [{q['question_style']:8s} tier={q['tier']}] {q['query_text']}")
        print(f"    gold={q['gold_state']}, distractors={q['distractor_states']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
