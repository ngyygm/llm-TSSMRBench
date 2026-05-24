"""Evaluation pipeline for the temporal version-differentiation benchmark.

Loads real data (GitHub versioned files + novel chunks with planted state
paragraphs) into a retrieval system, runs queries, and measures retrieval
accuracy at chunk level.

Usage:
    python evaluate.py --system bm25
    python evaluate.py --system faiss
    python evaluate.py --system full_context
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retrievers import (
    BM25Retriever,
    ChromaDBRetriever,
    CrossEncoderRetriever,
    DatabaseEntry,
    FAISSRetriever,
    FullContextRetriever,
    GraphitiRetriever,
    HybridRetriever,
    Mem0Retriever,
    RandomRetriever,
    Retriever,
    SimpleKGRetriever,
    TFIDFRetriever,
)


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


# ─── Data structures ───


@dataclass
class QueryResult:
    answer: str = ""
    retrieved_context: str = ""
    retrieved_indices: list[int] = field(default_factory=list)
    retrieved_texts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0


@dataclass
class Question:
    qid: str
    query_text: str
    qtype: str  # single_version, multi_version, insufficient_information
    difficulty: str
    gold_doc_ids: list[int] = field(default_factory=list)
    gold_state_summary: str = ""
    dynamic_top_k: int = 5
    scenario_id: str = ""
    question_style: str = ""
    scenario_doc_ids: list[int] = field(default_factory=list)
    question_source: str = ""  # "github" or "narrative"


@dataclass
class EvalResult:
    qid: str
    query_text: str
    qtype: str
    difficulty: str
    gold_count: int
    top_k: int
    retrieved_gold_count: int
    recall: float
    precision_at_k: float
    latency_ms: float
    mrr: float = 0.0
    localized: bool = False
    scenario_id: str = ""
    question_style: str = ""


# ─── Database loading ───


def load_github_entries() -> list[DatabaseEntry]:
    """Load all (file, version) pairs as database entries."""
    entries = []
    path = DATA_DIR / "benchmark_github" / "all_versioned_files.jsonl"
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            entries.append(DatabaseEntry(
                doc_id=i,
                source="github",
                text=f"[{r['repo']} @ {r['version']}] {r['file_path']}\n{r['content']}",
                metadata={
                    "repo": r["repo"],
                    "version": r["version"],
                    "file_path": r["file_path"],
                    "content_hash": r["content_hash"],
                },
            ))
    return entries


def load_narrative_entries(github_count: int = 0) -> list[DatabaseEntry]:
    """Load novel chunks (original + planted) as database entries."""
    entries = []
    path = DATA_DIR / "benchmark_narrative" / "novel_chunks.jsonl"
    if not path.exists():
        return entries

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            chunk_type = r.get("type", "original")
            text = f"[Pride and Prejudice]\n{r['text']}"

            metadata: dict[str, Any] = {"chunk_type": chunk_type}
            if chunk_type in ("planted", "mutated"):
                metadata["scenario_id"] = r["scenario_id"]
                metadata["state_id"] = r["state_id"]

            entries.append(DatabaseEntry(
                doc_id=github_count + i,
                source="narrative",
                text=text,
                metadata=metadata,
            ))
    return entries


def load_questions(
    entries: list[DatabaseEntry],
) -> list[Question]:
    """Load questions and resolve gold evidence to doc_ids."""
    github_lookup: dict[tuple[str, str], int] = {}
    planted_lookup: dict[tuple[str, str], int] = {}
    for e in entries:
        if e.source == "github":
            key = (e.metadata["file_path"], e.metadata["version"])
            github_lookup[key] = e.doc_id
        elif e.source == "narrative" and e.metadata.get("chunk_type") in ("planted", "mutated"):
            key = (e.metadata["scenario_id"], e.metadata["state_id"])
            planted_lookup[key] = e.doc_id

    questions = []

    # GitHub questions
    qpath = DATA_DIR / "benchmark_github" / "questions_v2.jsonl"
    if qpath.exists():
        with open(qpath, encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                gold_ids = []
                for fpath, ver in q.get("gold_files", []):
                    key = (fpath, ver)
                    if key in github_lookup:
                        gold_ids.append(github_lookup[key])
                questions.append(Question(
                    qid=q["id"],
                    query_text=q["query_text"],
                    qtype=q["type"],
                    difficulty=q["difficulty"],
                    gold_doc_ids=gold_ids,
                    gold_state_summary=q.get("gold_state_summary", ""),
                    dynamic_top_k=q.get("dynamic_top_k", 5),
                    question_source="github",
                ))

    # Narrative questions
    qpath = DATA_DIR / "benchmark_narrative" / "questions.jsonl"
    if qpath.exists():
        with open(qpath, encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                gold_ids = []
                gold_scenario = q.get("gold_scenario", "")
                gold_state = q.get("gold_state", "")
                key = (gold_scenario, gold_state)
                if key in planted_lookup:
                    gold_ids.append(planted_lookup[key])

                distractor_states = q.get("distractor_states", [])
                scenario_doc_ids = []
                for sid in [gold_state] + distractor_states:
                    skey = (gold_scenario, sid)
                    if skey in planted_lookup:
                        scenario_doc_ids.append(planted_lookup[skey])

                questions.append(Question(
                    qid=q["id"],
                    query_text=q["query_text"],
                    qtype=q["type"],
                    difficulty=q["difficulty"],
                    gold_doc_ids=gold_ids,
                    gold_state_summary=q.get("gold_state_summary", ""),
                    dynamic_top_k=q.get("dynamic_top_k", 5),
                    scenario_id=gold_scenario,
                    question_style=q.get("question_style", ""),
                    scenario_doc_ids=scenario_doc_ids,
                    question_source="narrative",
                ))

    return questions


# ─── Unified evaluation ───


def evaluate(
    retriever: Retriever,
    questions: list[Question],
) -> list[EvalResult]:
    """Run retrieval evaluation with any Retriever implementation."""
    # Filter questions by supported sources
    sources = retriever.supported_sources
    if sources is not None:
        questions = [q for q in questions if q.question_source in sources]
        if not questions:
            print("  No questions match supported sources, skipping evaluation.")
            return []

    results = []
    for q in questions:
        start = time.time()
        probe_k = max(q.dynamic_top_k, 100)
        retrieved_doc_ids = retriever.find(q.query_text, top_k=probe_k)
        latency = (time.time() - start) * 1000

        # For ranked retrievers: only count top-k. For unranked (FullContext): use all.
        effective_k = q.dynamic_top_k if retriever.is_ranked else len(retrieved_doc_ids)
        top_k_ids = set(retrieved_doc_ids[:effective_k])
        gold_set = set(q.gold_doc_ids)

        gold_retrieved = len(gold_set & top_k_ids)
        gold_count = len(gold_set)
        recall = gold_retrieved / gold_count if gold_count > 0 else 0.0
        precision = gold_retrieved / effective_k if effective_k > 0 else 0.0

        # MRR + localization (narrative only)
        mrr = 0.0
        localized = False
        if q.scenario_doc_ids:
            if retriever.is_ranked:
                scenario_set = set(q.scenario_doc_ids)
                scenario_ranked = [d for d in retrieved_doc_ids if d in scenario_set]
                for rank, doc_id in enumerate(scenario_ranked, 1):
                    if doc_id in gold_set:
                        mrr = 1.0 / rank
                        break
                top3 = set(retrieved_doc_ids[:3])
                localized = bool(scenario_set & top3)
            else:
                # Unranked (FullContext): trivially finds everything
                mrr = 1.0 if gold_count > 0 else 0.0
                localized = True

        results.append(EvalResult(
            qid=q.qid,
            query_text=q.query_text,
            qtype=q.qtype,
            difficulty=q.difficulty,
            gold_count=gold_count,
            top_k=effective_k,
            retrieved_gold_count=gold_retrieved,
            recall=recall,
            precision_at_k=precision,
            latency_ms=latency,
            mrr=mrr,
            localized=localized,
            scenario_id=q.scenario_id,
            question_style=q.question_style,
        ))

    return results


# ─── Reporting ───


def print_report(results: list[EvalResult], system_name: str) -> None:
    """Print evaluation summary."""
    total = len(results)
    if total == 0:
        print(f"No results for {system_name}")
        return

    avg_recall = sum(r.recall for r in results) / total
    avg_precision = sum(r.precision_at_k for r in results) / total
    avg_latency = sum(r.latency_ms for r in results) / total

    narrative = [r for r in results if r.scenario_id]
    avg_mrr = sum(r.mrr for r in narrative) / len(narrative) if narrative else 0.0

    by_type: dict[str, list[EvalResult]] = {}
    for r in results:
        by_type.setdefault(r.qtype, []).append(r)

    by_diff: dict[str, list[EvalResult]] = {}
    for r in results:
        by_diff.setdefault(r.difficulty, []).append(r)

    github_results = [r for r in results if not r.scenario_id]
    narrative_results = [r for r in results if r.scenario_id]

    print(f"\n{'='*60}")
    print(f"  System: {system_name}")
    print(f"  Total questions: {total}")
    print(f"  Average Recall@k: {avg_recall:.3f}")
    print(f"  Average Precision@k: {avg_precision:.3f}")
    if narrative:
        print(f"  Narrative MRR (in-group): {avg_mrr:.3f}")
    print(f"  Average latency: {avg_latency:.1f} ms")
    print(f"{'='*60}")

    for qtype, group in sorted(by_type.items()):
        r = sum(x.recall for x in group) / len(group)
        print(f"  {qtype:30s}: recall={r:.3f}  (n={len(group)})")

    print()
    for diff, group in sorted(by_diff.items()):
        r = sum(x.recall for x in group) / len(group)
        print(f"  {diff:30s}: recall={r:.3f}  (n={len(group)})")

    if github_results:
        r = sum(x.recall for x in github_results) / len(github_results)
        print(f"\n  {'GitHub questions':30s}: recall={r:.3f}  (n={len(github_results)})")
    if narrative_results:
        r = sum(x.recall for x in narrative_results) / len(narrative_results)
        m = sum(x.mrr for x in narrative_results) / len(narrative_results)
        loc = sum(1 for x in narrative_results if x.localized) / len(narrative_results)
        print(f"  {'Narrative questions':30s}: recall={r:.3f}, MRR={m:.3f}, Loc@3={loc:.3f}  (n={len(narrative_results)})")

    # Two-stage breakdown (narrative only)
    if narrative_results:
        loc_n = [r for r in narrative_results if r.localized]
        not_loc_n = [r for r in narrative_results if not r.localized]
        print(f"\n  --- Two-Stage Analysis (Narrative) ---")
        print(f"  Localization@3:             {len(loc_n)}/{len(narrative_results)} ({len(loc_n)/len(narrative_results)*100:.1f}%)")
        if loc_n:
            ver_recall = sum(r.recall for r in loc_n) / len(loc_n)
            ver_mrr = sum(r.mrr for r in loc_n) / len(loc_n)
            print(f"  Version Recall@1 (given loc): {ver_recall:.3f}  (n={len(loc_n)})")
            print(f"  Version MRR (given loc):      {ver_mrr:.3f}  (n={len(loc_n)})")
        if not_loc_n:
            print(f"  Failed to localize:          {len(not_loc_n)}/{len(narrative_results)} ({len(not_loc_n)/len(narrative_results)*100:.1f}%)")

    # By question style (narrative only)
    by_style: dict[str, list[EvalResult]] = {}
    for r in narrative_results:
        by_style.setdefault(r.question_style, []).append(r)
    if len(by_style) > 1:
        print()
        for style, group in sorted(by_style.items()):
            r = sum(x.recall for x in group) / len(group)
            m = sum(x.mrr for x in group) / len(group)
            print(f"  {style:30s}: recall={r:.3f}, MRR={m:.3f}  (n={len(group)})")

    # Show some failure cases
    failures = [r for r in results if r.recall < 1.0 and r.gold_count > 0]
    if failures:
        print(f"\n  Failure cases ({len(failures)}):")
        for f in failures[:5]:
            style_tag = f" [{f.question_style}]" if f.question_style else ""
            print(f"    [{f.qtype}{style_tag}] recall={f.recall:.2f}, MRR={f.mrr:.2f}"
                  f" (found {f.retrieved_gold_count}/{f.gold_count} gold, top_k={f.top_k})")
            print(f"      Q: {f.query_text[:100]}...")


def main():
    parser = argparse.ArgumentParser(description="New benchmark evaluation")
    parser.add_argument("--system",
                        choices=["bm25", "faiss", "full_context", "tfidf", "random", "hybrid",
                                 "cross_encoder", "chroma", "simple_kg", "mem0", "graphiti", "all"],
                        default="all")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in words for FAISS")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap in words for FAISS")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence-transformer model")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    github_entries = load_github_entries()
    narrative_entries = load_narrative_entries(github_count=len(github_entries))
    entries = github_entries + narrative_entries
    print(f"  Database: {len(entries)} entries")
    print(f"    GitHub: {sum(1 for e in entries if e.source == 'github')}")
    print(f"    Narrative: {sum(1 for e in entries if e.source == 'narrative')}")

    total_chars = sum(len(e.text) for e in entries)
    print(f"    Total chars: {total_chars:,}")
    print(f"    Total words: ~{total_chars // 5:,}")

    # Load questions
    questions = load_questions(entries)
    print(f"\n  Questions: {len(questions)}")
    by_type = {}
    for q in questions:
        by_type.setdefault(q.qtype, []).append(q)
    for qt, qs in sorted(by_type.items()):
        print(f"    {qt}: {len(qs)}")
    print(f"    Questions with gold evidence: {sum(1 for q in questions if q.gold_doc_ids)}")

    # Build retrievers
    system_configs = []
    if args.system in ("bm25", "all"):
        system_configs.append(("BM25", BM25Retriever()))
    if args.system in ("tfidf", "all"):
        system_configs.append(("TF-IDF", TFIDFRetriever()))
    if args.system in ("random", "all"):
        system_configs.append(("Random", RandomRetriever()))
    if args.system in ("faiss", "all"):
        system_configs.append(("FAISS", FAISSRetriever(
            model_name=args.model, chunk_size=args.chunk_size, overlap=args.chunk_overlap,
        )))
    if args.system in ("hybrid", "all"):
        system_configs.append(("Hybrid", HybridRetriever(
            model_name=args.model, chunk_size=args.chunk_size, overlap=args.chunk_overlap,
        )))
    if args.system in ("cross_encoder", "all"):
        system_configs.append(("Cross-Encoder", CrossEncoderRetriever()))
    if args.system in ("chroma", "all"):
        system_configs.append(("ChromaDB", ChromaDBRetriever()))
    if args.system in ("simple_kg", "all"):
        system_configs.append(("Simple KG", SimpleKGRetriever()))
    if args.system in ("mem0",):
        system_configs.append(("Mem0", Mem0Retriever()))
    if args.system in ("graphiti",):
        system_configs.append(("Graphiti", GraphitiRetriever()))
    if args.system in ("full_context", "all"):
        system_configs.append(("Full Context", FullContextRetriever()))

    output_dir = BASE_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    for name, retriever in system_configs:
        print(f"\n--- Building {name} index ---")
        retriever.add_documents(entries)

        print(f"\n--- Evaluating {name} ---")
        results = evaluate(retriever, questions)
        print_report(results, name)

        out_path = output_dir / f"{name.lower().replace(' ', '_')}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"qid": r.qid, "recall": r.recall, "precision": r.precision_at_k,
                  "gold_count": r.gold_count, "retrieved_gold": r.retrieved_gold_count,
                  "top_k": r.top_k, "latency_ms": r.latency_ms,
                  "mrr": r.mrr, "localized": r.localized,
                  "scenario_id": r.scenario_id,
                  "question_style": r.question_style}
                 for r in results],
                f, indent=2, ensure_ascii=False,
            )
        print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
