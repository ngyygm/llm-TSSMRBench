# BiTempQA V3: Temporal Version-Differentiation Benchmark

A benchmark for evaluating retrieval systems' ability to distinguish the correct version of information among near-identical alternatives, across two domains: **real-world GitHub code** and **narrative text with planted state changes**.

## Overview

The benchmark tests a fundamental challenge in retrieval: when multiple versions of a document exist, can a system retrieve the *right* version? Each question targets a specific version of a document, and the system must distinguish it from other versions of the same file (GitHub) or from planted alternative states (narrative).

**Key metrics:**
- **Recall@k** — whether the gold document appears in top-k results
- **MRR** (Mean Reciprocal Rank) — rank of the gold document among scenario-matched results
- **Localization@3** — whether any document from the target scenario appears in top-3

## Dataset

| Domain | Documents | Questions | Scenarios |
|--------|-----------|-----------|-----------|
| GitHub code | 38,746 versioned files (13 repos) | 150 | — |
| Narrative text | 2,191 chunks (5 novels) | 134 | 50 |
| **Total** | **40,937** | **284** | **50** |

### Data Sources

**GitHub**: 13 popular open-source repositories at multiple release versions. Each `(file, version)` pair is a database entry.

**Narrative**: Classic novels (Pride and Prejudice, Emma, Great Expectations, Jane Eyre, Sense and Sensibility, Wuthering Heights) with 50 planted state-change scenarios. Each scenario mutates a paragraph to create an alternative version, and questions ask about details that differ between versions.

### Downloading Data

The full GitHub corpus (`all_versioned_files.jsonl`, ~451MB) is hosted on HuggingFace:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download the large GitHub data file
huggingface-cli download heihei/BiTempQA-v3 \
  benchmark_github/all_versioned_files.jsonl \
  --repo-type dataset \
  --local-dir new_benchmark/data/
```

After download, your `data/` directory should look like:

```
data/
  benchmark_github/
    all_versioned_files.jsonl    ← from HuggingFace (451MB)
    questions_v2.jsonl           ← included in repo (104KB)
  benchmark_narrative/
    novel_chunks.jsonl           ← included in repo (2.1MB)
    questions.jsonl              ← included in repo (61KB)
    selected_chunks.jsonl
    mutated_chunks.jsonl
    pride_and_prejudice.txt
    novels/                      ← additional novel source texts
      emma.txt, great_expectations.txt, jane_eyre.txt, ...
```

### Data Format

**all_versioned_files.jsonl** — one entry per `(file, version)`:
```json
{"repo": "fastapi/fastapi", "version": "v0.100.0", "file_path": "fastapi/routing.py",
 "content": "...", "content_hash": "abc123"}
```

**questions_v2.jsonl** / **questions.jsonl** — one question per line:
```json
{"id": "gh_001", "query_text": "What is the value of X in file Y at version Z?",
 "type": "single_version", "difficulty": "medium",
 "gold_files": [["path/to/file.py", "v1.0"]], "dynamic_top_k": 3}
```

**novel_chunks.jsonl** — original + mutated chunks:
```json
{"text": "...", "type": "original"}
{"text": "...", "type": "mutated", "scenario_id": "sc_001", "state_id": "state_A"}
```

## Evaluation

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single system
python evaluate.py --system bm25
python evaluate.py --system faiss
python evaluate.py --system full_context

# Run all systems
python evaluate.py --system all

# Run with custom parameters
python evaluate.py --system faiss --model all-MiniLM-L6-v2 --chunk-size 512
```

### Supported Systems

| System | Type | External Deps | Source |
|--------|------|---------------|--------|
| **BM25** | Lexical | None | All |
| **TF-IDF** | Lexical | scikit-learn | All |
| **FAISS** | Dense retrieval | sentence-transformers, faiss | All |
| **Hybrid** (BM25+FAISS) | Hybrid | sentence-transformers, faiss | All |
| **Cross-Encoder** | Reranking | sentence-transformers | All |
| **ChromaDB** | Vector DB | chromadb | All |
| **Simple KG** | Knowledge graph | (built-in) | All |
| **Random** | Baseline | None | All |
| **Full Context** | Oracle (upper bound) | None | All |
| **Mem0** | Memory-based | mem0ai, qdrant, LLM API | Narrative only |
| **Graphiti** | Temporal KG | graphiti-core, neo4j, LLM API | Narrative only |

> Mem0 and Graphiti only evaluate narrative questions (134) because they require LLM-powered fact extraction, which is not meaningful for raw code.

### Results

10-system benchmark results (284 total questions):

| System | Recall@k | Precision@k | Narrative MRR | Loc@3 |
|--------|----------|-------------|---------------|-------|
| Full Context | 1.000 | 1.000 | 1.000 | 1.000 |
| BM25 | 0.236 | 0.065 | 0.634 | 0.866 |
| Cross-Encoder | 0.236 | 0.065 | 0.629 | 0.731 |
| TF-IDF | 0.092 | 0.025 | 0.629 | 0.694 |
| Hybrid | 0.218 | 0.059 | 0.616 | 0.373 |
| FAISS | 0.183 | 0.050 | 0.215 | 0.067 |
| ChromaDB | 0.102 | 0.028 | 0.213 | 0.067 |
| Simple KG | 0.081 | 0.022 | 0.428 | 0.045 |
| Mem0 | 0.015 | 0.015 | 0.231 | 0.142 |
| Random | 0.000 | 0.000 | 0.000 | 0.000 |

> Mem0 evaluates 134 narrative questions only; all other systems evaluate all 284 questions.

## Directory Structure

```
new_benchmark/
  evaluate.py                    # Main evaluation pipeline
  README.md                      # This file
  retrievers/
    base.py                      # Retriever ABC + DatabaseEntry + tokenize()
    bm25_retriever.py            # BM25
    tfidf_retriever.py           # TF-IDF
    faiss_retriever.py           # FAISS dense retrieval
    hybrid_retriever.py          # BM25 + FAISS hybrid
    cross_encoder.py             # Cross-encoder reranking
    chroma_retriever.py          # ChromaDB vector store
    full_context.py              # Oracle (all docs returned)
    random_retriever.py          # Random baseline
    simple_kg_retriever.py       # Simple knowledge graph
    mem0_retriever.py            # Mem0 memory system
    graphiti_retriever.py        # Graphiti temporal KG
  scripts/
    batch_download_repos.py      # Download repos at multiple versions
    batch_extract_files.py       # Extract versioned files from repos
    expand_novel_corpus.py       # Download additional novels
    select_mutable_chunks.py     # Select paragraphs for mutation
    generate_mutations.py        # Generate mutated variants
    generate_planted_narrative.py # Build planted narrative dataset
    generate_narrative_questions.py # Generate narrative questions
    generate_version_diff_questions.py # Generate GitHub questions
    validate_benchmark_v2.py     # Validate questions
    run_experiments.py           # Run experiment suite
  data/                          # Benchmark data (see Download section)
  results/                       # Per-system JSON results
```

## Extending with New Retrievers

Implement the `Retriever` interface:

```python
from retrievers import Retriever, DatabaseEntry

class MyRetriever(Retriever):
    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        # Index documents
        ...

    def find(self, query: str, top_k: int = 10) -> list[int]:
        # Return ordered doc_ids by relevance
        ...

    @property
    def supported_sources(self) -> list[str] | None:
        return None  # None = evaluate all question sources
```

Then register in `evaluate.py` and `retrievers/__init__.py`.

## Citation

```bibtex
@inproceedings{bitempqa2026,
  title={BiTempQA: Bitemporal Question Answering Benchmark for Retrieval-Augmented Generation},
  author={...},
  booktitle={ACL},
  year={2026}
}
```
