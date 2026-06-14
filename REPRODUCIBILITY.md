# Reproducibility

This document records what is reproducible from the committed data alone, what
requires an API key or external services, and the exact commands to refresh each
artifact. It was added alongside the research-integrity fixes.

## What reproduces from the committed data (no API needed)

The four retrieval-system results (BM25, FAISS, Mem0 `internal_fact_k=10`,
Graphiti) are stored as compressed per-question logs and reproduce exactly.

```bash
# Regenerate the paper tables + paper_metrics.json from the committed .gz files.
python benchmark/scripts/84_generate_official300_paper_artifacts.py

# Regenerate the mixed-effects regression + system CIs + paired k=2 test.
python paper/scripts/build_task_effect_regression.py

# Deterministic, retrieval-free answer-position baselines (always-A / majority / uniform).
python benchmark/scripts/90_option_position_baselines.py

# Validate the released dataset (support-size invariants, gold-id existence, position skew).
python benchmark/scripts/92_validate_official300.py

# Prove Oracle Cov/CSR = 1.0 by construction on the current dataset.
python benchmark/scripts/93_verify_oracle_coverage.py

# Emit a position-balanced dataset variant for re-evaluation.
python benchmark/scripts/91_shuffle_balance_options.py
```

Confirmed (run during the audit):
- 300 repositories, 9,000 memory nodes, 900 multiple-choice questions (300 per task family).
- The four systems' ACC / Cov / CSR / latency match `paper_metrics.json` and the
  paper prose exactly (e.g. FAISS latency 195.12 ms, Graphiti 34425.76 ms,
  decoupling buckets FAISS all-recalled&correct 73.44%, regression odds ratios
  0.26x cross / 0.11x temporal).
- The saved per-question logs contain exactly the same 900 question ids as the
  released `official_300_merged.json`.

## What requires an API key / external services to refresh

Answer accuracy (ACC) depends on the DeepSeek-V4 answer model and therefore on
API access. The following must be re-run by someone with credentials:

```bash
# 1. Set credentials (see .env.example).
export DEEPSEEK_API_KEY=...
export SILICONFLOW_API_KEY=...

# 2. Refresh Oracle ACC (Oracle raw file is currently missing).
python benchmark/scripts/82_run_merged_github_release_unified_global_pool_evaluation.py \
  --config benchmark/configs/state_version_experiment_config_deepseek_flash_memory.yaml \
  --merged-json benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/official_300_merged.json \
  --system full_context \
  --output-dir benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_simple_baselines_conda_taskk

# 3. Re-evaluate any system on the position-BALANCED dataset to remove the
#    answer-position artifact from ACC.
python benchmark/scripts/82_run_merged_github_release_unified_global_pool_evaluation.py \
  --config benchmark/configs/state_version_experiment_config_deepseek_flash_memory.yaml \
  --merged-json benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/official_300_merged_balanced.json \
  --system faiss \
  --output-dir benchmark/data/prototype_eval_results/balanced_re_eval/faiss
#   (repeat with --system bm25 / mem0 / graphiti / full_context)

# 4. Re-aggregate.
python benchmark/scripts/84_generate_official300_paper_artifacts.py
```

Mem0 additionally requires a local Qdrant instance (`localhost:6333`) and
Graphiti requires a local Neo4j instance (`bolt://localhost:7687`); see the
PowerShell runner scripts under `benchmark/scripts/` for the full chain.

## Model versions

- Answer generation + dataset construction: `deepseek-v4-pro` (DeepSeek API,
  temperature 0, thinking disabled). No public DeepSeek-V4 technical report
  exists at the time of writing; the exact API model strings and endpoint are
  in `benchmark/configs/`.
- Internal memory processing (Mem0, Graphiti): `deepseek-v4-flash`.
- Embeddings: BGE-M3 via a hosted endpoint (`Pro/BAAI/bge-m3`).
- Reranking (Graphiti): BGE Reranker v2 M3.

ACC values are only fully reproducible with the same answer-model snapshot.
Cov/CSR and the deterministic position baselines are reproducible from the
committed data alone.

## Known issues pending refresh

- Oracle ACC row is from an earlier dataset revision (raw file missing);
  rerun step 2 above. Cov/CSR are 1.0 by construction (verified by
  `93_verify_oracle_coverage.py`).
- Two questions violate the support-size invariant (one cross question with
  |G|=1, one single question with |G|=3); flagged by `92_validate_official300.py`
  for regeneration.
- Dependency versions are lower-bounded (`>=`) in the requirements files, not
  pinned; Mem0/Graphiti behavior can drift across versions. Pin exact versions
  before a formal reproduction.
