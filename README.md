# BiTempQA

BiTempQA is a benchmark project for evaluating memory systems on **bi-temporal / multi-version retrieval and reasoning**.  
The current final artifact in this repository is a **GitHub release-note based benchmark** built from influential open-source projects, together with unified mixed-pool evaluation pipelines for:

- `BM25`
- `FAISS`
- `Mem0`
- `Graphiti`

This README only documents the **final release-note unified dataset and the final evaluation pipeline**. Earlier prototype routes are not part of the recommended workflow.

## Final Deliverables

### 1. Final dataset

The current final dataset is the **formal 300-repository unified release-note dataset**:

- [benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1](benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1)

Key files:

- `official_300_merged.json`
  - merged formal dataset entry point used by full experiments
- `prototype_index_official_300.jsonl`
  - official list of the 300 repository windows
- each `<repo>_release_window/prototype.json`
  - one project sample containing:
    - `chunks`
    - `questions`

### 2. Final scripts

The main scripts for the final pipeline are:

- [benchmark/scripts/77_build_github_release_note_unified_prototype.py](benchmark/scripts/77_build_github_release_note_unified_prototype.py)
  - builds one unified project-level release-note sample
- [benchmark/scripts/78_generate_github_release_unified_formal.py](benchmark/scripts/78_generate_github_release_unified_formal.py)
  - constructs the large formal dataset with incremental persistence
- [benchmark/scripts/79_merge_github_release_unified_formal.py](benchmark/scripts/79_merge_github_release_unified_formal.py)
  - merges the per-project formal samples into one experiment-ready JSON
- [benchmark/scripts/82_run_merged_github_release_unified_global_pool_evaluation.py](benchmark/scripts/82_run_merged_github_release_unified_global_pool_evaluation.py)
  - runs the **global mixed-pool** evaluation, i.e. all project memories are stored first and then queried from one shared memory pool
- [benchmark/scripts/68_run_state_version_evaluation.py](benchmark/scripts/68_run_state_version_evaluation.py)
  - shared evaluation system factory and config entry helpers

### 3. Final configs

The main configs used in the final experiments are:

- [benchmark/configs/state_version_build_config.yaml](benchmark/configs/state_version_build_config.yaml)
- [benchmark/configs/state_version_experiment_config_deepseek_flash_memory.yaml](benchmark/configs/state_version_experiment_config_deepseek_flash_memory.yaml)
- [benchmark/configs/state_version_experiment_config_deepseek_flash_memory_mem0_internal10.yaml](benchmark/configs/state_version_experiment_config_deepseek_flash_memory_mem0_internal10.yaml)

## Final Dataset Format

Each project sample is a unified JSON file:

```json
{
  "prototype_id": "...",
  "repo": "...",
  "window_title": "...",
  "window_summary": "...",
  "chunks": [...],
  "questions": [...]
}
```

Important properties:

- `chunks`
  - memory units derived from recent GitHub release notes
- `questions`
  - three task types:
    - `single_state_lookup`
    - `cross_version_comparison`
    - `temporal_version_ordering`
- each question uses `source_chunk_ids` to indicate the chunks required for answering

## Final Evaluation Setting

### Global mixed-pool setting

The final experiments use a **single mixed memory pool**:

1. store all memories from all selected projects
2. query from the same global pool
3. evaluate answer generation with task-specific `top-k` slicing

This is different from earlier prototype experiments that evaluated one project window at a time.

### Task-specific top-k

Retrieval is done once with `top-10`, then answer generation reuses the retrieved list with task-specific slicing:

- `single_state_lookup`: `k = 1 / 2 / 3`
- `cross_version_comparison`: `k = 2 / 5 / 8`
- `temporal_version_ordering`: `k = 5 / 8 / 10`

### Main reported metrics

The main table metrics are:

- `ACC`
- `Cov`
- `CSR`
- `Latency`

The result files also support additional analysis such as:

- `Zero-Gold`
- `D/G`
- `Context Tokens`
- `Top-1 vs Top-k stability`
- correct answers without retrieving required gold chunks

## Final Result Directories

### Simple baselines

- BM25:
  - [benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_bm25_globalpool_taskk_v1](benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_bm25_globalpool_taskk_v1)
- FAISS:
  - [benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_faiss_globalpool_taskk_v1](benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_faiss_globalpool_taskk_v1)

### Memory systems

- Mem0, `internal_fact_k = 60`:
  - [benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2](benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2)
- Mem0, `internal_fact_k = 10`:
  - [benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10](benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10)
- Graphiti:
  - [benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1](benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1)

## How To Reproduce

### 1. Build / refresh formal dataset

```powershell
python benchmark/scripts/78_generate_github_release_unified_formal.py
python benchmark/scripts/79_merge_github_release_unified_formal.py
```

### 2. Run simple baselines on the global mixed pool

```powershell
python benchmark/scripts/82_run_merged_github_release_unified_global_pool_evaluation.py `
  --config benchmark/configs/state_version_experiment_config_deepseek_flash_memory.yaml `
  --merged-json benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/official_300_merged.json `
  --system bm25 `
  --output-dir benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_bm25_globalpool_taskk_v1
```

Replace `--system bm25` with:

- `faiss`
- `mem0`
- `graphiti`

### 3. Reuse an existing Mem0 mixed pool with `internal_fact_k = 10`

```powershell
python benchmark/scripts/82_run_merged_github_release_unified_global_pool_evaluation.py `
  --config benchmark/configs/state_version_experiment_config_deepseek_flash_memory_mem0_internal10.yaml `
  --merged-json benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/official_300_merged.json `
  --system mem0 `
  --output-dir benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10
```

## Notes

- The final benchmark uses **release-note memory units**, not raw code blocks.
- The final formal experiments should use the **official 300 merged dataset**.
- Large per-question result files may exceed GitHub single-file upload limits; summary files are the most convenient entry point for quick inspection.

## File Guide

For a Chinese file-level guide to the final dataset, result directories, and intermediate files, see:

- [benchmark/docs/FORMAL_RELEASE_UNIFIED_FILE_GUIDE_ZH.md](benchmark/docs/FORMAL_RELEASE_UNIFIED_FILE_GUIDE_ZH.md)

